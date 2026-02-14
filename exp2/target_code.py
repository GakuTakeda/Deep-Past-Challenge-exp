# Import standard libraries
import os

# Set environment variables before importing PyTorch
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# ---------------------------
# 2) Imports & Setup
# ---------------------------

# Import standard libraries
import re
import json
import random
import logging
import warnings
from pathlib import Path
from dataclasses import dataclass
from typing import List

# Import third-party libraries
import pandas as pd
import numpy as np

# Import PyTorch
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.cuda.amp import autocast

# Import Transformers
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Import progress bar
from tqdm.auto import tqdm

# Suppress warnings
warnings.filterwarnings("ignore")


# ---------------------------
# 3) Configuration
# ---------------------------

@dataclass
class UltraConfig:
    # ============ PATHS ============
    test_data_path: str = "/kaggle/input/deep-past-initiative-machine-translation/test.csv"
    model_path: str = "/kaggle/input/final-byt5/byt5-akkadian-optimized-34x"
    output_dir: str = "/kaggle/working/"

    # ============ PROCESSING ============
    max_length: int = 512
    batch_size: int = 8
    num_workers: int = 4

    # ============ GENERATION ============
    num_beams: int = 8
    max_new_tokens: int = 512
    length_penalty: float = 1.3
    early_stopping: bool = True
    no_repeat_ngram_size: int = 0

    # ============ OPTIMIZATIONS ============
    use_mixed_precision: bool = True
    use_better_transformer: bool = True
    use_bucket_batching: bool = True
    use_vectorized_postproc: bool = True
    use_adaptive_beams: bool = True
    use_auto_batch_size: bool = False

    # ============ OTHER ============
    aggressive_postprocessing: bool = True
    checkpoint_freq: int = 100
    num_buckets: int = 4

    def __post_init__(self):
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create output directory
        Path(self.output_dir).mkdir(exist_ok=True, parents=True)

        # Disable CUDA-dependent optimizations if no GPU
        if not torch.cuda.is_available():
            self.use_mixed_precision = False
            self.use_better_transformer = False


# ---------------------------
# 4) Logging Setup
# ---------------------------

def setup_logging(output_dir: str) -> logging.Logger:
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    # Define log file path
    log_file = Path(output_dir) / "inference_ultra.log"

    # Remove existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file),
        ],
    )

    # Return logger
    return logging.getLogger(__name__)


# ---------------------------
# 5) Optimized Text Preprocessor
# ---------------------------

class OptimizedPreprocessor:
    # Initialize precompiled regex patterns
    def __init__(self):
        # Pre-compile patterns
        self.patterns = {
            "big_gap": re.compile(r"(\.{3,}|…+|……)"),
            "small_gap": re.compile(r"(xx+|\s+x\s+)"),
        }

    # Preprocess a single input text
    def preprocess_input_text(self, text: str) -> str:
        # Handle NaN
        if pd.isna(text):
            return ""

        # Convert to string
        cleaned_text = str(text)

        # Replace gaps
        cleaned_text = self.patterns["big_gap"].sub("<big_gap>", cleaned_text)
        cleaned_text = self.patterns["small_gap"].sub("<gap>", cleaned_text)

        return cleaned_text

    # Preprocess a batch of texts using vectorized pandas ops
    def preprocess_batch(self, texts: List[str]) -> List[str]:
        # Convert to Series and sanitize
        s = pd.Series(texts).fillna("")
        s = s.astype(str)

        # Apply vectorized replacements
        s = s.str.replace(self.patterns["big_gap"], "<big_gap>", regex=True)
        s = s.str.replace(self.patterns["small_gap"], "<gap>", regex=True)

        return s.tolist()


# ---------------------------
# 6) Vectorized Postprocessor
# ---------------------------

class VectorizedPostprocessor:
    # Initialize postprocessor patterns and translation tables
    def __init__(self, aggressive: bool = True):
        # Store mode
        self.aggressive = aggressive

        # Pre-compile patterns
        self.patterns = {
            "gap": re.compile(r"(\[x\]|\(x\)|\bx\b)", re.I),
            "big_gap": re.compile(r"(\.{3,}|…|\[\.+\])"),
            "annotations": re.compile(r"\((fem|plur|pl|sing|singular|plural|\?|!)\..\s*\w*\)", re.I),
            "repeated_words": re.compile(r"\b(\w+)(?:\s+\1\b)+"),
            "whitespace": re.compile(r"\s+"),
            "punct_space": re.compile(r"\s+([.,:])"),
            "repeated_punct": re.compile(r"([.,])\1+"),
        }

        # Create character translation tables
        self.subscript_trans = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
        self.special_chars_trans = str.maketrans("ḫḪ", "hH")

        # Define forbidden characters
        self.forbidden_chars = '!?()"——<>⌈⌋⌊[]+ʾ/;'

        # Create forbidden translate table
        self.forbidden_trans = str.maketrans("", "", self.forbidden_chars)

    # Vectorized postprocessing
    def postprocess_batch(self, translations: List[str]) -> List[str]:
        # Convert to Series
        s = pd.Series(translations)

        # Validate entries
        valid_mask = s.apply(lambda x: isinstance(x, str) and x.strip())
        if not valid_mask.all():
            s[~valid_mask] = ""

        # Basic cleaning
        s = s.str.translate(self.special_chars_trans)
        s = s.str.translate(self.subscript_trans)
        s = s.str.replace(self.patterns["whitespace"], " ", regex=True)
        s = s.str.strip()

        # Aggressive postprocessing
        if self.aggressive:
            # Normalize gaps
            s = s.str.replace(self.patterns["gap"], "<gap>", regex=True)
            s = s.str.replace(self.patterns["big_gap"], "<big_gap>", regex=True)

            # Merge adjacent gaps
            s = s.str.replace("<gap> <gap>", "<big_gap>", regex=False)
            s = s.str.replace("<big_gap> <big_gap>", "<big_gap>", regex=False)

            # Remove annotations
            s = s.str.replace(self.patterns["annotations"], "", regex=True)

            # Protect gaps during forbidden-character removal
            s = s.str.replace("<gap>", "\x00GAP\x00", regex=False)
            s = s.str.replace("<big_gap>", "\x00BIG\x00", regex=False)

            # Remove forbidden characters
            s = s.str.translate(self.forbidden_trans)

            # Restore gaps
            s = s.str.replace("\x00GAP\x00", " <gap> ", regex=False)
            s = s.str.replace("\x00BIG\x00", " <big_gap> ", regex=False)

            # Fractions
            s = s.str.replace(r"(\d+)\.5\b", r"\1½", regex=True)
            s = s.str.replace(r"\b0\.5\b", "½", regex=True)
            s = s.str.replace(r"(\d+)\.25\b", r"\1¼", regex=True)
            s = s.str.replace(r"\b0\.25\b", "¼", regex=True)
            s = s.str.replace(r"(\d+)\.75\b", r"\1¾", regex=True)
            s = s.str.replace(r"\b0\.75\b", "¾", regex=True)

            # Remove repeated words
            s = s.str.replace(self.patterns["repeated_words"], r"\1", regex=True)

            # Remove repeated n-grams (4 -> 2)
            for n in range(4, 1, -1):
                pattern = r"\b((?:\w+\s+){" + str(n - 1) + r"}\w+)(?:\s+\1\b)+"
                s = s.str.replace(pattern, r"\1", regex=True)

            # Fix punctuation spacing and repeats
            s = s.str.replace(self.patterns["punct_space"], r"\1", regex=True)
            s = s.str.replace(self.patterns["repeated_punct"], r"\1", regex=True)

            # Final whitespace cleanup
            s = s.str.replace(self.patterns["whitespace"], " ", regex=True)
            s = s.str.strip().str.strip("-").str.strip()

        return s.tolist()


# ---------------------------
# 7) Bucket Batch Sampler
# ---------------------------

class BucketBatchSampler(Sampler):
    # Initialize sampler
    def __init__(self, dataset: Dataset, batch_size: int, num_buckets: int, logger: logging.Logger, shuffle: bool = False):
        # Store settings
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.logger = logger

        # Compute lengths for bucketing
        lengths = [len(text.split()) for _, text in dataset]

        # Sort indices by length
        sorted_indices = sorted(range(len(lengths)), key=lambda i: lengths[i])

        # Create buckets
        bucket_size = max(1, len(sorted_indices) // max(1, num_buckets))
        self.buckets = []

        for i in range(num_buckets):
            start = i * bucket_size
            end = None if i == num_buckets - 1 else (i + 1) * bucket_size
            self.buckets.append(sorted_indices[start:end])

        # Log bucket details
        self.logger.info(f"Created {num_buckets} buckets:")
        for i, bucket in enumerate(self.buckets):
            bucket_lengths = [lengths[idx] for idx in bucket] if len(bucket) > 0 else [0]
            self.logger.info(
                f"  Bucket {i}: {len(bucket)} samples, length range [{min(bucket_lengths)}, {max(bucket_lengths)}]"
            )

    # Yield batches
    def __iter__(self):
        for bucket in self.buckets:
            if self.shuffle:
                random.shuffle(bucket)

            for i in range(0, len(bucket), self.batch_size):
                yield bucket[i : i + self.batch_size]

    # Return number of batches
    def __len__(self):
        total = 0
        for bucket in self.buckets:
            total += (len(bucket) + self.batch_size - 1) // self.batch_size
        return total


# ---------------------------
# 8) Dataset
# ---------------------------

class AkkadianDataset(Dataset):
    # Initialize dataset
    def __init__(self, dataframe: pd.DataFrame, preprocessor: OptimizedPreprocessor, logger: logging.Logger):
        # Store ids
        self.sample_ids = dataframe["id"].tolist()

        # Preprocess in batch
        raw_texts = dataframe["transliteration"].tolist()
        preprocessed = preprocessor.preprocess_batch(raw_texts)

        # Add task prefix
        self.input_texts = ["translate Akkadian to English: " + text for text in preprocessed]

        # Log dataset info
        logger.info(f"Dataset created with {len(self.sample_ids)} samples")

    # Return size
    def __len__(self):
        return len(self.sample_ids)

    # Return item
    def __getitem__(self, index: int):
        return self.sample_ids[index], self.input_texts[index]


# ---------------------------
# 9) Ultra-Optimized Inference Engine
# ---------------------------

class UltraInferenceEngine:
    # Initialize engine
    def __init__(self, config: UltraConfig, logger: logging.Logger):
        # Store config and logger
        self.config = config
        self.logger = logger

        # Create helpers
        self.preprocessor = OptimizedPreprocessor()
        self.postprocessor = VectorizedPostprocessor(aggressive=config.aggressive_postprocessing)

        # Initialize results
        self.results = []

        # Load model and tokenizer
        self._load_model()

    # Load and optimize model
    def _load_model(self):
        # Log model load
        self.logger.info(f"Loading model from {self.config.model_path}")

        # Load model
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_path)
        self.model = self.model.to(self.config.device)
        self.model = self.model.eval()

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)

        # Log parameter count
        num_params = sum(p.numel() for p in self.model.parameters())
        self.logger.info(f"Model loaded: {num_params:,} parameters")

        # Apply BetterTransformer if enabled
        if self.config.use_better_transformer and torch.cuda.is_available():
            try:
                # Import optimum lazily
                from optimum.bettertransformer import BetterTransformer

                # Apply transformation
                self.logger.info("Applying BetterTransformer...")
                self.model = BetterTransformer.transform(self.model)
                self.logger.info("✅ BetterTransformer applied (20-50% speedup)")
            except ImportError:
                self.logger.warning("⚠️  'optimum' not installed, skipping BetterTransformer")
                self.logger.warning("   Install with: !pip install optimum")
            except Exception as exc:
                self.logger.warning(f"⚠️  BetterTransformer failed: {exc}")

    # Collate function for DataLoader
    def _collate_fn(self, batch_samples):
        # Extract ids and texts
        batch_ids = [s[0] for s in batch_samples]
        batch_texts = [s[1] for s in batch_samples]

        # Tokenize
        tokenized = self.tokenizer(
            batch_texts,
            max_length=self.config.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        return batch_ids, tokenized

    # Adaptive beam selection
    def _get_adaptive_beam_size(self, attention_mask: torch.Tensor) -> int:
        # Return fixed beams if disabled
        if not self.config.use_adaptive_beams:
            return self.config.num_beams

        # Compute lengths
        lengths = attention_mask.sum(dim=1)

        # Choose beams based on length
        short_beams = max(4, self.config.num_beams // 2)
        beam_sizes = torch.where(
            lengths < 100,
            torch.tensor(short_beams, device=lengths.device),
            torch.tensor(self.config.num_beams, device=lengths.device),
        )

        # Use first element (batch-wise adaptive is possible, but keep simple/fast)
        return int(beam_sizes[0].item())

    # Save periodic checkpoints
    def _save_checkpoint(self):
        # Checkpoint only when frequency matches
        if len(self.results) > 0 and len(self.results) % self.config.checkpoint_freq == 0:
            checkpoint_path = Path(self.config.output_dir) / f"checkpoint_{len(self.results)}.csv"

            # Create DataFrame
            df = pd.DataFrame(self.results, columns=["id", "translation"])

            # Save
            df.to_csv(checkpoint_path, index=False)

            # Log
            self.logger.info(f"💾 Checkpoint: {len(self.results)} translations")

    # Optional: auto-tune batch size
    def find_optimal_batch_size(self, dataset: Dataset, start_bs: int = 32) -> int:
        # Log
        self.logger.info("🔍 Finding optimal batch size...")

        # Initialize binary search
        max_bs = start_bs
        min_bs = 1

        # Binary search loop
        while max_bs - min_bs > 1:
            # Choose midpoint
            test_bs = (max_bs + min_bs) // 2

            try:
                # Build test batch
                test_batch = [dataset[i] for i in range(min(test_bs, len(dataset)))]

                # Collate
                _, inputs = self._collate_fn(test_batch)

                # Run a tiny generation
                with torch.inference_mode():
                    if self.config.use_mixed_precision:
                        with autocast():
                            _ = self.model.generate(
                                input_ids=inputs.input_ids.to(self.config.device),
                                attention_mask=inputs.attention_mask.to(self.config.device),
                                num_beams=self.config.num_beams,
                                max_new_tokens=64,
                                use_cache=True,
                            )
                    else:
                        _ = self.model.generate(
                            input_ids=inputs.input_ids.to(self.config.device),
                            attention_mask=inputs.attention_mask.to(self.config.device),
                            num_beams=self.config.num_beams,
                            max_new_tokens=64,
                            use_cache=True,
                        )

                # Update min bound
                min_bs = test_bs
                self.logger.info(f"  ✅ Batch size {test_bs} works")

                # Cleanup
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except RuntimeError as exc:
                # Handle OOM
                if "out of memory" in str(exc).lower():
                    max_bs = test_bs
                    self.logger.info(f"  ❌ Batch size {test_bs} OOM")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                else:
                    raise

        # Log result
        self.logger.info(f"🎯 Optimal batch size: {min_bs}")
        return min_bs

    # Run inference end-to-end
    def run_inference(self, test_df: pd.DataFrame) -> pd.DataFrame:
        # Log start
        self.logger.info("🚀 Starting ULTRA-OPTIMIZED inference")

        # Create dataset
        dataset = AkkadianDataset(test_df, self.preprocessor, self.logger)

        # Auto-tune batch size if enabled
        if self.config.use_auto_batch_size:
            optimal_bs = self.find_optimal_batch_size(dataset)
            self.config.batch_size = optimal_bs

        # Create DataLoader
        if self.config.use_bucket_batching:
            batch_sampler = BucketBatchSampler(
                dataset=dataset,
                batch_size=self.config.batch_size,
                num_buckets=self.config.num_buckets,
                logger=self.logger,
                shuffle=False,
            )

            dataloader = DataLoader(
                dataset,
                batch_sampler=batch_sampler,
                num_workers=self.config.num_workers,
                collate_fn=self._collate_fn,
                pin_memory=True,
                prefetch_factor=2,
                persistent_workers=True if self.config.num_workers > 0 else False,
            )
        else:
            dataloader = DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
                collate_fn=self._collate_fn,
                pin_memory=True,
                prefetch_factor=2,
                persistent_workers=True if self.config.num_workers > 0 else False,
            )

        # Log dataloader and optimizations
        self.logger.info(f"DataLoader created: {len(dataloader)} batches")
        self.logger.info("Active optimizations:")
        self.logger.info(f"  ✅ Mixed Precision: {self.config.use_mixed_precision}")
        self.logger.info(f"  ✅ BetterTransformer: {self.config.use_better_transformer}")
        self.logger.info(f"  ✅ Bucket Batching: {self.config.use_bucket_batching}")
        self.logger.info(f"  ✅ Vectorized Postproc: {self.config.use_vectorized_postproc}")
        self.logger.info(f"  ✅ Adaptive Beams: {self.config.use_adaptive_beams}")

        # Build base generation config
        base_gen_config = {
            "max_new_tokens": self.config.max_new_tokens,
            "length_penalty": self.config.length_penalty,
            "early_stopping": self.config.early_stopping,
            "use_cache": True,
        }

        # Add no-repeat constraint if requested
        if self.config.no_repeat_ngram_size > 0:
            base_gen_config["no_repeat_ngram_size"] = self.config.no_repeat_ngram_size

        # Reset results
        self.results = []

        # Inference loop
        with torch.inference_mode():
            for batch_idx, (batch_ids, tokenized) in enumerate(tqdm(dataloader, desc="🚀 Translating")):
                try:
                    # Move inputs
                    input_ids = tokenized.input_ids.to(self.config.device)
                    attention_mask = tokenized.attention_mask.to(self.config.device)

                    # Adaptive beams
                    beam_size = self._get_adaptive_beam_size(attention_mask)

                    # Merge gen config
                    gen_config = dict(base_gen_config)
                    gen_config["num_beams"] = beam_size

                    # Generate
                    if self.config.use_mixed_precision:
                        with autocast():
                            outputs = self.model.generate(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                **gen_config,
                            )
                    else:
                        outputs = self.model.generate(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            **gen_config,
                        )

                    # Decode
                    translations = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

                    # Postprocess
                    if self.config.use_vectorized_postproc:
                        cleaned = self.postprocessor.postprocess_batch(translations)
                    else:
                        cleaned = [self.postprocessor.postprocess_batch([t])[0] for t in translations]

                    # Store
                    self.results.extend(list(zip(batch_ids, cleaned)))

                    # Save checkpoint
                    self._save_checkpoint()

                    # Periodic cache clearing
                    if torch.cuda.is_available() and batch_idx % 10 == 0:
                        torch.cuda.empty_cache()

                except Exception as exc:
                    # Log error
                    self.logger.error(f"❌ Batch {batch_idx} error: {exc}")

                    # Fill empties for this batch
                    self.results.extend([(bid, "") for bid in batch_ids])

        # Log completion
        self.logger.info("✅ Inference completed")

        # Build results DataFrame
        results_df = pd.DataFrame(self.results, columns=["id", "translation"])

        # Validate
        self._validate_results(results_df)

        return results_df

    # Print validation report
    def _validate_results(self, df: pd.DataFrame):
        # Header
        print("\n" + "=" * 60)
        print("📊 VALIDATION REPORT")
        print("=" * 60)

        # Empty count
        empty = df["translation"].astype(str).str.strip().eq("").sum()
        print(f"\nEmpty: {empty} ({(empty / max(1, len(df))) * 100:.2f}%)")

        # Length stats
        lengths = df["translation"].astype(str).str.len()
        print("\n📏 Length stats:")
        print(f"   Mean: {lengths.mean():.1f}, Median: {lengths.median():.1f}")
        print(f"   Min: {lengths.min()}, Max: {lengths.max()}")

        # Very short translations
        short = ((lengths < 5) & (lengths > 0)).sum()
        if short > 0:
            print(f"   ⚠️  {short} very short translations")

        # Samples
        print("\n📝 Sample translations:")

        # Choose indices robustly
        sample_indices = [0]
        if len(df) > 2:
            sample_indices.append(len(df) // 2)
        if len(df) > 1:
            sample_indices.append(len(df) - 1)

        for idx in sample_indices:
            row = df.iloc[idx]
            text = str(row["translation"])
            preview = text[:70] + "..." if len(text) > 70 else text
            print(f"   ID {int(row['id']):4d}: {preview}")

        print("\n" + "=" * 60 + "\n")


# ---------------------------
# 10) IO Helpers
# ---------------------------

def print_environment_info():
    # Print PyTorch and CUDA info
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        total_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU Memory: {total_mem_gb:.2f} GB")

    # Verify optimum availability (optional)
    try:
        from optimum.bettertransformer import BetterTransformer  # noqa: F401

        print("✅ BetterTransformer available!")
    except ImportError:
        print("❌ BetterTransformer NOT available")


def save_outputs(
    results_df: pd.DataFrame,
    config: UltraConfig,
    output_dir: str,
    logger: logging.Logger,
):
    # Define output paths
    output_path = Path(output_dir) / "submission.csv"
    config_path = Path(output_dir) / "ultra_config.json"

    # Save submission
    results_df.to_csv(output_path, index=False)
    logger.info(f"✅ Submission saved to {output_path}")

    # Build config dict
    config_dict = {
        "batch_size": config.batch_size,
        "num_beams": config.num_beams,
        "length_penalty": config.length_penalty,
        "no_repeat_ngram_size": config.no_repeat_ngram_size,
        "optimizations": {
            "mixed_precision": config.use_mixed_precision,
            "better_transformer": config.use_better_transformer,
            "bucket_batching": config.use_bucket_batching,
            "vectorized_postproc": config.use_vectorized_postproc,
            "adaptive_beams": config.use_adaptive_beams,
        },
    }

    # Save config json
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("🎉 ULTRA-OPTIMIZED INFERENCE COMPLETE!")
    print("=" * 60)
    print(f"Submission file: {output_path}")
    print(f"Config file: {config_path}")
    print(f"Log file: {Path(output_dir) / 'inference_ultra.log'}")
    print(f"Total translations: {len(results_df)}")
    print("=" * 60)


def inspect_results(output_dir: str):
    # Load submission
    submission_path = Path(output_dir) / "submission.csv"
    submission = pd.read_csv(submission_path)

    # Print quick view
    print(f"Submission shape: {submission.shape}")

    print("\nFirst 10 translations:")
    print(submission.head(10))

    print("\nLast 10 translations:")
    print(submission.tail(10))

    # Length statistics
    lengths = submission["translation"].astype(str).str.len()
    print("\nLength distribution:")
    print(lengths.describe())

    # Empty checks
    empty = submission["translation"].astype(str).str.strip().eq("").sum()
    print(f"\nEmpty translations: {empty}")

    if empty > 0:
        print("\nEmpty translation IDs:")
        print(submission[submission["translation"].astype(str).str.strip().eq("")]["id"].tolist())


# ---------------------------
# 11) Main
# ---------------------------

def main():
    # Create config
    config = UltraConfig()

    # Setup logger
    logger = setup_logging(config.output_dir)
    logger.info("Logging initialized")

    # Print environment info
    print_environment_info()

    # Log configuration
    logger.info("Configuration:")
    logger.info(f"  Device: {config.device}")
    logger.info(f"  Batch size: {config.batch_size}")
    logger.info(f"  Beams: {config.num_beams}")
    logger.info("Optimizations:")
    logger.info(f"  Mixed Precision: {config.use_mixed_precision}")
    logger.info(f"  BetterTransformer: {config.use_better_transformer}")
    logger.info(f"  Bucket Batching: {config.use_bucket_batching}")
    logger.info(f"  Vectorized Postproc: {config.use_vectorized_postproc}")
    logger.info(f"  Adaptive Beams: {config.use_adaptive_beams}")

    # Load test data
    logger.info(f"Loading test data from {config.test_data_path}")
    test_df = pd.read_csv(config.test_data_path, encoding="utf-8")
    logger.info(f"✅ Loaded {len(test_df)} test samples")

    # Print first samples
    print("\nFirst 5 samples:")
    print(test_df.head())

    # Create engine
    engine = UltraInferenceEngine(config, logger)

    # Run inference
    results_df = engine.run_inference(test_df)

    # Save results and config
    save_outputs(results_df, config, config.output_dir, logger)

    # Optional inspection
    inspect_results(config.output_dir)


if __name__ == "__main__":
    main()
