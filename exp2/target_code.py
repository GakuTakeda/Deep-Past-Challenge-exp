# Import standard libraries
import os

# Set environment variables before importing PyTorch
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# ---------------------------
# Imports
# ---------------------------
import re
import json
import random
import logging
import warnings
from pathlib import Path
from typing import List

import pandas as pd
import numpy as np
from omegaconf import DictConfig, OmegaConf

import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.cuda.amp import autocast

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from tqdm.auto import tqdm

warnings.filterwarnings("ignore")


# ---------------------------
# Logging Setup
# ---------------------------

def setup_logging(output_dir: str) -> logging.Logger:
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    return logging.getLogger(__name__)


# ---------------------------
# Optimized Text Preprocessor
# ---------------------------

class OptimizedPreprocessor:
    def __init__(self):
        self.patterns = {
            "big_gap": re.compile(r"(\.{3,}|…+|……)"),
            "small_gap": re.compile(r"(xx+|\s+x\s+)"),
        }

    def preprocess_input_text(self, text: str) -> str:
        if pd.isna(text):
            return ""
        cleaned_text = str(text)
        cleaned_text = self.patterns["big_gap"].sub("<big_gap>", cleaned_text)
        cleaned_text = self.patterns["small_gap"].sub("<gap>", cleaned_text)
        return cleaned_text

    def preprocess_batch(self, texts: List[str]) -> List[str]:
        s = pd.Series(texts).fillna("")
        s = s.astype(str)
        s = s.str.replace(self.patterns["big_gap"], "<big_gap>", regex=True)
        s = s.str.replace(self.patterns["small_gap"], "<gap>", regex=True)
        return s.tolist()


# ---------------------------
# Vectorized Postprocessor
# ---------------------------

class VectorizedPostprocessor:
    def __init__(self, aggressive: bool = True):
        self.aggressive = aggressive
        self.patterns = {
            "gap": re.compile(r"(\[x\]|\(x\)|\bx\b)", re.I),
            "big_gap": re.compile(r"(\.{3,}|…|\[\.+\])"),
            "annotations": re.compile(r"\((fem|plur|pl|sing|singular|plural|\?|!)\..\s*\w*\)", re.I),
            "repeated_words": re.compile(r"\b(\w+)(?:\s+\1\b)+"),
            "whitespace": re.compile(r"\s+"),
            "punct_space": re.compile(r"\s+([.,:])"),
            "repeated_punct": re.compile(r"([.,])\1+"),
        }
        self.subscript_trans = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
        self.special_chars_trans = str.maketrans("ḫḪ", "hH")
        self.forbidden_chars = '!?()"——<>⌈⌋⌊[]+ʾ/;'
        self.forbidden_trans = str.maketrans("", "", self.forbidden_chars)

    def postprocess_batch(self, translations: List[str]) -> List[str]:
        s = pd.Series(translations)
        valid_mask = s.apply(lambda x: isinstance(x, str) and bool(x.strip()))
        if not valid_mask.all():
            s[~valid_mask] = ""

        s = s.str.translate(self.special_chars_trans)
        s = s.str.translate(self.subscript_trans)
        s = s.str.replace(self.patterns["whitespace"], " ", regex=True)
        s = s.str.strip()

        if self.aggressive:
            s = s.str.replace(self.patterns["gap"], "<gap>", regex=True)
            s = s.str.replace(self.patterns["big_gap"], "<big_gap>", regex=True)
            s = s.str.replace("<gap> <gap>", "<big_gap>", regex=False)
            s = s.str.replace("<big_gap> <big_gap>", "<big_gap>", regex=False)
            s = s.str.replace(self.patterns["annotations"], "", regex=True)

            s = s.str.replace("<gap>", "\x00GAP\x00", regex=False)
            s = s.str.replace("<big_gap>", "\x00BIG\x00", regex=False)
            s = s.str.translate(self.forbidden_trans)
            s = s.str.replace("\x00GAP\x00", " <gap> ", regex=False)
            s = s.str.replace("\x00BIG\x00", " <big_gap> ", regex=False)

            s = s.str.replace(r"(\d+)\.5\b", r"\1½", regex=True)
            s = s.str.replace(r"\b0\.5\b", "½", regex=True)
            s = s.str.replace(r"(\d+)\.25\b", r"\1¼", regex=True)
            s = s.str.replace(r"\b0\.25\b", "¼", regex=True)
            s = s.str.replace(r"(\d+)\.75\b", r"\1¾", regex=True)
            s = s.str.replace(r"\b0\.75\b", "¾", regex=True)

            s = s.str.replace(self.patterns["repeated_words"], r"\1", regex=True)
            for n in range(4, 1, -1):
                pattern = r"\b((?:\w+\s+){" + str(n - 1) + r"}\w+)(?:\s+\1\b)+"
                s = s.str.replace(pattern, r"\1", regex=True)

            s = s.str.replace(self.patterns["punct_space"], r"\1", regex=True)
            s = s.str.replace(self.patterns["repeated_punct"], r"\1", regex=True)
            s = s.str.replace(self.patterns["whitespace"], " ", regex=True)
            s = s.str.strip().str.strip("-").str.strip()

        return s.tolist()


# ---------------------------
# Bucket Batch Sampler
# ---------------------------

class BucketBatchSampler(Sampler):
    def __init__(self, dataset: Dataset, batch_size: int, num_buckets: int, logger: logging.Logger, shuffle: bool = False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.logger = logger

        lengths = [len(text.split()) for _, text in dataset]
        sorted_indices = sorted(range(len(lengths)), key=lambda i: lengths[i])

        bucket_size = max(1, len(sorted_indices) // max(1, num_buckets))
        self.buckets = []

        for i in range(num_buckets):
            start = i * bucket_size
            end = None if i == num_buckets - 1 else (i + 1) * bucket_size
            self.buckets.append(sorted_indices[start:end])

        self.logger.info(f"Created {num_buckets} buckets:")
        for i, bucket in enumerate(self.buckets):
            bucket_lengths = [lengths[idx] for idx in bucket] if len(bucket) > 0 else [0]
            self.logger.info(
                f"  Bucket {i}: {len(bucket)} samples, length range [{min(bucket_lengths)}, {max(bucket_lengths)}]"
            )

    def __iter__(self):
        for bucket in self.buckets:
            if self.shuffle:
                random.shuffle(bucket)
            for i in range(0, len(bucket), self.batch_size):
                yield bucket[i : i + self.batch_size]

    def __len__(self):
        total = 0
        for bucket in self.buckets:
            total += (len(bucket) + self.batch_size - 1) // self.batch_size
        return total


# ---------------------------
# Dataset
# ---------------------------

class AkkadianDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, preprocessor: OptimizedPreprocessor, task_prefix: str, logger: logging.Logger):
        self.sample_ids = dataframe["id"].tolist()
        raw_texts = dataframe["transliteration"].tolist()
        preprocessed = preprocessor.preprocess_batch(raw_texts)
        self.input_texts = [task_prefix + text for text in preprocessed]
        logger.info(f"Dataset created with {len(self.sample_ids)} samples")

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, index: int):
        return self.sample_ids[index], self.input_texts[index]


# ---------------------------
# Ultra-Optimized Inference Engine
# ---------------------------

class UltraInferenceEngine:
    def __init__(self, cfg: DictConfig, logger: logging.Logger):
        self.cfg = cfg
        self.logger = logger
        self.preprocessor = OptimizedPreprocessor()
        self.postprocessor = VectorizedPostprocessor(aggressive=cfg.postprocessing.aggressive)
        self.results = []
        self._load_model()

    def _load_model(self):
        self.logger.info(f"Loading model from {self.cfg.model.path}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("/home/tkdgk/deep_past_challenge/exp2/outputs/2026-02-16/19-59-39/output/best_model", local_files_only=True)
        self.model = self.model.to(self.device)
        self.model = self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained("/home/tkdgk/deep_past_challenge/exp2/outputs/2026-02-16/19-59-39/output/best_model", local_files_only=True)

        num_params = sum(p.numel() for p in self.model.parameters())
        self.logger.info(f"Model loaded: {num_params:,} parameters")

        if self.cfg.optimizations.use_better_transformer and torch.cuda.is_available():
            try:
                from optimum.bettertransformer import BetterTransformer
                self.logger.info("Applying BetterTransformer...")
                self.model = BetterTransformer.transform(self.model)
                self.logger.info("BetterTransformer applied")
            except ImportError:
                self.logger.warning("'optimum' not installed, skipping BetterTransformer")
            except Exception as exc:
                self.logger.warning(f"BetterTransformer failed: {exc}")

    def _collate_fn(self, batch_samples):
        batch_ids = [s[0] for s in batch_samples]
        batch_texts = [s[1] for s in batch_samples]

        tokenized = self.tokenizer(
            batch_texts,
            max_length=self.cfg.processing.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        return batch_ids, tokenized

    def _get_adaptive_beam_size(self, attention_mask: torch.Tensor) -> int:
        if not self.cfg.optimizations.use_adaptive_beams:
            return self.cfg.generation.num_beams

        lengths = attention_mask.sum(dim=1)
        short_beams = max(4, self.cfg.generation.num_beams // 2)
        beam_sizes = torch.where(
            lengths < 100,
            torch.tensor(short_beams, device=lengths.device),
            torch.tensor(self.cfg.generation.num_beams, device=lengths.device),
        )
        return int(beam_sizes[0].item())

    def _save_checkpoint(self, output_dir: str):
        if len(self.results) > 0 and len(self.results) % self.cfg.postprocessing.checkpoint_freq == 0:
            checkpoint_path = Path(output_dir) / f"checkpoint_{len(self.results)}.csv"
            df = pd.DataFrame(self.results, columns=["id", "translation"])
            df.to_csv(checkpoint_path, index=False)
            self.logger.info(f"Checkpoint: {len(self.results)} translations")

    def find_optimal_batch_size(self, dataset: Dataset, start_bs: int = 32) -> int:
        self.logger.info("Finding optimal batch size...")
        max_bs = start_bs
        min_bs = 1

        while max_bs - min_bs > 1:
            test_bs = (max_bs + min_bs) // 2
            try:
                test_batch = [dataset[i] for i in range(min(test_bs, len(dataset)))]
                _, inputs = self._collate_fn(test_batch)

                with torch.inference_mode():

                    _ = self.model.generate(
                        input_ids=inputs.input_ids.to(self.device),
                        attention_mask=inputs.attention_mask.to(self.device),
                        num_beams=self.cfg.generation.num_beams,
                        max_new_tokens=64,
                        use_cache=True,
                    )

                min_bs = test_bs
                self.logger.info(f"  Batch size {test_bs} works")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    max_bs = test_bs
                    self.logger.info(f"  Batch size {test_bs} OOM")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                else:
                    raise

        self.logger.info(f"Optimal batch size: {min_bs}")
        return min_bs

    def run_inference(self, test_df: pd.DataFrame, output_dir: str) -> pd.DataFrame:
        self.logger.info("Starting inference")

        dataset = AkkadianDataset(test_df, self.preprocessor, self.cfg.model.task_prefix, self.logger)

        batch_size = self.cfg.processing.batch_size
        if self.cfg.optimizations.use_auto_batch_size:
            batch_size = self.find_optimal_batch_size(dataset)

        if self.cfg.optimizations.use_bucket_batching:
            batch_sampler = BucketBatchSampler(
                dataset=dataset,
                batch_size=batch_size,
                num_buckets=self.cfg.postprocessing.num_buckets,
                logger=self.logger,
                shuffle=False,
            )
            dataloader = DataLoader(
                dataset,
                batch_sampler=batch_sampler,
                num_workers=self.cfg.processing.num_workers,
                collate_fn=self._collate_fn,
                pin_memory=True,
                prefetch_factor=2,
                persistent_workers=True if self.cfg.processing.num_workers > 0 else False,
            )
        else:
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=self.cfg.processing.num_workers,
                collate_fn=self._collate_fn,
                pin_memory=True,
                prefetch_factor=2,
                persistent_workers=True if self.cfg.processing.num_workers > 0 else False,
            )

        self.logger.info(f"DataLoader created: {len(dataloader)} batches")
        self.logger.info("Active optimizations:")
        self.logger.info(f"  Mixed Precision: {self.cfg.optimizations.use_mixed_precision}")
        self.logger.info(f"  BetterTransformer: {self.cfg.optimizations.use_better_transformer}")
        self.logger.info(f"  Bucket Batching: {self.cfg.optimizations.use_bucket_batching}")
        self.logger.info(f"  Vectorized Postproc: {self.cfg.optimizations.use_vectorized_postproc}")
        self.logger.info(f"  Adaptive Beams: {self.cfg.optimizations.use_adaptive_beams}")

        base_gen_config = {
            "max_new_tokens": self.cfg.generation.max_new_tokens,
            "length_penalty": self.cfg.generation.length_penalty,
            "early_stopping": self.cfg.generation.early_stopping,
            "use_cache": True,
        }
        if self.cfg.generation.no_repeat_ngram_size > 0:
            base_gen_config["no_repeat_ngram_size"] = self.cfg.generation.no_repeat_ngram_size

        self.results = []

        with torch.inference_mode():
            for batch_idx, (batch_ids, tokenized) in enumerate(tqdm(dataloader, desc="Translating")):
                try:
                    input_ids = tokenized.input_ids.to(self.device)
                    attention_mask = tokenized.attention_mask.to(self.device)

                    beam_size = self._get_adaptive_beam_size(attention_mask)
                    gen_config = dict(base_gen_config)
                    gen_config["num_beams"] = beam_size

                    outputs = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        **gen_config,
                    )
                    

                    translations = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                    print(translations)
                    if self.cfg.optimizations.use_vectorized_postproc:
                        cleaned = self.postprocessor.postprocess_batch(translations)
                    else:
                        cleaned = [self.postprocessor.postprocess_batch([t])[0] for t in translations]

                    self.results.extend(list(zip(batch_ids, cleaned)))
                    self._save_checkpoint(output_dir)

                    if torch.cuda.is_available() and batch_idx % 10 == 0:
                        torch.cuda.empty_cache()

                except Exception as exc:
                    self.logger.error(f"Batch {batch_idx} error: {exc}")
                    self.results.extend([(bid, "") for bid in batch_ids])

        self.logger.info("Inference completed")

        results_df = pd.DataFrame(self.results, columns=["id", "translation"])
        self._validate_results(results_df)
        return results_df

    def _validate_results(self, df: pd.DataFrame):
        print("\n" + "=" * 60)
        print("VALIDATION REPORT")
        print("=" * 60)

        empty = df["translation"].astype(str).str.strip().eq("").sum()
        print(f"\nEmpty: {empty} ({(empty / max(1, len(df))) * 100:.2f}%)")

        lengths = df["translation"].astype(str).str.len()
        print("\nLength stats:")
        print(f"   Mean: {lengths.mean():.1f}, Median: {lengths.median():.1f}")
        print(f"   Min: {lengths.min()}, Max: {lengths.max()}")

        short = ((lengths < 5) & (lengths > 0)).sum()
        if short > 0:
            print(f"   {short} very short translations")

        print("\nSample translations:")
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
# IO Helpers
# ---------------------------

def print_environment_info():
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        total_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU Memory: {total_mem_gb:.2f} GB")

    try:
        from optimum.bettertransformer import BetterTransformer  # noqa: F401
        print("BetterTransformer available")
    except ImportError:
        print("BetterTransformer NOT available")


def save_outputs(results_df: pd.DataFrame, cfg: DictConfig, output_dir: str, logger: logging.Logger):
    output_path = Path(output_dir) / "submission.csv"

    results_df.to_csv(output_path, index=False)
    logger.info(f"Submission saved to {output_path}")

    config_dict = OmegaConf.to_container(cfg, resolve=True)

    print("\n" + "=" * 60)
    print("INFERENCE COMPLETE")
    print("=" * 60)
    print(f"Submission file: {output_path}")
    print(f"Total translations: {len(results_df)}")
    print("=" * 60)


def inspect_results(output_dir: str):
    submission_path = Path(output_dir) / "submission.csv"
    submission = pd.read_csv(submission_path)

    print(f"Submission shape: {submission.shape}")

    print("\nFirst 10 translations:")
    print(submission.head(10))

    print("\nLast 10 translations:")
    print(submission.tail(10))

    lengths = submission["translation"].astype(str).str.len()
    print("\nLength distribution:")
    print(lengths.describe())

    empty = submission["translation"].astype(str).str.strip().eq("").sum()
    print(f"\nEmpty translations: {empty}")

    if empty > 0:
        print("\nEmpty translation IDs:")
        print(submission[submission["translation"].astype(str).str.strip().eq("")]["id"].tolist())


def main(cfg: DictConfig):
    # Hydraの出力ディレクトリを使用
    output_dir = "./"

    print("=" * 60)
    print("Akkadian Translation - Inference")
    print("=" * 60)
    print(OmegaConf.to_yaml(cfg))

    logger = setup_logging(output_dir)
    logger.info("Logging initialized")

    print_environment_info()

    # テストデータ読み込み
    test_data_path = "/home/tkdgk/deep_past_challenge/data/test.csv"
    logger.info(f"Loading test data from {test_data_path}")
    test_df = pd.read_csv(test_data_path, encoding="utf-8")
    logger.info(f"Loaded {len(test_df)} test samples")

    print("\nFirst 5 samples:")
    print(test_df.head())

    # 推論実行
    engine = UltraInferenceEngine(cfg, logger)
    results_df = engine.run_inference(test_df, output_dir)

    # 結果保存
    save_outputs(results_df, cfg, output_dir, logger)
    inspect_results(output_dir)


if __name__ == "__main__":
    cfg = OmegaConf.load("/home/tkdgk/deep_past_challenge/exp2/configs/inference.yaml")
    main(cfg)
