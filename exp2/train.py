# Import standard libraries
import os

# Set environment variables before importing PyTorch
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# ---------------------------
# Imports
# ---------------------------
import re
import json
import math
import warnings
from pathlib import Path
from typing import List

import pandas as pd
import numpy as np
import sacrebleu

import torch
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
    TrainerCallback,
)
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")


# ---------------------------
# Constants
# ---------------------------

TRAIN_DATA_PATH = "../data/train.csv"
BASE_MODEL_NAME = "google/byt5-small"
OUTPUT_DIR = "./exp2/output"
TASK_PREFIX = "translate Akkadian to English: "
MAX_SOURCE_LENGTH = 512
MAX_TARGET_LENGTH = 512
VAL_RATIO = 0.1
SEED = 42


# ---------------------------
# Preprocessor (推論コードと同一)
# ---------------------------

class OptimizedPreprocessor:
    def __init__(self):
        self.patterns = {
            "big_gap": re.compile(r"(\.{3,}|…+|……)"),
            "small_gap": re.compile(r"(xx+|\s+x\s+)"),
        }

    def preprocess_batch(self, texts: List[str]) -> List[str]:
        s = pd.Series(texts).fillna("")
        s = s.astype(str)
        s = s.str.replace(self.patterns["big_gap"], "<big_gap>", regex=True)
        s = s.str.replace(self.patterns["small_gap"], "<gap>", regex=True)
        return s.tolist()


# ---------------------------
# Postprocessor (推論コードと同一)
# ---------------------------

class VectorizedPostprocessor:
    def __init__(self, aggressive: bool = True):
        self.aggressive = aggressive
        self.patterns = {
            "gap": re.compile(r"(\[x\]|\(x\)|\bx\b)", re.I),
            "big_gap": re.compile(r"(\.{3,}|…|\[\.+\])"),
            "annotations": re.compile(
                r"\((fem|plur|pl|sing|singular|plural|\?|!)\..\s*\w*\)", re.I
            ),
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
                pattern = (
                    r"\b((?:\w+\s+){" + str(n - 1) + r"}\w+)(?:\s+\1\b)+"
                )
                s = s.str.replace(pattern, r"\1", regex=True)

            s = s.str.replace(self.patterns["punct_space"], r"\1", regex=True)
            s = s.str.replace(self.patterns["repeated_punct"], r"\1", regex=True)
            s = s.str.replace(self.patterns["whitespace"], " ", regex=True)
            s = s.str.strip().str.strip("-").str.strip()

        return s.tolist()


# ---------------------------
# Dataset
# ---------------------------

class AkkadianTrainDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, preprocessor, tokenizer, max_source_length, max_target_length):
        raw_texts = dataframe["transliteration"].tolist()
        preprocessed = preprocessor.preprocess_batch(raw_texts)
        self.input_texts = [TASK_PREFIX + t for t in preprocessed]
        self.target_texts = dataframe["translation"].fillna("").astype(str).tolist()
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.input_texts)

    def __getitem__(self, idx):
        source = self.tokenizer(
            self.input_texts[idx],
            max_length=self.max_source_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        target = self.tokenizer(
            self.target_texts[idx],
            max_length=self.max_target_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        labels = target.input_ids.squeeze().clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": source.input_ids.squeeze(),
            "attention_mask": source.attention_mask.squeeze(),
            "labels": labels,
        }


# ---------------------------
# Evaluation (コンペ公式指標: √(BLEU × chrF++))
# ---------------------------

def compute_competition_score(predictions: List[str], references: List[str]) -> dict:
    bleu = sacrebleu.corpus_bleu(predictions, [references])
    chrf = sacrebleu.corpus_chrf(predictions, [references], word_order=2)
    geo_mean = math.sqrt(max(bleu.score * chrf.score, 0.0))

    return {
        "bleu": bleu.score,
        "chrf": chrf.score,
        "score": geo_mean,
    }


# ---------------------------
# Tokenizer互換性修正コールバック
# ---------------------------

def fix_tokenizer_config(save_path: str):
    config_path = Path(save_path) / "tokenizer_config.json"
    if config_path.exists():
        with open(config_path) as f:
            data = json.load(f)
        if "extra_special_tokens" in data and isinstance(data["extra_special_tokens"], list):
            data["extra_special_tokens"] = {}
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)


class FixTokenizerCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        fix_tokenizer_config(str(checkpoint_dir))


# ---------------------------
# compute_metrics factory
# ---------------------------

def make_compute_metrics(tokenizer, postprocessor):
    def compute_metrics(eval_preds):
        preds, label_ids = eval_preds

        # -100 をpad_token_idに戻す（デコード用）
        label_ids = np.where(label_ids == -100, tokenizer.pad_token_id, label_ids)

        # デコード
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        # 推論コードと同一の後処理
        cleaned_preds = postprocessor.postprocess_batch(decoded_preds)
        cleaned_labels = postprocessor.postprocess_batch(decoded_labels)

        metrics = compute_competition_score(cleaned_preds, cleaned_labels)

        # サンプル出力
        for k in range(min(3, len(cleaned_preds))):
            print(f"  PRED: {cleaned_preds[k][:120]}")
            print(f"  REF:  {cleaned_labels[k][:120]}")
            print()

        return metrics

    return compute_metrics


# ---------------------------
# Main
# ---------------------------

def main():
    print("=" * 60)
    print("Akkadian Translation - Training (Seq2SeqTrainer)")
    print("=" * 60)

    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU Memory: {mem_gb:.2f} GB")

    # データ読み込み
    print(f"Loading data from {TRAIN_DATA_PATH}")
    df = pd.read_csv(TRAIN_DATA_PATH, encoding="utf-8")
    df = df.dropna(subset=["transliteration", "translation"]).reset_index(drop=True)
    print(f"Loaded {len(df)} samples (after dropping NaN)")

    # Train / Validation 分割
    train_df, val_df = train_test_split(df, test_size=VAL_RATIO, random_state=SEED)
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    print(f"Train: {len(train_df)}, Val: {len(val_df)}")

    # モデル・トークナイザ読み込み
    print(f"Loading model: {BASE_MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL_NAME)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,}")

    # 前処理・後処理
    preprocessor = OptimizedPreprocessor()
    postprocessor = VectorizedPostprocessor(aggressive=True)

    # Dataset
    train_dataset = AkkadianTrainDataset(
        train_df, preprocessor, tokenizer, MAX_SOURCE_LENGTH, MAX_TARGET_LENGTH
    )
    val_dataset = AkkadianTrainDataset(
        val_df, preprocessor, tokenizer, MAX_SOURCE_LENGTH, MAX_TARGET_LENGTH
    )

    # TrainingArguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=20,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=8,
        learning_rate=3e-4,
        weight_decay=0.01,
        warmup_ratio=0.1,
        max_grad_norm=1.0,
        fp16=False,  # ByT5はfp16非互換（相対位置バイアスがoverflow）
        predict_with_generate=True,
        generation_num_beams=4,
        generation_max_length=MAX_TARGET_LENGTH,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=10,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="score",
        greater_is_better=True,
        seed=SEED,
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
        report_to="none",
    )

    # Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=make_compute_metrics(tokenizer, postprocessor),
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=5),
            FixTokenizerCallback(),
        ],
    )

    # 学習
    print("Starting training...")
    trainer.train()

    # Best model を保存
    best_path = Path(OUTPUT_DIR) / "best_model"
    trainer.save_model(str(best_path))
    tokenizer.save_pretrained(str(best_path))
    fix_tokenizer_config(str(best_path))
    print(f"Best model saved -> {best_path}")

    # 設定をJSON保存
    config_dict = {
        "base_model": BASE_MODEL_NAME,
        "epochs": 20,
        "batch_size": 4,
        "gradient_accumulation_steps": 8,
        "learning_rate": 3e-4,
        "max_source_length": MAX_SOURCE_LENGTH,
        "max_target_length": MAX_TARGET_LENGTH,
        "task_prefix": TASK_PREFIX,
    }
    config_path = Path(OUTPUT_DIR) / "train_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)

    print("=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
