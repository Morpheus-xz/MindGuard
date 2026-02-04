"""
MindGuard v2.0 - BERT Mental Health Classifier
===============================================
Clean, simple, and effective training script.

Principles:
- No text augmentation (preserves clinical meaning)
- ONE balancing strategy (class weights OR data balance, not both)
- No label smoothing (keeps probabilities interpretable for SHAP)
- Minimal regularization (just what's needed)
- f1_weighted for best model (reflects class importance)
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import random
from pathlib import Path
from datetime import datetime
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    precision_score,
    classification_report,
    confusion_matrix,
)
from sklearn.utils.class_weight import compute_class_weight
import warnings
import json

warnings.filterwarnings("ignore")

# ============================================================
# Configuration
# ============================================================
DATA_DIR = Path(__file__).parent.parent.parent / "data" / "processed"
MODEL_DIR = Path(__file__).parent.parent.parent / "models"
SEED = 42


def set_seed(seed):
    """Set all seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# ============================================================
# Dataset
# ============================================================
class MentalHealthDataset(Dataset):
    """Simple PyTorch Dataset for text classification."""
    
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]),
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ============================================================
# Weighted Trainer (ONLY class weights, no other tricks)
# ============================================================
class WeightedTrainer(Trainer):
    """Trainer with class-weighted loss to handle imbalance."""
    
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        if self.class_weights is not None:
            weight = self.class_weights.to(logits.device)
            loss_fct = nn.CrossEntropyLoss(weight=weight)
        else:
            loss_fct = nn.CrossEntropyLoss()
        
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


# ============================================================
# Metrics
# ============================================================
def compute_metrics(eval_pred):
    """Compute evaluation metrics."""
    preds = np.argmax(eval_pred.predictions, axis=1)
    labels = eval_pred.label_ids
    
    recall_per_class = recall_score(labels, preds, average=None, zero_division=0)
    precision_per_class = precision_score(labels, preds, average=None, zero_division=0)
    
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_weighted": f1_score(labels, preds, average="weighted"),
        "f1_macro": f1_score(labels, preds, average="macro"),
        "recall_low": recall_per_class[0],
        "recall_medium": recall_per_class[1],
        "recall_high": recall_per_class[2],
        "precision_low": precision_per_class[0],
        "precision_medium": precision_per_class[1],
        "precision_high": precision_per_class[2],
    }


# ============================================================
# Training
# ============================================================
def train(
    model_name: str = "distilbert-base-uncased",
    epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    max_length: int = 256,
    warmup_ratio: float = 0.1,
    weight_decay: float = 0.01,
    eval_steps: int = 500,
    early_stopping_patience: int = 3,
    use_class_weights: bool = True,
):
    """
    Train mental health classifier.
    
    Simple and effective:
    - Uses class weights to handle imbalance (no data augmentation)
    - Standard dropout from pretrained model
    - Early stopping to prevent overfitting
    - f1_weighted as metric (respects class distribution)
    """
    
    print("=" * 70)
    print("🧠 MindGuard v2.0 - Mental Health Classifier")
    print("=" * 70)
    
    # ----------------------------------------------------------
    # Device
    # ----------------------------------------------------------
    if torch.cuda.is_available():
        device = "cuda"
        device_name = torch.cuda.get_device_name(0)
    elif torch.backends.mps.is_available():
        device = "mps"
        device_name = "Apple Silicon (MPS)"
    else:
        device = "cpu"
        device_name = "CPU"
    
    print(f"\n📱 Device: {device_name}")
    
    # ----------------------------------------------------------
    # Load Data (NO augmentation, NO resampling)
    # ----------------------------------------------------------
    print("\n📊 Loading data...")
    
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    val_df = pd.read_csv(DATA_DIR / "val.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    
    print(f"   Train: {len(train_df):,}")
    print(f"   Val  : {len(val_df):,}")
    print(f"   Test : {len(test_df):,}")
    
    # Show class distribution
    print("\n📈 Class distribution (Train):")
    for label, name in [(0, "Low"), (1, "Medium"), (2, "High")]:
        count = (train_df["label"] == label).sum()
        pct = count / len(train_df) * 100
        print(f"   {name:8}: {count:,} ({pct:.1f}%)")
    
    # ----------------------------------------------------------
    # Compute Class Weights (handles imbalance)
    # ----------------------------------------------------------
    class_weights = None
    if use_class_weights:
        classes = np.array([0, 1, 2])
        weights = compute_class_weight(
            class_weight='balanced',
            classes=classes,
            y=train_df['label'].values
        )
        class_weights = torch.tensor(weights, dtype=torch.float32)
        print(f"\n⚖️ Class weights: Low={weights[0]:.2f}, Medium={weights[1]:.2f}, High={weights[2]:.2f}")
    
    # ----------------------------------------------------------
    # Tokenizer & Datasets
    # ----------------------------------------------------------
    print(f"\n🤖 Model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    train_ds = MentalHealthDataset(
        train_df["text"].tolist(),
        train_df["label"].tolist(),
        tokenizer,
        max_length,
    )
    val_ds = MentalHealthDataset(
        val_df["text"].tolist(),
        val_df["label"].tolist(),
        tokenizer,
        max_length,
    )
    test_ds = MentalHealthDataset(
        test_df["text"].tolist(),
        test_df["label"].tolist(),
        tokenizer,
        max_length,
    )
    
    # ----------------------------------------------------------
    # Model (use default dropout, no extra regularization)
    # ----------------------------------------------------------
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3,
        id2label={0: "Low", 1: "Medium", 2: "High"},
        label2id={"Low": 0, "Medium": 1, "High": 2},
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {total_params:,}")
    
    # ----------------------------------------------------------
    # Output Directory
    # ----------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = MODEL_DIR / f"mindguard-{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # ----------------------------------------------------------
    # Training Arguments
    # ----------------------------------------------------------
    training_args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        logging_steps=100,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model="f1_weighted",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",
        seed=SEED,
    )
    
    # ----------------------------------------------------------
    # Trainer
    # ----------------------------------------------------------
    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)],
    )
    
    # ----------------------------------------------------------
    # Train
    # ----------------------------------------------------------
    print("\n" + "=" * 70)
    print("🏋️ Training")
    print("=" * 70)
    print(f"   Epochs         : {epochs}")
    print(f"   Batch size     : {batch_size}")
    print(f"   Learning rate  : {learning_rate}")
    print(f"   Weight decay   : {weight_decay}")
    print(f"   Class weights  : {'Yes' if use_class_weights else 'No'}")
    print()
    
    trainer.train()
    
    # ----------------------------------------------------------
    # Evaluate
    # ----------------------------------------------------------
    print("\n" + "=" * 70)
    print("📈 Test Set Evaluation")
    print("=" * 70)
    
    results = trainer.evaluate(test_ds)
    
    print(f"\n   Overall:")
    print(f"   Accuracy    : {results['eval_accuracy']:.4f}")
    print(f"   F1 Weighted : {results['eval_f1_weighted']:.4f}")
    print(f"   F1 Macro    : {results['eval_f1_macro']:.4f}")
    
    print(f"\n   Per-Class Recall:")
    print(f"   Low    : {results['eval_recall_low']:.4f}")
    print(f"   Medium : {results['eval_recall_medium']:.4f}")
    print(f"   High   : {results['eval_recall_high']:.4f}")
    
    print(f"\n   Per-Class Precision:")
    print(f"   Low    : {results['eval_precision_low']:.4f}")
    print(f"   Medium : {results['eval_precision_medium']:.4f}")
    print(f"   High   : {results['eval_precision_high']:.4f}")
    
    # ----------------------------------------------------------
    # Classification Report
    # ----------------------------------------------------------
    print("\n" + "=" * 70)
    print("📋 Classification Report")
    print("=" * 70)
    
    predictions = trainer.predict(test_ds)
    preds = np.argmax(predictions.predictions, axis=1)
    labels = test_df["label"].tolist()
    
    print(classification_report(
        labels, preds,
        target_names=["Low", "Medium", "High"],
        digits=4,
    ))
    
    # Confusion Matrix
    cm = confusion_matrix(labels, preds)
    print("Confusion Matrix:")
    print(f"{'':>10} {'Low':>8} {'Medium':>8} {'High':>8}")
    for i, name in enumerate(["Low", "Medium", "High"]):
        print(f"{name:>10} {cm[i][0]:>8} {cm[i][1]:>8} {cm[i][2]:>8}")
    
    # ----------------------------------------------------------
    # Save Model
    # ----------------------------------------------------------
    final_path = MODEL_DIR / "mindguard-bert-final"
    
    if final_path.exists():
        import shutil
        shutil.rmtree(final_path)
    
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    
    # Save config
    config = {
        "model_name": model_name,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "max_length": max_length,
        "use_class_weights": use_class_weights,
        "class_weights": class_weights.tolist() if class_weights is not None else None,
        "test_results": {
            "accuracy": float(results["eval_accuracy"]),
            "f1_weighted": float(results["eval_f1_weighted"]),
            "f1_macro": float(results["eval_f1_macro"]),
            "recall_low": float(results["eval_recall_low"]),
            "recall_medium": float(results["eval_recall_medium"]),
            "recall_high": float(results["eval_recall_high"]),
        },
        "trained_at": timestamp,
    }
    
    with open(final_path / "training_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"\n💾 Saved: {final_path}")
    
    # ----------------------------------------------------------
    # SOW Check
    # ----------------------------------------------------------
    print("\n" + "=" * 70)
    print("✅ SOW Compliance")
    print("=" * 70)
    
    checks = [
        ("Accuracy ≥ 85%", results["eval_accuracy"] >= 0.85),
        ("Recall High ≥ 90%", results["eval_recall_high"] >= 0.90),
        ("Precision ≥ 80%", (results["eval_precision_low"] + results["eval_precision_medium"] + results["eval_precision_high"]) / 3 >= 0.80),
        ("F1 ≥ 85%", results["eval_f1_weighted"] >= 0.85),
    ]
    
    all_pass = True
    for name, passed in checks:
        print(f"   {name}: {'✅' if passed else '❌'}")
        if not passed:
            all_pass = False
    
    print("\n" + "=" * 70)
    print("🎉 Done!" if all_pass else "⚠️ Some targets not met")
    print("=" * 70)
    
    return trainer, results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--no-weights", action="store_true", help="Disable class weights")
    
    args = parser.parse_args()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        eval_steps=args.eval_steps,
        use_class_weights=not args.no_weights,
    )
