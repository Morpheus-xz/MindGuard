
"""Prepare the Mental Health Dataset for training."""

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import re

DATA_DIR = Path(__file__).parent.parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"


def setup_directories():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    print("✓ Directories ready")


def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[^\w\s.,!?\'-]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def map_to_risk_level(status):
    status = str(status).lower().strip()
    if status in ['normal']:
        return 0, "Low"
    elif status in ['anxiety', 'stress']:
        return 1, "Medium"
    elif status in ['depression', 'suicidal', 'bi-polar', 'bipolar', 
                    'personality disorder', 'personality_disorder']:
        return 2, "High"
    return 1, "Medium"


def load_and_process_dataset():
    possible_names = ["mental_health_dataset.csv", "Combined Data.csv", "combined_data.csv"]
    
    raw_file = None
    for name in possible_names:
        path = RAW_DIR / name
        if path.exists():
            raw_file = path
            break
    
    if raw_file is None:
        print("❌ Dataset not found!")
        print(f"   Please save dataset to: {RAW_DIR}/mental_health_dataset.csv")
        return None
    
    print(f"✓ Found: {raw_file}")
    df = pd.read_csv(raw_file)
    print(f"✓ Loaded {len(df)} rows")
    print(f"  Columns: {df.columns.tolist()}")
    
    text_col = label_col = None
    for col in df.columns:
        col_lower = col.lower()
        if 'statement' in col_lower or 'text' in col_lower:
            text_col = col
        if 'status' in col_lower or 'label' in col_lower:
            label_col = col
    
    if not text_col or not label_col:
        print(f"❌ Could not find columns")
        return None
    
    print(f"✓ Text: {text_col}, Label: {label_col}")
    print(f"\n📊 Original labels:\n{df[label_col].value_counts()}")
    
    df = df.dropna(subset=[text_col, label_col])
    df['clean_text'] = df[text_col].apply(clean_text)
    df = df[df['clean_text'].str.len() > 10]
    df['label'], df['risk_level'] = zip(*df[label_col].apply(map_to_risk_level))
    
    df = df[['clean_text', 'label', 'risk_level']].copy()
    df.columns = ['text', 'label', 'risk_level']
    
    print(f"\n✓ Processed: {len(df)} rows")
    print(f"\n📊 Risk levels:\n{df['risk_level'].value_counts()}")
    return df


def create_splits(df, test_size=0.15, val_size=0.15):
    print("\n✂️ Splitting...")
    train_val, test_df = train_test_split(df, test_size=test_size, stratify=df['label'], random_state=42)
    val_ratio = val_size / (1 - test_size)
    train_df, val_df = train_test_split(train_val, test_size=val_ratio, stratify=train_val['label'], random_state=42)
    
    train_df.to_csv(PROCESSED_DIR / "train.csv", index=False)
    val_df.to_csv(PROCESSED_DIR / "val.csv", index=False)
    test_df.to_csv(PROCESSED_DIR / "test.csv", index=False)
    
    print(f"✓ Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")


def main():
    print("=" * 50)
    print("MindGuard Dataset Preparation")
    print("=" * 50)
    setup_directories()
    df = load_and_process_dataset()
    if df is None:
        return
    df.to_csv(PROCESSED_DIR / "full_dataset.csv", index=False)
    create_splits(df)
    print("\n✅ Done!")


if __name__ == "__main__":
    main()
