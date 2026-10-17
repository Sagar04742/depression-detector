"""
preprocess.py
--------------
Loads the raw Reddit depression-severity dataset, cleans the text,
and splits it into train/validation/test sets.

Dataset source: usmaann/Depression_Severity_Dataset (Reddit posts labeled
by self-reported depression severity: minimum, mild, moderate, severe).
"""

import re
import pandas as pd
from sklearn.model_selection import train_test_split

RAW_PATH = "data/reddit_depression_dataset.csv"
OUT_TRAIN = "data/train.csv"
OUT_VAL = "data/val.csv"
OUT_TEST = "data/test.csv"

LABELS = ["minimum", "mild", "moderate", "severe"]


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)          # URLs
    text = re.sub(r"[^a-z0-9\s']", " ", text)               # punctuation/symbols
    text = re.sub(r"\s+", " ", text).strip()                # extra whitespace
    return text


def main():
    df = pd.read_csv(RAW_PATH)
    df = df.dropna(subset=["text", "label"])
    df["label"] = df["label"].str.strip().str.lower()
    df = df[df["label"].isin(LABELS)]

    df["text"] = df["text"].apply(clean_text)
    df = df[df["text"].str.len() > 5]  # drop empty/near-empty rows

    print("Class distribution:\n", df["label"].value_counts())

    # 60/20/20 split, stratified so each class is represented in every split
    train_df, temp_df = train_test_split(
        df, test_size=0.4, stratify=df["label"], random_state=42
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df["label"], random_state=42
    )

    train_df.to_csv(OUT_TRAIN, index=False)
    val_df.to_csv(OUT_VAL, index=False)
    test_df.to_csv(OUT_TEST, index=False)

    print(f"\nSaved: {len(train_df)} train / {len(val_df)} val / {len(test_df)} test rows")


if __name__ == "__main__":
    main()
