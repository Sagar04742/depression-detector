"""
evaluate.py
-----------
Evaluates the trained model on the held-out test set:
accuracy, precision, recall, F1 (per class + macro), and a confusion matrix.
"""

import json
import pickle

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


def main():
    with open("config.json") as f:
        config = json.load(f)
    labels = config["labels"]
    max_len = config["max_len"]

    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    model = load_model("model.keras")

    df = pd.read_csv("data/test.csv")
    texts = df["text"].astype(str).tolist()
    y_true = df["label"].map({l: i for i, l in enumerate(labels)}).values

    X_test = pad_sequences(tokenizer.texts_to_sequences(texts), maxlen=max_len, padding="post", truncating="post")
    y_pred = model.predict(X_test, verbose=0).argmax(axis=1)

    print("=== Classification Report ===")
    print(classification_report(y_true, y_pred, target_names=labels, digits=3))

    print("=== Confusion Matrix ===")
    cm = confusion_matrix(y_true, y_pred)
    header = "        " + "  ".join(f"{l[:8]:>8}" for l in labels)
    print(header)
    for label, row in zip(labels, cm):
        print(f"{label[:8]:>8} " + "  ".join(f"{v:>8}" for v in row))


if __name__ == "__main__":
    main()
