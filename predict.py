"""
predict.py
----------
Simple CLI demo: type/paste a piece of text and get the model's
predicted depression-severity level with confidence scores.

IMPORTANT DISCLAIMER
---------------------
This is a small educational/demo project, NOT a medical or diagnostic
tool. It is trained on a limited public dataset and can be wrong. It
must never be used to make real decisions about someone's mental
health. If you or someone you know is struggling, please reach out to
a mental health professional or a crisis helpline.
"""

import json
import pickle
import sys

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from preprocess import clean_text


def load_artifacts():
    with open("config.json") as f:
        config = json.load(f)
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    model = load_model("model.keras")
    return model, tokenizer, config["labels"], config["max_len"]


def predict(text, model, tokenizer, labels, max_len):
    cleaned = clean_text(text)
    seq = pad_sequences(tokenizer.texts_to_sequences([cleaned]), maxlen=max_len, padding="post", truncating="post")
    probs = model.predict(seq, verbose=0)[0]
    ranked = sorted(zip(labels, probs), key=lambda x: -x[1])
    return ranked


def main():
    print("=" * 70)
    print(" Depression-Severity Text Classifier (demo project)")
    print(" DISCLAIMER: Not a medical tool. For education/demo purposes only.")
    print("=" * 70)

    model, tokenizer, labels, max_len = load_artifacts()

    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
        ranked = predict(text, model, tokenizer, labels, max_len)
        print(f"\nInput: {text}")
        print("Predicted severity:", ranked[0][0])
        print("Confidence scores:")
        for label, prob in ranked:
            print(f"  {label:<10} {prob*100:5.1f}%")
        return

    print("\nType a sentence/paragraph and press Enter (type 'quit' to exit).\n")
    while True:
        text = input("> ").strip()
        if text.lower() in ("quit", "exit"):
            break
        if not text:
            continue
        ranked = predict(text, model, tokenizer, labels, max_len)
        print("Predicted severity:", ranked[0][0])
        for label, prob in ranked:
            print(f"  {label:<10} {prob*100:5.1f}%")
        print()


if __name__ == "__main__":
    main()
