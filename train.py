"""
train.py
--------
Trains a simple neural network (Embedding + Bidirectional LSTM + Dense)
to classify Reddit posts into depression-severity levels:
minimum, mild, moderate, severe.

Easy, standard tech stack: TensorFlow/Keras + scikit-learn + pandas.
"""

import json
import pickle

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers, models

LABELS = ["minimum", "mild", "moderate", "severe"]
LABEL_TO_IDX = {label: i for i, label in enumerate(LABELS)}

MAX_WORDS = 10000     # vocabulary size
MAX_LEN = 100          # max tokens per post
EMBED_DIM = 64
EPOCHS = 12
BATCH_SIZE = 32


def load_split(path):
    df = pd.read_csv(path)
    texts = df["text"].astype(str).tolist()
    labels = df["label"].map(LABEL_TO_IDX).values
    return texts, labels


def build_model(vocab_size):
    model = models.Sequential([
        layers.Input(shape=(MAX_LEN,)),
        layers.Embedding(vocab_size, EMBED_DIM, mask_zero=True),
        layers.Bidirectional(layers.LSTM(32)),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(len(LABELS), activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    train_texts, y_train = load_split("data/train.csv")
    val_texts, y_val = load_split("data/val.csv")

    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(train_texts)

    X_train = pad_sequences(tokenizer.texts_to_sequences(train_texts), maxlen=MAX_LEN, padding="post", truncating="post")
    X_val = pad_sequences(tokenizer.texts_to_sequences(val_texts), maxlen=MAX_LEN, padding="post", truncating="post")

    vocab_size = min(MAX_WORDS, len(tokenizer.word_index) + 1)
    model = build_model(vocab_size)
    model.summary()

    # Handle class imbalance (dataset is skewed toward "minimum")
    class_weights_arr = compute_class_weight(
        class_weight="balanced", classes=np.arange(len(LABELS)), y=y_train
    )
    class_weights = {i: w for i, w in enumerate(class_weights_arr)}
    print("Class weights:", class_weights)

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        callbacks=[early_stop],
        verbose=2,
    )

    # Save model + tokenizer + label map for reuse in evaluate.py / predict.py
    model.save("model.keras")
    with open("tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)
    with open("config.json", "w") as f:
        json.dump({"labels": LABELS, "max_len": MAX_LEN}, f)

    with open("history.json", "w") as f:
        json.dump({k: [float(x) for x in v] for k, v in history.history.items()}, f)

    print("\nSaved model.keras, tokenizer.pkl, config.json, history.json")


if __name__ == "__main__":
    main()
