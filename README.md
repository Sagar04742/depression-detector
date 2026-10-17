# Depression Severity Detection (Neural Network Demo)

A small, easy-to-run demo project that uses a neural network to classify
short pieces of text (Reddit-style posts) into a depression-severity
level: **minimum, mild, moderate, or severe**.

This is a **student/portfolio-style demo project**, not a production or
clinical system. It's meant to show an end-to-end neural network
pipeline (data → preprocessing → model → evaluation → demo) in a simple,
easy-to-follow way.

## ⚠️ Disclaimer

This project is for **educational/demo purposes only**. It is not a
medical device and must not be used to diagnose or make real decisions
about anyone's mental health. If you or someone you know is struggling,
please talk to a mental health professional or a crisis helpline.

## Tech stack (kept intentionally simple)

- Python 3
- **TensorFlow / Keras** — Embedding + Bidirectional LSTM + Dense neural
  network (nothing fancier like transformers, to keep things easy to
  understand and run on a normal laptop/CPU)
- pandas, scikit-learn — data handling, splitting, metrics

## Dataset

Public Reddit dataset of ~3,550 posts labeled by self-reported depression
severity (source: `usmaann/Depression_Severity_Dataset` on GitHub).
Already included in `data/reddit_depression_dataset.csv`.

Classes: `minimum`, `mild`, `moderate`, `severe` (imbalanced — mostly
"minimum" — handled with class weighting during training).

## Project structure

```
depression_detector/
├── data/
│   ├── reddit_depression_dataset.csv   # raw data
│   ├── train.csv / val.csv / test.csv  # created by preprocess.py
├── preprocess.py     # cleans text, splits into train/val/test
├── train.py           # builds + trains the neural network
├── evaluate.py         # prints accuracy/precision/recall/F1 + confusion matrix
├── predict.py           # CLI demo: type text, get a prediction
├── app.py               # Streamlit web UI demo (recommended)
├── requirements.txt
└── README.md
```

## How to run it

```bash
pip install -r requirements.txt

python3 preprocess.py   # cleans data, creates train/val/test splits
python3 train.py         # trains the model (a couple minutes on CPU)
python3 evaluate.py       # shows accuracy/precision/recall/F1 on test set
python3 predict.py "I feel so hopeless lately and can't get out of bed"
```

Or run `python3 predict.py` with no arguments for an interactive prompt.

## Web UI (recommended for demos/presentations)

Instead of the terminal, you can launch a simple local website:

```bash
streamlit run app.py
```

This opens a page in your browser (usually `http://localhost:8501`) with a
text box — paste in a sentence, click **Analyze**, and see the predicted
severity plus a confidence chart. This is the easiest way to *show* the
project to someone. Make sure you've already run `preprocess.py` and
`train.py` at least once before launching the app, since it loads the
saved model.

## How it works (short version)

1. **preprocess.py** — lowercases text, strips URLs/punctuation, removes
   near-empty rows, and does a stratified 60/20/20 train/val/test split.
2. **train.py** — tokenizes text into integer sequences (top 10,000
   words, padded/truncated to 100 tokens), then feeds them through:
   `Embedding → Bidirectional LSTM → Dense(ReLU) → Dropout → Dense(softmax, 4 classes)`.
   Class weights correct for the imbalance toward "minimum" severity.
   Early stopping prevents overfitting.
3. **evaluate.py** — loads the saved model and reports precision/recall/F1
   per class plus a confusion matrix on data the model never saw.
4. **predict.py** — loads the saved model/tokenizer and lets you type a
   sentence to see the predicted severity + confidence breakdown.

## Current results (on held-out test set)

Overall accuracy: ~58–61% on a 4-class problem (random guessing would be
~25%), with much stronger performance on the majority "minimum" class
than the minority classes — a known effect of dataset imbalance and small
dataset size. This is expected and totally fine for a demo project; it's
also a good, honest talking point about the real-world challenges of this
kind of task (imbalanced classes, subjective/noisy labels, limited data).

## Ideas to extend this later (optional)

- Swap the LSTM for a pretrained transformer (e.g. DistilBERT) for higher
  accuracy.
- Collapse to binary classification (depressed vs. not) for simpler,
  higher-accuracy results.
- Add a small web UI (Flask/Streamlit) instead of the CLI.
- Add attention/SHAP visualization to show which words drove a prediction.
