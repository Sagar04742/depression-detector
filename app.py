"""
app.py
------
Professional web UI (Streamlit) for the depression-severity text classifier.

Run with:
    streamlit run app.py

Opens a local website in your browser where you can paste text and see
the predicted severity level with a clean, presentation-ready layout.
"""

import json
import pickle
from datetime import datetime

import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from preprocess import clean_text

# ----------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="MindScan — Depression Severity Detector",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------
SEVERITY_STYLE = {
    "minimum": {"color": "#2E7D32", "bg": "#E8F5E9", "label": "Minimum"},
    "mild":    {"color": "#F9A825", "bg": "#FFF8E1", "label": "Mild"},
    "moderate":{"color": "#EF6C00", "bg": "#FFF3E0", "label": "Moderate"},
    "severe":  {"color": "#C62828", "bg": "#FFEBEE", "label": "Severe"},
}

st.markdown("""
<style>
    #MainMenu, footer, header {visibility: hidden;}

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 950px;
    }

    .app-header {
        background: linear-gradient(135deg, #4A5FE8 0%, #6C4AE8 100%);
        padding: 2.2rem 2.5rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 1.5rem;
    }
    .app-header h1 {
        margin: 0;
        font-size: 2rem;
        font-weight: 700;
    }
    .app-header p {
        margin: 0.4rem 0 0 0;
        opacity: 0.9;
        font-size: 1.02rem;
    }

    .disclaimer-box {
        background: #FFF3E0;
        border-left: 4px solid #EF6C00;
        padding: 0.9rem 1.2rem;
        border-radius: 8px;
        font-size: 0.92rem;
        color: #4a3b1f;
        margin-bottom: 1.5rem;
    }

    .card {
        background: white;
        border: 1px solid #ECECF3;
        border-radius: 14px;
        padding: 1.6rem 1.8rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.04);
        margin-bottom: 1.2rem;
    }

    .section-title {
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: #8A8AA3;
        margin-bottom: 0.6rem;
    }

    .result-badge {
        display: inline-block;
        padding: 0.35rem 1rem;
        border-radius: 999px;
        font-weight: 700;
        font-size: 1.05rem;
        margin-bottom: 0.3rem;
    }

    .conf-row {
        display: flex;
        align-items: center;
        gap: 0.8rem;
        margin: 0.5rem 0;
    }
    .conf-label {
        width: 90px;
        font-size: 0.9rem;
        font-weight: 600;
        color: #444;
    }
    .conf-pct {
        width: 48px;
        text-align: right;
        font-size: 0.85rem;
        color: #666;
    }
    .conf-track {
        flex: 1;
        background: #F0F0F5;
        border-radius: 6px;
        height: 14px;
        overflow: hidden;
    }
    .conf-fill {
        height: 100%;
        border-radius: 6px;
    }

    .metric-box {
        background: #F7F8FC;
        border-radius: 10px;
        padding: 0.9rem 1rem;
        text-align: center;
    }
    .metric-box .num {
        font-size: 1.4rem;
        font-weight: 700;
        color: #4A5FE8;
    }
    .metric-box .lbl {
        font-size: 0.78rem;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    .stButton>button {
        background: linear-gradient(135deg, #4A5FE8 0%, #6C4AE8 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 1.6rem;
        font-weight: 600;
        font-size: 1rem;
        width: 100%;
    }
    .stButton>button:hover {
        opacity: 0.92;
        color: white;
    }

    .example-chip {
        display: inline-block;
        background: #F0F1FA;
        color: #4A5FE8;
        padding: 0.3rem 0.8rem;
        border-radius: 8px;
        font-size: 0.82rem;
        margin: 0.15rem;
    }
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------
# Model loading
# ----------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    with open("config.json") as f:
        config = json.load(f)
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    model = load_model("model.keras")
    return model, tokenizer, config["labels"], config["max_len"]


def predict(text, model, tokenizer, labels, max_len):
    cleaned = clean_text(text)
    seq = pad_sequences(
        tokenizer.texts_to_sequences([cleaned]),
        maxlen=max_len, padding="post", truncating="post",
    )
    probs = model.predict(seq, verbose=0)[0]
    return dict(zip(labels, probs))


# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🧠 About this project")
    st.write(
        "A neural network (Bidirectional LSTM) trained on Reddit posts "
        "to classify text into a depression-severity level."
    )
    st.markdown("---")
    st.markdown("### 🛠️ How it works")
    st.write(
        "1. Text is cleaned and tokenized\n"
        "2. Fed through an Embedding + BiLSTM network\n"
        "3. Softmax output gives a probability per severity class"
    )
    st.markdown("---")
    st.markdown("### 💬 Try an example")
    examples = [
        "I've been laughing with friends all week, life feels great!",
        "I'm a bit stressed about exams but otherwise doing okay.",
        "I feel numb most days and don't enjoy things like I used to.",
        "I feel hopeless and can't get out of bed, nothing matters anymore.",
    ]
    for ex in examples:
        if st.button(ex, key=ex, use_container_width=True):
            st.session_state["text_input"] = ex
    st.markdown("---")
    st.caption("Built with TensorFlow/Keras + Streamlit · Educational demo")


# ----------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------
st.markdown("""
<div class="app-header">
    <h1>🧠 MindScan</h1>
    <p>Neural network–based depression severity text classifier — demo project</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="disclaimer-box">
    ⚠️ <b>Disclaimer:</b> This is an educational/demo project, not a medical
    or diagnostic tool. Predictions can be wrong and must never be used to
    make real decisions about anyone's mental health. If you or someone you
    know is struggling, please reach out to a mental health professional or
    a crisis helpline.
</div>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# Load model
# ----------------------------------------------------------------------
try:
    model, tokenizer, labels, max_len = load_artifacts()
except Exception:
    st.error(
        "⚠️ Model files not found. Please run `python3 preprocess.py` and "
        "`python3 train.py` in the project folder before launching this app."
    )
    st.stop()

# ----------------------------------------------------------------------
# Input card
# ----------------------------------------------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Enter text to analyze</div>', unsafe_allow_html=True)
text = st.text_area(
    label="text_input_area",
    label_visibility="collapsed",
    height=140,
    placeholder="Type or paste a journal entry, message, or social media post...",
    key="text_input",
)
col1, col2 = st.columns([1, 3])
with col1:
    analyze = st.button("🔍 Analyze Text")
with col2:
    st.caption(f"{len(text)} characters" if text else "")
st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------------------------------------------------
# Results
# ----------------------------------------------------------------------
if analyze:
    if not text or not text.strip():
        st.warning("Please enter some text first.")
    else:
        with st.spinner("Analyzing..."):
            result = predict(text, model, tokenizer, labels, max_len)
        ranked = sorted(result.items(), key=lambda x: -x[1])
        top_label, top_prob = ranked[0]
        style = SEVERITY_STYLE.get(top_label, {"color": "#333", "bg": "#eee", "label": top_label})

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Prediction</div>', unsafe_allow_html=True)
        st.markdown(
            f'<span class="result-badge" style="background:{style["bg"]}; color:{style["color"]};">'
            f'{style["label"].upper()} SEVERITY</span>',
            unsafe_allow_html=True,
        )
        st.caption(f"Top confidence: {top_prob*100:.1f}% · Analyzed at {datetime.now().strftime('%H:%M:%S')}")

        st.markdown('<br>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Confidence breakdown</div>', unsafe_allow_html=True)
        for label, prob in ranked:
            s = SEVERITY_STYLE.get(label, {"color": "#4A5FE8"})
            st.markdown(f"""
            <div class="conf-row">
                <div class="conf-label">{label.capitalize()}</div>
                <div class="conf-track">
                    <div class="conf-fill" style="width:{prob*100:.1f}%; background:{s['color']};"></div>
                </div>
                <div class="conf-pct">{prob*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Model performance context
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Model context</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown('<div class="metric-box"><div class="num">~60%</div><div class="lbl">Test Accuracy</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="metric-box"><div class="num">4</div><div class="lbl">Severity Classes</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown('<div class="metric-box"><div class="num">~3,550</div><div class="lbl">Training Posts</div></div>', unsafe_allow_html=True)
        st.caption(
            "Trained on a small public Reddit dataset with imbalanced classes — "
            "accuracy reflects the real-world difficulty of this task, not a production-grade system."
        )
        st.markdown('</div>', unsafe_allow_html=True)
else:
    st.info("👆 Enter some text above (or pick an example from the sidebar) and click **Analyze Text**.")
