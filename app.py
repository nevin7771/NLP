"""
Week 1 Day 4: Streamlit UI for SMS spam vs ham prediction.
Uses TF-IDF + LogisticRegression(class_weight='balanced') trained on the full dataset.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from nlp_utils import clean_text, default_csv_path, ensure_nltk


@st.cache_resource(show_spinner="Loading model (first run trains on full data)...")
def load_tfidf_and_model():
    ensure_nltk()
    csv_path = default_csv_path()
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing dataset: {csv_path}")

    df = pd.read_csv(csv_path, encoding="latin-1")
    df = df[["v1", "v2"]].copy()
    df.columns = ["label", "text"]
    df["label_num"] = df["label"].map({"ham": 0, "spam": 1})

    df["cleaned_text"] = df["text"].apply(clean_text)
    df["cleaned_text_str"] = df["cleaned_text"].apply(lambda x: " ".join(x))

    tfidf = TfidfVectorizer(max_features=3000)
    X = tfidf.fit_transform(df["cleaned_text_str"])
    y = df["label_num"]

    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight="balanced",
        solver="liblinear",
    )
    model.fit(X, y)
    return tfidf, model


def predict(text: str, tfidf: TfidfVectorizer, model: LogisticRegression):
    cleaned_str = " ".join(clean_text(text))
    X = tfidf.transform([cleaned_str])
    proba = model.predict_proba(X)[0]
    pred = int(model.predict(X)[0])
    return pred, proba


def main():
    st.set_page_config(page_title="SMS Spam Detector", page_icon="📧", layout="centered")
    st.title("SMS Spam Detector")
    st.caption("Week 1 · Day 4 · TF-IDF + balanced Logistic Regression")

    tfidf, model = load_tfidf_and_model()

    text = st.text_area(
        "Paste an SMS message",
        height=120,
        placeholder='e.g. "Congratulations! You won a free iPhone. Click here now!"',
    )

    if st.button("Predict", type="primary"):
        if not text.strip():
            st.warning("Enter some text first.")
            return
        pred, proba = predict(text, tfidf, model)
        ham_p, spam_p = float(proba[0]), float(proba[1])
        label = "SPAM" if pred == 1 else "HAM"
        st.subheader(label)
        col1, col2 = st.columns(2)
        col1.metric("HAM probability", f"{ham_p:.1%}")
        col2.metric("SPAM probability", f"{spam_p:.1%}")
        st.caption("Spam score (visual)")
        st.progress(min(1.0, max(0.0, spam_p)))

    with st.expander("How this works"):
        st.markdown(
            """
            - Same preprocessing as Days 1–3 (lowercase, tokenize, remove stopwords).
            - **TF-IDF** turns text into numbers; **Logistic Regression** with `class_weight='balanced'`
              matches the Day 3 setup that improved spam recall.
            - The model is trained on the **full** `spam.csv` for this demo (good for a small app;
              in production you would load a saved artifact trained offline).
            """
        )


if __name__ == "__main__":
    main()
