# Week 1 - Day 1 (NLP): SMS Spam Preprocessing Notebook

This project is Day 1 of a 30-day NLP build journey.

## Dataset
- File: `spam.csv`
- Source: SMS Spam Collection Dataset (Kaggle/UCI)
- Approx size: ~5,500 messages
- Labels:
  - `ham` = normal message
  - `spam` = unwanted message

## What this Day 1 notebook does
`day1_nlp_spam.ipynb`:
- Loads the dataset
- Keeps only label and text columns
- Cleans text using NLTK:
  - lowercase
  - tokenization
  - alphanumeric filtering
  - stopword removal
- Inspects class distribution (ham vs spam)
- Converts labels to numeric (`ham -> 0`, `spam -> 1`)
- Includes step-by-step markdown explanations for presentation and LinkedIn sharing

## Run with uv
1. Install dependencies (already in `pyproject.toml`):
   ```bash
   uv sync
   ```
2. Open Jupyter Notebook:
   ```bash
   uv run jupyter notebook
   ```
3. Open and run `day1_nlp_spam.ipynb`.

## Execute notebook from terminal (optional)
```bash
uv run jupyter nbconvert --to notebook --execute --inplace day1_nlp_spam.ipynb
```

## Day 1 checklist
- [x] Dataset loaded
- [x] Text cleaned
- [x] Labels converted
- [x] ML-ready base dataframe prepared

## Next (Day 2)
- Convert cleaned text into numerical features using TF-IDF
- Train first baseline classifier (e.g., Naive Bayes)

## Day 2 notebook
- File: `day2_nlp_tfidf_model.ipynb`
- Includes:
  - TF-IDF feature engineering
  - Naive Bayes training
  - Accuracy and classification report
  - Custom message prediction demo

## Day 3 notebook
- File: `day3_nlp_model_evaluation.ipynb`
- Includes:
  - Why accuracy alone is misleading on imbalanced data
  - Confusion matrix + classification report (precision, recall, F1)
  - Naive Bayes vs Logistic Regression vs LR with `class_weight='balanced'`
  - Side-by-side comparison table (accuracy vs spam metrics)

## Day 4 — Streamlit app
- Files: `app.py`, `nlp_utils.py`
- Notebook guide: `day4_streamlit_app.ipynb` (how to run + what to demo)
- Run:
  ```bash
  uv sync
  uv run streamlit run app.py
  ```
- The app shows **HAM/SPAM** and **class probabilities** for pasted text (same preprocessing as earlier days; model uses balanced Logistic Regression + TF-IDF).
