import pandas as pd
import nltk
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def ensure_nltk_resources() -> None:
    local_nltk_dir = os.path.join(os.getcwd(), ".nltk_data")
    os.makedirs(local_nltk_dir, exist_ok=True)
    if local_nltk_dir not in nltk.data.path:
        nltk.data.path.insert(0, local_nltk_dir)
    nltk.download("punkt", quiet=True, download_dir=local_nltk_dir)
    nltk.download("punkt_tab", quiet=True, download_dir=local_nltk_dir)
    nltk.download("stopwords", quiet=True, download_dir=local_nltk_dir)


def clean_text(text: str) -> list[str]:
    text = str(text).lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.isalnum()]
    stop_words = set(stopwords.words("english"))
    tokens = [word for word in tokens if word not in stop_words]
    return tokens


def load_sms_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="latin-1")

    # Kaggle SMS dataset usually uses v1 (label) and v2 (text).
    if "v1" in df.columns and "v2" in df.columns:
        df = df[["v1", "v2"]].copy()
        df.columns = ["label", "text"]
    elif "label" in df.columns and "text" in df.columns:
        df = df[["label", "text"]].copy()
    else:
        raise ValueError(
            "Expected columns (v1, v2) or (label, text) in the dataset."
        )
    return df


def main() -> None:
    ensure_nltk_resources()

    df = load_sms_dataset("spam.csv")

    print("\n=== First 5 rows ===")
    print(df.head())

    df["cleaned_text"] = df["text"].apply(clean_text)

    print("\n=== Original vs cleaned sample ===")
    print(df[["text", "cleaned_text"]].head())

    print("\n=== Dataset inspection ===")
    print("Total messages:", len(df))
    print(df["label"].value_counts())

    df["label_num"] = df["label"].map({"ham": 0, "spam": 1})

    print("\n=== Label mapping check ===")
    print(df[["label", "label_num"]].head())

    print("\nDay 1 pipeline complete: dataset loaded, text cleaned, labels converted.")


if __name__ == "__main__":
    main()
