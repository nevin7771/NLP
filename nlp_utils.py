"""Shared NLP helpers (same preprocessing as Day 1–3 notebooks)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

PROJECT_ROOT = Path(__file__).resolve().parent


def _writable_nltk_dir(project_root: Path) -> Path:
    """Prefer project-local NLTK data; fall back to /tmp if the app dir is not writable (e.g. some cloud hosts)."""
    primary = project_root / ".nltk_data"
    try:
        primary.mkdir(parents=True, exist_ok=True)
        probe = primary / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return primary
    except OSError:
        pass
    fallback = Path(tempfile.gettempdir()) / "sms_spam_nltk_data"
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


def ensure_nltk(project_root: Path | None = None) -> None:
    root = project_root or PROJECT_ROOT
    local_nltk_dir = _writable_nltk_dir(root)
    p = str(local_nltk_dir)
    if p not in nltk.data.path:
        nltk.data.path.insert(0, p)
    nltk.download("punkt", quiet=True, download_dir=p)
    nltk.download("punkt_tab", quiet=True, download_dir=p)
    nltk.download("stopwords", quiet=True, download_dir=p)


def clean_text(text: str) -> list[str]:
    text = str(text).lower()
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w.isalnum()]
    sw = set(stopwords.words("english"))
    return [w for w in tokens if w not in sw]


def default_csv_path() -> Path:
    return PROJECT_ROOT / "spam.csv"
