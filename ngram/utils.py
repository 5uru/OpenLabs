import unicodedata
import re

def clean_text(text: str) -> str:
    """Normalize text: Unicode NFKC, ASCII, remove digits/punct, lowercase, normalize spaces."""
    text = unicodedata.normalize("NFKC", text)
    text = text.encode("ascii", "ignore").decode()
    text = re.sub(r"\d", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)  # delete punctuation
    return text