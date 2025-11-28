import re
import spacy
from functools import lru_cache
from spacy.language import Language

APOSTROPHES = ["’", "ʼ", "`", "´", "ʻ"]

@lru_cache(maxsize=1)
def _get_nlp(model_name: str = "uk_core_news_sm") -> Language:
    return spacy.load(model_name)

def normalize_soft(s: str) -> str:
    s = s.replace("\r", "")
    for a in APOSTROPHES:
        s = s.replace(a, "'")
    s = s.replace("\u00A0", " ").replace("\u200B", "")
    s = re.sub(r"https?://\S+|www\.\S+", "", s)
    s = re.sub(r"[ \t]+", " ", s)
    s = "\n".join(line.strip() for line in s.splitlines())
    return s.strip()

def normalize_for_spacy(s: str) -> str:
    s = s.replace("\r", " ")
    for a in APOSTROPHES:
        s = s.replace(a, "'")
    s = s.replace("\u00A0", " ").replace("\u200B", "")
    s = re.sub(r"https?://\S+|www\.\S+", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def get_documents_by_text(text):
    text = normalize_for_spacy(text)
    nlp = _get_nlp()
    doc = nlp(text)
    return doc

def get_sentences_by_text(text):
    doc = get_documents_by_text(text)
    sentences = [s.text.strip() for s in doc.sents if s.text.strip()]
    return sentences or []

def get_paragraphs_by_text(text):
    text = normalize_soft(text)
    paragraphs = [
        p.strip()
        for p in re.split(r"\n\s*\n+", text or "")
        if p.strip()
    ]
    return paragraphs

