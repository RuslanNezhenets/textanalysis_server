import json
import re
from typing import List, Dict, Any


def split_sentences_generic(text: str, nlp) -> List[str]:
    doc = nlp(re.sub(r"\s+", " ", text.strip()))
    return [s.text.strip() for s in doc.sents if s.text.strip()]

def load_templates(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)