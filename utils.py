import re
from typing import List


def split_sentences_generic(text: str, nlp) -> List[str]:
    doc = nlp(re.sub(r"\s+", " ", text.strip()))
    return [s.text.strip() for s in doc.sents if s.text.strip()]