from typing import Dict, Any, List
from compat_topic_classifier import TopicsClassifier

def annotate_blocks_with_topics(doc: Dict[str, Any],
                                clf: TopicsClassifier,
                                top_k: int = 5,
                                field_name: str = "topics_top") -> Dict[str, Any]:
    """
    doc: структура с ключом "blocks": [{index, text, ...}, ...]
    Заполняет для каждого блока doc["blocks"][i][field_name] = [ [key, conf, sim, low_conf], ... ]
    Возвращает модифицированную копию doc.
    """
    blocks = doc.get("blocks", []) or []
    texts = [b.get("text", "") for b in blocks]
    preds: List[List] = clf.classify_sentences(texts, top_k=top_k)

    out = {**doc}
    new_blocks = []
    for b, hits in zip(blocks, preds):
        rows = []
        for h in hits:
            # сохраняем компактно, как ты выводил для intents
            rows.append([h.intent, round(h.conf, 6), round(h.score, 6), bool(getattr(h, "low_conf", False))])
        bb = dict(b)
        bb[field_name] = rows
        new_blocks.append(bb)

    out["blocks"] = new_blocks
    return out
