from typing import Any, Dict, List, Optional

import numpy as np
import spacy
import re
from sentence_transformers import SentenceTransformer

from nlpclf.topics_agg import aggregate_block_topics_with_override
from nlpclf.compat.base_compat import BaseCompatClassifier
from nlpclf.compat.common import load_templates, labels_with_margin
from nlpclf.segmentation.pipeline import text_division


from models.models import TopicHit, SegmentationRequest, SentenceResult, ClassifyResponse


class TopicsClassifier(BaseCompatClassifier):
    """
    Сумісна обгортка для тематичної класифікації.
    Працює поверх TextClassifier, конфіг читає BaseCompatClassifier.
    """

    def classify_sentences(self, sentences: List[str]) -> List[List[TopicHit]]:
        """
        Класифікує список речень і повертає для кожного впорядкований набір тем.

        Args:
            sentences (List[str]): Речення або короткі фрагменти тексту.

        Returns:
            List[List[TopicHit]]: Для кожного речення — список TopicHit (спадно за ймовірністю),
                структуровано як:
                  - topic (str), name (str), conf (float 0..1),
                  - score=log(conf), sim (змішаний логіт),
                  - low_conf (bool), conf_label ("Low/Medium/High").
        """
        preds = self._predict_hits(sentences)
        out: List[List[TopicHit]] = []

        for sp in preds:
            confs = [lp.prob for lp in sp.labels]
            labels = labels_with_margin(confs, margin_hi=0.10, margin_lo=0.06)

            hits: List[TopicHit] = []
            for i, lp in enumerate(sp.labels):
                h = TopicHit(
                    topic=lp.key,
                    name=lp.name,
                    score=float(np.log(lp.prob + 1e-12)),
                    conf=float(lp.prob),
                    sim=float(lp.mix_logit),
                    low_conf=bool(lp.low_conf),
                    conf_label=labels[i] if i < len(labels) else "Low"
                )
                hits.append(h)
            out.append(hits)
        return out


def build_topics_classifier(
    templates_path: str,
    *,
    config_path: Optional[str] = None,
    cfg: Optional[Dict[str, Any]] = None,
    model=None,
) -> TopicsClassifier:
    """
    Створити TopicsClassifier, що бере всі налаштування з конфіга.

    Args:
        templates_path (str): Шлях до JSON зі схемою тем (ключ, назва, приклади).
        config_path (str | None): Шлях до YAML/JSON конфіга.
        cfg (dict | None): Уже завантажений конфіг (перекриває файл).
        model: Готова SentenceTransformer (опційно).

        Returns:
            TopicsClassifier: Готовий класифікатор тем.
    """
    templates = load_templates(templates_path)
    return TopicsClassifier(templates, config_path=config_path, cfg=cfg, model=model)

def _split_sentences(text: str, nlp: Optional[spacy.language.Language]) -> List[str]:
    """
    Розбиває текст на речення за spaCy (якщо надано), або простим правилом.

    Args:
        text (str): Вхідний текст.
        nlp (spacy.language.Language | None): Пайплайн spaCy або None.

    Returns:
        List[str]: Список речень без порожніх рядків.
    """
    if nlp is None:
        return [s.strip() for s in re.split(r'(?<=[\.\!\?…])\s+', text or "") if s.strip()]
    doc = nlp(text or "")
    return [s.text.strip() for s in doc.sents if s.text.strip()]


def topics_definition(
    req: SegmentationRequest,
    nlp: Optional[spacy.language.Language] = None,
    clf: Any = None,
    st_model: Optional[SentenceTransformer] = None
):
    t0 = __import__("time").perf_counter()

    cfg = (getattr(clf, "config", {}) or {}).get("segmentation", {})
    doc = text_division(req, nlp=nlp, st_model=st_model, cfg=cfg)
    if not getattr(doc, "blocks", None):
        return doc

    # Витягуємо конфіг секцій для агрегації
    full_cfg: Dict[str, Any] = getattr(clf, "config", {}) or {}
    agg_cfg = full_cfg.get("topics_agg", {}) or {}
    override_cfg = full_cfg.get("topics_override", {}) or {}

    results: List[SentenceResult] = []

    def _split_sentences(text: str) -> List[str]:
        return [s.strip() for s in re.split(r'(?<=[\.\!\?…])\s+', text or "") if s.strip()]

    for idx, blk in enumerate(doc.blocks):
        block_text = blk.text
        if hasattr(blk, "sentences") and blk.sentences:
            sentences = [s if isinstance(s, str) else getattr(s, "text", "") for s in blk.sentences]
            sentences = [s for s in sentences if s]
        else:
            sentences = _split_sentences(block_text)

        per_sent_hits: List[List[Dict[str, Any]]] = []
        if sentences:
            sent_raw = clf.classify_sentences(sentences)
            for per_sent in sent_raw:
                per_sent_hits.append([
                    {
                        "topic": h.topic, "name": h.name,
                        "conf": float(h.conf), "low_conf": bool(getattr(h, "low_conf", False)),
                        "score": float(getattr(h, "score", 0.0)), "sim": float(getattr(h, "sim", 0.0)),
                        "is_rhetorical": bool(getattr(h, "is_rhetorical", False)),
                    } for h in per_sent
                ])

        top_topics, _mass, _dbg = aggregate_block_topics_with_override(per_sent_hits, agg_cfg=agg_cfg, override_cfg=override_cfg)

        results.append(SentenceResult(
            id=idx,
            text=block_text,
            top=top_topics,
            span_range=getattr(blk, "span_range", None),
        ))

    metrics = {
        "latency_ms": round((__import__("time").perf_counter() - t0) * 1000, 2),
        "n_blocks": len(doc.blocks),
        "model_used": req.model_name,
    }
    return ClassifyResponse(results=results, metrics=metrics)
