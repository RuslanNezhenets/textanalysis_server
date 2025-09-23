
from __future__ import annotations
import json, re
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import spacy
from sentence_transformers import SentenceTransformer

from text_division import text_division
from universal_text_classifier import LabelSchema, UniversalTextClassifier
from models import TopicHit, SegmentationRequest, SentenceResult, ClassifyResponse

# ---------- helpers: идентичны интентовым ----------
_LEVELS = ("Low", "Medium", "High")

def _label_from_conf(p: float) -> str:
    if p >= 0.60: return "High"
    if p >= 0.45: return "Medium"
    return "Low"

def _bump(level: str, delta: int) -> str:
    i = _LEVELS.index(level)
    i = max(0, min(len(_LEVELS) - 1, i + delta))
    return _LEVELS[i]

def _labels_per_span(hits: List[TopicHit]) -> List[str]:
    """
    Формирует список словесных меток для топ-k тем одного фрагмента:
    база — пороги по conf; для top-1 учитываем margin (p1 - p2).
    """
    if not hits:
        return []
    confs_sorted = sorted((h.conf for h in hits), reverse=True)
    p1 = confs_sorted[0]
    p2 = confs_sorted[1] if len(confs_sorted) > 1 else 0.0
    margin = p1 - p2

    labels = [_label_from_conf(h.conf) for h in hits]
    if margin >= 0.10:
        labels[0] = _bump(labels[0], +1)
    elif margin < 0.06:
        labels[0] = _bump(labels[0], -1)
    return labels

def _simple_split(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", (text or "").strip())
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]

# ---------- основной класс ----------
class TopicsClassifier:
    """
    Универсальная обёртка под тематики.
    Параметры и поведение синхронизированы с IntentClassifier.
    """

    def __init__(
        self,
        templates: Dict[str, Any],
        model_name: str = "paraphrase-xlm-r-multilingual-v1",
        device: Optional[str] = None,
        use_mahalanobis: bool = True,
        agg_weights: Tuple[float, float] = (0.7, 0.3),
        T_maha: float = 0.75,
        T_maxcos: float = 0.85,
        prob_threshold: float = 0.30,
        margin_delta: float = 0.035,
        per_label_thresholds: Optional[Dict[str, float]] = None,
        model: Optional[SentenceTransformer] = None
    ):
        schema = LabelSchema(templates)

        temperatures = {
            "MahalanobisChannel": float(T_maha) if use_mahalanobis else 1.0,
            "CosineMaxChannel": float(T_maxcos),
        }
        weights = {
            "MahalanobisChannel": float(agg_weights[0]) if use_mahalanobis else 0.0,
            "CosineMaxChannel": float(agg_weights[1]),
        }

        logs = []
        def jsonl_logger(rec: dict):
            # пример: складываем в список или пишем в файл
            logs.append(rec)

        self._clf = UniversalTextClassifier(
            schema=schema,
            model=model,
            model_name=model_name,
            device=device,
            temperatures=temperatures,
            weights=weights,
            prob_threshold=prob_threshold,
            margin_delta=margin_delta,
            per_label_thresholds=per_label_thresholds or {},
            collect_explanations=True,  # включаем возврат объяснений
            explain_top_m=5,  # top-5 ближайших примеров по CosineMax
            logger=jsonl_logger  # опционально: строчный JSON-лог
        )

        self.topic_list = list(schema.keys)
        self.topic_titles = {k: schema.titles.get(k, k) for k in self.topic_list}

    def encode(self, texts: List[str]) -> np.ndarray:
        return self._clf.encode(texts)

    def classify_sentences(self, sentences: List[str], top_k: int = 3) -> List[List[TopicHit]]:
        """
        Возвращает по каждому предложению список TopicHit (top-k),
        где у каждого уже проставлен conf_label по тем же правилам, что у интентов.
        """
        res = self._clf.predict(sentences, top_k=top_k, return_channel_scores=False)

        for p in res.predictions:
            print("=" * 80)
            print(f"[{p.id}] {p.text}")
            if not p.labels:
                continue
            top_label = p.labels[0]
            print("Top:", top_label.key, f"{top_label.prob:.3f}", "low_conf" if top_label.low_conf else "")
            print("Explain:")
            print(json.dumps(top_label.extras["explain"], ensure_ascii=False, indent=2))

        out: List[List[TopicHit]] = []
        for sp in res.predictions:
            hits: List[TopicHit] = []
            for lp in sp.labels:
                hits.append(
                    TopicHit(
                        topic=lp.key,            # внешнее имя
                        name=lp.name,
                        score=float(np.log(lp.prob + 1e-12)),  # как в интентах
                        conf=float(lp.prob),
                        sim=float(lp.mix_logit),
                        low_conf=bool(lp.low_conf)
                    )
                )
            # добавляем словесные метки уверенности (учитываем margin для top-1)
            labels = _labels_per_span(hits)
            for i, h in enumerate(hits):
                h.conf_label = labels[i] if i < len(labels) else _label_from_conf(h.conf)
            out.append(hits)
        return out

# ---------- загрузчики ----------
def load_templates(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_topics_classifier(
    templates_path: str,
    model_name: str = "paraphrase-xlm-r-multilingual-v1",
    **kwargs
) -> TopicsClassifier:
    """Дефолты совпадают с интент-обёрткой."""
    templates = load_templates(templates_path)
    defaults = dict(
        use_mahalanobis=True,
        agg_weights=(0.7, 0.3),
        T_maha=0.75,
        T_maxcos=0.85,
        prob_threshold=0.30,
        margin_delta=0.035,
    )
    defaults.update(kwargs)
    return TopicsClassifier(templates, model_name=model_name, **defaults)

def topics_definition(req: SegmentationRequest, nlp: Optional[spacy.language.Language] = None, clf: Any = None, st_model: Optional[SentenceTransformer] = None):
    t0 = time.perf_counter()

    doc = text_division(req, nlp=nlp, st_model=st_model)

    if not doc.blocks:
        return doc

    texts = [blk.text for blk in doc.blocks]
    # raw_hits = clf.classify_sentences(texts, top_k=req.top_k)
    raw_hits = clf.classify_sentences(texts, top_k=3)

    results: List[SentenceResult] = []
    for idx, (sent, per_sent) in enumerate(zip(doc.blocks, raw_hits)):
        topics = []
        conf_labels = _labels_per_span(per_sent)
        for i, h in enumerate(per_sent):
            payload = {
                "topic": h.topic,
                "name": h.name,
                "conf": float(h.conf),
                "low_conf": bool(getattr(h, "low_conf", False)),
                "conf_label": conf_labels[i]
            }
            if getattr(req, "debug", False):
                payload.update({"score": float(h.score), "sim": float(h.sim)})
            topics.append(payload)
        results.append(SentenceResult(id=idx, text=sent.text, top=topics, span_range=sent.span_range))

    metrics = {
        "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
        "n_blocks": len(doc.blocks),
        # "top_k": req.top_k,
        "top_k": 3,
        "model_used": req.model_name,
    }

    return ClassifyResponse(results=results, metrics=metrics)