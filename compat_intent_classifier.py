import json, re, time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import spacy
from sentence_transformers import SentenceTransformer

from universal_text_classifier import (
    LabelSchema,
    UniversalTextClassifier
)

# --- import your dataclasses (unchanged) ---
from models import IntentHit, ClassifyRequest, ClassifyResponse, SentenceResult

# ---------------- helpers (unchanged logic) ----------------

_LEVELS = ("Low", "Medium", "High")

def _label_from_conf(p: float) -> str:
    if p >= 0.60: return "High"
    if p >= 0.45: return "Medium"
    return "Low"

def _bump(level: str, delta: int) -> str:
    i = _LEVELS.index(level)
    i = max(0, min(len(_LEVELS)-1, i+delta))
    return _LEVELS[i]

def _labels_per_intent(per_sent: List[IntentHit]) -> List[str]:
    if not per_sent: return []
    p_sorted = sorted((h.conf for h in per_sent), reverse=True)
    p_top = p_sorted[0]
    p_2nd = p_sorted[1] if len(p_sorted) > 1 else 0.0
    margin = p_top - p_2nd
    labels = [_label_from_conf(h.conf) for h in per_sent]
    top_idx = 0
    if margin >= 0.10:
        labels[top_idx] = _bump(labels[top_idx], +1)
    elif margin < 0.06:
        labels[top_idx] = _bump(labels[top_idx], -1)
    return labels

def _simple_split(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", (text or "").strip())
    if not text: return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]

# ---------------- main compatibility class ----------------

class IntentClassifier:
    """
    Drop-in API-compatible wrapper.
    Internally delegates to UniversalTextClassifier with tuned params
    to match your original default behaviour.
    """

    def __init__(
        self,
        templates: Dict[str, Any],
        model_name: str = "paraphrase-xlm-r-multilingual-v1",
        device: Optional[str] = None,
        use_mahalanobis: bool = True,
        mahalanobis_reg: float = 0.10,
        agg_weights: Tuple[float, float] = (0.7, 0.3),
        T_maha: float = 0.75,
        T_maxcos: float = 0.85,
        prob_threshold: float = 0.48,
        margin_delta: float = 0.07,
        per_class_thresholds: Optional[Dict[str, float]] = None,
        model: Optional[SentenceTransformer] = None
    ):
        # Build LabelSchema from your templates
        schema = LabelSchema(templates)

        # Temperatures & weights mapped to channel names
        temperatures = {
            "MahalanobisChannel": float(T_maha) if use_mahalanobis else 1.0,
            "CosineMaxChannel": float(T_maxcos),
        }
        weights = {
            "MahalanobisChannel": float(agg_weights[0]) if use_mahalanobis else 0.0,
            "CosineMaxChannel": float(agg_weights[1]),
        }

        # Create underlying universal classifier
        self._clf = UniversalTextClassifier(
            schema=schema,
            model=model,
            model_name=model_name,
            device=device,
            # channels already include Mahalanobis + CosineMax by default
            temperatures=temperatures,
            weights=weights,
            prob_threshold=prob_threshold,
            margin_delta=margin_delta,
            per_label_thresholds=per_class_thresholds or {},
        )

        # expose for parity
        self.intent_list = list(schema.keys)
        self.intent_titles = {k: schema.titles.get(k, k) for k in self.intent_list}

    def _encode(self, texts: List[str]) -> np.ndarray:
        return self._clf.encode(texts)

    def classify_sentences(self, sentences: List[str], top_k: int = 3) -> List[List[IntentHit]]:
        res = self._clf.predict(sentences, top_k=top_k, return_channel_scores=False)

        out: List[List[IntentHit]] = []
        for sp in res.predictions:
            hits: List[IntentHit] = []
            for lp in sp.labels:
                # Note: score in your printout был log(prob), sim — это mix_logit
                hits.append(
                    IntentHit(
                        intent=lp.key,
                        name=lp.name,
                        score=float(np.log(lp.prob + 1e-12)),
                        conf=float(lp.prob),
                        sim=float(lp.mix_logit),
                        low_conf=bool(lp.low_conf),
                    )
                )
            out.append(hits)
        return out

# ---------------- convenience loaders ----------------

def load_templates(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_classifier(templates_path: str, model_name: str = "paraphrase-xlm-r-multilingual-v1", **kwargs) -> IntentClassifier:
    templates = load_templates(templates_path)
    # IMPORTANT: set defaults to your original demo CLI values
    defaults = dict(
        use_mahalanobis=True,
        mahalanobis_reg=0.10,
        agg_weights=(0.7, 0.3),
        T_maha=0.75,
        T_maxcos=0.85,
        prob_threshold=0.30,   # как в твоём __main__
        margin_delta=0.035,    # как в твоём __main__
    )
    defaults.update(kwargs)
    return IntentClassifier(templates, model_name=model_name, **defaults)

# ---------------- analysis function (API parity) ----------------

def intent_analysis(req: ClassifyRequest, nlp: Optional[spacy.language.Language] = None, clf: Any = None):
    t0 = time.perf_counter()
    # sentence split: keep your logic
    if nlp is not None:
        doc = nlp(req.text)
        sentences = [s.text.strip() for s in doc.sents if s.text.strip()]
    else:
        sentences = _simple_split(req.text)

    raw_hits = clf.classify_sentences(sentences, top_k=req.top_k)

    results: List[SentenceResult] = []
    for idx, (sent, per_sent) in enumerate(zip(sentences, raw_hits)):
        intents = []
        # compute conf_label exactly as before
        conf_labels = _labels_per_intent(per_sent)
        for i, h in enumerate(per_sent):
            payload = {
                "intent": h.intent,
                "name": h.name,
                "conf": float(h.conf),
                "low_conf": bool(getattr(h, "low_conf", False)),
                "conf_label": conf_labels[i]
            }
            if getattr(req, "debug", False):
                payload.update({"score": float(h.score), "sim": float(h.sim)})
            intents.append(payload)
        results.append(SentenceResult(id=idx, text=sent, top=intents))

    metrics = {
        "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
        "n_sentences": len(sentences),
        "top_k": req.top_k,
        "model_used": req.model_name,
    }
    return ClassifyResponse(results=results, metrics=metrics)
