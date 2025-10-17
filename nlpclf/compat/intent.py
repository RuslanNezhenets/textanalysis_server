from typing import Any, List, Optional, Dict

import numpy as np

from nlpclf.compat.base_compat import BaseCompatClassifier
from nlpclf.compat.common import load_templates, labels_with_margin

from models.models import IntentHit, ClassifyRequest, ClassifyResponse, SentenceResult

class IntentClassifier(BaseCompatClassifier):
    """
    Сумісна обгортка для класифікації інтенцій.
    Працює поверх TextClassifier, конфіг читає BaseCompatClassifier.
    """

    def classify_sentences(self, sentences: List[str]) -> List[List[IntentHit]]:
        """
        Класифікує список речень і повертає для кожного впорядкований набір інтенцій.

        Args:
            sentences (List[str]): Речення або короткі фрагменти тексту.

        Returns:
            List[List[IntentHit]]: Для кожного речення — список IntentHit (спадно за ймовірністю),
                де:
                  - intent (str): ключ інтенції;
                  - name (str): назва;
                  - conf (float): ймовірність (0..1);
                  - score (float): log(conf), для сумісності;
                  - sim (float): змішаний логіт;
                  - low_conf (bool): ознака низької впевненості;
                  - conf_label (str): «Low/Medium/High» з урахуванням відриву.
        """
        preds = self._predict_hits(sentences)
        out: List[List[IntentHit]] = []

        for sp in preds:
            confs = [lp.prob for lp in sp.labels]
            labels = labels_with_margin(confs, margin_hi=0.10, margin_lo=0.06)

            hits: List[IntentHit] = []
            for i, lp in enumerate(sp.labels):
                h = IntentHit(
                    intent=lp.key,
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


def build_intent_classifier(
    templates_path: str,
    *,
    config_path: Optional[str] = None,
    cfg: Optional[Dict[str, Any]] = None,
    model=None,
) -> IntentClassifier:
    """
    Створити IntentClassifier, що бере всі налаштування з конфіга.

    Args:
        templates_path (str): Шлях до JSON зі схемою інтенцій (ключ, назва, приклади).
        config_path (str | None): Шлях до YAML/JSON конфіга.
        cfg (dict | None): Уже завантажений конфіг (перекриває файл).
        model: Готова SentenceTransformer (опційно).

    Returns:
        IntentClassifier: Готовий класифікатор інтенцій.
    """
    templates = load_templates(templates_path)
    return IntentClassifier(templates, config_path=config_path, cfg=cfg, model=model)

def intent_analysis(req: ClassifyRequest, clf: Any) -> ClassifyResponse:
    """
    Совместимый entry-point анализа интентов (без печати/логов).

    Args:
        req: запрос с текстом и настройками.
        clf: собранный IntentClassifier.

    Returns:
        ClassifyResponse: список предложений с top-интентами и метриками.
    """
    import time, re
    t0 = time.perf_counter()

    text = re.sub(r"\s+", " ", (req.text or "").strip())
    sentences = [p for p in re.split(r"(?<=[.!?])\s+", text) if p]  # простое разбиение — как в источнике

    raw_hits = clf.classify_sentences(sentences)

    results: List[SentenceResult] = []
    for idx, (sent, per_sent) in enumerate(zip(sentences, raw_hits)):
        intents = []
        for h in per_sent:
            payload = {
                "intent": h.intent,
                "name": h.name,
                "conf": float(h.conf),
                "low_conf": bool(getattr(h, "low_conf", False)),
                "conf_label": getattr(h, "conf_label", "Low"),
            }
            if getattr(req, "debug", False):
                payload.update({"score": float(h.score), "sim": float(h.sim)})
            intents.append(payload)
        results.append(SentenceResult(id=idx, text=sent, top=intents[:req.top_k]))

    metrics = {
        "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
        "n_sentences": len(sentences),
        "model_used": req.model_name,
    }
    return ClassifyResponse(results=results, metrics=metrics)
