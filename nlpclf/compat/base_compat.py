from typing import Any, Dict, List, Optional

import numpy as np

from nlpclf.classifier import LabelSchema, TextClassifier
from nlpclf.channels.cosine import CosineExemplarMax
from nlpclf.channels.mahalanobis import CenterMahalanobis
from nlpclf.config import load_config

_CHANNELS_REGISTRY = {
    "CosineExemplarMax": CosineExemplarMax,
    "CenterMahalanobis": CenterMahalanobis,
}

class BaseCompatClassifier:
    """
    Узкая обёртка над TextClassifier с совместимыми дефолтами и полями
    (чтобы старые вызовы работали без изменений).
    """

    def __init__(self, templates: Dict[str, Any], *, config_path: Optional[str] = None, cfg: Optional[Dict[str, Any]] = None, model=None) -> None:
        """
        Args:
            templates (dict): Схема міток (ключ -> {title, examples}).
            config_path (str | None): Шлях до YAML/JSON конфіга.
            cfg (dict | None): Готовий конфіг (перекриває файл).
            model: Готова SentenceTransformer (опційно).

        Returns:
            None
        """
        schema = LabelSchema(templates)
        conf = load_config(config_path, cfg)

        # --- Пристрої: "auto" -> None (нехай вирішує сама модель)
        device = conf.get("device", "auto")
        device = None if str(device).lower() == "auto" else str(device).lower()

        # --- Канали: інстансуємо за назвою, збираємо ваги/температури
        channels = []
        temps: Dict[str, float] = {}
        weights: Dict[str, float] = {}

        for item in conf.get("channels", []):
            name = item["name"]
            cls = _CHANNELS_REGISTRY.get(name)
            if cls is None:
                raise ValueError(f"Unknown channel: {name}")
            inst = cls(**(item.get("args") or {}))
            channels.append(inst)
            temps[name] = float(item.get("temperature", 1.0))
            weights[name] = float(item.get("weight", 1.0))

        # --- Пороги
        th = conf.get("thresholds", {}) or {}
        prob_threshold = float(th.get("default", 0.5))
        per_label_thresholds = th.get("per_label", {}) or {}
        margin_delta = float(conf.get("margin_delta", 0.07))

        # --- Збираємо ядро
        self._clf = TextClassifier(
            schema=schema,
            model=model,
            model_name=str(conf.get("model")),
            device=device,
            channels=channels,
            temperatures=temps,
            weights=weights,
            prob_threshold=prob_threshold,
            margin_delta=margin_delta,
            per_label_thresholds=per_label_thresholds,
        )

        # Плоскі довідники (зручно в сумісному шарі)
        self.label_keys = list(schema.keys)
        self.label_titles = {k: schema.titles.get(k, k) for k in self.label_keys}

    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Args:
            texts: список строк.
        Returns:
            np.ndarray: матрица эмбеддингов (N, d), float32.
        """
        return self._clf.encode(texts)

    def _predict_hits(self, sentences: List[str]):
        """
        Внутренняя утилита: вызывает TextClassifier и отдаёт список объектов
        с полями key/name/prob/mix_logit/low_conf для каждого предложения.
        """
        res = self._clf.predict(sentences, return_channel_scores=False)
        return res.predictions
