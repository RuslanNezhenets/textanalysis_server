import time
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from nlpclf.channels.base import ScoreChannel
from nlpclf.channels.cosine import CosineExemplarMax
from nlpclf.channels.mahalanobis import CenterMahalanobis
from nlpclf.schema import LabelSchema
from nlpclf.utils import _softmax


class TextClassifier:
    """
    Класифікатор, що поєднує кілька каналів оцінювання та повертає
    відсортований список міток для кожного вхідного речення.
    """

    def __init__(
        self,
        schema: LabelSchema,
        model: Optional[SentenceTransformer] = None,
        model_name: str = "paraphrase-xlm-r-multilingual-v1",
        device: Optional[str] = None,
        channels: Optional[List[ScoreChannel]] = None,
        temperatures: Optional[Dict[str, float]] = None,
        weights: Optional[Dict[str, float]] = None,
        prob_threshold: float = 0.48,
        margin_delta: float = 0.07,
        per_label_thresholds: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Args:
            schema (LabelSchema): Схема міток із назвами та прикладами.
            model (SentenceTransformer | None): Готова модель (якщо вже створена).
            model_name (str): Назва моделі, якщо `model` не заданий.
            device (str | None): Пристрій для моделі ('cpu', 'cuda', тощо). Якщо None — за замовчуванням моделі.
            channels (List[ScoreChannel] | None): Канали оцінювання. Якщо None — використати стандартні.
            temperatures (Dict[str, float] | None): Карта канал→температура softmax.
            weights (Dict[str, float] | None): Карта канал→вага в суміші.
            prob_threshold (float): Поріг довіри для top-мітки.
            margin_delta (float): Мінімальна різниця між 1-ю та 2-ю міткою, щоб не позначати low_conf.
            per_label_thresholds (Dict[str, float] | None): Індивідуальні пороги для окремих міток.
        """
        self.schema = schema
        self.encoder = model or SentenceTransformer(model_name)
        self.device = device
        if self.device:
            # Перемістити модель на заданий пристрій, якщо потрібно
            self.encoder.to(self.device)

        # Канали за замовчуванням (можна переозначити через config.py)
        if channels is None:
            channels = [CenterMahalanobis(reg=0.10), CosineExemplarMax()]
        self.channels = channels
        for ch in self.channels:
            ch.prepare(schema, self.encoder)

        # Температури та ваги каналів
        self.temperatures = temperatures or {ch.name: 0.85 for ch in self.channels}
        default_weights = {ch.name: (0.7 if isinstance(ch, CenterMahalanobis) else 0.3) for ch in self.channels}
        if weights:
            default_weights.update(weights)
        self.weights = default_weights

        # Пороги впевненості
        self.prob_threshold = float(prob_threshold)
        self.margin_delta = float(margin_delta)
        self.per_label_thresholds = per_label_thresholds or {}

        # Кеш-структури
        self._keys = list(schema.keys)
        self._titles = schema.titles

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """
        Закодувати тексти у вектори ембеддингів.

        Args:
            texts (Sequence[str]): Послідовність рядків (речення/фрази).

        Returns:
            np.ndarray: Матриця розміру (N, d) у форматі float32,
                де N — кількість текстів, d — розмірність ембеддинга.
        """
        embs = self.encoder.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=False,
            device=self.device,
        )
        return embs.astype(np.float32)

    def predict(
        self,
        sentences: Sequence[str],
        return_channel_scores: bool = False,
    ):
        """
        Класифікувати речення: для кожного повернути повний список міток,
        відсортований за ймовірністю, та службові метрики.

        Args:
            sentences (Sequence[str]): Список речень для класифікації.
            return_channel_scores (bool): Додати в кожну мітку «сирі» оцінки по каналах.

        Returns:
            SimpleNamespace:
                predictions: List[SimpleNamespace]
                    Для кожного речення об’єкт з полями:
                        - id (int): індекс у вхідному списку.
                        - text (str): саме речення.
                        - labels (List[SimpleNamespace]): відсортований список міток;
                            кожен елемент має:
                                * key (str): ключ мітки (напр., 'DEF').
                                * name (str): читабельна назва мітки.
                                * prob (float): підсумкова ймовірність (0..1).
                                * mix_logit (float): підсумковий змішаний логіт.
                                * low_conf (bool): ознака низької впевненості для top-мітки.
                                * extras (dict | None): опційно, «сирі» оцінки по каналах.
                        - raw_scores (dict | None): якщо return_channel_scores=True — повні вектори оцінок по каналах.
                metrics: SimpleNamespace
                    latency_ms (float): час обробки всього батчу, мс.
                    n_sentences (int): кількість речень.
                    model_used (str): клас моделі енкодера.
        """
        t0 = time.perf_counter()
        embs = self.encode(sentences)
        out: List[SimpleNamespace] = []

        for i, (txt, emb) in enumerate(zip(sentences, embs)):
            # Агрегація каналів у ймовірності
            probs_mix = np.zeros(len(self._keys), np.float32)
            mix_logit = np.zeros_like(probs_mix)
            raws: Dict[str, np.ndarray] = {}

            for ch in self.channels:
                r = ch.score(emb)  # (L,)
                raws[ch.name] = r
                probs_mix += float(self.weights.get(ch.name, 0.0)) * _softmax(
                    r, float(self.temperatures.get(ch.name, 1.0))
                )
                mix_logit += float(self.weights.get(ch.name, 0.0)) * r

            # Нормалізуємо суміш, щоб сума = 1
            probs_mix /= (np.sum(probs_mix) + 1e-12)

            # Відсортований порядок міток (найвища ймовірність — спочатку)
            order = np.argsort(-probs_mix)

            # Формуємо повний список міток
            labels: List[SimpleNamespace] = []
            for j in order:
                key = self._keys[j]
                name = self._titles.get(key, key)

                extras: Optional[Dict[str, Any]] = None
                if return_channel_scores:
                    extras = {"channel_scores": {n: float(raws[n][j]) for n in raws}}

                labels.append(
                    SimpleNamespace(
                        key=key,
                        name=name,
                        prob=float(probs_mix[j]),
                        mix_logit=float(mix_logit[j]),
                        low_conf=False,   # оновимо для top-мітки нижче
                        extras=extras,
                    )
                )

            # Визначаємо low_conf для top-мітки (поріг і різниця з другою)
            if len(order) >= 2 and labels:
                b, s = order[0], order[1]
                p_top = float(probs_mix[b])
                p_2 = float(probs_mix[s])
                tau = float(self.per_label_thresholds.get(self._keys[b], self.prob_threshold))
                if (p_top < tau) or ((p_top - p_2) < self.margin_delta):
                    labels[0].low_conf = True

            # Пакуємо результат для одного речення
            pred = SimpleNamespace(
                id=i,
                text=txt,
                labels=labels,  # повний відсортований список
                top={n: raws[n].tolist() for n in raws} if return_channel_scores else None,
            )
            out.append(pred)

        metrics = SimpleNamespace(
            latency_ms=round((time.perf_counter() - t0) * 1000, 2),
            n_sentences=len(sentences),
            model_used=self.encoder.__class__.__name__,
        )
        return SimpleNamespace(predictions=out, metrics=metrics)


def load_schema(path: str) -> LabelSchema:
    """
    Завантажити схему міток з файлу.

    Args:
        path (str): Шлях до JSON-файла зі схемою.

    Returns:
        LabelSchema: Об’єкт схеми міток.
    """
    return LabelSchema.load(path)
