from typing import Dict, List

import numpy as np
from sentence_transformers import SentenceTransformer

from nlpclf.channels.base import ScoreChannel
from nlpclf.schema import LabelSchema
from nlpclf.utils import _l2n


class CosineExemplarMax(ScoreChannel):
    """
    Канал, що повертає для кожної мітки оцінку у вигляді
    максимального значення косинусної подібності між запитом
    та прикладами цієї мітки.
    """
    def __init__(self):
        self._keys: List[str] = []
        self._embeddings_by_label: Dict[str, np.ndarray] = {}
        self._dim: int = 0

    def prepare(self, schema: LabelSchema, encoder: SentenceTransformer) -> None:
        """
        Підготувати канал до роботи: закодувати та L2-нормалізувати
        приклади для кожної мітки.

        Args:
            schema (LabelSchema): Схема міток із прикладами (списки текстових зразків).
            encoder (SentenceTransformer): Модель для отримання векторних подань текстів.

        Returns:
            None
        """
        self._keys = list(schema.keys)
        self._embeddings_by_label.clear()
        self._dim = encoder.get_sentence_embedding_dimension()

        for k in self._keys:
            ex = schema.examples[k]
            if not ex:
                self._embeddings_by_label[k] = np.zeros((0, self._dim), dtype=np.float32)
                continue

            emb = encoder.encode(ex, convert_to_numpy=True, normalize_embeddings=False)
            emb = emb.astype(np.float32)

            self._embeddings_by_label[k] = _l2n(emb)

    def score(self, emb: np.ndarray) -> np.ndarray:
        """
        Обчислює для кожної мітки (класу) числову оцінку схожості поточного речення
        з прикладами цієї мітки. Чим більше число, тим ближче речення до прикладів мітки.

        Args:
            emb (np.ndarray): Вектор ембеддинга поточного речення форми

        Returns:
            np.ndarray:
            Масив оцінок форми (L,), де L — кількість міток у схемі.
            Кожен елемент — число від приблизно -1 до 1:
              • ближче до 1 — речення дуже схоже на приклади цієї мітки;
              • близько до 0 — нейтральна/слабка схожість;
              • від’ємне значення — протилежний напрямок (антисхожість).
            Якщо в мітки немає жодного прикладу, повертається -inf для цієї мітки
            (щоб така мітка не могла обіграти інші).
        """

        if not self._keys:
            return np.zeros((0,), dtype=np.float32)

        # L2-нормалізуємо вектор запиту (поточного речення)
        q = _l2n(emb.reshape(1, -1))[0]

        sims: List[float] = []
        # Для кожної мітки: косинусна подібність до всіх її прикладів
        for k in self._keys:
            E = self._embeddings_by_label[k]    # (n_k, d), L2-нормовані приклади мітки
            if E.shape[0] == 0:
                sims.append(-np.inf)
            else:
                s = E @ q                       # косинусні подібності (скалярні добутки)
                sims.append(float(np.max(s)))   # беремо максимум як оцінку мітки

        return np.asarray(sims, dtype=np.float32)
