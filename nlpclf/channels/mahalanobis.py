from typing import Dict, List

import numpy as np
from sentence_transformers import SentenceTransformer

from nlpclf.channels.base import ScoreChannel
from nlpclf.schema import LabelSchema


class CenterMahalanobis(ScoreChannel):
    """
    Канал, що повертає для кожної мітки оцінку у вигляді мінус-відстані
    Махаланобіса між вектором речення та центром цієї мітки.
    """

    def __init__(self, reg: float = 0.10) -> None:
        """
        Args:
            reg (float): Додатня регуляризація до коваріації (lambda * I),
                щоб уникати виродження та зробити обернення стабільним.
        """
        self.reg: float = float(reg)
        self._keys: List[str] = []                    # усі ключі міток у фіксованому порядку
        self._centers: Dict[str, np.ndarray] = {}     # мапа: мітка -> центр (d,)
        self._inv_covs: Dict[str, np.ndarray] = {}    # мапа: мітка -> обернена коваріація (d,d)
        self._dim: int = 0

    def prepare(self, schema: LabelSchema, encoder: SentenceTransformer) -> None:
        """
        Підготувати канал: закодувати приклади, порахувати центри та обернені коваріації.

        Args:
            schema (LabelSchema): Схема міток із прикладами (списки текстових зразків).
            encoder (SentenceTransformer): Модель для отримання ембеддингів.

        Returns:
            None

        Примітка:
            Якщо у мітки немає прикладів, центр = нульовий вектор,
            обернена коваріація = (1/reg) * I (щоб формально мати матрицю).
        """
        self._keys = list(schema.keys)
        self._centers.clear()
        self._inv_covs.clear()
        self._dim = encoder.get_sentence_embedding_dimension()

        for k in self._keys:
            examples = schema.examples[k]
            if not examples:
                # порожня мітка: ставимо «заглушки», щоб score() міг працювати
                self._centers[k] = np.zeros((self._dim,), dtype=np.float32)
                self._inv_covs[k] = (1.0 / self.reg) * np.eye(self._dim, dtype=np.float32)
                continue

            E = encoder.encode(examples, convert_to_numpy=True, normalize_embeddings=False).astype(np.float32)
            mu = E.mean(axis=0)                            # центр мітки
            self._centers[k] = mu

            if E.shape[0] >= 2:
                # коваріація з регуляризацією: S = cov + reg * I
                X = E - mu
                S = (X.T @ X) / max(E.shape[0] - 1, 1)
                S = S.astype(np.float32)
                S += self.reg * np.eye(S.shape[0], dtype=np.float32)

                # обертаємо: звичайне або псевдообернене, якщо вироджена
                try:
                    inv = np.linalg.inv(S)
                except np.linalg.LinAlgError:
                    inv = np.linalg.pinv(S)
            else:
                # одного прикладу замало для коваріації — ставимо (1/reg) * I
                inv = (1.0 / self.reg) * np.eye(self._dim, dtype=np.float32)

            self._inv_covs[k] = inv

    def score(self, emb: np.ndarray) -> np.ndarray:
        """
        Обчислити оцінки для всіх міток як мінус-відстань Махаланобіса
        до їхніх центрів (більше число = ближче до центру мітки).

        Args:
            emb (np.ndarray): Вектор ембеддинга поточного речення форми (d,).

        Returns:
            np.ndarray: Масив оцінок форми (L,), dtype=float32, де L — кількість міток.
                Якщо у мітки немає валідного центру (усі нулі), повертається -inf для цієї мітки.
        """
        if not self._keys:
            return np.zeros((0,), dtype=np.float32)

        x = emb.astype(np.float32)
        out: List[float] = []

        for k in self._keys:
            mu = self._centers[k]
            if not np.any(mu):                 # немає центру — мітка не «змагається»
                out.append(-np.inf)
                continue

            d = x - mu                         # зсув від центру
            inv = self._inv_covs[k]            # обернена коваріація

            # відстань Махаланобіса: sqrt(d^T * inv * d)
            dist_sq = float(d @ (inv @ d))     # квадрат відстані (стисліше та стабільніше)

            # беремо мінус відстані (монотонне перетворення): більше = ближче
            out.append(-float(np.sqrt(dist_sq + 1e-12)))

        return np.asarray(out, dtype=np.float32)
