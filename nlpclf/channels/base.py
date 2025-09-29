from abc import ABC, abstractmethod

import numpy as np
from sentence_transformers import SentenceTransformer

from nlpclf.schema import LabelSchema


class ScoreChannel(ABC):
    """
    Абстрактний базовий клас для каналів оцінювання.
    Кожен канал має підтримувати підготовку (prepare) і підрахунок оцінок (score).
    """

    @abstractmethod
    def prepare(self, schema: LabelSchema, encoder: SentenceTransformer) -> None:
        """
        Підготувати канал до роботи (закодувати/агрегувати приклади тощо).

        Args:
            schema (LabelSchema): Схема міток із прикладами.
            encoder (SentenceTransformer): Модель для отримання ембеддингів.

        Returns:
            None
        """
        ...

    @abstractmethod
    def score(self, emb: np.ndarray) -> np.ndarray:
        """
        Обчислити оцінки для всіх міток за одним вектором введення.

        Args:
            emb (np.ndarray): Вектор ембеддинга форми (d,).

        Returns:
            np.ndarray: Масив оцінок форми (L,), dtype=float32,
                де L — кількість міток у схемі.
        """
        ...

    @property
    def name(self) -> str:
        """
        Повертає зручне ім’я каналу (класове ім’я).

        Returns:
            str: Назва каналу.
        """
        return self.__class__.__name__
