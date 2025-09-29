from .cosine import CosineExemplarMax
from .mahalanobis import CenterMahalanobis

CHANNELS = {
    "CosineExemplarMax": CosineExemplarMax,
    "CenterMahalanobis": CenterMahalanobis,
}

__all__ = ["CosineExemplarMax", "CenterMahalanobis", "CHANNELS"]
