from typing import List, Optional, Dict, Any, Tuple
from pydantic import BaseModel, Field
from pydantic import ConfigDict

__all__ = [
    # Сегментация
    "SmoothingConfig",
    "SegmentationRequest",
    "SegmentationResponse",
    "SegmentBlock",
    # Інтент-аналіз
    "IntentHit",
    "TopicHit",
    "ClassifyRequest",
    "ClassifyResponse",
    "SentenceResult"
]

# =========================
# СЕКЦІЯ 1. СЕГМЕНТАЦІЯ
# =========================

class SmoothingConfig(BaseModel):
    """Налаштування згладжування коротких речень при сегментації."""
    enabled: bool = Field(True, description="Включити згладжування коротких речень")
    min_length: int = Field(5, ge=1, le=50, description="Поріг довжини короткого речення (у словах)")


class SegmentBlock(BaseModel):
    """Один семантичний блок тексту після сегментації."""
    index: int
    text: str
    span_range: Optional[Tuple[int, int]] = Field(
        default=None,
        description="Глобальні індекси речень (start, end) у всьому тексті"
    )
    topics_top: Optional[List[List[Any]]] = Field(
        default=None,
        description="Результати тематичної класифікації"
    )

class SegmentationRequest(BaseModel):
    """Вхід для /segment — параметри розбиття тексту на блоки."""
    text: str = Field(..., description="Вхідний текст для розбиття")
    window_size: int = Field(3, ge=1, le=50)
    model_name: str = Field("paraphrase-xlm-r-multilingual-v1", description="Назва SentenceTransformer-моделі")
    smoothing: SmoothingConfig = Field(default_factory=SmoothingConfig, description="Конфіг згладжування коротких речень")
    debug: bool = Field(False, description="Повернути діагностику обчислень")


class SegmentationResponse(BaseModel):
    """Вихід з /segment — параметри, блоки, метрики і діагностика (за бажанням)."""
    params: Dict[str, Any]
    blocks: List[SegmentBlock]
    metrics: Dict[str, Any]
    debug: Optional[Dict[str, Any]] = None

# =========================
# СЕКЦІЯ 2. ІНТЕНТ-АНАЛІЗ
# =========================

class _BaseLabelHit(BaseModel):
    name: str = Field(..., description="Людська назва мітки")
    conf: float = Field(..., description="Впевненість")
    low_conf: bool = Field(False, description="Позначка низької впевненості")
    conf_label: Optional[str] = Field(None, description="High / Medium / Low")

    # диагностические (используйте только при debug)
    score: Optional[float] = Field(default=None, description="Логарифм ймовірності")
    sim: Optional[float] = Field(default=None, description="Гібридний показник схожості")
    lex: Optional[float] = Field(default=None, description="Лексичний бонус")

    model_config = ConfigDict(
        extra="ignore",
        ser_json_inf_nan="null",
        populate_by_name=True
    )


class IntentHit(_BaseLabelHit):
    # хотим сериализовать поле key под именем "intent"
    intent: Optional[str] = Field(
        default=None,
        validation_alias="key",
        serialization_alias="intent",
        description="Технічна назва інтенту (ключ) для зовнішнього API",
    )


class TopicHit(_BaseLabelHit):
    # хотим сериализовать поле key под именем "topic"
    topic: Optional[str] = Field(
        default=None,
        validation_alias="key",
        serialization_alias="topic",
        description="Технічний ключ теми для зовнішнього API",
    )

class ClassifyRequest(BaseModel):
    """Вхід для /intent — сирий текст + налаштування класифікатора."""
    text: str
    top_k: int = Field(3, ge=1, le=10)
    model_name: str = Field(
        "paraphrase-xlm-r-multilingual-v1",
        description="SentenceTransformer-модель для класифікації"
    )

    # Параметри класифікатора (опційні, із дефолтами)
    use_mahalanobis: bool = True
    mahalanobis_reg: float = 0.10
    T_maha: float = 0.75
    T_maxcos: float = 0.85
    prob_threshold: float = 0.30
    margin_delta: float = 0.035
    per_class_thresholds: Dict[str, float] = Field(
        default_factory=dict,
        description="Персональні пороги ймовірності на клас"
    )


class SentenceResult(BaseModel):
    """Результат класифікації одного текстового фрагмента (речення чи блоку)."""
    id: int
    text: str
    top: List[IntentHit|TopicHit]
    span_range: Optional[Tuple[int, int]] = None

class ClassifyResponse(BaseModel):
    """Вихід з /intent — список речень і топ-k гіпотез по кожному."""
    results: List[SentenceResult]
    metrics: Dict[str, Any]

