from typing import List, Optional, Dict, Any, Tuple, Literal
from pydantic import BaseModel, Field, constr, conint
from pydantic import ConfigDict

__all__ = [
    "SmoothingConfig",
    "SegmentationRequest",
    "SegmentationResponse",
    "SegmentBlock",
    "IntentHit",
    "TopicHit",
    "ClassifyRequest",
    "ClassifyResponse",
    "SentenceResult",
    "StatsRequest",
    "SentimentRequest",
    "SentimentResponse"
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
    tab_id: str
    window_size: int = Field(3, ge=1, le=50)
    model_name: str = Field("paraphrase-xlm-r-multilingual-v1", description="Назва SentenceTransformer-моделі")
    smoothing: SmoothingConfig = Field(default_factory=SmoothingConfig, description="Конфіг згладжування коротких речень")
    top_k: int = Field(3, ge=1, le=12)


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

    score: Optional[float] = Field(default=None, description="Логарифм ймовірності")
    sim: Optional[float] = Field(default=None, description="Гібридний показник схожості")
    lex: Optional[float] = Field(default=None, description="Лексичний бонус")

    model_config = ConfigDict(
        extra="ignore",
        ser_json_inf_nan="null",
        populate_by_name=True
    )


class IntentHit(_BaseLabelHit):
    intent: Optional[str] = Field(
        default=None,
        validation_alias="key",
        serialization_alias="intent",
        description="Технічна назва інтенту (ключ) для зовнішнього API",
    )


class TopicHit(_BaseLabelHit):
    topic: Optional[str] = Field(
        default=None,
        validation_alias="key",
        serialization_alias="topic",
        description="Технічний ключ теми для зовнішнього API",
    )

class ClassifyRequest(BaseModel):
    """Вхід для /intent — сирий текст + налаштування класифікатора."""
    text: str
    tab_id: str
    top_k: int = Field(3, ge=1, le=10)
    model_name: str = Field(
        "paraphrase-xlm-r-multilingual-v1",
        description="SentenceTransformer-модель для класифікації"
    )

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

class StatsRequest(BaseModel):
    text: constr(strip_whitespace=True, min_length=2) = Field(..., description="Сырой текст")
    tab_id: str
    top_n_words: conint(ge=1, le=100) = Field(10, description="Сколько топ-слов вернуть")
    top_n_bigrams: conint(ge=1, le=100) = Field(10, description="Сколько топ-биграм вернуть")
    spacy_model: Optional[str] = Field(None, description="Имя модели spaCy (по умолчанию uk_core_news_sm)")

class TopEntry(BaseModel):
    name: str
    conf: float
    sentiment: str

class SentSentenceOut(BaseModel):
    idx: int
    text: str
    label: str
    score: float
    low_conf: bool
    top: Optional[List[TopEntry]] = None
    span_range: Optional[Tuple[int, int]] = None

class SummaryOut(BaseModel):
    label: str
    score: float
    avg: Dict[str, float]
    counts: Dict[str, int]

class SentimentRequest(BaseModel):
    text: str
    tab_id: str
    include_raw: bool = Field(False, description="Возвращать ли raw_scores по предложениям"),
    low_conf_threshold: float = Field(0.55, ge=0.0, le=1.0)

class SentimentResponse(BaseModel):
    summary: SummaryOut
    sentences: List[SentSentenceOut]


class ReportAnalysis(BaseModel):
    stats: Optional[dict] = None
    sentiment: Optional[dict] = None
    segment: Optional[dict] = None
    intent: Optional[dict] = None

class ReportPayload(BaseModel):
    title: str
    analysis: ReportAnalysis