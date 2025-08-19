from typing import List, Optional, Dict, Any, Tuple
from pydantic import BaseModel, Field


class SmoothingConfig(BaseModel):
    enabled: bool = Field(True, description="Включить сглаживание коротких предложений")
    min_length: int = Field(5, ge=1, le=50, description="Порог длины короткого предложения (слов)")


# ВХОД
class TextDivisionRequest(BaseModel):
    text: str = Field(..., description="Исходный текст для разбиения")
    window_size: int = Field(3, ge=1, le=50)
    model_name: str = Field("paraphrase-xlm-r-multilingual-v1", description="SentenceTransformer модель")
    smoothing: SmoothingConfig = Field(default_factory=SmoothingConfig)
    debug: bool = False


# ВЫХОД
class Block(BaseModel):
    index: int
    text: str
    span_range: Tuple[int, int] | None = Field(default=None)
    intents_top: Optional[List[str]] = None
    extra: Optional[Dict[str, Any]] = None

class TextDivisionResponse(BaseModel):
    params: Dict[str, Any]
    blocks: List[Block]
    metrics: Dict[str, Any]
    debug: Optional[Dict[str, Any]] = None
