from pydantic import BaseModel
from datetime import datetime
from typing import Optional, Any


class AnalysisBlock(BaseModel):
    stats: Optional[Any] = None
    sentiment: Optional[Any] = None
    segment: Optional[Any] = None
    intent: Optional[Any] = None


class WorkspaceTabFull(BaseModel):
    tab_id: str
    title: str
    created_at: datetime
    updated_at: datetime

    text_id: int
    text: str
    analysis: AnalysisBlock


class WorkspaceTabLight(BaseModel):
    tab_id: str
    title: str
    created_at: datetime


class NewTabRequest(BaseModel):
    title: str | None = None
