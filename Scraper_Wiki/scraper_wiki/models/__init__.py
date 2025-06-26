from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any


class DatasetRecord(BaseModel):
    """Dataset record produced by scraping plugins."""

    id: str
    language: str
    category: Optional[str] = None
    content: str
    created_at: str
    title: Optional[str] = None
    summary: Optional[str] = None
    content_embedding: Optional[List[float]] = None
    summary_embedding: Optional[List[float]] = None
    questions: Optional[List[Dict[str, Any]]] = None
    answers: Optional[List[Dict[str, Any]]] = None
    relations: Optional[List[Any]] = None
    metadata: Optional[Dict[str, Any]] = None

    class Config:
        extra = "allow"


class QARecord(BaseModel):
    """Simple question/answer pair."""

    question: str
    answer: str


__all__ = ["DatasetRecord", "QARecord"]
