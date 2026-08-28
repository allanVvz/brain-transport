from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class InsightCreate(BaseModel):
    persona_id: Optional[str] = None
    severity: str  # critical | warning | info
    category: str  # performance | reliability | architecture | business
    title: str
    description: Optional[str] = None
    recommendation: Optional[str] = None
    affected_component: Optional[str] = None
    score_impact: int = 0


class InsightUpdate(BaseModel):
    status: str  # open | acknowledged | resolved


class Insight(InsightCreate):
    id: str
    status: str = "open"
    created_at: datetime
    resolved_at: Optional[datetime] = None
