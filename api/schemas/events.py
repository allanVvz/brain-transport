from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class LeadEvent(BaseModel):
    lead_id: str
    lead_ref: Optional[int] = None
    nome: Optional[str] = None
    stage: str = "novo"
    canal: str = "whatsapp"
    mensagem: str
    interesse_produto: Optional[str] = None
    cidade: Optional[str] = None
    cep: Optional[str] = None
    whatsapp_phone_number_id: Optional[str] = None
    persona_slug: str = "global"
    timestamp: datetime = Field(default_factory=datetime.utcnow)
