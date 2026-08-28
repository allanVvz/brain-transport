from pydantic import BaseModel
from typing import Optional, List


class Lead(BaseModel):
    id: str
    ref: Optional[int] = None
    nome: Optional[str] = None
    stage: str = "novo"
    canal: str = "whatsapp"
    interesse_produto: Optional[str] = None
    cidade: Optional[str] = None
    cep: Optional[str] = None
    ai_enabled: bool = True
    metadata: dict = {}


class Context(BaseModel):
    lead: Lead
    mensagem: str
    historico: List[dict] = []
    kb_chunks: List[str] = []
    persona_slug: str = "global"
    classification: Optional[dict] = None
    score: int = 0
    tags: List[str] = []
    route_hint: str = "SDR"
    funnel_stage: str = "novo"
    metadata: dict = {}

