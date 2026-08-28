from pydantic import BaseModel
from typing import Optional, List, Dict


class PersonaCreate(BaseModel):
    slug: str
    name: str
    tone: Optional[str] = None
    products: List[str] = []
    prompts: Dict[str, str] = {}
    config: Dict = {}
    catalog_url: Optional[str] = None


class PersonaUpdate(BaseModel):
    name: Optional[str] = None
    tone: Optional[str] = None
    products: Optional[List[str]] = None
    prompts: Optional[Dict[str, str]] = None
    config: Optional[Dict] = None
    catalog_url: Optional[str] = None
