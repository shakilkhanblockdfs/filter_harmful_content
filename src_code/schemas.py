# schemas.py
from pydantic import BaseModel
from typing import Dict

class ModerateRequest(BaseModel):
    text: str
    user_id: str

class ModerateResponse(BaseModel):
    action: str
    risk_score: float
    scores: Dict[str, float]
