from typing import Dict, List, Optional
from pydantic import BaseModel, field_validator

from .config import ALL_STYLES


class Task(BaseModel):
    task_id: str
    video_url: str
    styles: List[str] = ALL_STYLES

    @field_validator("styles", mode="before")
    @classmethod
    def default_styles(cls, v):
        # Contract requires all four styles present in output regardless; treat
        # missing/empty styles[] as "all four".
        if not v:
            return list(ALL_STYLES)
        return v


class Captions(BaseModel):
    formal: str
    sarcastic: str
    humorous_tech: str
    humorous_non_tech: str

    @field_validator("*")
    @classmethod
    def non_empty(cls, v: str) -> str:
        v = (v or "").strip()
        if not v:
            raise ValueError("caption must be non-empty")
        return v


class Result(BaseModel):
    task_id: str
    captions: Captions


class GroundedFacts(BaseModel):
    subjects: List[str] = []
    actions: List[str] = []
    setting: str = ""
    on_screen_text: List[str] = []
    mood: str = ""
    audio_summary: str = ""
    temporal_arc: str = ""
    salient_objects: List[str] = []
    uncertainty_notes: List[str] = []

    def compact(self) -> Dict:
        d = self.model_dump()
        return {k: v for k, v in d.items() if v}
