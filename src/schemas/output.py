"""
Pydantic schemas for the extraction output.
These define the contract of the JSON that leaves the pipeline.
"""
from __future__ import annotations

from typing import List, Optional, Tuple
from pydantic import BaseModel, Field, ConfigDict


class BBox(BaseModel):
    """Axis-aligned bounding box in pixel coordinates: [x1, y1, x2, y2]."""
    model_config = ConfigDict(frozen=True)

    x1: int
    y1: int
    x2: int
    y2: int

    def as_list(self) -> List[int]:
        return [self.x1, self.y1, self.x2, self.y2]

    @classmethod
    def from_list(cls, arr) -> "BBox":
        return cls(x1=int(arr[0]), y1=int(arr[1]), x2=int(arr[2]), y2=int(arr[3]))


class PresenceField(BaseModel):
    """Used for signature and stamp — boolean presence plus bbox when present."""
    present: bool
    bbox: Optional[List[int]] = None
    confidence: float = 0.0


class ExtractedFields(BaseModel):
    """The six fields specified by the problem statement."""
    dealer_name: Optional[str] = None
    model_name: Optional[str] = None
    horse_power: Optional[int] = None
    asset_cost: Optional[int] = None
    signature: PresenceField = Field(default_factory=lambda: PresenceField(present=False))
    stamp: PresenceField = Field(default_factory=lambda: PresenceField(present=False))


class FieldConfidences(BaseModel):
    """Per-field confidence scores (0.0–1.0)."""
    dealer_name: float = 0.0
    model_name: float = 0.0
    horse_power: float = 0.0
    asset_cost: float = 0.0
    signature: float = 0.0
    stamp: float = 0.0


class DocumentOutput(BaseModel):
    """Top-level JSON returned for each document."""
    doc_id: str
    fields: ExtractedFields
    field_confidences: FieldConfidences
    confidence: float = 0.0          # document-level aggregate
    processing_time_sec: float = 0.0
    cost_estimate_usd: float = 0.0
    ocr_language_detected: Optional[str] = None
    num_pages: int = 1
    errors: List[str] = Field(default_factory=list)


class OCRToken(BaseModel):
    """One OCR token with its bbox and confidence. Consumed internally."""
    text: str
    bbox: List[int]               # [x1, y1, x2, y2]
    confidence: float
    page: int = 0
