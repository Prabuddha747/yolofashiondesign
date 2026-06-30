"""
Shared data contracts for Document 10 — one detection per predicted box, one
FrameResult per processed frame (single photo or one tick of the live loop).
"""

from dataclasses import dataclass, field
from typing import Any, List, Tuple


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    box_xyxy: Tuple[int, int, int, int]


@dataclass
class FrameResult:
    annotated_frame: Any           # np.ndarray, BGR, boxes/labels already drawn
    detections: List[Detection] = field(default_factory=list)
    inference_ms: float = 0.0
