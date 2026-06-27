"""
Data contracts shared by every module in this document.

These are plain containers, not logic — defining them is the senior-engineer
"architecture" part. Implementing the functions that populate them is yours.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class BoundingBox:
    """One annotated object, in normalized YOLO format (all four values in [0, 1])."""
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float


@dataclass
class Annotation:
    """Everything parsed from one label .txt file."""
    label_path: Path
    boxes: List[BoundingBox] = field(default_factory=list)
    malformed_lines: List[str] = field(default_factory=list)


@dataclass
class ImageMeta:
    """Metadata read from one image file. height/width/channels are 0 if is_corrupted."""
    height: int
    width: int
    channels: int
    is_corrupted: bool
    error: Optional[str] = None


@dataclass
class DatasetRecord:
    """One paired (image, label) entry — the unit Dataset Loader produces."""
    stem: str
    split: str  # "train" or "val"
    image_path: Path
    label_path: Path
