"""
Shared data contract for Document 3 — every pipeline step (Original, BGR,
RGB, HSV, LAB, Gray, Gaussian, ...) is represented the same way, so the
Visualizer can render any of them without caring which function produced it.
"""

from dataclasses import dataclass
from typing import Any, Tuple


@dataclass
class StepResult:
    name: str
    image: Any              # np.ndarray — the actual pixel data for this step
    shape: Tuple[int, ...]
    channels: int
    execution_time_ms: float
    explanation: str
    color_space: str        # "BGR" | "RGB" | "GRAY" | "HSV" | "LAB" | "BINARY"
