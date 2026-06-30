"""
Shared data contracts for Document 6 — training and evaluation each return
one of these so main.py can print/save a summary without reaching back into
Ultralytics' internal result objects.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict


@dataclass
class TrainingResult:
    best_weights_path: Path
    data_yaml_path: Path
    epochs_run: int
    train_time_s: float
    device: str
    final_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    map50: float
    map50_95: float
    per_class_map50: Dict[str, float]
    precision: float
    recall: float
