"""
Single source of truth for Document 6 (YOLO Training) paths, hyperparameter
defaults, and device selection.

Reuses src/dataset_analysis/config.py for dataset paths and class names —
per docs/00_Project_Architecture.md Section 7, training reads the same
verified-clean dataset Document 1 already validated, not a re-derived copy.
"""

from pathlib import Path

import torch

from src.dataset_analysis import config as dataset_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "yolo_training"
DATA_YAML_PATH = OUTPUT_DIR / "data.yaml"
RUNS_DIR = OUTPUT_DIR / "runs"
RUN_NAME = "train"
BEST_WEIGHTS_PATH = OUTPUT_DIR / "best.pt"

CLASS_NAMES = dataset_config.CLASS_NAMES
NUM_CLASSES = dataset_config.NUM_CLASSES

# Nano variant — smallest/fastest Ultralytics checkpoint, a safe default on
# memory-constrained machines or machines with no dedicated GPU. Bump to
# yolov8s/m/l/x if your hardware has the RAM/VRAM to spare.
BASE_MODEL = "yolov8n.pt"
IMGSZ = 640
EPOCHS = 50
BATCH = 16
PATIENCE = 15


def get_device() -> str:
    """Picks the fastest available backend: CUDA (NVIDIA GPU) > MPS (Apple
    GPU) > CPU. Calibrate batch size on your own hardware before committing
    to a long run — see docs/06_YOLO_Training.md."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
