"""
Single source of truth for Document 10 (Real-Time Detection) defaults.

Falls back to the stock COCO-pretrained yolov8n.pt when no garment-trained
weights exist yet, purely so the camera/window/draw-loop plumbing in this
module is testable on day one. COCO has no garment classes — swap in
src/yolo_training's BEST_WEIGHTS_PATH (Document 6's output) to actually
detect clothing.
"""

from pathlib import Path

from src.yolo_training import config as yolo_training_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "realtime_detection"

TRAINED_WEIGHTS_PATH = yolo_training_config.BEST_WEIGHTS_PATH
FALLBACK_WEIGHTS = "yolov8n.pt"


def resolve_weights_path(weights_path: str = None) -> str:
    """Picks (in order): an explicit path, the trained garment model if it
    exists on disk, else the stock COCO checkpoint."""
    if weights_path is not None:
        return weights_path
    if TRAINED_WEIGHTS_PATH.is_file():
        return str(TRAINED_WEIGHTS_PATH)
    return FALLBACK_WEIGHTS


CAMERA_INDEX = 0
CONFIDENCE_THRESHOLD = 0.25
WINDOW_NAME = "Garment Detection — press q to quit"
