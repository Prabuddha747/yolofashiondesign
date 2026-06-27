"""
Single source of truth for dataset paths and class metadata.

This is the one place (per docs/00_Project_Architecture.md, Section 6) that
knows about kagglehub's cache layout. Every other module receives plain
Path objects and does not know or care where they came from.
"""

from pathlib import Path

import kagglehub

KAGGLE_DATASET_SLUG = "lahbibfedi/fashion-dataset-with-annotation"

# Verified on disk (docs/00_Project_Architecture.md, Section 3) — the uploader's
# zip produced these exact (slightly odd) nested folder names. If a future
# dataset version changes this layout, this is the only place to fix it.
TRAIN_IMAGES_SUBDIR = Path("new_train (1)/new_t/images")
TRAIN_LABELS_SUBDIR = Path("new_train (1)/new_t/labels")
VAL_IMAGES_SUBDIR = Path("new_validation (1)/new_v/images")
VAL_LABELS_SUBDIR = Path("new_validation (1)/new_v/labels")

NUM_CLASSES = 13

# Inferred from visually inspecting cropped bounding boxes per class id —
# this dataset ships NO classes.txt / data.yaml. Matches the DeepFashion2
# 13-category taxonomy. Re-verify with your own draw_bbox_overlay() samples
# (Document 1, Visualization module) before trusting this for training.
CLASS_NAMES = {
    0: "short_sleeve_top",
    1: "long_sleeve_top",
    2: "short_sleeve_outwear",
    3: "long_sleeve_outwear",
    4: "vest",
    5: "sling",
    6: "shorts",
    7: "trousers",
    8: "skirt",
    9: "short_sleeve_dress",
    10: "long_sleeve_dress",
    11: "vest_dress",
    12: "sling_dress",
}

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "dataset_analysis"


def get_dataset_root() -> Path:
    """
    Returns the local path to the downloaded dataset, downloading it first
    if it isn't already cached. Safe to call every run — kagglehub checks
    its cache before touching the network.
    """
    return Path(kagglehub.dataset_download(KAGGLE_DATASET_SLUG))
