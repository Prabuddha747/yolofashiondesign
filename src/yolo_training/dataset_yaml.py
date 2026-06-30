"""
Builds the data.yaml Ultralytics needs to train: dataset root + train/val
image directories + class names. No images are copied — Ultralytics
discovers each image's labels by swapping "images" for "labels" in the path,
which already matches the raw kagglehub layout (docs/00_Project_Architecture.md
Section 3), so this just points at the existing cache in place.

Receives: nothing (reads dataset_config for paths/classes).
Checks: the resolved train/val image directories actually exist on disk.
Computes: a YOLO-format data.yaml dict (path, train, val, names).
Returns: the Path the yaml was written to.
Saves: outputs/yolo_training/data.yaml.
Failures: raises FileNotFoundError if an expected image directory is missing
(fail loudly — training on a wrong/empty path silently produces a useless
model instead of an obvious error).
"""

from pathlib import Path

import yaml

from src.dataset_analysis import config as dataset_config

from . import config


def build_data_yaml() -> Path:
    dataset_root = dataset_config.get_dataset_root()
    train_images_dir = dataset_root / dataset_config.TRAIN_IMAGES_SUBDIR
    val_images_dir = dataset_root / dataset_config.VAL_IMAGES_SUBDIR

    for images_dir in (train_images_dir, val_images_dir):
        if not images_dir.is_dir():
            raise FileNotFoundError(f"Expected image directory missing: {images_dir}")

    data = {
        "path": str(dataset_root),
        "train": str(train_images_dir.relative_to(dataset_root)),
        "val": str(val_images_dir.relative_to(dataset_root)),
        "names": {class_id: name for class_id, name in config.CLASS_NAMES.items()},
    }

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config.DATA_YAML_PATH.write_text(yaml.dump(data, sort_keys=False))
    return config.DATA_YAML_PATH
