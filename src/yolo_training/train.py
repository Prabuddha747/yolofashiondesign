"""
Trains a YOLOv8 detector on the fashion dataset.

Receives: data_yaml path + hyperparameters (model checkpoint, image size,
epochs, batch size, device, early-stop patience, optional `fraction` of the
training set for fast smoke tests).
Checks: nothing extra — Ultralytics validates the yaml itself and raises on
a malformed config.
Computes: runs the full Ultralytics training loop (forward/backward passes,
augmentation, validation-per-epoch, early stopping).
Returns: a TrainingResult with the path to the best checkpoint and final
validation metrics.
Saves: Ultralytics writes weights/curves/confusion-matrix under
outputs/yolo_training/runs/<name>/; this also copies best.pt to a stable
outputs/yolo_training/best.pt so downstream modules (realtime_detection)
never need to know the run name.
Failures: lets Ultralytics' own exceptions propagate — a bad data.yaml or an
OOM on this machine's 8GB RAM should fail loudly, not silently downgrade.
"""

import shutil
import time

from ultralytics import YOLO

from . import config
from .schema import TrainingResult


def train_model(
    data_yaml_path,
    model_name: str = config.BASE_MODEL,
    imgsz: int = config.IMGSZ,
    epochs: int = config.EPOCHS,
    batch: int = config.BATCH,
    patience: int = config.PATIENCE,
    device: str = None,
    fraction: float = 1.0,
    run_name: str = config.RUN_NAME,
    val: bool = True,
) -> TrainingResult:
    device = device or config.get_device()
    model = YOLO(model_name)

    start = time.perf_counter()
    results = model.train(
        data=str(data_yaml_path),
        imgsz=imgsz,
        epochs=epochs,
        batch=batch,
        patience=patience,
        device=device,
        fraction=fraction,
        project=str(config.RUNS_DIR),
        name=run_name,
        exist_ok=True,
        verbose=True,
        val=val,
    )
    elapsed_s = time.perf_counter() - start

    best_weights = results.save_dir / "weights" / "best.pt"
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_weights, config.BEST_WEIGHTS_PATH)

    metrics = getattr(results, "results_dict", {}) or {}
    epochs_run = int(model.trainer.epoch + 1) if getattr(model, "trainer", None) else epochs

    return TrainingResult(
        best_weights_path=config.BEST_WEIGHTS_PATH,
        data_yaml_path=data_yaml_path,
        epochs_run=epochs_run,
        train_time_s=elapsed_s,
        device=device,
        final_metrics={k: float(v) for k, v in metrics.items()},
    )
