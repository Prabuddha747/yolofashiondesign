"""
Runs Ultralytics' validation loop against the held-out validation split and
extracts the headline metrics.

Receives: trained weights path + data_yaml path.
Checks: weights file exists.
Computes: precision/recall/mAP50/mAP50-95, overall and per class.
Returns: an EvaluationResult.
Saves: Ultralytics writes its own PR curve / confusion matrix plots under
outputs/yolo_training/runs/<name>/ as a side effect of model.val().
Failures: raises FileNotFoundError if weights are missing — evaluating a
model that was never trained is a programming error, not a recoverable case.
"""

from pathlib import Path

from ultralytics import YOLO

from . import config
from .schema import EvaluationResult


def evaluate_model(weights_path, data_yaml_path, device: str = None) -> EvaluationResult:
    weights_path = Path(weights_path)
    if not weights_path.is_file():
        raise FileNotFoundError(f"No trained weights at {weights_path} — run train_model() first.")

    device = device or config.get_device()
    model = YOLO(str(weights_path))
    results = model.val(data=str(data_yaml_path), device=device, project=str(config.RUNS_DIR), name="val")

    names = results.names
    per_class_map50 = {
        names[class_id]: float(results.box.ap50[i])
        for i, class_id in enumerate(results.box.ap_class_index)
    }

    return EvaluationResult(
        map50=float(results.box.map50),
        map50_95=float(results.box.map),
        per_class_map50=per_class_map50,
        precision=float(results.box.mp),
        recall=float(results.box.mr),
    )
