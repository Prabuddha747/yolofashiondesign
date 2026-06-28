"""
Document 3 orchestrator — runs the full OpenCV fundamentals pipeline on one
sample image from the dataset, adds a Document-1-sourced YOLO ground-truth
overlay for comparison, and saves every artifact to disk.

Run with:
    python -m src.opencv_fundamentals.main
"""

if __name__ == "__main__":
    # Only force a headless backend for a real CLI run, and only before
    # pyplot gets imported (by visualizer, below) for the first time. A
    # notebook importing `run` from this file keeps its own inline backend —
    # this guard never executes when __name__ != "__main__".
    import matplotlib
    matplotlib.use("Agg")

import time
from pathlib import Path

import cv2

from src.dataset_analysis import config as dataset_config
from src.dataset_analysis.annotation_parser import parse_annotation

from . import visualizer
from .schema import StepResult

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "opencv_fundamentals"


def _add_yolo_groundtruth_step(steps, original_bgr, sample_stem: str) -> None:
    """Not a classical-CV step — overlays the dataset's human-annotated YOLO
    labels (Document 1's annotation_parser) for direct visual comparison
    against the contour-based bounding boxes above."""
    root = dataset_config.get_dataset_root()
    label_path = root / dataset_config.TRAIN_LABELS_SUBDIR / f"{sample_stem}.txt"
    annotation = parse_annotation(label_path, dataset_config.NUM_CLASSES)

    start = time.perf_counter()
    output = original_bgr.copy()
    height, width = output.shape[:2]
    for box in annotation.boxes:
        x1 = int((box.x_center - box.width / 2) * width)
        y1 = int((box.y_center - box.height / 2) * height)
        x2 = int((box.x_center + box.width / 2) * width)
        y2 = int((box.y_center + box.height / 2) * height)
        name = dataset_config.CLASS_NAMES.get(box.class_id, str(box.class_id))
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(output, name, (x1, max(y1 - 6, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    elapsed_ms = (time.perf_counter() - start) * 1000

    steps.append(StepResult(
        f"YOLO Ground Truth ({len(annotation.boxes)} boxes, Document 1 labels)",
        output, output.shape, 3, elapsed_ms,
        "Bridge to Document 1/6, not a classical-CV step: these boxes come "
        "directly from the dataset's human-annotated YOLO labels, not from "
        "any pixel analysis. Compare to the 'Bounding Boxes' step above — "
        "same output format (a rectangle + label), entirely different source "
        "of truth. A trained YOLO model (Document 6) aims to predict boxes "
        "like THESE, not the contour-based ones.",
        "BGR",
    ))


def run(sample_stem: str = "000060") -> None:
    root = dataset_config.get_dataset_root()
    image_path = root / dataset_config.TRAIN_IMAGES_SUBDIR / f"{sample_stem}.jpg"

    steps, _contours = visualizer.run_pipeline(image_path)
    _add_yolo_groundtruth_step(steps, steps[0].image, sample_stem)

    grid_path = visualizer.save_step_grid(steps, OUTPUT_DIR / f"{sample_stem}_pipeline_grid.png")
    individual_paths = visualizer.save_individual_steps(steps, OUTPUT_DIR / sample_stem)

    visualizer.print_step_table(steps)
    print(f"\nWrote grid: {grid_path}")
    print(f"Wrote {len(individual_paths)} individual step images to: {OUTPUT_DIR / sample_stem}")


if __name__ == "__main__":
    run()
