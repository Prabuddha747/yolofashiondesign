"""
Visualizer — Document 3, Function 7.

Responsibility: run every step in order, collect the StepResults, and turn
them into the artifacts a human looks at — one labeled grid image, one PNG
per individual step, and a printed shape/channels/timing table. This module
contains no image-processing logic of its own; it only calls the other six
functions in order and renders what they returned.
"""

from pathlib import Path
from typing import List, Tuple

import cv2
import matplotlib.pyplot as plt

# Deliberately NOT calling matplotlib.use() here — backend selection is an
# application-level decision, not a library one. Forcing "Agg" in a module
# that a notebook also imports silently kills the notebook's inline
# rendering (no error, just zero images shown). The CLI entry point
# (main.py) sets Agg for itself, only when actually run as __main__.

from . import blur_processor, color_converter, contour_detector, edge_detector, threshold_processor
from .image_reader import read_image
from .schema import StepResult


def run_pipeline(image_path: Path) -> Tuple[List[StepResult], list]:
    """
    Runs Original -> BGR -> RGB -> HSV -> LAB -> Gray -> Gaussian -> Median ->
    Bilateral -> Threshold -> Adaptive Threshold -> Morphology -> Edges ->
    Contours -> Bounding Box, in that order. Returns (steps, contours) so a
    caller (e.g. main.py) can add further steps, like overlaying ground-truth
    YOLO boxes, on top of the same pipeline.
    """
    steps: List[StepResult] = []

    original = read_image(image_path)
    steps.append(original)
    bgr = original.image

    rgb = color_converter.to_rgb(bgr)
    hsv = color_converter.to_hsv(bgr)
    lab = color_converter.to_lab(bgr)
    gray = color_converter.to_gray(bgr)
    steps.extend([rgb, hsv, lab, gray])

    gaussian = blur_processor.apply_gaussian(gray.image)
    median = blur_processor.apply_median(gray.image)
    bilateral = blur_processor.apply_bilateral(gray.image)
    steps.extend([gaussian, median, bilateral])

    global_thresh = threshold_processor.apply_global_threshold(gaussian.image)
    adaptive_thresh = threshold_processor.apply_adaptive_threshold(gaussian.image)
    morph = threshold_processor.apply_morphology(adaptive_thresh.image)
    steps.extend([global_thresh, adaptive_thresh, morph])

    edges = edge_detector.detect_edges(gray.image)
    steps.append(edges)

    contours_step, contours = contour_detector.find_contours(morph.image)
    bbox_step = contour_detector.draw_bounding_boxes(bgr, contours)
    steps.extend([contours_step, bbox_step])

    return steps, contours


def _imshow_step(ax, step: StepResult) -> None:
    img = step.image
    if step.color_space == "BGR_RAW":
        # Intentionally NOT converted — this is the Original step, and showing
        # the raw BGR array through an RGB-assuming renderer IS the lesson.
        ax.imshow(img)
    elif step.color_space == "BGR":
        # Downstream BGR arrays (bounding-box overlays etc.) are corrected for
        # display — their job is to show the result clearly, not re-teach BGR/RGB.
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    elif step.color_space in ("GRAY", "BINARY"):
        ax.imshow(img, cmap="gray")
    else:  # RGB, HSV, LAB — shown exactly as stored (HSV/LAB will look like false color, on purpose)
        ax.imshow(img)
    ax.set_title(f"{step.name}\n{step.shape} | {step.execution_time_ms:.2f}ms", fontsize=8)
    ax.axis("off")


def save_step_grid(steps: List[StepResult], output_path: Path, cols: int = 4) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = (len(steps) + cols - 1) // cols
    plt.figure(figsize=(cols * 3.4, rows * 3.4))
    for i, step in enumerate(steps):
        ax = plt.subplot(rows, cols, i + 1)
        _imshow_step(ax, step)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()
    return output_path


def save_individual_steps(steps: List[StepResult], output_dir: Path) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, step in enumerate(steps):
        safe_name = (
            step.name.lower()
            .replace(" ", "_").replace("(", "").replace(")", "")
            .replace(",", "").replace(">", "to").replace("=", "")
        )
        path = output_dir / f"{i:02d}_{safe_name}.png"
        fig, ax = plt.subplots(figsize=(5, 5))
        _imshow_step(ax, step)
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        paths.append(path)
    return paths


def print_step_table(steps: List[StepResult]) -> None:
    print(f"{'Step':<32} {'Shape':<18} {'Channels':<9} {'Time (ms)':<10}")
    print("-" * 71)
    for step in steps:
        print(f"{step.name:<32} {str(step.shape):<18} {step.channels:<9} {step.execution_time_ms:<10.3f}")
