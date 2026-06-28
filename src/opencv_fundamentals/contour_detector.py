"""
Contour Detector — Document 3, Function 6.

Responsibility: trace outlines from a binary image (Threshold Processor's
cleaned-up output), then draw bounding boxes around the meaningful ones on
a color image for display. This is the classical-CV equivalent of what a
trained detector like YOLO outputs directly: a box around "something."
"""

import time

import cv2
import numpy as np

from .schema import StepResult


def find_contours(binary_image):
    start = time.perf_counter()
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    elapsed_ms = (time.perf_counter() - start) * 1000

    canvas = np.zeros((*binary_image.shape, 3), dtype="uint8")
    cv2.drawContours(canvas, contours, -1, (0, 255, 0), 2)

    step = StepResult(
        f"Contours ({len(contours)} found)", canvas, canvas.shape, 3, elapsed_ms,
        f"cv2.findContours() traced {len(contours)} external outlines from "
        "the binary mask (RETR_EXTERNAL ignores nested/internal contours). "
        "Each contour is just a list of (x, y) boundary points — this is the "
        "bridge from 'pixels' to 'objects' in classical CV; a trained model "
        "like YOLO learns to make this same leap instead of being told "
        "explicitly where brightness changes.",
        "RGB",
    )
    return step, contours


def draw_bounding_boxes(bgr_image, contours, min_area: int = 200) -> StepResult:
    start = time.perf_counter()
    output = bgr_image.copy()
    kept = 0
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
        kept += 1
    elapsed_ms = (time.perf_counter() - start) * 1000

    return StepResult(
        f"Bounding Boxes ({kept} kept, area>={min_area}px)", output, output.shape, 3, elapsed_ms,
        f"cv2.boundingRect() turns each contour into the smallest axis-aligned "
        f"rectangle containing it, filtered here to {kept} boxes with area "
        f">= {min_area}px to drop noise contours. Compare this directly to "
        "the 'YOLO Ground Truth' step at the end of this notebook: same "
        "visual output (a green rectangle), completely different source of "
        "truth — 'has a clean outline after thresholding' vs. 'a human "
        "labeled this as a garment.'",
        "BGR",
    )
