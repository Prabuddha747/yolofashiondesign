"""
Edge Detector — Document 3, Function 5.

Responsibility: Canny edge detection on the grayscale image — a different
route to "structure" than thresholding. Threshold answers "which pixels are
above/below a brightness cutoff" (filled regions); Canny answers "where does
intensity change sharply" (thin one-pixel-wide lines).
"""

import time

import cv2

from .schema import StepResult


def detect_edges(gray_image, low_threshold: int = 50, high_threshold: int = 150) -> StepResult:
    start = time.perf_counter()
    edges = cv2.Canny(gray_image, low_threshold, high_threshold)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return StepResult(
        "Canny Edges", edges, edges.shape, 1, elapsed_ms,
        f"Flags pixels where the intensity-gradient magnitude crosses between "
        f"{low_threshold} and {high_threshold} (with hysteresis linking weak "
        "edges to strong ones so real edges don't break into dashes). "
        "Produces thin outlines rather than filled regions — a complementary "
        "view to thresholding, useful when foreground/background don't "
        "separate cleanly by brightness alone.",
        "BINARY",
    )
