"""
Threshold Processor — Document 3, Function 4.

Responsibility: turn a grayscale image into a binary (0/255) mask two
different ways, then clean the result up with morphological operations
before it's handed to the Contour Detector.
"""

import time

import cv2
import numpy as np

from .schema import StepResult


def apply_global_threshold(gray_image) -> StepResult:
    start = time.perf_counter()
    threshold_value, binary = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return StepResult(
        "Global Threshold (Otsu)", binary, binary.shape, 1, elapsed_ms,
        f"Otsu's method picked ONE cutoff (threshold={threshold_value:.0f}) "
        "that best splits the *whole image's* pixel-intensity histogram into "
        "two classes. Fast and simple, but fails badly when lighting is "
        "uneven across the frame — a shadow on one side gets misclassified.",
        "BINARY",
    )


def apply_adaptive_threshold(gray_image, block_size: int = 11, c: int = 2) -> StepResult:
    start = time.perf_counter()
    binary = cv2.adaptiveThreshold(
        gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, c,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000
    return StepResult(
        "Adaptive Threshold", binary, binary.shape, 1, elapsed_ms,
        f"Computes a DIFFERENT threshold for every {block_size}x{block_size} "
        f"neighborhood (minus constant {c}), instead of one global value. "
        "Handles uneven lighting that defeats Otsu — at the cost of being "
        "more sensitive to local noise/texture.",
        "BINARY",
    )


def apply_morphology(binary_image, kernel_size: int = 5) -> StepResult:
    start = time.perf_counter()
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    closed = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return StepResult(
        "Morphology (Close then Open)", opened, opened.shape, 1, elapsed_ms,
        f"Close ({kernel_size}x{kernel_size} kernel) fills small holes inside "
        "white regions; Open then strips small white speckle noise. Cleans a "
        "binary mask before contour detection so you get a handful of "
        "meaningful contours instead of hundreds of single-pixel-noise ones.",
        "BINARY",
    )
