"""
Blur Processor — Document 3, Function 3.

Responsibility: apply the three classic denoising/smoothing filters to a
grayscale image and make their tradeoffs (speed vs. edge preservation vs.
noise type) visible via the timing + explanation on each StepResult.
"""

import time

import cv2

from .schema import StepResult


def _timed(name: str, fn, explanation: str) -> StepResult:
    start = time.perf_counter()
    result = fn()
    elapsed_ms = (time.perf_counter() - start) * 1000
    channels = result.shape[2] if result.ndim == 3 else 1
    return StepResult(name, result, result.shape, channels, elapsed_ms, explanation, "GRAY")


def apply_gaussian(gray_image, ksize: int = 5) -> StepResult:
    return _timed(
        "Gaussian Blur",
        lambda: cv2.GaussianBlur(gray_image, (ksize, ksize), 0),
        f"Weighted average over a {ksize}x{ksize} window, weights follow a "
        "Gaussian curve so nearby pixels matter more than far ones. Smooths "
        "noise while softening (not destroying) edges — the standard "
        "pre-blur before thresholding or edge detection.",
    )


def apply_median(gray_image, ksize: int = 5) -> StepResult:
    return _timed(
        "Median Blur",
        lambda: cv2.medianBlur(gray_image, ksize),
        f"Replaces each pixel with the median of its {ksize}x{ksize} "
        "neighborhood instead of a weighted average. Far better than "
        "Gaussian at removing salt-and-pepper noise (isolated extreme-value "
        "pixels), because one outlier can't drag a median the way it drags "
        "a mean.",
    )


def apply_bilateral(gray_image, d: int = 9, sigma_color: int = 75, sigma_space: int = 75) -> StepResult:
    return _timed(
        "Bilateral Filter",
        lambda: cv2.bilateralFilter(gray_image, d, sigma_color, sigma_space),
        "Like Gaussian blur, but the weight also drops off across large "
        "*intensity* differences, not just spatial distance — so flat "
        "regions get smoothed while strong edges stay sharp. Compare its "
        "execution_time_ms to the other two: this is the slowest filter "
        "here, only worth it when edge preservation actually matters.",
    )
