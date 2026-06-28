"""
Color Converter — Document 3, Function 2.

Responsibility: convert a BGR image (Image Reader's output) into the other
color spaces this pipeline teaches — RGB, HSV, LAB, grayscale. Every
function here takes a BGR np.ndarray in and returns one StepResult.
"""

import time

import cv2

from .schema import StepResult


def _timed(name: str, convert_fn, bgr_image, explanation: str, color_space: str) -> StepResult:
    start = time.perf_counter()
    converted = convert_fn(bgr_image)
    elapsed_ms = (time.perf_counter() - start) * 1000
    channels = converted.shape[2] if converted.ndim == 3 else 1
    return StepResult(name, converted, converted.shape, channels, elapsed_ms, explanation, color_space)


def to_rgb(bgr_image) -> StepResult:
    return _timed(
        "RGB", lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2RGB), bgr_image,
        "Swaps channel order B<->R, nothing else changes. Needed any time a "
        "cv2-decoded image is handed to something that assumes RGB — "
        "matplotlib, PIL, and most ML model input pipelines, including this "
        "project's own YOLO training in Document 6.",
        "RGB",
    )


def to_hsv(bgr_image) -> StepResult:
    return _timed(
        "HSV", lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2HSV), bgr_image,
        "Hue / Saturation / Value. Separates color identity (Hue) from how "
        "vivid (Saturation) and how bright (Value) it is. Note this image "
        "looks visually wrong if you just imshow() it raw — H/S/V are not "
        "R/G/B, so a viewer that assumes 3-channel-means-RGB renders nonsense "
        "colors here on purpose, to make that exact point.",
        "HSV",
    )


def to_lab(bgr_image) -> StepResult:
    return _timed(
        "LAB", lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2LAB), bgr_image,
        "Lightness / a (green-red) / b (blue-yellow). Designed for perceptual "
        "uniformity: equal numeric distance in LAB corresponds to roughly "
        "equal *perceived* color difference, which plain RGB distance does "
        "not guarantee. Same false-color caveat as HSV applies when displayed "
        "raw.",
        "LAB",
    )


def to_gray(bgr_image) -> StepResult:
    return _timed(
        "Grayscale", lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), bgr_image,
        "Collapses 3 channels to 1 via a weighted sum "
        "(~0.299*R + 0.587*G + 0.114*B) approximating human brightness "
        "perception. Almost every step later in this pipeline (blur, "
        "threshold, edges, contours) operates on this single channel, not "
        "color — most classical CV operations care about structure, not hue.",
        "GRAY",
    )
