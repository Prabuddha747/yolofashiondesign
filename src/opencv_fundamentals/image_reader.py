"""
Image Reader — Document 3, Function 1.

Responsibility: read one image file into memory exactly the way OpenCV does
by default — BGR channel order, uint8. Every other function in this package
assumes its *input* is already in this BGR form unless its name says
otherwise.
"""

import time
from pathlib import Path

import cv2

from .schema import StepResult


def read_image(image_path: Path) -> StepResult:
    start = time.perf_counter()
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"cv2.imread returned None for {image_path}")
    elapsed_ms = (time.perf_counter() - start) * 1000

    return StepResult(
        name="Original (BGR)",
        image=img,
        shape=img.shape,
        channels=img.shape[2] if img.ndim == 3 else 1,
        execution_time_ms=elapsed_ms,
        explanation=(
            "cv2.imread() always decodes to BGR channel order, never RGB — a "
            "historical default from early Windows camera APIs that OpenCV "
            "never changed for backwards compatibility. The panel below is "
            "shown DELIBERATELY UNCORRECTED — this raw BGR array handed "
            "straight to a renderer that assumes RGB (like matplotlib) is "
            "exactly the bug. Compare directly to the 'RGB' panel next to it: "
            "same pixels, only the channel order differs, and that alone is "
            "enough to turn reds blue."
        ),
        color_space="BGR_RAW",
    )
