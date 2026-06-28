"""
Image Loader — Document 1, Module 2.

Responsibility:
    Read image -> Check corruption -> Store metadata.

Decode each image exactly once. Statistics Generator must reuse this result,
never re-open a file that already failed here.
"""

from pathlib import Path

import cv2

from .schema import ImageMeta


def load_image_meta(image_path: Path) -> ImageMeta:
    """
    What does it receive?
        image_path: path to a single .jpg file.

    What does it check?
        - File exists (defensive — don't assume upstream pairing was perfect).
        - File actually decodes as an image. Wrap ONLY the decode call in
          try/except, not the whole function, so you don't accidentally
          swallow a bug elsewhere as "corrupted image."
        - Number of channels actually found (this dataset scanned 100% RGB
          per docs/00_Project_Architecture.md, but don't hardcode that
          assumption — record what you actually see).

    What does it compute?
        height, width, channels from the decoded image.
        Pick ONE library (cv2.imread or PIL.Image.open) and use it
        consistently across this whole project. This choice matters: cv2
        decodes to BGR, PIL decodes to RGB — Document 3 (OpenCV Fundamentals)
        is built entirely around that distinction, so make the choice here
        deliberately, not by accident.

    What does it return?
        ImageMeta(height, width, channels, is_corrupted, error).
        - Success: is_corrupted=False, error=None.
        - Failure: is_corrupted=True, error=<short message>,
          height=width=channels=0 (use 0, not None, so downstream code
          never needs a None-check).

    What should it save?
        Nothing — pure read -> metadata.

    How should failures be handled?
        Never raise for a corrupted/unreadable file — that is the expected
        case this function exists to detect, not an exceptional program
        state. Catch the decode error, fill in is_corrupted=True, return
        normally.
    """
    if not image_path.exists():
        return ImageMeta(height=0, width=0, channels=0, is_corrupted=True,
                          error=f"file not found: {image_path}")

    img = cv2.imread(str(image_path))
    if img is None:
        return ImageMeta(height=0, width=0, channels=0, is_corrupted=True,
                          error="cv2.imread returned None (corrupted or unsupported file)")

    height, width = img.shape[0], img.shape[1]
    channels = img.shape[2] if img.ndim == 3 else 1
    return ImageMeta(height=height, width=width, channels=channels, is_corrupted=False, error=None)
