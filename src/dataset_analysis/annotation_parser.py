"""
Annotation Parser — Document 1, Module 3.

Responsibility:
    Read txt -> Extract (class, bounding box) per line -> Validate -> Store object.

This is the ONLY module allowed to apply these validation rules:
    1. class_id is an integer in [0, num_classes - 1]
    2. x_center, y_center, width, height are floats in [0.0, 1.0]
    3. width > 0 and height > 0 (zero-area boxes are invalid)
    4. the derived box (x_center - w/2, y_center - h/2, x_center + w/2,
       y_center + h/2) stays within [0, 1] on both axes — this catches
       overflow even when center and width individually looked valid.

A line failing any rule is NOT silently dropped: record its raw text in
Annotation.malformed_lines and exclude it from Annotation.boxes, so
Statistics Generator can report a real count instead of objects vanishing.
"""

from pathlib import Path

from .schema import Annotation


def parse_annotation(label_path: Path, num_classes: int) -> Annotation:
    """
    What does it receive?
        label_path: path to one YOLO-format .txt file.
        num_classes: valid class ids are integers in [0, num_classes - 1].

    What does it check? (per non-blank line)
        1. Line splits into exactly 5 whitespace-separated tokens.
        2. Token 0 parses as int and falls in [0, num_classes - 1].
        3. Tokens 1..4 parse as float.
        4. Each of x_center, y_center, width, height is in [0.0, 1.0].
        5. width > 0 and height > 0.
        6. Derived (x1, y1, x2, y2) stays within [0, 1].

    What does it compute?
        A BoundingBox per line that passes all six checks.

    What does it return?
        Annotation(label_path, boxes=[valid...], malformed_lines=[raw text of invalid lines]).
        An empty file (zero objects) is VALID, not malformed — an image can
        legitimately have zero annotated garments. Don't flag it.

    What should it save?
        Nothing directly — the caller (Report Generator) decides whether
        accumulated malformed_lines across the dataset become a CSV.

    How should failures be handled?
        - label_path doesn't exist: raise FileNotFoundError. This should
          never happen if Dataset Loader's pairing was correct; if it does,
          that's a real upstream bug and must surface loudly, not be papered over.
        - A malformed line: never raise. Record + skip + keep parsing the rest.
    """
    raise NotImplementedError(
        "TODO: implement parse_annotation. "
        "Hint: iterate label_path.read_text().strip().splitlines(), validate each, "
        "collect good ones into boxes and bad ones (raw line text) into malformed_lines."
    )
