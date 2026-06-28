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

from .schema import Annotation, BoundingBox


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
    if not label_path.exists():
        raise FileNotFoundError(f"label file not found: {label_path}")

    boxes = []
    malformed_lines = []

    text = label_path.read_text().strip()
    if not text:
        return Annotation(label_path=label_path, boxes=boxes, malformed_lines=malformed_lines)

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) != 5:
            malformed_lines.append(line)
            continue

        try:
            class_id = int(parts[0])
            x_center, y_center, width, height = (float(p) for p in parts[1:5])
        except ValueError:
            malformed_lines.append(line)
            continue

        if not (0 <= class_id < num_classes):
            malformed_lines.append(line)
            continue
        if not all(0.0 <= v <= 1.0 for v in (x_center, y_center, width, height)):
            malformed_lines.append(line)
            continue
        if width <= 0.0 or height <= 0.0:
            malformed_lines.append(line)
            continue

        x1, y1 = x_center - width / 2, y_center - height / 2
        x2, y2 = x_center + width / 2, y_center + height / 2
        if x1 < -1e-6 or y1 < -1e-6 or x2 > 1.0 + 1e-6 or y2 > 1.0 + 1e-6:
            malformed_lines.append(line)
            continue

        boxes.append(BoundingBox(class_id=class_id, x_center=x_center, y_center=y_center,
                                  width=width, height=height))

    return Annotation(label_path=label_path, boxes=boxes, malformed_lines=malformed_lines)
