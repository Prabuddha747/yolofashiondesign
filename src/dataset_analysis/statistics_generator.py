"""
Statistics Generator — Document 1, Module 4.

Responsibility:
    Reduce everything Dataset Loader + Image Loader + Annotation Parser
    produced for one split into the numbers a human needs to trust the data.

This module COMPUTES; it does not plot (Visualization) and does not write
files (Report Generator). Keep it a pure function of its inputs so it stays
trivially testable — same inputs, same DatasetStatistics, every time.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from .dataset_loader import LoadResult
from .schema import Annotation, ImageMeta


@dataclass
class DatasetStatistics:
    total_images: int
    total_labels: int
    images_without_label: int
    labels_without_image: int
    corrupted_files: int
    malformed_annotation_count: int
    class_distribution: Dict[int, int]          # class_id -> instance count
    avg_resolution: Tuple[float, float]          # (avg_width, avg_height)
    avg_bbox_area_norm: float                    # mean of (width * height), normalized
    largest_object: dict                         # {"stem", "class_id", "area_norm"}
    smallest_object: dict                        # {"stem", "class_id", "area_norm"}


def generate_statistics(
    load_result: LoadResult,
    image_meta_by_stem: Dict[str, ImageMeta],
    annotations_by_stem: Dict[str, Annotation],
) -> DatasetStatistics:
    """
    What does it receive?
        load_result: output of Dataset Loader (pairs + orphan counts).
        image_meta_by_stem: output of Image Loader, one ImageMeta per stem.
        annotations_by_stem: output of Annotation Parser, one Annotation per stem.
        All three are assumed already computed — this function does no I/O.

    What does it check?
        Nothing new — it trusts upstream validation but must MEASURE upstream
        failure, e.g. corrupted_files = count of image_meta_by_stem entries
        where is_corrupted is True.

    What does it compute?
        - total_images / total_labels: len(load_result.records) for both
          (by definition a DatasetRecord only exists for a matched pair).
        - images_without_label / labels_without_image: straight from load_result.
        - corrupted_files: count where ImageMeta.is_corrupted is True.
        - malformed_annotation_count: sum of len(a.malformed_lines) across
          all Annotation objects.
        - class_distribution: count every BoundingBox.class_id across every
          Annotation's boxes.
        - avg_resolution: mean width, mean height across image_meta_by_stem
          (skip corrupted entries — they have width=height=0, which would
          silently drag the average down).
        - avg_bbox_area_norm: mean of (box.width * box.height) across every box.
        - largest_object / smallest_object: the single box with max/min
          (width * height) — keep its stem and class_id so a human can go
          look directly at that image.

    What does it return?
        One DatasetStatistics. Call this once per split — never mix train
        and val records in a single call, or the averages become meaningless.

    What should it save?
        Nothing. Pure computation.

    How should failures be handled?
        If load_result.records is empty, that's a real upstream problem
        (nothing loaded at all) — raise ValueError rather than returning a
        DatasetStatistics full of zeros that could be misread as "a
        perfectly clean, just empty, dataset."
    """
    raise NotImplementedError(
        "TODO: implement generate_statistics. "
        "Hint: build it up incrementally — get total_images/total_labels right first, "
        "print() the partial result, then add the next field."
    )
