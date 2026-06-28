"""
Report Generator — Document 1, Module 6.

Responsibility:
    The only module allowed to write the final artifact files. Serializes
    a DatasetStatistics into the deliverables docs/01_Dataset_Analysis.md
    promises:
        {split}_dataset_report.csv
        {split}_bbox_statistics.csv
    (the two PNG deliverables — class_distribution.png and sample_grid.png —
    are produced by visualization.py; call those from main.py or from here,
    pick one place and be consistent.)
"""

import csv
from pathlib import Path
from typing import Dict

from . import config
from .statistics_generator import DatasetStatistics


def generate_report(stats: DatasetStatistics, output_dir: Path, split: str) -> Dict[str, Path]:
    """
    What does it receive?
        stats: a computed DatasetStatistics for exactly one split.
        output_dir: base folder, e.g. outputs/dataset_analysis/.
        split: "train" or "val" — namespaces filenames so train and val
        reports never collide or overwrite each other.

    What does it check?
        output_dir exists — create it (with parents=True) if not. This is
        the one module allowed to create directories on disk.

    What does it compute?
        Nothing new. If you find yourself computing a number here that
        isn't already a field on `stats`, that number belongs in
        Statistics Generator instead — move it there.

    What does it return?
        Dict[str, Path] mapping artifact name -> path written, e.g.:
        {"dataset_report": Path(...), "bbox_statistics": Path(...)}
        so main.py can print exactly what was written and where, without
        re-deriving filenames.

    What should it save?
        {split}_dataset_report.csv   — the headline DatasetStatistics fields,
            one row is fine (this isn't a per-image table).
        {split}_bbox_statistics.csv  — per-class breakdown: class_id,
            class_name, instance_count, share_of_total.

    How should failures be handled?
        Disk write errors should raise — there's nothing sensible to
        silently fall back to. Avoid partial writes: build the full CSV
        content in memory first, then write it in one call, so a crash
        mid-run never leaves a half-written file that looks complete.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    report_rows = [
        ["metric", "value"],
        ["total_images", stats.total_images],
        ["total_labels", stats.total_labels],
        ["images_without_label", stats.images_without_label],
        ["labels_without_image", stats.labels_without_image],
        ["corrupted_files", stats.corrupted_files],
        ["malformed_annotation_count", stats.malformed_annotation_count],
        ["avg_resolution_width_px", round(stats.avg_resolution[0], 2)],
        ["avg_resolution_height_px", round(stats.avg_resolution[1], 2)],
        ["avg_bbox_area_norm", round(stats.avg_bbox_area_norm, 6)],
        ["largest_object_stem", stats.largest_object["stem"]],
        ["largest_object_class_id", stats.largest_object["class_id"]],
        ["largest_object_area_norm", round(stats.largest_object["area_norm"], 6)],
        ["smallest_object_stem", stats.smallest_object["stem"]],
        ["smallest_object_class_id", stats.smallest_object["class_id"]],
        ["smallest_object_area_norm", round(stats.smallest_object["area_norm"], 6)],
    ]
    report_path = output_dir / f"{split}_dataset_report.csv"
    with report_path.open("w", newline="") as f:
        csv.writer(f).writerows(report_rows)

    total_instances = sum(stats.class_distribution.values())
    bbox_rows = [["class_id", "class_name", "instance_count", "share_of_total"]]
    for class_id, count in sorted(stats.class_distribution.items()):
        share = count / total_instances if total_instances else 0.0
        bbox_rows.append([
            class_id,
            config.CLASS_NAMES.get(class_id, str(class_id)),
            count,
            round(share, 4),
        ])
    bbox_path = output_dir / f"{split}_bbox_statistics.csv"
    with bbox_path.open("w", newline="") as f:
        csv.writer(f).writerows(bbox_rows)

    return {"dataset_report": report_path, "bbox_statistics": bbox_path}
