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

from pathlib import Path
from typing import Dict

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
    raise NotImplementedError("TODO: implement generate_report")
