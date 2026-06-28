"""
Dataset Loader — Document 1, Module 1.

Responsibility (docs/01_Dataset_Analysis.md):
    Locate dataset -> Verify folders -> Collect image paths ->
    Collect annotation paths -> Pair them.

Boundary rule: this module deals ONLY in paths. It must not decode an image
(Image Loader's job) or read the contents of a label file (Annotation
Parser's job). If you find yourself opening a file's contents here, stop —
that logic belongs in a different module.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List

from .schema import DatasetRecord


@dataclass
class LoadResult:
    """What load_dataset returns: the good pairs, plus everything that didn't pair."""
    records: List[DatasetRecord]
    images_without_label: List[str]   # stems
    labels_without_image: List[str]   # stems


def load_dataset(images_dir: Path, labels_dir: Path, split: str) -> LoadResult:
    """
    What does it receive?
        images_dir: folder containing .jpg images for one split.
        labels_dir: folder containing .txt YOLO labels for the same split.
        split: a tag ("train" / "val") stored on each record for later filtering.

    What does it check?
        - images_dir exists and is a directory.
        - labels_dir exists and is a directory.
        - For every image stem, whether a matching label stem exists, and
          vice versa. Mismatches are NOT fatal — collect them, don't crash.

    What does it compute?
        The set intersection of image stems and label stems -> one
        DatasetRecord per matched pair. The two set differences -> the
        orphan lists.

    What does it return?
        LoadResult(records, images_without_label, labels_without_image).
        Never silently drop the orphan counts — a caller must be able to
        learn "37 images had no label" without re-scanning the filesystem.

    What should it save?
        Nothing. (Report Generator decides what becomes a file on disk.)

    How should failures be handled?
        - images_dir or labels_dir missing entirely: raise FileNotFoundError
          naming the exact missing path — don't let this surface later as a
          confusing empty-list bug three modules downstream.
        - Individual orphan files: never raise. Record + exclude from `records`.
    """
    if not images_dir.is_dir():
        raise FileNotFoundError(f"images_dir not found: {images_dir}")
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"labels_dir not found: {labels_dir}")

    image_stems = {f.stem for f in images_dir.glob("*.jpg")}
    label_stems = {f.stem for f in labels_dir.glob("*.txt")}

    paired_stems = image_stems & label_stems
    images_without_label = sorted(image_stems - label_stems)
    labels_without_image = sorted(label_stems - image_stems)

    records = [
        DatasetRecord(
            stem=stem,
            split=split,
            image_path=images_dir / f"{stem}.jpg",
            label_path=labels_dir / f"{stem}.txt",
        )
        for stem in sorted(paired_stems)
    ]

    return LoadResult(
        records=records,
        images_without_label=images_without_label,
        labels_without_image=labels_without_image,
    )
