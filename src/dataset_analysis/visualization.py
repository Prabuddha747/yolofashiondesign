"""
Visualization — Document 1, Module 5.

Responsibility:
    Turn DatasetStatistics + raw records into the PNGs a human actually looks
    at. Every function here SAVES a file and returns the Path it saved to,
    so Report Generator / main.py can log "wrote X" without re-deriving the path.
"""

from pathlib import Path
from typing import Dict, List

from .dataset_loader import LoadResult
from .schema import Annotation, DatasetRecord, ImageMeta
from .statistics_generator import DatasetStatistics


def plot_class_histogram(stats: DatasetStatistics, class_names: Dict[int, str], output_path: Path) -> Path:
    """
    What does it receive? stats.class_distribution and the id->name mapping
        from config.CLASS_NAMES, plus where to save.
    What does it check? That every class_id present has a name; fall back to
        showing the raw id (don't crash) if config.CLASS_NAMES is incomplete.
    What does it compute? A bar chart, classes sorted by count DESCENDING —
        the point is to make the 3,755-vs-28 imbalance immediately visible,
        not buried in alphabetical order.
    What does it return? output_path, after creating parent dirs if needed.
    What should it save? {split}_class_distribution.png
    How should failures be handled? Let matplotlib/IO errors raise — there's
        no sensible silent fallback for "couldn't save a plot."
    """
    raise NotImplementedError("TODO: implement plot_class_histogram")


def plot_bbox_size_histogram(annotations_by_stem: Dict[str, Annotation], output_path: Path) -> Path:
    """
    Histogram of normalized bbox area (width * height) across every box in
    annotations_by_stem. Same input/output contract shape as
    plot_class_histogram. This is where you'll see whether tiny objects
    dominate — relevant later for augmentation (Document 5) and anchor/loss
    behavior (Document 6).
    """
    raise NotImplementedError("TODO: implement plot_bbox_size_histogram")


def plot_resolution_distribution(image_meta_by_stem: Dict[str, ImageMeta], output_path: Path) -> Path:
    """
    Scatter or 2D histogram of (width, height) across all non-corrupted
    images. This turns the "2,693 distinct resolutions" fact from
    docs/00_Project_Architecture.md into something you can actually see —
    and is the visual argument for why Document 4's letterboxing step matters.
    """
    raise NotImplementedError("TODO: implement plot_resolution_distribution")


def show_random_samples(records: List[DatasetRecord], n: int, output_path: Path) -> Path:
    """
    Grid of n randomly chosen raw images, no boxes drawn — the most basic
    sanity check: do these actually look like fashion photos? Use a fixed
    random seed so re-runs are reproducible while you're debugging.
    """
    raise NotImplementedError("TODO: implement show_random_samples")


def draw_bbox_overlay(
    record: DatasetRecord,
    annotation: Annotation,
    class_names: Dict[int, str],
    output_path: Path,
) -> Path:
    """
    Draw every box in `annotation` on top of the image at record.image_path,
    label each with its class NAME (not just the id), and save.

    This is the most important function in this module: run it across many
    samples per class id and use the result to confirm or correct the
    class-name hypothesis in docs/00_Project_Architecture.md / config.py
    BEFORE that mapping gets baked into data.yaml in Document 6.

    Remember the normalized-box-to-pixel conversion:
        x1 = (x_center - width / 2) * image_width
        y1 = (y_center - height / 2) * image_height
        x2 = (x_center + width / 2) * image_width
        y2 = (y_center + height / 2) * image_height
    """
    raise NotImplementedError("TODO: implement draw_bbox_overlay")
