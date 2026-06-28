"""
Visualization — Document 1, Module 5.

Responsibility:
    Turn DatasetStatistics + raw records into the PNGs a human actually looks
    at. Every function here SAVES a file and returns the Path it saved to,
    so Report Generator / main.py can log "wrote X" without re-deriving the path.
"""

import random
from pathlib import Path
from typing import Dict, List

import cv2
import matplotlib
matplotlib.use("Agg")  # headless — we only ever save to file, never show an interactive window
import matplotlib.pyplot as plt

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
    output_path.parent.mkdir(parents=True, exist_ok=True)

    items = sorted(stats.class_distribution.items(), key=lambda kv: kv[1], reverse=True)
    labels = [class_names.get(class_id, str(class_id)) for class_id, _ in items]
    counts = [count for _, count in items]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, counts, color="#4C72B0")
    plt.bar_label(bars, fontsize=8)
    plt.xticks(rotation=60, ha="right")
    plt.ylabel("Instance count")
    plt.title("Class distribution (sorted by count, descending)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()
    return output_path


def plot_bbox_size_histogram(annotations_by_stem: Dict[str, Annotation], output_path: Path) -> Path:
    """
    Histogram of normalized bbox area (width * height) across every box in
    annotations_by_stem. Same input/output contract shape as
    plot_class_histogram. This is where you'll see whether tiny objects
    dominate — relevant later for augmentation (Document 5) and anchor/loss
    behavior (Document 6).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    areas = [box.width * box.height for ann in annotations_by_stem.values() for box in ann.boxes]

    plt.figure(figsize=(8, 5))
    plt.hist(areas, bins=50, color="#55A868")
    plt.xlabel("Normalized bbox area (width * height, fraction of image)")
    plt.ylabel("Count")
    plt.title("Bounding box size distribution")
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()
    return output_path


def plot_resolution_distribution(image_meta_by_stem: Dict[str, ImageMeta], output_path: Path) -> Path:
    """
    Scatter or 2D histogram of (width, height) across all non-corrupted
    images. This turns the "2,693 distinct resolutions" fact from
    docs/00_Project_Architecture.md into something you can actually see —
    and is the visual argument for why Document 4's letterboxing step matters.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    widths = [m.width for m in image_meta_by_stem.values() if not m.is_corrupted]
    heights = [m.height for m in image_meta_by_stem.values() if not m.is_corrupted]

    plt.figure(figsize=(7, 7))
    plt.scatter(widths, heights, s=6, alpha=0.25, color="#C44E52")
    plt.xlabel("Width (px)")
    plt.ylabel("Height (px)")
    plt.title(f"Image resolution distribution ({len(set(zip(widths, heights)))} distinct sizes)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()
    return output_path


def show_random_samples(records: List[DatasetRecord], n: int, output_path: Path) -> Path:
    """
    Grid of n randomly chosen raw images, no boxes drawn — the most basic
    sanity check: do these actually look like fashion photos? Use a fixed
    random seed so re-runs are reproducible while you're debugging.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(42)
    chosen = rng.sample(records, min(n, len(records)))

    cols = 4
    rows = (len(chosen) + cols - 1) // cols
    plt.figure(figsize=(cols * 3, rows * 3))
    for i, record in enumerate(chosen):
        img = cv2.imread(str(record.image_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img_rgb)
        plt.axis("off")
        plt.title(record.stem, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()
    return output_path


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
    output_path.parent.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(record.image_path))
    height, width = img.shape[:2]

    for box in annotation.boxes:
        x1 = int((box.x_center - box.width / 2) * width)
        y1 = int((box.y_center - box.height / 2) * height)
        x2 = int((box.x_center + box.width / 2) * width)
        y2 = int((box.y_center + box.height / 2) * height)
        name = class_names.get(box.class_id, str(box.class_id))

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, name, (x1, max(y1 - 6, 0)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 1, cv2.LINE_AA)

    cv2.imwrite(str(output_path), img)
    return output_path
