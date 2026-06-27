"""
Document 1 orchestrator.

This file IS written for you — it's wiring, not the learning exercise. It
spells out, at the top level, the project-wide pattern from
docs/00_Project_Architecture.md Section 8:

    Initialize -> Load -> Validate -> Process -> Analyze -> Visualize -> Save -> Log -> Return

Each module you implement covers one or two of these stages; this script
just calls them in order. Run it after implementing dataset_loader.py at
minimum — it will run until it hits the next NotImplementedError, which
tells you exactly what to build next.

Run with:
    python -m src.dataset_analysis.main
"""

from . import config, visualization
from .annotation_parser import parse_annotation
from .dataset_loader import load_dataset
from .image_loader import load_image_meta
from .report_generator import generate_report
from .statistics_generator import generate_statistics


def run(split: str) -> None:
    # Initialize
    dataset_root = config.get_dataset_root()
    if split == "train":
        images_dir = dataset_root / config.TRAIN_IMAGES_SUBDIR
        labels_dir = dataset_root / config.TRAIN_LABELS_SUBDIR
    else:
        images_dir = dataset_root / config.VAL_IMAGES_SUBDIR
        labels_dir = dataset_root / config.VAL_LABELS_SUBDIR

    # Load + Validate (Dataset Loader)
    load_result = load_dataset(images_dir, labels_dir, split)
    print(f"[{split}] paired records: {len(load_result.records)} "
          f"| images_without_label: {len(load_result.images_without_label)} "
          f"| labels_without_image: {len(load_result.labels_without_image)}")

    # Load (Image Loader) + Validate (Annotation Parser), per record
    image_meta_by_stem = {r.stem: load_image_meta(r.image_path) for r in load_result.records}
    annotations_by_stem = {r.stem: parse_annotation(r.label_path, config.NUM_CLASSES) for r in load_result.records}

    # Analyze (Statistics Generator)
    stats = generate_statistics(load_result, image_meta_by_stem, annotations_by_stem)
    print(f"[{split}] stats: {stats}")

    # Visualize
    out_dir = config.OUTPUT_DIR
    visualization.plot_class_histogram(stats, config.CLASS_NAMES, out_dir / f"{split}_class_distribution.png")
    visualization.plot_bbox_size_histogram(annotations_by_stem, out_dir / f"{split}_bbox_size_histogram.png")
    visualization.plot_resolution_distribution(image_meta_by_stem, out_dir / f"{split}_resolution_distribution.png")
    visualization.show_random_samples(load_result.records, n=16, output_path=out_dir / f"{split}_sample_grid.png")

    # Save + Log (Report Generator)
    artifacts = generate_report(stats, out_dir, split)
    print(f"[{split}] wrote: {artifacts}")


if __name__ == "__main__":
    run("train")
    run("val")
