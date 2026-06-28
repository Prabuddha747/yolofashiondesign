# Implementation & Observation Guide

Fast-path companion to [LEARNER_GUIDE.md](LEARNER_GUIDE.md). No "why," no
prose — just: what to implement, the command to run it, and the result you
should see. Use this once you already understand a module's purpose (from
`docs/NN_*.md` or `LEARNER_GUIDE.md`) and just want to move quickly. Same
results stay on disk in `outputs/<document>/` either way.

One-time setup (do once per machine):
```bash
cd "/Users/prabuddhaverma/Visual Studio Code /yolo"
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Every session after that, just:
```bash
cd "/Users/prabuddhaverma/Visual Studio Code /yolo" && source venv/bin/activate
```

---

## Document 1 — Dataset Analysis ✅ done

Results on disk: `outputs/dataset_analysis/` (4 CSVs, 12 PNGs, 4.1MB).

| # | File → function | Run | Result |
|---|---|---|---|
| 1 | `dataset_loader.py` → `load_dataset()` | `python3 -c "from src.dataset_analysis import config; from src.dataset_analysis.dataset_loader import load_dataset; r=config.get_dataset_root(); res=load_dataset(r/config.TRAIN_IMAGES_SUBDIR, r/config.TRAIN_LABELS_SUBDIR, 'train'); print(len(res.records), len(res.images_without_label), len(res.labels_without_image))"` | `10000 0 0` |
| 2 | `image_loader.py` → `load_image_meta()` | `python3 -c "from src.dataset_analysis import config; from src.dataset_analysis.image_loader import load_image_meta; r=config.get_dataset_root(); print(load_image_meta(r/config.TRAIN_IMAGES_SUBDIR/'000060.jpg'))"` | `ImageMeta(height=968, width=750, channels=3, is_corrupted=False, error=None)` |
| 3 | `annotation_parser.py` → `parse_annotation()` | `python3 -c "from src.dataset_analysis import config; from src.dataset_analysis.annotation_parser import parse_annotation; r=config.get_dataset_root(); a=parse_annotation(r/config.TRAIN_LABELS_SUBDIR/'000060.txt', config.NUM_CLASSES); print(len(a.boxes), a.malformed_lines)"` | `2 []` |
| 4 | `statistics_generator.py` → `generate_statistics()` | (exercised by step 7 below — too long for one line) | 13/13 class counts match Document 0 exactly |
| 5 | `visualization.py` (5 fns) | (exercised by step 7 below) | 12 PNGs |
| 6 | `report_generator.py` → `generate_report()` | (exercised by step 7 below) | 4 CSVs |
| 7 | **Full pipeline** | `python -m src.dataset_analysis.main` | train+val stats printed, 12 files + 4 CSVs written to `outputs/dataset_analysis/`, **~62s total** |

Headline numbers (train / val):
```
total_images        10000 / 2000
corrupted_files          0 / 0
malformed_annotations    0 / 0
largest class      0 short_sleeve_top  (3755 / 806)
smallest class     2 short_sleeve_outwear (28 / 13)
distinct resolutions  2693 (train)
```

Class-name spot check (`draw_bbox_overlay`, the two lower-confidence ones):
```python
from pathlib import Path
from src.dataset_analysis import config
from src.dataset_analysis.dataset_loader import load_dataset
from src.dataset_analysis.annotation_parser import parse_annotation
from src.dataset_analysis.visualization import draw_bbox_overlay

root = config.get_dataset_root()
result = load_dataset(root / config.TRAIN_IMAGES_SUBDIR, root / config.TRAIN_LABELS_SUBDIR, "train")
for r in result.records[:500]:
    ann = parse_annotation(r.label_path, config.NUM_CLASSES)
    for box in ann.boxes:
        if box.class_id in (10, 11):
            draw_bbox_overlay(r, ann, config.CLASS_NAMES,
                Path(f"outputs/dataset_analysis/class_verification/{box.class_id}_{r.stem}.jpg"))
```
→ both confirmed correct by eye, no `config.CLASS_NAMES` changes needed.

---

## Document 2 — Annotation Validation ⏳ deferred

Will append here in the same table format once we start it. Known from
Document 1: 0 malformed lines exist in the real data, so the validator needs
a deliberately-corrupted copy of a few label files to prove it actually
catches problems (see `LEARNER_GUIDE.md` Section 9).

---

## Document 3 — OpenCV Fundamentals ✅ done (built before Document 2, on request)

Results on disk: `outputs/opencv_fundamentals/{stem}_pipeline_grid.png` +
`outputs/opencv_fundamentals/{stem}/*.png` (15 files/sample) +
`notebooks/03_opencv_fundamentals.ipynb` (already executed, images embedded).

| # | File → function | Run | Result |
|---|---|---|---|
| 1 | `image_reader.py` → `read_image()` | part of pipeline | `(624,468,3)`, BGR |
| 2 | `color_converter.py` → `to_rgb/to_hsv/to_lab/to_gray()` | part of pipeline | RGB/HSV/LAB/Gray, see grid |
| 3 | `blur_processor.py` → `apply_gaussian/median/bilateral()` | part of pipeline | 3 smoothed variants |
| 4 | `threshold_processor.py` → `apply_global_threshold/adaptive_threshold/morphology()` | part of pipeline | 2 binary masks + cleanup |
| 5 | `edge_detector.py` → `detect_edges()` | part of pipeline | Canny edges |
| 6 | `contour_detector.py` → `find_contours/draw_bounding_boxes()` | part of pipeline | outlines → rectangles |
| 7 | **Full pipeline (CLI)** | `python -m src.opencv_fundamentals.main` | 15-step table printed, grid + per-step PNGs written |
| 8 | **Full pipeline (notebook)** | `jupyter nbconvert --to notebook --execute --inplace notebooks/03_opencv_fundamentals.ipynb` | same steps, markdown explanations, output embedded |

Pick a colorful sample — the BGR/RGB lesson is invisible on a black/white
image:
```python
from src.opencv_fundamentals.main import run
run("000150")  # red dress — use this one, not 000060, to actually see BGR vs RGB differ
```

Two real bugs caught and fixed while building this (full detail in
`LEARNER_GUIDE.md` Section 8):
1. Visualizer was auto-correcting BGR before display → both BGR and RGB
   panels looked identical. Fixed: Original step shown raw on purpose.
2. `visualizer.py` forced `matplotlib.use("Agg")` at import time → broke the
   notebook's inline images with zero errors. Fixed: moved to `main.py`'s
   `if __name__=="__main__"` guard only.

LAB color conversion: first call ~102-111ms, every call after ~0.27ms (one-time
internal lookup-table build, verified by repeating the call 4x).
