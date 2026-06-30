# Implementation & Observation Guide

Fast-path companion to [LEARNER_GUIDE.md](LEARNER_GUIDE.md). No "why," no
prose — just: what to implement, the command to run it, and the result you
should see. Use this once you already understand a module's purpose (from
`docs/NN_*.md` or `LEARNER_GUIDE.md`) and just want to move quickly. Same
results stay on disk in `outputs/<document>/` either way.

One-time setup (do once per machine):
```bash
cd /path/to/yolofashiondesign   # your local clone of this repo
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Every session after that, just:
```bash
cd /path/to/yolofashiondesign && source venv/bin/activate
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

---

## Document 6 — YOLO Training ✅ done (jumped ahead of Documents 2/4/5, on request)

Results on disk: `outputs/yolo_training/best.pt`, `data.yaml`,
`runs/train/` (curves, confusion matrix, weights), `runs/val/` (final
eval plots).

| # | File → function | Run | Result |
|---|---|---|---|
| 1 | `config.py` → `get_device()` | part of pipeline | auto-picks `"cuda"` > `"mps"` > `"cpu"` for your machine |
| 2 | `dataset_yaml.py` → `build_data_yaml()` | part of pipeline | `outputs/yolo_training/data.yaml` |
| 3 | `train.py` → `train_model()` | part of pipeline | `best.pt`, 3 epochs, 69 min compute time |
| 4 | `evaluate.py` → `evaluate_model()` | part of pipeline | mAP50=0.454, mAP50-95=0.351 |
| 5 | **Full pipeline** | `python -m src.yolo_training.main` | trains + evaluates + prints summary |

**Calibrate batch size on your own hardware first** — don't assume.
Example from an 8GB-RAM, no-dedicated-GPU machine: batch=16 → 1.43 img/s,
batch=8 → 2.78 img/s, batch=4 → 6.10 img/s (smaller batch was faster,
opposite of typical GPU scaling, due to memory swapping). Used `batch=4`
for that machine's real run — your fastest batch size will likely differ.

**Real run (example machine above): 3 epochs, full 10,000-image train set:**
```
mAP50: 0.454 | mAP50-95: 0.351 | precision: 0.626 | recall: 0.464
```
Per-class mAP50 tracks training-instance count almost exactly: `trousers`
0.858 (2,807 instances) down to `short_sleeve_outwear` 0.015 (28
instances) — numeric proof of Document 1's 134:1 imbalance warning.

Wall-clock said 5.2 hours; actual compute was ~69 minutes — the machine
slept mid-run. Use `caffeinate` (macOS) or your OS's equivalent for
unattended long training next time.

---

## Document 10 — Real-Time Detection ✅ done

Results on disk: `outputs/realtime_detection/captured_photo.png` (and
`live_last_frame.png` when run with `--max-frames`).

| # | File → function | Run | Result |
|---|---|---|---|
| 1 | `capture.py` → `capture_single_frame()` | part of pipeline | real BGR frame from the system's default camera, verified by saving + viewing |
| 2 | `detector.py` → `GarmentDetector.detect()` | part of pipeline | `FrameResult` (annotated frame, detections, inference ms) |
| 3 | **Single photo** | `python -m src.realtime_detection.main --mode photo` | captures, detects, saves, prints |
| 4 | **Live window** | `python -m src.realtime_detection.main --mode live` | continuous detection + FPS overlay, `q`/ESC to quit |

Pipeline verified twice: once with stock `yolov8n.pt` (COCO) — correctly
found `person 0.79` and `cup 0.35` on a test photo — and once with
Document 6's trained weights, which needed `--conf 0.15` (down from the
0.25 default) to show any signal on an out-of-distribution selfie, given
only 3 training epochs.

One real snag: `cam.read()` returned `False` on every frame mid-session
while the screen was locked, despite `cam.isOpened()` being `True` — not a
bug, macOS blocks camera frames during screen lock. Resolved itself once
unlocked.
