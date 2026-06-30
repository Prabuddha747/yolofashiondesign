# 01 — Dataset Analysis

## Goal

Understand everything inside the dataset — by writing the code that proves it,
not by trusting Document 0's numbers. Document 0 told you *what* we found
(10,000 train / 2,000 val images, 13 classes, zero corruption, etc.) and *how*
it was checked. Your job here is to write the six modules that derive those
same facts independently, and go deeper (per-class box-size distributions,
resolution scatter, a real sample grid).

## Learning Objectives

- Dataset organization (how a YOLO-style `images/` + `labels/` split is laid out)
- Annotation format (`class_id x_center y_center width height`, normalized)
- Image properties (resolution, channels, corruption)
- Categories (13 classes, and why their names had to be *inferred*, not read
  from a file)
- Bounding boxes (how to validate one, and what "invalid" actually means)

## Code already provided for you

| File | What it is |
|---|---|
| `src/dataset_analysis/schema.py` | Data contracts (`BoundingBox`, `Annotation`, `ImageMeta`, `DatasetRecord`) — written, not a stub. |
| `src/dataset_analysis/config.py` | Dataset paths + the 13 inferred class names — written, not a stub. |
| `src/dataset_analysis/main.py` | Orchestrator wiring all six modules in the `Initialize → Load → Validate → Process → Analyze → Visualize → Save → Log → Return` order — written, not a stub. |

You should not need to edit these three. Everything else is a stub with
`raise NotImplementedError(...)` where the logic goes — the docstring on each
function already answers the six questions (input / validation / processing /
output / artifacts / error handling). That's the spec; the implementation is
yours.

## Modules — implement in this order

Each module only becomes testable once the one before it works, so build
top-to-bottom:

1. **`dataset_loader.py` → `load_dataset()`**
   Locate dataset → verify folders → collect image paths → collect
   annotation paths → pair them by filename stem.
2. **`image_loader.py` → `load_image_meta()`**
   Read image → check corruption → store height/width/channels.
3. **`annotation_parser.py` → `parse_annotation()`**
   Read txt → extract class + box per line → validate → store structured object.
4. **`statistics_generator.py` → `generate_statistics()`**
   Reduce the above into one `DatasetStatistics` per split.
5. **`visualization.py`** (5 functions)
   Turn statistics + raw records into PNGs.
6. **`report_generator.py` → `generate_report()`**
   Write the final CSV deliverables.

## How to test each module as you go (don't wait until the end)

After implementing `load_dataset()`, before touching anything else, sanity
check it directly:

```bash
cd /path/to/yolofashiondesign   # your local clone of this repo
source venv/bin/activate
python3 -c "
from src.dataset_analysis import config
from src.dataset_analysis.dataset_loader import load_dataset

root = config.get_dataset_root()
result = load_dataset(root / config.TRAIN_IMAGES_SUBDIR, root / config.TRAIN_LABELS_SUBDIR, 'train')
print('paired records:', len(result.records))
print('images_without_label:', len(result.images_without_label))
print('labels_without_image:', len(result.labels_without_image))
"
```

**Expected output, if correct** (these are the verified ground-truth numbers
from Document 0 — your code should reproduce them exactly):

```
paired records: 10000
images_without_label: 0
labels_without_image: 0
```

If you don't get exactly `10000` / `0` / `0`, that's a bug in your pairing
logic, not a dataset problem — Document 0 already confirmed the raw data is
clean. Do the same kind of direct, single-function smoke test after
`load_image_meta()` (run it on one known-good path) and after
`parse_annotation()` (run it on `000060.txt`, which Document 0 shows in full —
you should get back exactly 2 boxes: class 6 and class 0).

Once all six modules are implemented, the full pipeline is:

```bash
python -m src.dataset_analysis.main
```

This runs `run("train")` then `run("val")` and should produce, in
`outputs/dataset_analysis/`:

```
train_dataset_report.csv
train_bbox_statistics.csv
train_class_distribution.png
train_bbox_size_histogram.png
train_resolution_distribution.png
train_sample_grid.png
val_dataset_report.csv
val_bbox_statistics.csv
val_class_distribution.png
val_bbox_size_histogram.png
val_resolution_distribution.png
val_sample_grid.png
```

## Validation Rules (enforced inside `annotation_parser.py`)

| Rule | Check |
|---|---|
| Image exists | handled in `dataset_loader.py`, not here |
| Annotation exists | handled in `dataset_loader.py`, not here |
| Class valid | `0 <= class_id <= 12` |
| Bounding box inside image | derived `(x1,y1,x2,y2)` within `[0,1]` |
| Width > 0 | `width > 0.0` |
| Height > 0 | `height > 0.0` |

This dataset, per the full scan in Document 0, has **zero** violations of any
of these — so if your validation logic reports violations, suspect your own
parsing first (e.g. off-by-one indexing into `parts[]`, or comparing against
the wrong bound) before suspecting the data.

## Expected headline numbers (train split) — your code should reproduce these

| Metric | Expected value |
|---|---|
| Total images | 10,000 |
| Total labels | 10,000 |
| Images without label | 0 |
| Labels without image | 0 |
| Corrupted files | 0 |
| Malformed annotation lines | 0 |
| Objects per image | 1: 4,005 · 2: 5,812 · 3: 125 · 4: 58 |
| Largest class | `0` (`short_sleeve_top`) — 3,755 instances |
| Smallest class | `2` (`short_sleeve_outwear`) — 28 instances |
| Distinct image resolutions | 2,693 |

## Outputs (deliverables for this document)

- `train_dataset_report.csv`, `val_dataset_report.csv`
- `train_bbox_statistics.csv`, `val_bbox_statistics.csv`
- `*_class_distribution.png`
- `*_sample_grid.png`
- `*_bbox_size_histogram.png`, `*_resolution_distribution.png` (extra, beyond the original four — added because the resolution spread and bbox-size spread are both directly relevant to Document 4/5/6 decisions)

---

**Next:** once `main.py` runs clean for both splits and you've eyeballed
`train_sample_grid.png` and a handful of `draw_bbox_overlay()` outputs per
class (confirming or correcting the class-name hypothesis), we move to
Document 2 — Annotation Validation, which formalizes the malformed-annotation
logging into its own reusable pipeline with a CSV report.
