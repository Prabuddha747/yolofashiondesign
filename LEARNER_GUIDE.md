# Learner Guide — FashionVisionAI / yolofashiondesign

One continuous, step-wise walkthrough: every command run so far, why it was
run, and what result you should see if it worked. This file grows as we move
through Documents 2–12 — each new document gets appended at the end in the
same format. The `docs/NN_*.md` files hold the deeper architectural reasoning;
this file is the "do this, see this" companion.

---

## 0. Prerequisites

- Python 3.12 (check with `python3 --version`)
- macOS/zsh (commands below assume this; adjust paths for other shells)
- A Kaggle account is **not** required to download this specific dataset —
  `kagglehub` can pull public datasets anonymously. If you ever hit an auth
  error, see Troubleshooting at the bottom.

---

## 1. Create an isolated environment

Never install packages globally — every dependency for this project lives in
its own `venv`.

```bash
cd "/Users/prabuddhaverma/Visual Studio Code /yolo"
python3 -m venv venv
source venv/bin/activate
python3 -m pip install --upgrade pip
```

**Expected result:** your shell prompt gains a `(venv)` prefix. `which python3`
now points inside `./venv/bin/`.

---

## 2. Install dependencies

`requirements.txt` is intentionally added to *one dependency at a time, at the
document where it's first needed* — not all at once — so it's always obvious
why a package is there. Current contents:

```
kagglehub
Pillow
opencv-python
numpy
pandas
matplotlib
```

Install them:

```bash
pip install -r requirements.txt
```

**Expected result:**

```bash
python3 -c "import cv2, numpy, pandas, matplotlib, kagglehub; print('all OK')"
```
prints `all OK` with no import errors.

---

## 3. Download the dataset

```python
import kagglehub
path = kagglehub.dataset_download("lahbibfedi/fashion-dataset-with-annotation")
print("Path to dataset files:", path)
```

Run it:

```bash
python3 -c "
import kagglehub
path = kagglehub.dataset_download('lahbibfedi/fashion-dataset-with-annotation')
print('Path to dataset files:', path)
"
```

**Expected result:** a ~627MB download the first time (cached after that —
re-running is instant), ending with:

```
Path to dataset files: /Users/<you>/.cache/kagglehub/datasets/lahbibfedi/fashion-dataset-with-annotation/versions/1
```

This path is **not** inside the project repo — kagglehub manages its own
cache outside of git, which is why you won't see a `data/` folder full of
raw images in this repository.

---

## 4. Verify the raw structure yourself

Don't trust a dataset — inspect it. This is exactly what Document 1 turns
into reusable code, but first, do it by hand once so you know what "correct"
looks like:

```bash
DSPATH="$(python3 -c "import kagglehub; print(kagglehub.dataset_download('lahbibfedi/fashion-dataset-with-annotation'))")"
find "$DSPATH" -maxdepth 4 -type d
```

**Expected result:**

```
.../versions/1
.../versions/1/new_train (1)
.../versions/1/new_train (1)/new_t
.../versions/1/new_train (1)/new_t/images
.../versions/1/new_train (1)/new_t/labels
.../versions/1/new_validation (1)
.../versions/1/new_validation (1)/new_v
.../versions/1/new_validation (1)/new_v/images
.../versions/1/new_validation (1)/new_v/labels
```

Look at one label file:

```bash
cat "$DSPATH/new_train (1)/new_t/labels/000060.txt"
```

**Expected result:**

```
6 0.5266666666666666 0.29235537190082644 0.4666666666666667 0.3140495867768595
0 0.5253333333333333 0.1559917355371901 0.464 0.3119834710743802
```

Format: `class_id x_center y_center width height`, all four numbers
normalized to `[0, 1]` — standard YOLO label format. This image has two
garments: class `6` (trousers) and class `0` (short_sleeve_top).

**Full findings from this kind of inspection are written up in
[docs/00_Project_Architecture.md](docs/00_Project_Architecture.md)** — 12,000
images total (10,000 train / 2,000 val), zero corrupted files, zero
missing pairs, zero invalid boxes, 13 classes (ids `0`–`12`), heavy class
imbalance (3,755 vs 28 instances), and 2,693 distinct image resolutions in
the train split alone. Read that file before writing any code — it tells you
exactly what your Document 1 code should reproduce.

---

## 5. Project structure (current)

```
yolo/
├── venv/                       (gitignored — your isolated environment)
├── requirements.txt
├── README.md
├── LEARNER_GUIDE.md            ← this file
├── .gitignore
├── docs/
│   ├── 00_Project_Architecture.md   (architecture + verified dataset facts)
│   └── 01_Dataset_Analysis.md       (Document 1 spec + implementation order)
└── src/
    └── dataset_analysis/
        ├── schema.py            (written — data contracts)
        ├── config.py            (written — paths + class names)
        ├── main.py              (written — orchestrator)
        ├── dataset_loader.py    (IMPLEMENTED — reference, see Section 7.1)
        ├── image_loader.py      (IMPLEMENTED — reference, see Section 7.2)
        ├── annotation_parser.py (IMPLEMENTED — reference, see Section 7.3)
        ├── statistics_generator.py (IMPLEMENTED — reference, see Section 7.4)
        ├── visualization.py     (IMPLEMENTED — reference, see Section 7.5)
        └── report_generator.py  (IMPLEMENTED — reference, see Section 7.6)
```

**Status note:** all six modules now contain a working reference
implementation (run end-to-end on 2026-06-28, see Section 7 for every
command + real output). The docstrings — the actual spec — are unchanged.
If you want the practice of writing these yourself rather than reading the
answer, the exercise is the same: read each docstring, ignore the function
body below it, and write your own version, then diff against what's there.
The `raise NotImplementedError(...)` stub state is preserved in git history
(`git log -- src/dataset_analysis/`) if you want to check out that earlier
commit and start from a clean stub.

Folders for Documents 2–12 (`annotation_validation/`, `opencv_fundamentals/`,
etc.) don't exist yet — they get created at the start of each document, not
in advance. An empty folder commits nothing useful to git anyway.

---

## 6. File & Function Reference — what does what, and why

The data flow through Document 1, end to end:

```
raw image/label paths
    │
    ▼
dataset_loader.load_dataset()          paths in  → paired (image, label) records out
    │
    ▼
image_loader.load_image_meta()         one image path → pixel metadata        (per record)
annotation_parser.parse_annotation()   one label path → validated boxes       (per record)
    │
    ▼
statistics_generator.generate_statistics()   metadata + boxes → counted-up numbers
    │
    ▼
visualization.*()                      numbers + records → PNG files
report_generator.generate_report()     numbers → CSV files
```

Every arrow is a module boundary, and every module boundary exists for a
specific reason — not arbitrarily. Below is each file, its function(s), and
*why* the boundary is drawn exactly there.

### `schema.py` — shared data contracts (written for you)

**What:** four `@dataclass` definitions — `BoundingBox`, `Annotation`,
`ImageMeta`, `DatasetRecord`. No logic, no I/O, just named, typed shapes.

**Why it exists:** without a shared contract, every module would invent its
own tuple or dict for "a bounding box" — one might use
`(class_id, x, y, w, h)`, another `(x, y, w, h, class_id)`, and a mismatch
between them would fail silently (wrong number assigned to wrong field,
no error, just corrupted statistics three modules later). A dataclass is
one definition that every other file imports — change a field name once,
every caller that's wrong fails loudly with an `AttributeError`, immediately,
at the point of the mistake.

### `config.py` — single source of truth for paths & class names (written for you)

**What:** `get_dataset_root()` (downloads/locates the cached dataset via
`kagglehub`), the four subdirectory path constants
(`TRAIN_IMAGES_SUBDIR`, etc.), `NUM_CLASSES = 13`, the `CLASS_NAMES` dict,
and `OUTPUT_DIR`.

**Why it exists:** the real on-disk folder names are oddly nested
(`new_train (1)/new_t/images`) — an artifact of how the dataset was zipped.
If six different files each hardcoded that path string, fixing a typo or
adapting to a future dataset version means editing six files and hoping you
didn't miss one. Centralizing it means changing one constant. The same logic
applies to `CLASS_NAMES`: it's an inferred hypothesis (Document 0, Section 3),
not a fact read from a file — it needs to live in exactly one place you can
correct after you run `draw_bbox_overlay()` yourself.

### `main.py` — orchestrator (written for you)

**What:** `run(split)` calls, strictly in this order: `load_dataset()` →
`load_image_meta()` + `parse_annotation()` per record → `generate_statistics()`
→ four `visualization` calls → `generate_report()`.

**Why it exists:** so that sequencing logic and data-processing logic never
mix. `main.py` contains zero statistics math, zero validation rules, zero
plotting code — it only calls things in order and prints progress. This is
exactly why you could test `load_dataset()` directly in Section 7.1 without
running the whole pipeline: every module is independently callable, and
`main.py` is the only file that needs to know about all of them at once.

### `dataset_loader.py` → `load_dataset(images_dir, labels_dir, split)`

**What:** lists `*.jpg` and `*.txt` files, matches them by filename stem,
returns `LoadResult(records, images_without_label, labels_without_image)`.

**Why it exists, and why it's first:** nothing downstream can run until you
know which (image, label) pairs are even valid to process — this is the
gatekeeper. It deliberately knows **nothing** about pixel content or box
content (no `cv2.imread`, no parsing `.txt` contents) — that boundary keeps
it fast (pure filesystem listing over 12,000 files) and means a bug in image
decoding can never be mistaken for a pairing bug, because they're physically
different functions.

### `image_loader.py` → `load_image_meta(image_path)`

**What:** opens one image, returns `ImageMeta(height, width, channels, is_corrupted, error)`.

**Why it exists:** decoding is the expensive, failure-prone step (vs. just
listing a filename). Isolating it in one function means: (1) it's decoded
exactly once per image — `statistics_generator.py` reuses this result instead
of re-opening the file, and (2) a corrupted file is caught in exactly one
place, with one consistent meaning (`is_corrupted=True`), instead of crashing
differently depending on which downstream module happened to touch the file
first.

### `annotation_parser.py` → `parse_annotation(label_path, num_classes)`

**What:** reads a `.txt` label file line by line, validates each one, returns
`Annotation(boxes=[valid...], malformed_lines=[raw invalid lines])`.

**Why it exists:** this is the *only* place the six validation rules
(class id in range, coordinates in `[0,1]`, positive width/height, box stays
in-bounds) are implemented. Keeping validation in exactly one function means
Document 2 (Annotation Validation) can reuse this same logic rather than
re-deriving the rules from scratch, and means "what counts as a malformed
annotation" has exactly one definition in the whole project, not a slightly
different interpretation in every module that touches a label file.

### `statistics_generator.py` → `generate_statistics(load_result, image_meta_by_stem, annotations_by_stem)`

**What:** takes the *already computed* outputs of the three functions above
and reduces them to one `DatasetStatistics` (totals, class distribution,
average resolution, largest/smallest object, corruption/malformed counts).

**Why it exists, and why it does no I/O:** it's deliberately a pure function
— same inputs always produce the same `DatasetStatistics`, with no file reads
inside it. That means you can unit-test it with three fake records in
milliseconds, without touching the real 627MB dataset, and a bug in "the
math" can never be confused with a bug in "reading the file" — they're
different functions you can test in isolation.

### `visualization.py` — 5 functions

**What:** `plot_class_histogram`, `plot_bbox_size_histogram`,
`plot_resolution_distribution`, `show_random_samples`, `draw_bbox_overlay` —
each takes already-computed data plus an output path, saves one PNG, returns
the path it wrote.

**Why it exists, separately from statistics:** separating "compute the
numbers" from "draw a picture of the numbers" means you can redesign a chart
(colors, sort order, bins) without touching any statistic, and vice versa.
`draw_bbox_overlay()` is the most important function in this file for a
different reason: it's the only output in Document 1 you must *look at*, not
just print — it's how you personally confirm or correct the inferred
`CLASS_NAMES` mapping in `config.py` before it gets baked into training data
in Document 6.

### `report_generator.py` → `generate_report(stats, output_dir, split)`

**What:** serializes one `DatasetStatistics` into
`{split}_dataset_report.csv` and `{split}_bbox_statistics.csv`, returns a
dict of `{artifact_name: path_written}`.

**Why it exists, and why it's the only writer:** concentrating every disk
write in one function means there's exactly one place to check if you ever
need to answer "what files does this project create, and where" — you don't
have to grep six files to find a stray `open(..., "w")`. It also means a
partial-write bug (a crash mid-save leaving a half-written CSV) can only ever
happen in one place, which makes it easy to guard against (build the full
content in memory, then write once).

---

## 7. Document 1 — Dataset Analysis: implementation walkthrough

Implement the six modules **in this order** — each depends on the one before
it being testable. (See Section 6 above for *what* each function does and
*why* it's structured this way — this section is the "do it, test it" steps.)

> **Every "Test it" command below (7.1–7.6), and the full pipeline command in
> 7.7, will raise `NotImplementedError` until you've actually written the
> body of the function it calls.** That is correct, expected behavior — the
> stub is designed to fail loudly and tell you exactly which function to
> implement next, instead of failing silently or doing the wrong thing. For
> example, right now, *before* implementing anything, running 7.7 gives:
> ```
> NotImplementedError: TODO: implement load_dataset. Hint: ...
> ```
> That traceback is proof the wiring (`main.py` → `dataset_loader.load_dataset`)
> is correct: it ran `Initialize → Load` and stopped at the very first
> unimplemented piece, as designed. Don't run 7.7 until 7.1–7.6 are all done —
> each section below is a checkpoint, run its own "Test it" command first and
> confirm the "Expected result" before moving to the next section.
>
> **Update:** all six modules below now have a working reference
> implementation, and every command in 7.1–7.7 has actually been run, with
> the real output recorded under **"✅ Observed result"** beneath each
> "Expected result." The `NotImplementedError` explanation above stays in
> this guide because it's still exactly what you'll see if you reset any one
> of these files to a stub and work through it yourself.

### 7.1 `dataset_loader.py` → `load_dataset()`

Locate dataset → verify folders exist → collect image paths → collect label
paths → pair them by filename stem. Full spec is in the docstring already in
the file.

**Test it:**

```bash
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

**Expected result:**

```
paired records: 10000
images_without_label: 0
labels_without_image: 0
```

**✅ Observed result (actually run, 2026-06-28):** exact match —
```
paired records: 10000
images_without_label: 0
labels_without_image: 0
sample record: DatasetRecord(stem='000024', split='train',
  image_path=PosixPath('.../new_train (1)/new_t/images/000024.jpg'),
  label_path=PosixPath('.../new_train (1)/new_t/labels/000024.txt'))
```
The reference implementation: build a `set` of `.stem` from each `.glob()`,
intersect for pairs, set-difference for orphans. ~15 lines.

### 7.2 `image_loader.py` → `load_image_meta()`

Read one image → detect corruption → return height/width/channels.

**Test it** (using a real filename from Section 4 above):

```bash
python3 -c "
from src.dataset_analysis import config
from src.dataset_analysis.image_loader import load_image_meta

root = config.get_dataset_root()
img = root / config.TRAIN_IMAGES_SUBDIR / '000060.jpg'
print(load_image_meta(img))
"
```

**Expected result:** an `ImageMeta` with `is_corrupted=False`, `error=None`,
and real positive `height`/`width`/`channels` values (`channels` should be 3).

**✅ Observed result (actually run):**
```
ImageMeta(height=968, width=750, channels=3, is_corrupted=False, error=None)
```
Also tested the failure path directly (a path that doesn't exist), to prove
it returns instead of crashing:
```
ImageMeta(height=0, width=0, channels=0, is_corrupted=True,
          error='file not found: .../does_not_exist.jpg')
```
**Observation:** notice `000060.jpg` is 968×750 — neither dimension matches
the "common" resolutions from Document 0 (468×624, 640×960, etc). That's the
2,693-distinct-resolutions fact made concrete on one real file: there is no
single "typical" image here, which is exactly the argument for why Document 4
can't skip resizing.

### 7.3 `annotation_parser.py` → `parse_annotation()`

Read a `.txt` label file → validate each line → return structured boxes +
any malformed lines.

**Test it:**

```bash
python3 -c "
from src.dataset_analysis import config
from src.dataset_analysis.annotation_parser import parse_annotation

root = config.get_dataset_root()
lbl = root / config.TRAIN_LABELS_SUBDIR / '000060.txt'
ann = parse_annotation(lbl, config.NUM_CLASSES)
print('boxes:', ann.boxes)
print('malformed:', ann.malformed_lines)
"
```

**Expected result:** `boxes` is a list of exactly 2 `BoundingBox` objects
(`class_id=6` and `class_id=0`), `malformed` is an empty list.

**✅ Observed result (actually run):**
```
boxes: [BoundingBox(class_id=6, x_center=0.5267, y_center=0.2924, width=0.4667, height=0.3140),
        BoundingBox(class_id=0, x_center=0.5253, y_center=0.1560, width=0.4640, height=0.3120)]
malformed: []
```
Exact match — two boxes, class `6` (trousers) and class `0` (short_sleeve_top),
nothing malformed.

### 7.4 `statistics_generator.py` → `generate_statistics()`

Reduce the outputs of 7.1–7.3 (run across *all* records, not just one) into
one `DatasetStatistics` object.

**Expected result when run across the full train split** — these are the
real numbers, your code should reproduce them exactly (see
[docs/01_Dataset_Analysis.md](docs/01_Dataset_Analysis.md) for the full table):

| Metric | Expected |
|---|---|
| total_images | 10000 |
| total_labels | 10000 |
| corrupted_files | 0 |
| malformed_annotation_count | 0 |
| largest class | `0` (3,755 instances) |
| smallest class | `2` (28 instances) |

**✅ Observed result (actually run across all 10,000 train images + labels):**
```
DatasetStatistics(total_images=10000, total_labels=10000,
  images_without_label=0, labels_without_image=0,
  corrupted_files=0, malformed_annotation_count=0,
  class_distribution={0: 3755, 1: 1830, 2: 28, 3: 675, 4: 831, 5: 92,
                       6: 1889, 7: 2807, 8: 1682, 9: 912, 10: 429,
                       11: 993, 12: 313},
  avg_resolution=(603.6171, 757.1521),
  avg_bbox_area_norm=0.2794430877704816,
  largest_object={'stem': '177974', 'class_id': 3, 'area_norm': 0.9974},
  smallest_object={'stem': '187643', 'class_id': 7, 'area_norm': 0.00047})
```
Every count matches Document 0's full scan exactly — this is the real proof
that the implementation is correct, not just "looks plausible."

**Timing observation:** decoding all 10,000 images with `cv2.imread` +
parsing all 10,000 label files took **~24 seconds** on this machine (M-series
MacBook Air, single-threaded, no multiprocessing). Keep that number in mind
for Document 6 — training will iterate over this data far more than once per
run, so image-decode speed is a real cost, not a one-time fee.

**Val split, for comparison** (2,000 images — same command, `'val'` instead of `'train'`):
```
class_distribution={0: 806, 1: 347, 2: 13, 3: 111, 4: 163, 5: 32, 6: 269,
                     7: 610, 8: 419, 9: 181, 10: 98, 11: 197, 12: 67}
avg_resolution=(605.4115, 755.306)
largest_object={'stem': '008097', 'class_id': 0, 'area_norm': 0.9967}
smallest_object={'stem': '004940', 'class_id': 7, 'area_norm': 0.00639}
```
**Observation:** dividing each val count by its train count gives roughly
`0.20–0.23` across every single class (e.g. `806/3755=0.215`, `13/28=0.464` —
class `2` is the exception simply because 28 is too small a population for
the ratio to be stable). This tells you the train/val split was done by
*random* sampling, not stratified by class — for a class this rare (28
instances total), that's worth remembering in Document 6 when you look at
why its validation metrics are noisy.

### 7.5 `visualization.py` (5 functions)

Each function saves one PNG and returns its path. Implement
`draw_bbox_overlay()` last and use it across several samples per class —
this is also how you'll personally confirm or correct the class-name
hypothesis in `config.CLASS_NAMES` (it was inferred by visual inspection,
not read from a file the dataset doesn't provide).

**✅ Observed result — class-name verification actually performed:**
Document 0 flagged two class names as lower-confidence: class `10`
(`long_sleeve_dress`) and class `11` (`vest_dress`). Ran `draw_bbox_overlay()`
on 2 real samples of each and looked at the output images directly:
- **Class 11 (`vest_dress`):** both samples are unambiguous — a black
  sleeveless dress and a red sleeveless dress, box drawn tight around the
  full dress. **Confirmed.**
- **Class 10 (`long_sleeve_dress`):** both samples show a dress-length white
  garment with half/elbow sleeves (one a shirt-dress, one a tunic over
  leggings, with the leggings correctly boxed separately as class `7`
  trousers). Consistent with the "long sleeve" naming pattern already
  confirmed for class `1` (sleeves read as "long" relative to the
  short-sleeve classes, even when not full wrist-length in casual photos).
  **Confirmed, same confidence level as the other classes.**

Both `CLASS_NAMES` entries in `config.py` stayed as originally inferred — no
correction needed. If your own spot-check ever disagrees, that's the moment
to edit `config.CLASS_NAMES`, not later.

### 7.6 `report_generator.py` → `generate_report()`

Writes the final CSV deliverables. The only module allowed to create
directories / write files to disk.

**✅ Observed result — actual `train_bbox_statistics.csv` contents:**
```
class_id,class_name,instance_count,share_of_total
0,short_sleeve_top,3755,0.2313
1,long_sleeve_top,1830,0.1127
2,short_sleeve_outwear,28,0.0017
3,long_sleeve_outwear,675,0.0416
4,vest,831,0.0512
5,sling,92,0.0057
6,shorts,1889,0.1163
7,trousers,2807,0.1729
8,skirt,1682,0.1036
9,short_sleeve_dress,912,0.0562
10,long_sleeve_dress,429,0.0264
11,vest_dress,993,0.0612
12,sling_dress,313,0.0193
```

### 7.7 Run the whole pipeline

**⚠️ Prerequisite: sections 7.1–7.6 must all be implemented and individually
tested first.** This command exercises all six modules together — if any one
of them is still a stub, this is where it'll surface, naming that exact
function.

```bash
python -m src.dataset_analysis.main
```

**Expected result:** prints stats for `train` then `val`, and creates:

```
outputs/dataset_analysis/
├── train_dataset_report.csv
├── train_bbox_statistics.csv
├── train_class_distribution.png
├── train_bbox_size_histogram.png
├── train_resolution_distribution.png
├── train_sample_grid.png
├── val_dataset_report.csv
├── val_bbox_statistics.csv
├── val_class_distribution.png
├── val_bbox_size_histogram.png
├── val_resolution_distribution.png
└── val_sample_grid.png
```

(`outputs/` is gitignored — it's regenerated by running this command, never
committed.)

**✅ Observed result (actually run, full pipeline, both splits):**
```
[train] paired records: 10000 | images_without_label: 0 | labels_without_image: 0
[train] stats: DatasetStatistics(total_images=10000, ... corrupted_files=0, malformed_annotation_count=0, ...)
[train] wrote: {'dataset_report': outputs/dataset_analysis/train_dataset_report.csv,
                'bbox_statistics': outputs/dataset_analysis/train_bbox_statistics.csv}
[val] paired records: 2000 | images_without_label: 0 | labels_without_image: 0
[val] stats: DatasetStatistics(total_images=2000, ... corrupted_files=0, malformed_annotation_count=0, ...)
[val] wrote: {'dataset_report': outputs/dataset_analysis/val_dataset_report.csv,
              'bbox_statistics': outputs/dataset_analysis/val_bbox_statistics.csv}
```
**Total runtime: ~62 seconds** (both splits, 12,000 images decoded + parsed +
plotted, single core, no caching). All 12 files listed above were created —
confirmed with `ls -la outputs/dataset_analysis/`, file sizes ranged from
~22KB (histograms) to ~1.9MB (`train_sample_grid.png`, a 4×4 grid of full
JPEGs).

**Looked at the actual PNGs, not just the file listing:**
- `train_class_distribution.png` — bar chart, sorted descending, confirms the
  134:1 imbalance (`short_sleeve_top` 3,755 vs `short_sleeve_outwear` 28) is
  immediately, visually obvious — exactly the point of sorting by count
  instead of alphabetically.
- `train_resolution_distribution.png` — scatter plot titled "2693 distinct
  sizes" (computed live from the data, matching Document 0's manual count
  exactly). Visually, points cluster into **vertical streaks** at specific
  widths (≈468px, ≈640px, ≈750px, ≈800px, ≈880px) rather than spreading
  smoothly — these are common stock-photo/e-commerce crop widths with
  *varying* heights, not arbitrary camera resolutions. That pattern is only
  visible once you plot it; the raw "2,693 distinct sizes" number alone
  doesn't tell you that.

---

## 8. Document 3 — OpenCV Fundamentals

**Built out of order, on request** — Document 2 (Annotation Validation) is
deferred, not skipped (see Section 9 below for why it still matters). This
section exists because seeing classical CV operations run on real images —
grayscale, BGR↔RGB, thresholding, contours — mattered more right now than
finishing validation plumbing. Full writeup: [docs/03_OpenCV_Fundamentals.md](docs/03_OpenCV_Fundamentals.md).

### What got built

| File | Function | Produces |
|---|---|---|
| `image_reader.py` | `read_image()` | Original (BGR) |
| `color_converter.py` | `to_rgb/to_hsv/to_lab/to_gray()` | RGB, HSV, LAB, Grayscale |
| `blur_processor.py` | `apply_gaussian/median/bilateral()` | 3 smoothing filters |
| `threshold_processor.py` | `apply_global_threshold/adaptive_threshold/morphology()` | 2 binary masks + cleanup |
| `edge_detector.py` | `detect_edges()` | Canny edges |
| `contour_detector.py` | `find_contours/draw_bounding_boxes()` | Outlines → rectangles |
| `visualizer.py` | `run_pipeline/save_step_grid/save_individual_steps/print_step_table()` | All artifacts |
| `main.py` | `run(sample_stem)` | Orchestrates all of the above + a Document-1 YOLO ground-truth bridge step |

Plus `notebooks/03_opencv_fundamentals.ipynb` — same functions, called
cell-by-cell with markdown explanations between them, **already executed**
(every image below is embedded in the file, not regenerated on open).

### Run it

```bash
python -m src.opencv_fundamentals.main
```
Saves `outputs/opencv_fundamentals/{stem}_pipeline_grid.png` (one grid, all
15 steps labeled with shape/channels/time) plus one PNG per step in
`outputs/opencv_fundamentals/{stem}/`. Default sample is `000060`; pass a
different stem (`from src.opencv_fundamentals.main import run; run("000150")`)
to try another image.

### ✅ Observed results (actually run)

**First attempt used the wrong sample image.** `000060.jpg` (a black/white
striped outfit) was tried first — and the BGR-vs-RGB panels looked almost
identical, because that photo barely has saturated color to swap. Switched
to `000150.jpg` (a red dress) specifically because the lesson needs a
colorful image to be visible at all. Worth remembering generally: a demo
that "shows no difference" sometimes means the demo input was a poor
choice, not that there's no difference to show.

**The BGR/RGB step, done right** — `image_reader.py`'s Original step is
displayed **deliberately uncorrected** (no `cv2.cvtColor` before
`imshow`), and the RGB step is the corrected version, side by side:

| Original (BGR) — shown uncorrected | RGB — corrected |
|---|---|
| dress renders **blue** | dress renders **red** |

This caught a real bug while building it: the first version of
`visualizer.py` *auto-corrected* every BGR image before display "for
convenience," which silently defeated the entire lesson (both panels showed
red, looking identical). Fixed by tagging the Original step `"BGR_RAW"`
(shown raw) vs. downstream BGR arrays like bounding-box overlays `"BGR"`
(shown corrected, since *those* steps aren't trying to teach the bug).

**Timing — verified, not assumed:** the first `cv2.cvtColor(..., COLOR_BGR2LAB)`
call in a fresh process took **~102–111ms**; three immediate repeat calls
took `0.29ms, 0.27ms, 0.25ms`. That's a 400x difference between call 1 and
call 2 — confirmed by literally re-running the conversion 4 times in the
same process before writing this down, rather than trusting the first
number. It's OpenCV building an internal LAB lookup table once per process.

**Classical contours vs. YOLO ground truth, side by side** — the
`main.py`-added bonus step overlays Document 1's real annotation on the
same image the contour detector just processed. On both test images,
contour-based bounding boxes either wrapped the whole person+background as
one noisy blob, or fragmented into several small boxes around
background text/logos — nothing resembling "a garment." The YOLO
ground-truth box, by contrast, is exactly the dress. This is the concrete,
visual argument for why Document 6 trains a model instead of relying on
thresholding+contours.

**A real bug caught while building this, kept here on purpose:**
`visualizer.py` originally called `matplotlib.use("Agg")` at module level.
The CLI script worked fine. The **notebook silently broke** — no errors, 0
embedded images, every `plt.show()` was a no-op — because importing the
same `visualizer` module forced the notebook's backend to a non-interactive
one too. Fixed by moving the `Agg` call into `main.py`'s
`if __name__ == "__main__":` guard, so it only fires for a real CLI run,
never for an import. **Lesson:** library modules should never call
`matplotlib.use()` — that's an application-level decision, and a library
that makes it silently breaks whatever *other* rendering context imports it.

---

## 9. What's next

Document 2 — Annotation Validation — formalizes the malformed-annotation
logging from Document 1 into its own reusable pipeline with a CSV report.
That section will be appended here once we start it.

Two real findings from Document 1's run directly motivate it:
- **Zero malformed lines were found in this dataset** (`malformed_annotation_count=0`
  on both splits) — so Document 2's validation logic has nothing to exercise
  *yet*. Document 2 should therefore include deliberately injecting a few
  broken lines into a copy of the data, to prove the validator actually
  catches them, rather than trusting a clean pass on data that was never
  going to fail.
- **The train/val split isn't stratified by class** (Section 7.4's
  val/train ratio observation) — worth deciding in Document 2 or Document 6
  whether to re-split with stratification, given class `2`
  (`short_sleeve_outwear`) has only 28 train + 13 val examples total.

After Document 2: Document 4 — Image Preprocessing (resize/pad/normalize to
the 640x640 input YOLO expects) is the natural next step after Document 3's
classical-CV foundation.

---

## Troubleshooting

- **`kagglehub` asks for Kaggle credentials / fails to download:** this
  specific dataset is public and should download anonymously. If it still
  fails, create a Kaggle API token (Kaggle account → Settings → API → Create
  New Token) and place it at `~/.kaggle/kaggle.json`.
- **`ModuleNotFoundError` for `cv2`/`numpy`/etc.:** your `venv` isn't
  activated — run `source venv/bin/activate` first, every new terminal
  session.
- **`ImportError: attempted relative import` when running a file directly:**
  run modules as `python -m src.dataset_analysis.main`, not
  `python src/dataset_analysis/main.py` — the `-m` form is required for the
  relative imports (`from . import config`) inside the package to resolve.
- **`NotImplementedError: TODO: implement ...`:** not a bug — see the callout
  at the top of Section 7. It means the function named in the message hasn't
  been implemented yet. Go implement it.
