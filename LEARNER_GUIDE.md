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
        ├── dataset_loader.py    (STUB — implement load_dataset())
        ├── image_loader.py      (STUB — implement load_image_meta())
        ├── annotation_parser.py (STUB — implement parse_annotation())
        ├── statistics_generator.py (STUB — implement generate_statistics())
        ├── visualization.py     (STUB — implement 5 plotting functions)
        └── report_generator.py  (STUB — implement generate_report())
```

Folders for Documents 2–12 (`annotation_validation/`, `opencv_fundamentals/`,
etc.) don't exist yet — they get created at the start of each document, not
in advance. An empty folder commits nothing useful to git anyway.

---

## 6. Document 1 — Dataset Analysis: implementation walkthrough

Implement the six modules **in this order** — each depends on the one before
it being testable.

### 6.1 `dataset_loader.py` → `load_dataset()`

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

### 6.2 `image_loader.py` → `load_image_meta()`

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

### 6.3 `annotation_parser.py` → `parse_annotation()`

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

### 6.4 `statistics_generator.py` → `generate_statistics()`

Reduce the outputs of 6.1–6.3 (run across *all* records, not just one) into
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

### 6.5 `visualization.py` (5 functions)

Each function saves one PNG and returns its path. Implement
`draw_bbox_overlay()` last and use it across several samples per class —
this is also how you'll personally confirm or correct the class-name
hypothesis in `config.CLASS_NAMES` (it was inferred by visual inspection,
not read from a file the dataset doesn't provide).

### 6.6 `report_generator.py` → `generate_report()`

Writes the final CSV deliverables. The only module allowed to create
directories / write files to disk.

### 6.7 Run the whole pipeline

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

---

## 7. What's next

Document 2 — Annotation Validation — formalizes the malformed-annotation
logging from Document 1 into its own reusable pipeline with a CSV report.
That section will be appended here once we start it.

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
