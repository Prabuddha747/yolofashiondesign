# 00 — Project Architecture

## Status of this document
Written **before** any model code, using ground-truth facts pulled directly from the
downloaded dataset (not assumptions). Every number below came from scripts we ran
against the raw files — see "Verified Dataset Facts".

---

## 1. Problem Statement

Build an end-to-end **Computer Vision system** that:

1. Detects fashion garments in an image or live camera feed (YOLO object detection).
2. Explains *how* each detection was produced — not just the answer, but the pixel
   pipeline that led to it (color conversions, resizing, tensor shapes, NMS, confidence).
3. Given a detected garment, recommends visually similar garments from the dataset
   (embedding + similarity search — a content-based recommender, not collaborative
   filtering).
4. Surfaces all of the above in a live dashboard with analytics (FPS, confidence,
   per-class counts, inference time).

YOLO is **one module** inside this system (Document 6). The system also requires:
classical OpenCV (Document 3–5), evaluation methodology (Document 7), explainability
(Document 8), retrieval (Document 9), and real-time engineering (Document 10–11).

## 2. Objectives

- Learn to treat a dataset as something to be *interrogated*, not trusted — validate
  before training.
- Learn classical CV (color spaces, filters, thresholding, morphology, contours)
  before relying on a learned detector for everything.
- Learn the YOLO training/eval loop deeply enough to read a PR curve and a confusion
  matrix and know what's wrong.
- Learn embedding-based retrieval (the same idea that powers reverse image search,
  recommendation engines, and RAG — just on pixels instead of text).
- Ship something that runs live on a webcam, end to end.

## 3. Dataset — Verified Facts

Source: Kaggle `lahbibfedi/fashion-dataset-with-annotation`, downloaded via
`kagglehub.dataset_download(...)` to
`~/.cache/kagglehub/datasets/lahbibfedi/fashion-dataset-with-annotation/versions/1`.

**Raw layout actually on disk** (note: this differs from a typical clean repo —
folder names have a trailing `(1)` from how the uploader zipped it, and there's
an extra single-letter nesting level):

```
versions/1/
├── new_train (1)/
│   └── new_t/
│       ├── images/   10,000 .jpg
│       └── labels/   10,000 .txt
└── new_validation (1)/
    └── new_v/
        ├── images/   2,000 .jpg
        └── labels/   2,000 .txt
```

There is **no test split**, **no `classes.txt`**, **no `data.yaml`**, **no README**
shipped with the data. Anything resembling class names or a held-out test set has
to be built by us — that becomes explicit work in Document 1 (statistics) and
Document 6 (we will carve a test split out of validation, or use k-fold).

**Label format** — confirmed by reading raw `.txt` files: standard YOLO format,
one line per object:

```
<class_id> <x_center> <y_center> <width> <height>
```

All five values space-separated, last four normalized to `[0, 1]` relative to image
width/height. Example (`000060.txt`):

```
6 0.5266666666666666 0.29235537190082644 0.4666666666666667 0.3140495867768595
0 0.5253333333333333 0.1559917355371901 0.464 0.3119834710743802
```

**Integrity checks already run (full scan, not a sample):**

| Check | Train | Val |
|---|---|---|
| Images | 10,000 | 2,000 |
| Labels | 10,000 | 2,000 |
| Images missing a label | 0 | 0 |
| Labels missing an image | 0 | 0 |
| Corrupted images (failed to open) | 0 | 0 |
| Empty label files | 0 | 0 |
| Zero-width / zero-height boxes | 0 | 0 |
| Boxes extending outside `[0,1]` | 0 | 0 |
| Color mode | 100% RGB | 100% RGB |

This is a **clean** dataset — pairing is perfect and no malformed boxes exist. That
won't always be true of real datasets, which is exactly why Document 2 (Annotation
Validation) still gets built as a real, reusable pipeline rather than skipped.

**Objects per image (train):**

| Objects/image | Count |
|---|---|
| 1 | 4,005 |
| 2 | 5,812 |
| 3 | 125 |
| 4 | 58 |

Average ≈ 1.58 objects/image. Most images show one or two garments (e.g., top +
bottom), which matters for augmentation choices later (Mosaic helps; heavy
small-object augmentation matters less here than in crowded-scene datasets).

**Image resolution:** highly non-uniform — **2,693 distinct resolutions** in train
alone (most common: 468×624, 468×702, 640×960, 640×640). This is why Document 4
(resize/pad/letterbox to 640×640) is not optional boilerplate — it's load-bearing.

**Class distribution (train, 13 classes, ids `0`–`12`):**

| id | instances | inferred name | confidence |
|---|---|---|---|
| 0 | 3,755 | short_sleeve_top | verified visually |
| 1 | 1,830 | long_sleeve_top | verified visually |
| 2 | 28 | short_sleeve_outwear | verified visually |
| 3 | 675 | long_sleeve_outwear | verified visually |
| 4 | 831 | vest | verified visually |
| 5 | 92 | sling | verified visually |
| 6 | 1,889 | shorts | verified visually |
| 7 | 2,807 | trousers | verified visually |
| 8 | 1,682 | skirt | verified visually |
| 9 | 912 | short_sleeve_dress | verified visually |
| 10 | 429 | long_sleeve_dress | plausible, lower confidence |
| 11 | 993 | vest_dress | plausible, lower confidence |
| 12 | 313 | sling_dress | verified visually |

**How these names were derived:** the raw dataset gives no class names at all —
only integer ids 0–12. I cropped 2–3 real bounding boxes per class id and looked
at them directly. The ids, counts, and visual content line up with the
**DeepFashion2** category taxonomy (13 garment categories, same order). I'm
treating this as a strong working hypothesis, not certainty — it will get a
dedicated re-check with many more samples in Document 1 (Visualization /
"Random Samples" + "Bounding Box Overlay" sections), where you'll look at a full
grid per class yourself before we commit `classes.yaml` for training.

There is significant **class imbalance** (3,755 vs 28 — a 134:1 ratio between the
biggest and smallest class). This is a fact the Statistics Generator in Document 1
must surface loudly, and it directly affects: augmentation strategy (Document 5),
loss weighting / sampling during training (Document 6), and how we read per-class
recall in evaluation (Document 7) — a model can look "good" on mAP while being
useless on `short_sleeve_outwear`.

## 4. System Architecture (data flow)

```
Raw Dataset (Kaggle, YOLO-format labels)
        │
        ▼
Dataset Analysis ───────► dataset_report.csv, class_distribution.png
        │
        ▼
Annotation Validation ──► annotation_validation.csv, validation_summary.txt
        │
        ▼
OpenCV Fundamentals  (taught on sample images; not a pipeline stage)
        │
        ▼
Image Preprocessing ────► resize / pad / normalize → model-ready tensors
        │
        ▼
Data Augmentation ──────► augmented training set (flip/rotate/mosaic/etc.)
        │
        ▼
YOLO Dataset Conversion ► data.yaml + train/val/(test) folder layout Ultralytics expects
        │
        ▼
YOLO Training (YOLOv8) ─► weights (best.pt), training curves
        │
        ▼
Model Evaluation ───────► PR curve, confusion matrix, per-class metrics
        │
        ▼
        ├──────────────────────────────┐
        ▼                              ▼
Real-Time Detection            Explainable Detection
(camera → boxes on screen)     (every pixel-level step shown)
        │
        ▼
Feature Extraction (crop detected garment → embedding)
        │
        ▼
Similarity Search (FAISS / cosine) over embeddings of the whole dataset
        │
        ▼
Recommendation (top-5 visually similar garments)
        │
        ▼
Dashboard (camera + detections + recommendations + live analytics)
```

## 5. Technology Stack

| Concern | Choice | Why |
|---|---|---|
| Language | Python 3.12 | matches installed interpreter |
| Env isolation | `venv` (`./venv`) | no global installs |
| Classical CV | OpenCV (`opencv-python`) | industry-standard, what Document 3–5 teach |
| Detection model | Ultralytics YOLOv8 (`ultralytics`) | modern YOLO API, ONNX/torch export, built-in val/metrics |
| Tensor backend | PyTorch | required by Ultralytics |
| Data / stats | NumPy, Pandas | dataset_report.csv, bbox_statistics.csv |
| Plots | Matplotlib / Seaborn | histograms, PR curve, confusion matrix |
| Embeddings | a frozen CNN (e.g. ResNet18/MobileNet via `torchvision`) or YOLO backbone features | Document 9 |
| Similarity search | FAISS (`faiss-cpu`) | exact/approximate nearest-neighbor over embeddings |
| Real-time camera | OpenCV `VideoCapture` + `cv2.imshow` loop | simplest correct way to prove camera→detect→display works (Document 10) |
| Dashboard | Streamlit | fastest way to combine video, detections, recommendations, and live metrics in one page (Document 11); revisit only if webcam-in-browser latency becomes a problem |

`requirements.txt` will grow one dependency at a time, added at the document where
it's first *needed* — not all at once — so it's always obvious why a package is
there.

## 6. Folder Structure (as created)

```
yolo/                                  ← project root (this VS Code workspace)
├── venv/                              ← isolated Python env, not committed
├── requirements.txt
├── docs/                              ← 00–12, one doc per learning module
├── data/                              ← derivative data we generate (NOT raw Kaggle cache)
├── notebooks/                         ← exploratory notebooks per document
├── outputs/                           ← generated reports/plots/csv artifacts
└── src/
    ├── dataset_analysis/              ← Document 1
    ├── annotation_validation/         ← Document 2
    ├── opencv_fundamentals/           ← Document 3
    ├── preprocessing/                 ← Document 4
    ├── augmentation/                  ← Document 5
    ├── yolo_training/                 ← Document 6
    ├── evaluation/                    ← Document 7
    ├── explainable_detection/         ← Document 8
    ├── recommendation/                ← Document 9
    ├── realtime_detection/            ← Document 10
    └── dashboard/                     ← Document 11
```

The raw Kaggle download stays in kagglehub's own cache
(`~/.cache/kagglehub/datasets/...`) — we never copy 700MB of raw data into the repo.
Every module takes the dataset path as a parameter; nothing hardcodes that cache
path beyond a single config constant.

## 7. Module Dependency

Strict dependency chain — each module's *input* is the previous module's *output*,
not the raw dataset again (so a bug caught in Document 2 doesn't silently vanish
when Document 6 re-reads raw files):

```
01 Dataset Analysis
   └─requires→ raw dataset only

02 Annotation Validation
   └─requires→ 01's image/label pairing logic (reused, not reimplemented)

03 OpenCV Fundamentals
   └─requires→ a handful of sample images (any from 01) — standalone teaching module

04 Image Preprocessing
   └─requires→ 03's color/resize primitives

05 Data Augmentation
   └─requires→ 04's preprocessing primitives + 02's validated annotations
                (never augment an image whose annotation failed validation)

06 YOLO Training
   └─requires→ 02 (validated annotations) + 05 (augmentation config)

07 Model Evaluation
   └─requires→ 06's trained weights + the untouched validation split from 01

08 Explainable Detection
   └─requires→ 06's weights + 04's preprocessing (same pipeline, instrumented)

09 Fashion Recommendation
   └─requires→ 06's weights (to get crops) + the full image set from 01

10 Real-Time Detection
   └─requires→ 06's weights + 08's explainability hooks (optional overlay)

11 Dashboard
   └─requires→ 09 + 10 combined
```

## 8. Coding Pattern (applies to every module, every document)

Every function in this project is implemented against the same nine-stage mental
checklist:

```
Initialize → Load → Validate → Process → Analyze → Visualize → Save → Log → Return
```

And every function must be able to answer:

| Question | Purpose |
|---|---|
| What does it receive? | Input contract |
| What does it check? | Validation |
| What does it compute? | Processing |
| What does it return? | Output contract |
| What should it save? | Artifacts |
| How should failures be handled? | Error handling |

This is enforced starting Document 1 — function stubs in `src/` carry these six
answers in the docstring *before* a single line of logic is written.

## 9. Future Scope (deferred, see Document 12 for detail)

YOLOv11, segmentation masks (SAM), CLIP-based open-vocabulary tagging, Grounding
DINO, OCR on care labels, an LLM styling assistant, multi-camera support, and
cloud deployment. None of this is in scope until Documents 00–11 are solid.

---

**Next:** Document 1 — Dataset Analysis. We already know the headline numbers
(this document). Document 1's job is to write the *code* that derives them
independently and goes deeper (per-class bbox size distributions, resolution
histograms, sample grids) — and you'll write that code yourself, with specs
provided per function.
