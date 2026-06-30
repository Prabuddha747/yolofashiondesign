# 06 — YOLO Training

## Note on ordering

Built directly after Document 3, skipping Documents 2 (Annotation
Validation), 4 (Preprocessing), and 5 (Augmentation) — by explicit request
to get a trained model and a working real-time camera demo functioning
end-to-end, rather than build every intermediate pedagogical module first.
This is a scope decision, not an oversight: Document 1 already ran a full
integrity scan with zero malformed annotations (see
`docs/00_Project_Architecture.md` Section 3), and Ultralytics' own trainer
performs resize/letterbox/mosaic/flip augmentation internally during
`model.train()` — so training directly against the verified-clean dataset,
with Ultralytics handling preprocessing/augmentation, produces a real
working model without those documents being load-bearing for *this* step.
They remain valuable follow-up work (see "What's next").

## Goal

Train a YOLOv8 object detector on the 13-class garment taxonomy
(`docs/00_Project_Architecture.md` Section 3) so that, for the first time in
this project, a model — not a human label, not a classical-CV contour —
can look at a photo and propose "this is a `short_sleeve_top`, with this
confidence, in this box."

## Hardware reality — calibrate on your own machine, don't assume

`get_device()` (`src/yolo_training/config.py`) auto-picks CUDA > MPS > CPU,
so this runs on an NVIDIA GPU, an Apple Silicon Mac, or CPU-only without
code changes. But **batch size is hardware-specific and worth measuring,
not guessing** — on a dedicated GPU with plenty of VRAM, bigger batches are
usually faster; on a memory-constrained machine with no dedicated GPU,
they can be *slower*, because the bottleneck shifts from compute to memory
swapping. Before committing to a multi-hour run, time 1 epoch at a few
batch sizes against a small slice of your data (`fraction=0.05`,
`val=False`) and pick whichever is actually fastest on your hardware.

**Example from the machine this was developed on** (a laptop with 8GB RAM
and no dedicated GPU — MPS-only acceleration): three calibration runs (1
epoch, no per-epoch validation, 5% of the training data) found smaller
batches were *faster*, the opposite of typical GPU scaling:

| Batch size | Images | Wall time (train step only) | Throughput |
|---|---|---|---|
| 16 | 500 | 349.2s | 1.43 img/s |
| 8 | 500 | 180.0s | 2.78 img/s |
| 4 | 200 | 32.8s | 6.10 img/s |

This is consistent with memory pressure: at imgsz=640 with mosaic
augmentation, batch=16 needed enough RAM headroom that the machine started
swapping, and swapping dominated wall time far more than the lost
parallelism from a smaller batch. `batch=4` was chosen for that machine's
real run on this evidence — **your numbers will differ; run your own
calibration and use it.**

## Hyperparameters used (real run, example machine above)

| Param | Value | Why |
|---|---|---|
| Base checkpoint | `yolov8n.pt` | smallest Ultralytics checkpoint — a safe default with no dedicated GPU; bump up if your hardware allows |
| `imgsz` | 640 | matches the dataset's most common native resolutions and Document 00's stated 640×640 target |
| `batch` | 4 | fastest measured throughput on the example machine (table above) — calibrate your own |
| `epochs` | 3 | time-budget decision — a full ~28.5 min/epoch (train+val) made 50 epochs (~24h) impractical on that machine; 3 epochs proves the full pipeline with real (not state-of-the-art) detections. A machine with a dedicated GPU can likely afford far more epochs in the same wall time |
| `device` | auto-selected | `get_device()` picked `mps` on the example machine; will pick `cuda` automatically if you have an NVIDIA GPU |
| `val` | `True` | per-epoch validation kept on despite its ~1.5 min/epoch cost, so `best.pt` selection is based on real validation fitness, not just the last epoch |

## Functions

| Function | File | Produces |
|---|---|---|
| `get_device()` | `config.py` | `"mps"` or `"cpu"` |
| `build_data_yaml()` | `dataset_yaml.py` | `outputs/yolo_training/data.yaml` pointing at the existing kagglehub cache — **no images copied** |
| `train_model()` | `train.py` | `TrainingResult` (best weights path, epochs run, train time, final metrics) |
| `evaluate_model()` | `evaluate.py` | `EvaluationResult` (mAP50, mAP50-95, per-class mAP50, precision, recall) |
| `main.run()` | `main.py` | orchestrates all three, prints a summary |

## Run it

```bash
python -m src.yolo_training.main                                    # full run, current config.py defaults
python -m src.yolo_training.main --epochs 1 --batch 4 --fraction 0.02  # smoke test, ~4 min
```

## Outputs

```
outputs/yolo_training/
├── data.yaml             — generated, points at kagglehub cache in place
├── best.pt               — stable copy of the best checkpoint (any run)
└── runs/
    └── train/            — Ultralytics' own run dir: weights/, curves, confusion matrix, args.yaml
```

## Real findings from the actual run

**Wall-clock vs. compute time diverged sharply.** Ultralytics' own log
reported "3 epochs completed in 5.207 hours," but the `time.perf_counter()`
measurement wrapping the same `model.train()` call in `train.py` recorded
only **4150.8s (~69 minutes)**. The machine went to sleep partway through
the unattended background run; `perf_counter()` (a monotonic clock that
doesn't advance during system sleep) captured only the actual compute time,
while Ultralytics' wall-clock timer counted the sleep duration too. Lesson:
for unattended long-running training, either disable sleep (`caffeinate`)
or expect the wall-clock estimate to be unreliable — the 69-minute compute
figure matches the calibration-based prediction (3 × ~28.5 min/epoch ≈ 85
min) far better than 5.2 hours does.

**Real metrics, 3 epochs, full 10,000-image train set:**

| Metric | Value |
|---|---|
| mAP50 | 0.454 |
| mAP50-95 | 0.351 |
| Precision | 0.626 |
| Recall | 0.464 |

**Per-class mAP50 — the 134:1 class imbalance from Document 1 shows up
immediately, exactly as predicted:**

| Class | mAP50 | Train instances |
|---|---|---|
| trousers | 0.858 | 2,807 |
| short_sleeve_top | 0.825 | 3,755 |
| skirt | 0.691 | 1,682 |
| shorts | 0.677 | 1,889 |
| long_sleeve_top | 0.592 | 1,830 |
| vest_dress | 0.521 | 912 |
| long_sleeve_outwear | 0.491 | 675 |
| short_sleeve_dress | 0.452 | 912 |
| vest | 0.362 | 831 |
| long_sleeve_dress | 0.285 | 429 |
| sling_dress | 0.094 | 313 |
| sling | 0.044 | 92 |
| short_sleeve_outwear | 0.015 | 28 |

The two highest-instance-count classes (`short_sleeve_top`,
`trousers`) land in the top three by mAP50; the lowest-instance-count class
(`short_sleeve_outwear`, 28 examples) scores the worst by a wide margin
(0.015). This is the concrete number behind Document 00's warning that "a
model can look good on mAP while being useless on a rare class" — overall
mAP50 (0.454) looks reasonable, but a user trying to detect outwear
specifically would see it almost never fire.

**Loss curves were still descending, not plateaued, after 3 epochs**
(`outputs/yolo_training/runs/train/results.png`) — box/cls/dfl loss on both
train and val dropped every single epoch with no sign of flattening,
meaning more epochs would very likely keep improving mAP. The 3-epoch
budget was a deliberate "prove the pipeline" choice, not a claim that the
model has converged.

**Qualitative check on `val_batch0_pred.jpg`** (Ultralytics' own
predictions-vs-image grid): correctly localized boxes on plausible garment
regions across all 16 sampled validation images, with sensible labels
(`shorts`, `trousers`, `skirt`, `short_sleeve_top`, `vest_dress`) at
varying confidence (0.3-1.0) — including some visibly correct
high-confidence calls (`short_sleeve_top 1.0`, `shorts 1.0`).

## What's next

Document 2 (Annotation Validation), 4 (Preprocessing), 5 (Augmentation), and
7 (formal Model Evaluation — PR curves, confusion matrix walkthrough) remain
valuable, deferred work: they would let a *future*, longer training run
control variables this quick run didn't (stratified train/val split,
explicit augmentation policy for the 134:1 class imbalance, a held-out test
set). Document 10 — Real-Time Detection — consumes this document's
`best.pt` directly and does not wait for them.
