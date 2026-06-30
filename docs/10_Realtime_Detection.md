# 10 — Real-Time Detection

## Note on ordering

Built immediately after Document 6, ahead of Documents 7-9 (Evaluation,
Explainable Detection, Recommendation) — the explicit goal was "make the
camera work," not finish the full pipeline in numeric order. This module
only depends on Document 6's `best.pt`; it does not need Documents 7-9.

## Goal

Close the loop from `docs/00_Project_Architecture.md`'s Section 1 problem
statement: detect garments "in an image or live camera feed." Two modes,
one shared detector:

1. **Single photo** — open the camera, grab one frame, detect, save the
   annotated result. No GUI window needed; works from a plain terminal.
2. **Live video** — continuous `cv2.VideoCapture` loop, detection drawn on
   every frame, FPS overlay, `cv2.imshow` window, quit with `q`/ESC.

## Design decision: same `FrameResult` contract for both modes

`GarmentDetector.detect(frame) -> FrameResult` doesn't know or care whether
`frame` came from a single `capture_single_frame()` call or from the
`frames()` generator inside a `while` loop — same contract as
`opencv_fundamentals`'s `StepResult` pattern (Document 3). `main.py`'s
`analyze_photo()` and `run_live()` are both thin wrappers around the same
detector call.

## Functions

| Function | File | Produces |
|---|---|---|
| `resolve_weights_path()` | `config.py` | trained garment weights if present, else falls back to stock `yolov8n.pt` (COCO) so the camera plumbing is testable even before Document 6 finishes |
| `open_camera()` / `capture_single_frame()` / `frames()` | `capture.py` | a context-managed `cv2.VideoCapture`, one warmed-up frame, or a continuous frame generator |
| `GarmentDetector.detect()` | `detector.py` | a `FrameResult`: annotated frame + structured `Detection` list + inference time |
| `analyze_photo()` | `main.py` | single-shot capture → detect → save → print |
| `run_live()` | `main.py` | continuous capture → detect → FPS overlay → display loop |

## Run it

```bash
# Single photo — opens the camera, grabs one frame, detects, saves, exits.
python -m src.realtime_detection.main --mode photo

# Live camera window — press 'q' or ESC to quit.
python -m src.realtime_detection.main --mode live

# Headless smoke test (no GUI window, stops after N frames, saves the last one):
python -m src.realtime_detection.main --mode live --no-display --max-frames 30

# Lower confidence threshold (useful for an early/under-trained model):
python -m src.realtime_detection.main --mode photo --conf 0.15
```

By default both modes use `src/yolo_training`'s `best.pt` if it exists on
disk, falling back to the stock COCO `yolov8n.pt` otherwise (pass
`--weights path/to/file.pt` to override). COCO has no garment classes — it
will label a person "person," not "shirt." This fallback exists purely so
the camera/window/draw loop is independently testable before training
finishes, per the verification below.

## Outputs

```
outputs/realtime_detection/
├── captured_photo.png    — analyze_photo() output
└── live_last_frame.png   — run_live()'s last frame, when run headless/bounded
```

## Real findings from the actual run

**Camera access worked on the first try on the development machine (macOS),
no permission prompt encountered** — `cv2.VideoCapture(0)` opened the
system's default webcam and `cam.read()` returned real (non-black) frames
immediately, confirmed by saving a captured frame and visually inspecting
it (a real photo, not a gray/black placeholder). On macOS specifically,
the first run may prompt for camera permission (System Settings > Privacy
& Security > Camera) — grant it to your terminal/IDE if so; other OSes
have their own equivalent camera-permission prompts.

**Detector plumbing verified independently of training** — before
Document 6's training finished, `GarmentDetector` was run against the
captured photo using the stock `yolov8n.pt` (COCO). It correctly detected
`person` (confidence 0.79) and `cup` (confidence 0.35), with boxes and
labels drawn at the right coordinates — proof the camera → detect → draw →
save pipeline works end-to-end, independent of which weights are loaded.

**With Document 6's garment-trained weights (3-epoch model)** — a fresh
photo captured through the camera (a close, dim selfie showing mostly face
and one shoulder of a grey shirt) produced **zero detections at the
default 0.25 confidence threshold**. Lowering `--conf` to 0.15 revealed a
real but weak signal: a box roughly over the visible shirt region, labeled
`vest_dress` at confidence 0.18 — the right *location*, wrong *class*, low
confidence. This is an honest, expected result, not a bug: the training
images are clean, well-lit, full-garment e-commerce/social photos; a
close-up dim selfie showing a sliver of fabric is out-of-distribution for
a model trained 3 epochs on that data. Verified the pipeline itself is
correct, not just hopeful — the same `GarmentDetector` correctly drew
`person`/`cup` boxes earlier using stock COCO weights on the same kind of
photo (see above), so the weak/wrong result here is the *model's* current
capability, not a camera, drawing, or inference-wiring problem.

**Practical takeaway, added as a CLI option:** `--conf` was added to
`main.py` (`python -m src.realtime_detection.main --mode photo --conf 0.15`)
specifically because of this finding — the default 0.25 threshold is tuned
for a converged model, not a deliberate 3-epoch proof-of-pipeline run.

## What's next

Documents 7 (formal Evaluation), 8 (Explainable Detection — instrumenting
this same pipeline to show *why* a box was predicted), and 9
(Recommendation) remain deferred. `run_live()`'s `FrameResult.detections`
already carries everything Document 9 would need (class, confidence, box)
to crop a detected garment for embedding/similarity search.
