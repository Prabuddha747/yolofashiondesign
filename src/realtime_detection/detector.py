"""
Wraps an Ultralytics YOLO checkpoint behind one method: detect(frame) -> FrameResult.

Receives: a weights path (any .pt Ultralytics can load — COCO-pretrained or
this project's garment-trained weights) and a confidence threshold.
Checks: nothing extra — Ultralytics raises on an unreadable checkpoint.
Computes: per frame, runs model.predict(), times it, draws each kept box +
label + confidence onto a copy of the frame.
Returns: a FrameResult (annotated frame + structured Detection list +
inference time in ms) — same contract whether called once (single photo) or
in a tight loop (live video), so main.py doesn't need two code paths.
Saves: nothing — caller decides whether/where to persist the annotated frame.
Failures: lets Ultralytics' own exceptions propagate.
"""

import time

import cv2
from ultralytics import YOLO

from .schema import Detection, FrameResult

BOX_COLOR = (0, 200, 0)


class GarmentDetector:
    def __init__(self, weights_path: str, confidence_threshold: float = 0.25):
        self.model = YOLO(weights_path)
        self.confidence_threshold = confidence_threshold

    def detect(self, frame) -> FrameResult:
        start = time.perf_counter()
        results = self.model.predict(frame, conf=self.confidence_threshold, verbose=False)[0]
        inference_ms = (time.perf_counter() - start) * 1000

        annotated = frame.copy()
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = results.names[class_id]
            detections.append(Detection(class_id, class_name, confidence, (x1, y1, x2, y2)))

            cv2.rectangle(annotated, (x1, y1), (x2, y2), BOX_COLOR, 2)
            label = f"{class_name} {confidence:.2f}"
            cv2.putText(annotated, label, (x1, max(y1 - 8, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, BOX_COLOR, 2)

        return FrameResult(annotated_frame=annotated, detections=detections, inference_ms=inference_ms)
