"""
Document 10 orchestrator — two ways to run the camera -> YOLO -> boxes
pipeline:

  python -m src.realtime_detection.main --mode photo
      Opens the camera, grabs one frame, detects, saves an annotated image,
      prints what it found, exits. No GUI window required.

  python -m src.realtime_detection.main --mode live
      Opens a continuous cv2.imshow window with live detections + FPS
      overlay. Press 'q' or ESC to quit. Pass --max-frames to auto-stop
      after N frames (useful for headless/automated smoke tests) and
      --no-display to skip the window entirely (saves the last frame
      instead, for environments with no display attached).
"""

import argparse
import time

import cv2

from . import config
from .capture import capture_single_frame, frames
from .detector import GarmentDetector


def analyze_photo(weights_path: str = None, camera_index: int = config.CAMERA_INDEX, output_path=None,
                   confidence: float = config.CONFIDENCE_THRESHOLD):
    weights_path = config.resolve_weights_path(weights_path)
    detector = GarmentDetector(weights_path, confidence_threshold=confidence)

    frame = capture_single_frame(camera_index)
    result = detector.detect(frame)

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = output_path or config.OUTPUT_DIR / "captured_photo.png"
    cv2.imwrite(str(output_path), result.annotated_frame)

    print(f"weights: {weights_path}")
    print(f"inference: {result.inference_ms:.1f} ms | detections: {len(result.detections)}")
    for det in result.detections:
        print(f"  {det.class_name:24s} conf={det.confidence:.2f} box={det.box_xyxy}")
    print(f"Saved annotated photo to: {output_path}")
    return result


def run_live(weights_path: str = None, camera_index: int = config.CAMERA_INDEX,
             max_frames: int = None, display: bool = True,
             confidence: float = config.CONFIDENCE_THRESHOLD):
    weights_path = config.resolve_weights_path(weights_path)
    detector = GarmentDetector(weights_path, confidence_threshold=confidence)
    print(f"weights: {weights_path}")
    print("Press 'q' or ESC in the window to quit." if display else f"Headless mode, stopping after {max_frames} frames.")

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame_count = 0
    last_tick = time.perf_counter()
    last_result = None

    for frame in frames(camera_index):
        result = detector.detect(frame)
        last_result = result
        frame_count += 1

        now = time.perf_counter()
        fps = 1.0 / max(now - last_tick, 1e-6)
        last_tick = now
        cv2.putText(result.annotated_frame, f"FPS: {fps:.1f}", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if display:
            cv2.imshow(config.WINDOW_NAME, result.annotated_frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

        if max_frames is not None and frame_count >= max_frames:
            break

    if display:
        cv2.destroyAllWindows()
    if last_result is not None and (not display or max_frames is not None):
        output_path = config.OUTPUT_DIR / "live_last_frame.png"
        cv2.imwrite(str(output_path), last_result.annotated_frame)
        print(f"Processed {frame_count} frames. Saved last frame to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["photo", "live"], default="photo")
    parser.add_argument("--weights", default=None)
    parser.add_argument("--camera", type=int, default=config.CAMERA_INDEX)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--conf", type=float, default=config.CONFIDENCE_THRESHOLD)
    args = parser.parse_args()

    if args.mode == "photo":
        analyze_photo(weights_path=args.weights, camera_index=args.camera, confidence=args.conf)
    else:
        run_live(weights_path=args.weights, camera_index=args.camera,
                  max_frames=args.max_frames, display=not args.no_display, confidence=args.conf)
