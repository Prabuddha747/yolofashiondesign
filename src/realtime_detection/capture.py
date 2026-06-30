"""
Thin wrapper around cv2.VideoCapture — the only file in this module that
knows a camera index is a webcam and not, say, a video file.

Receives: a camera index (0 = default system camera) and how many warm-up
frames to discard before trusting a frame (the first frames out of most
webcams are under-exposed while auto-exposure/white-balance settle).
Checks: the camera actually opened, and each grabbed frame actually decoded.
Computes: nothing — pure I/O.
Returns: a single BGR np.ndarray (capture_single_frame) or yields a
continuous stream of them (frames).
Saves: nothing.
Failures: raises RuntimeError with the camera index in the message — a
silent black frame is worse than a loud crash when a learner is debugging
"why does my camera demo show nothing."
"""

from contextlib import contextmanager

import cv2


@contextmanager
def open_camera(camera_index: int):
    cam = cv2.VideoCapture(camera_index)
    if not cam.isOpened():
        raise RuntimeError(
            f"Could not open camera index {camera_index}. On macOS, check "
            f"System Settings > Privacy & Security > Camera and grant access "
            f"to the terminal/app running this process."
        )
    try:
        yield cam
    finally:
        cam.release()


def capture_single_frame(camera_index: int, warmup_frames: int = 5):
    with open_camera(camera_index) as cam:
        frame = None
        for _ in range(warmup_frames + 1):
            ok, frame = cam.read()
            if not ok:
                raise RuntimeError(f"Failed to read a frame from camera index {camera_index}.")
        return frame


def frames(camera_index: int):
    with open_camera(camera_index) as cam:
        while True:
            ok, frame = cam.read()
            if not ok:
                raise RuntimeError(f"Failed to read a frame from camera index {camera_index}.")
            yield frame
