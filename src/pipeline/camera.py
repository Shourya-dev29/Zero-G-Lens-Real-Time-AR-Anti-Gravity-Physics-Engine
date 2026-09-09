"""
CameraCapture: Thread-safe webcam frame producer using OpenCV.
Supports configurable resolution and FPS targeting.
"""
import cv2
import time
from typing import Generator
import numpy as np


class CameraCapture:
    def __init__(self, device_id: int = 0, resolution: tuple = (1280, 720), fps: int = 30):
        self.device_id  = device_id
        self.resolution = resolution
        self.fps        = fps
        self.cap        = None
        self._open()

    def _open(self):
        self.cap = cv2.VideoCapture(self.device_id, cv2.CAP_DSHOW)  # CAP_DSHOW for Windows
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera device {self.device_id}")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        self.cap.set(cv2.CAP_PROP_FPS,          self.fps)
        # Minimize internal buffer to reduce latency
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_f = self.cap.get(cv2.CAP_PROP_FPS)
        print(f"[CAM] Opened device {self.device_id}: {actual_w}×{actual_h} @ {actual_f:.0f} FPS")

    def stream(self) -> Generator[np.ndarray, None, None]:
        """
        Generator that yields BGR frames as fast as the camera allows.
        Caller is responsible for throttling if needed.
        """
        target_interval = 1.0 / self.fps
        last_time       = time.perf_counter()

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                print("[CAM] Frame grab failed — retrying...")
                time.sleep(0.01)
                continue

            # Soft FPS cap to avoid overwhelming downstream threads
            now     = time.perf_counter()
            elapsed = now - last_time
            if elapsed < target_interval:
                time.sleep(target_interval - elapsed)
            last_time = time.perf_counter()

            yield frame

    def release(self):
        if self.cap:
            self.cap.release()
            print("[CAM] Camera released.")
