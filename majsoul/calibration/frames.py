import math
from pathlib import Path

import cv2
import numpy as np


def write_image(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise ValueError(f"Cannot encode image: {path}")
    encoded.tofile(str(path))


def read_image(path):
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode image: {path}")
    return image


def extract_frame(video, frame_index):
    if type(frame_index) is not int or frame_index < 0:
        raise ValueError("frame_index must be a nonnegative integer")
    capture = cv2.VideoCapture(str(video))
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open video: {video}")
        fps = capture.get(cv2.CAP_PROP_FPS)
        # Sequential decoding avoids codec/keyframe seek ambiguity for the reference.
        for _ in range(frame_index + 1):
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"Video ended before frame {frame_index}")
        return frame, {"source": str(Path(video).resolve()), "frame_index": frame_index,
                       "seconds": frame_index / fps if math.isfinite(fps) and fps > 0 else None,
                       "resolution": [frame.shape[1], frame.shape[0]]}
    finally:
        capture.release()
