"""Detect tile blobs in a browser-captured Mahjong Soul frame.

Finds cream/white tile rectangles inside a rough search window and prints their
bounding boxes (full-frame coordinates) so browser-specific ROI boxes can be
anchored to measured positions instead of guessed offsets.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def detect_tiles(image: np.ndarray, window, min_area: int, max_area: int):
    x1, y1, x2, y2 = window
    roi = image[y1:y2, x1:x2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    # Cream tile faces: bright, low-to-moderate saturation.
    mask = cv2.inRange(hsv, np.array([0, 0, 140]), np.array([179, 80, 255]))
    mask = cv2.erode(mask, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    tiles = []
    for contour in contours:
        bx, by, bw, bh = cv2.boundingRect(contour)
        area = bw * bh
        if min_area <= area <= max_area and 0.5 <= bw / max(bh, 1) <= 2.0:
            tiles.append([x1 + bx, y1 + by, x1 + bx + bw, y1 + by + bh])
    tiles.sort(key=lambda b: (b[1] // 40, b[0]))
    return tiles, mask


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image")
    parser.add_argument("--window", type=int, nargs=4, required=True,
                        metavar=("X1", "Y1", "X2", "Y2"))
    parser.add_argument("--min-area", type=int, default=800)
    parser.add_argument("--max-area", type=int, default=20000)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    image = cv2.imread(args.image)
    if image is None:
        raise SystemExit(f"cannot read image: {args.image}")
    tiles, mask = detect_tiles(image, args.window, args.min_area, args.max_area)
    print(json.dumps({"window": args.window, "count": len(tiles), "tiles": tiles},
                     indent=1))
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        vis = image.copy()
        x1, y1, _, _ = args.window
        for box in tiles:
            cv2.rectangle(vis, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        cv2.imwrite(str(out), vis)
        cv2.imwrite(str(out.with_name(out.stem + "_mask.png")), mask)
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
