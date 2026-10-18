from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from ..domain import Box
from ..calibration.roi_config import ROIConfig


@dataclass(eq=False)
class ROICrop:
    name: str
    box: Box
    image: np.ndarray = field(repr=False)
    rotation_degrees: int = 0

    def map_box_to_screen(self, local: Box) -> Box:
        width = int(round(self.box.x2 - self.box.x1))
        height = int(round(self.box.y2 - self.box.y1))
        if self.rotation_degrees == 0:
            x1, y1, x2, y2 = local.x1, local.y1, local.x2, local.y2
        elif self.rotation_degrees == 90:
            x1, y1, x2, y2 = local.y1, height - local.x2, local.y2, height - local.x1
        elif self.rotation_degrees == -90:
            x1, y1, x2, y2 = width - local.y2, local.x1, width - local.y1, local.x2
        elif self.rotation_degrees == 180:
            x1, y1, x2, y2 = (width - local.x2, height - local.y2,
                              width - local.x1, height - local.y1)
        else:
            raise ValueError("Only right-angle ROI rotations are supported")
        return Box(self.box.x1 + x1, self.box.y1 + y1,
                   self.box.x1 + x2, self.box.y1 + y2)


def rotate_roi(crop: ROICrop, degrees: int) -> ROICrop:
    if degrees == 0:
        return crop
    if degrees == 90:
        image = cv2.rotate(crop.image, cv2.ROTATE_90_CLOCKWISE)
    elif degrees == -90:
        image = cv2.rotate(crop.image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif degrees == 180:
        image = cv2.rotate(crop.image, cv2.ROTATE_180)
    else:
        raise ValueError("Only right-angle ROI rotations are supported")
    return ROICrop(crop.name, crop.box, image, degrees)


def extract_roi(image: np.ndarray, config: ROIConfig, name: str) -> ROICrop:
    if not isinstance(image, np.ndarray) or image.ndim not in (2, 3) or image.size == 0:
        raise ValueError("image must be a non-empty grayscale or color array")
    height, width = image.shape[:2]
    boxes = config.pixel_boxes((width, height))
    if name not in config.regions:
        raise KeyError(f"Unknown ROI: {name}")
    if name not in boxes:
        raise ValueError(f"ROI is uncalibrated: {name}")
    x1, y1, x2, y2 = boxes[name]
    pixels = image[y1:y2, x1:x2].copy()
    if name in config.region_polygons:
        points = np.array([[round(x*width)-x1, round(y*height)-y1]
                           for x, y in config.region_polygons[name]], dtype=np.int32)
        mask = np.zeros(pixels.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [points], 255)
        pixels[mask == 0] = 0
    return ROICrop(name, Box(x1, y1, x2, y2), pixels)


def outside_occlusions(tiles, boxes):
    """Reject UI pixels and candidates touching a screen-space UI boundary.

    This runs before classification and keeps unaffected physical tile evidence.
    It does not prove that there are no hidden tiles behind the excluded boxes.
    """
    return tuple(tile for tile in tiles if not any(
        min(tile.box.x2, box.x2) >= max(tile.box.x1, box.x1) and
        min(tile.box.y2, box.y2) >= max(tile.box.y1, box.y1)
        for box in boxes))
