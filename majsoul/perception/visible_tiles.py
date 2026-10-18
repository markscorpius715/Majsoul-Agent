from __future__ import annotations

from dataclasses import replace
import numpy as np

from ..calibration.roi_config import ROIConfig
from .config import VisionConfig
from .hand_reader import HandReader
from .meld_reader import MeldReader
from .region_reader import TileRegionReader
from .river_reader import RiverReader
from .roi_extractor import extract_roi
from .schemas import MeldReading, TileSequenceReading, VisibleTilesReading
from .tile_classifier import TileTemplateClassifier


class VisibleTilesReader:
    """Read all visible tiles for an already locked match-level seat layout."""

    def __init__(self, config: VisionConfig, classifier: TileTemplateClassifier):
        self.config = config
        self.classifier = classifier

    def read(self, image: np.ndarray, roi: ROIConfig, active_slots, *,
             occlusion_boxes=()) -> VisibleTilesReading:
        slots = tuple(roi.layout_for_active_slots(active_slots))

        def read_region(name, reader, rotation=None, *, meld=False):
            crop = extract_roi(image, roi, name)
            covered = any(min(crop.box.x2, box.x2) > max(crop.box.x1, box.x1)
                          and min(crop.box.y2, box.y2) > max(crop.box.y1, box.y1)
                          for box in occlusion_boxes)
            if covered:
                if name.startswith(("river_", "kita_")):
                    reading = reader.read(crop, rotation or 0, excluded_boxes=occlusion_boxes)
                    return replace(reading, issues=tuple(dict.fromkeys(reading.issues +
                                   ("UI occlusion; region unavailable",))))
                # No group identity/openness is inferred from a truncated meld.
                # Empty means unavailable, not an established empty region.
                reading_type = MeldReading if meld else TileSequenceReading
                return reading_type(name, (), issues=("UI occlusion; region unavailable",))
            return reader.read(crop) if rotation is None else reader.read(crop, rotation)

        hand = read_region("self_hand", HandReader(self.config, self.classifier))
        river_reader = RiverReader(self.config, self.classifier)
        rivers = {
            slot: read_region(f"river_{slot}", river_reader,
                self.config.rotation_for_slot(slot))
            for slot in slots
        }
        melds = {
            slot: read_region(f"melds_{slot}",
                             MeldReader(self.config, self.classifier, slot), meld=True)
            for slot in slots
        }
        dora = read_region("dora_indicators",
                          TileRegionReader(self.config, self.classifier, self.config.dora))
        kita = {}
        if roi.kita:
            for slot in slots:
                kita[slot] = read_region(f"kita_{slot}", TileRegionReader(
                    self.config, self.classifier,
                    self.config.kita_for_slot(slot), "4z"),
                    self.config.rotation_for_slot(slot))
        return VisibleTilesReading(slots, hand, rivers, melds, dora, kita,
                                   tuple(occlusion_boxes))
