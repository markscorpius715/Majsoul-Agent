from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class ContinualConfig:
    persistent_frames: int = 4
    dedup_hamming_distance: int = 6
    minimum_confidence: float = 0.70
    minimum_margin: float = 0.04
    novelty_distance: float = 0.30
    training_batch_size: int = 50
    crop_dedup_hamming_distance: int = 1
    crop_dedup_mean_abs_difference: float = 0.025
    crop_minimum_frame_tiles: int = 5
    crop_checkpoint_frames: int = 100

    def __post_init__(self):
        if not 3 <= self.persistent_frames <= 5:
            raise ValueError("persistent_frames must be between 3 and 5")
        if not 0 <= self.dedup_hamming_distance <= 64:
            raise ValueError("dedup_hamming_distance must be between 0 and 64")
        for name in ("minimum_confidence", "minimum_margin", "novelty_distance"):
            value = getattr(self, name)
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.training_batch_size < 2:
            raise ValueError("training_batch_size must be at least two")
        if not 0 <= self.crop_dedup_hamming_distance <= 64:
            raise ValueError("crop_dedup_hamming_distance must be between 0 and 64")
        if not 0 <= self.crop_dedup_mean_abs_difference <= 1:
            raise ValueError(
                "crop_dedup_mean_abs_difference must be between 0 and 1")
        if self.crop_minimum_frame_tiles < 1:
            raise ValueError("crop_minimum_frame_tiles must be positive")
        if self.crop_checkpoint_frames < 1:
            raise ValueError("crop_checkpoint_frames must be positive")

    @classmethod
    def load(cls, path: str | Path) -> "ContinualConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        if data.get("schema_version") != 1:
            raise ValueError("Unsupported continual-learning config schema")
        values = {key: value for key, value in data.items() if key != "schema_version"}
        return cls(**values)
