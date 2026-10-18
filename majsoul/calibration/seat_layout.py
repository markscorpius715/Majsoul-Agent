"""Match-level tracking for the occupied Mahjong table screen slots."""

from __future__ import annotations

from dataclasses import dataclass, field

from .roi_config import ROIConfig


@dataclass
class SeatLayoutTracker:
    """Lock a stable slot layout for one match and retain it across hands."""

    config: ROIConfig
    confirmations_required: int = 2
    locked_layout: tuple[str, ...] | None = field(default=None, init=False)
    _pending_layout: tuple[str, ...] | None = field(default=None, init=False, repr=False)
    _pending_count: int = field(default=0, init=False, repr=False)

    def __post_init__(self):
        self.config.validate()
        if type(self.confirmations_required) is not int or self.confirmations_required < 2:
            raise ValueError("confirmations_required must be an integer >= 2")

    def observe(self, active_slots) -> tuple[str, ...] | None:
        """Accept one reliable occupancy observation and lock after repetition."""
        candidate = tuple(self.config.layout_for_active_slots(active_slots))
        if self.locked_layout is not None:
            if frozenset(candidate) != frozenset(self.locked_layout):
                raise ValueError("Seat layout cannot change within one match")
            return self.locked_layout
        if candidate == self._pending_layout:
            self._pending_count += 1
        else:
            self._pending_layout = candidate
            self._pending_count = 1
        if self._pending_count >= self.confirmations_required:
            self.locked_layout = candidate
        return self.locked_layout

    def active_region_names(self) -> list[str]:
        if self.locked_layout is None:
            raise ValueError("Seat layout has not been locked for this match")
        return self.config.active_region_names(self.locked_layout)

    @property
    def pending_layout(self) -> tuple[str, ...] | None:
        """Return the current uncommitted candidate for diagnostics."""
        return self._pending_layout

    @property
    def pending_count(self) -> int:
        """Return how many consecutive reliable observations support the candidate."""
        return self._pending_count

    def observe_unknown(self) -> tuple[str, ...] | None:
        """Break an unlocked confirmation streak after missing or ambiguous evidence."""
        if self.locked_layout is None:
            self._pending_layout = None
            self._pending_count = 0
        return self.locked_layout

    def reset_match(self) -> None:
        """Clear the lock only when a new match starts or explicit resync occurs."""
        self.locked_layout = None
        self._pending_layout = None
        self._pending_count = 0
