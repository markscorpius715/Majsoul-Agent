"""Shared deadline model for one operation window.

The on-screen countdown is read once (the first confirmed reading) and then
counted down locally on the monotonic clock. Later readings may only make the
deadline earlier, and unreadable clock frames are not failures. Without any
reading the window is assumed to hold only the guaranteed base time, counted
from the earliest moment the window could have started. The supervisor and the
Agent use the same model, so the Agent can always finish before a takeover.
"""
from dataclasses import dataclass, field
import math


def clock_total(value):
    """Return a usable displayed total in seconds, or None when unusable."""
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        return None
    return value


@dataclass
class OperationClock:
    base_seconds: float
    cap_seconds: float
    margin_seconds: float
    started_at: float | None = field(default=None, init=False)
    earliest_start: float | None = field(default=None, init=False)
    clock_due: float | None = field(default=None, init=False)
    readings: int = field(default=0, init=False)
    last_reading: tuple | None = field(default=None, init=False)

    @classmethod
    def from_protection(cls, config, *, extra_margin=0.):
        """The supervisor's model; ``extra_margin`` ends the Agent's own budget
        that much earlier so its input lands before any takeover."""
        return cls(base_seconds=config.shortest_turn_seconds,
                   cap_seconds=config.maximum_operation_seconds-extra_margin,
                   margin_seconds=(config.quantization_seconds+config.takeover_margin_seconds
                                   +extra_margin))

    @classmethod
    def for_agent(cls, emergency, *, action_margin):
        """The Agent's decision deadline: ``action_margin`` before its own
        tsumogiri rung (see ``EmergencyConfig``) would fire."""
        return cls(base_seconds=emergency.base_seconds,
                   cap_seconds=emergency.maximum_turn_seconds-action_margin,
                   margin_seconds=emergency.quantization_seconds+emergency.lead_seconds+action_margin)

    def reset(self):
        self.started_at = self.earliest_start = self.clock_due = self.last_reading = None
        self.readings = 0

    @property
    def open(self):
        return self.started_at is not None

    @property
    def confirmed(self):
        return self.clock_due is not None

    def start(self, at, earliest=None):
        """Open the window at the first actionable observation ``at``.

        ``earliest`` is the latest moment the window was proven closed (for
        example the last opponent-turn frame); the base-time fallback counts
        from there because an observation gap can hide the real start.
        """
        if self.started_at is not None:
            return
        self.started_at = at
        self.earliest_start = at if earliest is None or earliest > at else earliest

    def observe(self, captured_at, total):
        """Apply one clock reading; unknown readings are ignored."""
        if self.started_at is None:
            self.start(captured_at)
        total = clock_total(total)
        if total is None:
            return False
        due = captured_at+total-self.margin_seconds
        self.clock_due = due if self.clock_due is None else min(self.clock_due, due)
        self.readings += 1
        self.last_reading = (captured_at, total)
        return True

    @property
    def fallback_due(self):
        return None if self.earliest_start is None else self.earliest_start+self.base_seconds-self.margin_seconds

    @property
    def due(self):
        if self.started_at is None:
            return None
        cap = self.started_at+self.cap_seconds
        return min(cap, self.clock_due if self.clock_due is not None else self.fallback_due)

    def to_dict(self):
        return {'started_at': self.started_at, 'earliest_start': self.earliest_start,
                'source': 'confirmed_clock' if self.confirmed else 'base_time_fallback',
                'clock_due': self.clock_due, 'due': self.due, 'readings': self.readings}
