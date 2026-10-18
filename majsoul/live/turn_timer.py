"""The client's 7+20 time control, as the Agent models it.

* Every turn starts with ``base_seconds`` (7) of base time.
* Time beyond the base is taken from a reserve (20 s) shared by all turns of
  the current hand; the client restores it when a new hand starts.
* Once the reserve is spent, every later turn of the hand gets only
  ``timeout_turn_seconds`` (5) and no reserve.

The on-screen countdown is read once per turn and counted down locally; a
confirmed reading is the authority for both the base still left and the
reserve still left. Without a reading this hand, the reserve is assumed spent
and the turn gets the time-out base only (conservative). Deadlines inside a
turn are strictly ordered: model < local efficiency < tsumogiri < client.
"""
from dataclasses import dataclass, field
import math


def _seconds(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        return None
    return value


@dataclass
class MatchTimer:
    """Reserve and time-out state for the current hand."""
    base_seconds: float = 7.0
    reserve_seconds: float = 20.0
    timeout_turn_seconds: float = 5.0
    reserve_left: float | None = field(default=None, init=False)
    timeout: bool = field(default=False, init=False)
    hand: object = field(default=None, init=False)

    def new_hand(self, hand=None):
        """A new hand restores the full reserve."""
        self.hand, self.reserve_left, self.timeout = hand, float(self.reserve_seconds), False

    def unknown_hand(self, hand=None):
        """Joined mid-hand: the reserve left is unknown until a reading."""
        self.hand, self.reserve_left, self.timeout = hand, None, False

    def observe(self, countdown):
        """Take the reserve from a confirmed on-screen reading."""
        countdown = countdown or {}
        if countdown.get('status') != 'complete':
            return
        mode = countdown.get('mode')
        reserve = _seconds(countdown.get('reserve_seconds'))
        if mode in ('base_plus_reserve', 'reserve_only') and reserve is not None:
            self.reserve_left = min(float(reserve), float(self.reserve_seconds))
            self.timeout = False
        elif mode == 'base_only':
            # No reserve digits are shown once the reserve is spent.
            self.reserve_left, self.timeout = 0.0, True

    def spend(self, seconds):
        """Local accounting when a turn used time beyond its base."""
        if self.reserve_left is None or not seconds or seconds <= 0:
            return
        self.reserve_left = max(0.0, self.reserve_left-seconds)
        if self.reserve_left == 0:
            self.timeout = True

    @property
    def turn_base(self):
        return self.timeout_turn_seconds if self.timeout else self.base_seconds

    @property
    def turn_reserve(self):
        return 0.0 if self.timeout or self.reserve_left is None else self.reserve_left

    def to_dict(self):
        return {'reserve_left': self.reserve_left, 'timeout': self.timeout,
                'turn_base': self.turn_base, 'hand': self.hand}


@dataclass
class TurnTiming:
    quantization_seconds: float = 1.0   # the integer display may round up
    lead_seconds: float = 1.5           # tsumogiri fires this long before its turn end
    click_seconds: float = 0.5          # never start input with less than this left
    reserve_per_turn_seconds: float = 2.0  # reserve an ordinary turn may spend
    maximum_turn_seconds: float = 10.0  # absolute cap after the turn opened
    minimum_turn_seconds: float = 5.0   # time kept from first sight when base was lost to lag


class TurnClock:
    """Deadlines of one operation window, derived from the hand's timer.

    ``hard_due``: the last moment before the client acts (base + reserve left).
    ``ordinary_due``: an ordinary turn spends at most ``reserve_per_turn`` of
    the shared reserve. ``commit_at``: the tsumogiri rung fires. ``decision_due``:
    model and local efficiency must finish ``action_margin`` before that.
    ``selection_due``: a riichi tile selection may use the whole reserve.
    """
    def __init__(self, timer, timing=None, *, action_margin=0.0):
        self.timer = timer
        self.timing = timing or TurnTiming()
        self.action_margin = action_margin
        self.reset()

    def reset(self):
        self.started_at = self.earliest_start = None
        self.base = self.reserve = None
        self.base_end = self.hard_end = None
        self.readings = 0
        self.last_reading = None

    @property
    def open(self):
        return self.started_at is not None

    @property
    def confirmed(self):
        return self.readings > 0

    def start(self, at, earliest=None):
        if self.started_at is not None:
            return
        self.started_at = at
        self.earliest_start = at if earliest is None or earliest > at else earliest
        self.base, self.reserve = self.timer.turn_base, self.timer.turn_reserve

    def observe(self, captured_at, countdown):
        """Apply a reading; unknown readings are ignored. Only shortens."""
        if self.started_at is None:
            self.start(captured_at)
        countdown = countdown or {}
        if countdown.get('status') != 'complete':
            return False
        base = _seconds(countdown.get('base_seconds'))
        reserve = _seconds(countdown.get('reserve_seconds'))
        total = _seconds(countdown.get('total_seconds'))
        if base is None and reserve is None:
            if total is None:
                return False
            base, reserve = min(total, self.timer.turn_base), max(0.0, total-self.timer.turn_base)
        base, reserve = base or 0.0, reserve or 0.0
        self.timer.observe(countdown)
        base_end, hard_end = captured_at+base, captured_at+base+reserve
        self.base_end = base_end if self.base_end is None else min(self.base_end, base_end)
        self.hard_end = hard_end if self.hard_end is None else min(self.hard_end, hard_end)
        self.readings += 1
        self.last_reading = (captured_at, base+reserve)
        return True

    def close(self, at):
        """Charge the reserve with time this turn used beyond its base."""
        if self.started_at is not None and at is not None:
            # A reading already updated the reserve up to its base end.
            base_end = self.base_end if self.base_end is not None else self.earliest_start+self.base
            self.timer.spend(at-base_end)
        self.reset()

    def _q(self, value):
        return None if value is None else value-self.timing.quantization_seconds

    @property
    def base_due(self):
        if self.started_at is None:
            return None
        end = self.base_end if self.base_end is not None else self.earliest_start+self.base
        return self._q(end)

    @property
    def hard_due(self):
        if self.started_at is None:
            return None
        end = (self.hard_end if self.hard_end is not None
               else self.earliest_start+self.base+self.reserve)
        return self._q(end)

    @property
    def ordinary_due(self):
        if self.started_at is None:
            return None
        # Normally at most reserve_per_turn of the shared reserve; but when the
        # base already ran out before the Agent saw its turn (e.g. a long
        # verification), keep a usable window from first sight, still bounded
        # by the reserve actually left.
        ordinary = max(self.base_due+self.timing.reserve_per_turn_seconds,
                       self.started_at+self.timing.minimum_turn_seconds)
        return min(self.hard_due, ordinary)

    @property
    def commit_at(self):
        if self.started_at is None:
            return None
        return min(self.ordinary_due-self.timing.lead_seconds,
                   self.started_at+self.timing.maximum_turn_seconds)

    @property
    def decision_due(self):
        return None if self.started_at is None else self.commit_at-self.action_margin

    @property
    def selection_due(self):
        return None if self.started_at is None else self.hard_due-self.timing.lead_seconds

    @property
    def last_safe_input(self):
        return None if self.started_at is None else self.hard_due-self.timing.click_seconds

    def to_dict(self):
        return {'started_at': self.started_at, 'earliest_start': self.earliest_start,
                'base': self.base, 'reserve': self.reserve, 'readings': self.readings,
                'hard_due': self.hard_due, 'commit_at': self.commit_at,
                'decision_due': self.decision_due, 'timer': self.timer.to_dict()}
