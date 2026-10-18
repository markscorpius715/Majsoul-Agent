"""Local turn watchdog independent of global perception and model completion."""

from collections import Counter
from dataclasses import dataclass
import math

from ..domain import Action
from ..tiles import counts34
from .operation_clock import clock_total
from .turn_timer import MatchTimer, TurnClock, TurnTiming
from .verification import assess_discard_hand_change


@dataclass(frozen=True)
class EmergencyConfig:
    """Last rung of the Agent's own fallback chain, judged only from our own
    hand area and recognized buttons:

    * a recognized ron/tsumo button is clicked (a win is never declined);
    * on our own turn the drawn tile is discarded (tsumogiri), even while
      riichi/pass/nuki buttons are shown; during a riichi selection the drawn
      tile if selectable, else the first selectable tile;
    * PASS only in a call window, and never while a candidate looks like a
      win button that could not be recognized.

    Timing follows the hand's 7+20 timer (``turn_timer``): the rung fires
    ``lead_seconds`` before the turn's ordinary end, and never clicks after
    ``last_safe_input``.
    """
    enabled: bool = False
    timeout_seconds: float = 12.0
    verification_seconds: float = 2.0
    maximum_input_age: float = 1.5
    quantization_seconds: float = 1.0
    lead_seconds: float = 1.5
    click_seconds: float = 0.5
    reserve_per_turn_seconds: float = 2.0
    maximum_turn_seconds: float = 10.0
    minimum_turn_seconds: float = 5.0

    @classmethod
    def from_dict(cls, data):
        data = dict(data or {})
        if set(data)-set(cls.__dataclass_fields__):
            raise ValueError('emergency contains unsupported configuration')
        obj = cls(**data)
        if type(obj.enabled) is not bool:
            raise ValueError('emergency.enabled must be boolean')
        for key, value in vars(obj).items():
            if key != 'enabled' and (type(value) not in (int, float) or
                                    not math.isfinite(value) or not 0 < value <= 60):
                raise ValueError('Invalid emergency ' + key)
        if not obj.click_seconds < obj.lead_seconds:
            raise ValueError('emergency.click_seconds must be shorter than lead_seconds')
        return obj

    def timing(self):
        return TurnTiming(self.quantization_seconds, self.lead_seconds, self.click_seconds,
                          self.reserve_per_turn_seconds, self.maximum_turn_seconds,
                          self.minimum_turn_seconds)


WIN_BUTTONS = ('ron', 'tsumo')
NORTH = '4z'
MAX_NUKI = 4
OWN_TURN_SIZES = (14, 11, 8, 5, 2)


def window_kind(obs):
    hand = obs.hand
    if len(hand) in OWN_TURN_SIZES and type(obs.drawn_index) is int and 0 <= obs.drawn_index < len(hand):
        return 'own'
    if len(hand) % 3 == 1 and obs.drawn_index is None and obs.action_buttons:
        return 'response'
    return None


class TurnWatchdog:
    def __init__(self, config, timer=None):
        self.config = config
        self.timer = timer if timer is not None else MatchTimer()
        if timer is None:
            self.timer.new_hand()
        self.idle_at = None
        self.reset()

    def reset(self):
        self.since = None
        self.kind = None
        self.expires = None
        self.last_total = None
        self.last_hand = None
        self.sent = set()
        self.rebuild_pending = False
        self.extended = False  # a riichi selection may use the whole reserve
        self.nuki_clicks = 0
        self.clock = TurnClock(self.timer, self.config.timing())

    def _open(self, at, hand, kind):
        self.since, self.kind = at, kind
        self.clock.reset()
        self.clock.start(at, self.idle_at)
        self.sent.clear()
        self.nuki_clicks = 0
        self.last_hand = hand

    def _close(self, at):
        idle = self.idle_at
        rebuild = self.rebuild_pending
        self.clock.close(at)
        self.reset()
        self.rebuild_pending = rebuild
        self.idle_at = at if at is not None else idle

    @property
    def commit_at(self):
        """When the rung fires: lead before the turn's ordinary end (or, during
        a riichi selection, before the hard end), and never after the cap."""
        if self.since is None:
            return None
        due = self.clock.selection_due if self.extended else self.clock.commit_at
        return min(due, self.since+self.config.timeout_seconds)

    @property
    def last_safe_input(self):
        return None if self.since is None else self.clock.last_safe_input

    @staticmethod
    def hand_valid(obs, removed_tiles=()):
        if not obs.hand or any(i.startswith('hand') for i in obs.issues):
            return False
        labels = [t.tile for t in obs.hand]
        if any(t.face_down or t.tile is None or t.tile in removed_tiles for t in obs.hand):
            return False
        try:
            counts34(labels)
        except (ValueError, TypeError):
            return False
        return all(n <= 1 for t, n in Counter(labels).items() if t.startswith('0'))

    def observe(self, obs):
        clock = obs.countdown or {}
        total = clock_total(clock.get('total_seconds')) if clock.get('status') == 'complete' else None
        hand = tuple(t.tile for t in obs.hand)
        kind = window_kind(obs)
        if self.since is not None and kind is not None and self.kind is not None and kind != self.kind:
            # A call window and the following own draw are separate turns.
            self._close(obs.timestamp)
        if total is not None and total > 0:
            # A larger clock and a new physical hand prove a new operation,
            # including a turn whose intervening waiting frames were obscured.
            new_turn = (self.last_total is not None and total > self.last_total + 1 and
                        self.last_hand is not None and (Counter(hand) != Counter(self.last_hand) or
                        self.expires is not None and obs.timestamp >= self.expires))
            if new_turn and self.since is not None:
                self._close(obs.timestamp)
            if self.since is None:
                self._open(obs.timestamp, hand, kind)
            # Read once, count locally: later readings only shorten the window.
            self.clock.observe(obs.timestamp, clock)
            self.last_total, self.last_hand = total, hand
        # Absence of the timer alone is NOT proof that a blocked own turn ended.
        elif (self.hand_valid(obs) and len(hand) % 3 == 1 and obs.drawn_index is None
              and obs.action_buttons == () and not any(i.startswith('action_buttons') for i in obs.issues)):
            if self.since is not None:
                self._close(obs.timestamp)
            else:
                self.idle_at = obs.timestamp
        elif self.since is None and (bool(obs.action_buttons) or (
                self.hand_valid(obs) and kind == 'own')):
            # Local hand gap or recognized interaction establishes an operation
            # even when the clock is unreadable.
            self._open(obs.timestamp, hand, kind)
        if self.since is not None and self.kind is None:
            self.kind = kind
        if self.since is None:
            self.expires = None
            return False
        self.expires = self.clock.hard_due
        return obs.timestamp >= self.commit_at

    def select(self, obs, buttons, *, removed_tiles=(), pending_riichi=False, win_hint=False):
        if self.since is None:
            return None, 'operation_window_not_open'
        if obs.timestamp > self.last_safe_input:
            return None, 'operation_window_expired'
        recognized = set(buttons) & set(obs.action_buttons or ())
        win = next((b for b in WIN_BUTTONS if b in recognized), None)
        if win is not None:
            # A recognized client win button is always taken.
            return Action(f'emergency:{win}', win, button=win), None
        if win_hint:
            # Never PASS or discard over a candidate that looks like a win.
            return None, 'possible_win_button_unrecognized'
        kind = window_kind(obs)
        if kind == 'response':
            if 'pass' in recognized:
                return Action('emergency:pass', 'pass', button='pass'), None
            return None, 'call_window_without_recognized_pass'
        if not self.hand_valid(obs, removed_tiles):
            return None, 'own_hand_unrecognized_or_illegal'
        if obs.action_buttons is None:
            # Unreadable buttons could hide a tsumo; leave it to the client.
            return None, 'action_buttons_unrecognized'
        if (kind == 'own' and not pending_riichi and 'nuki' in recognized and
                any(t.tile == NORTH for t in obs.hand) and self.nuki_clicks < MAX_NUKI):
            # Kita before tsumogiri: a drawn North is never thrown away. A
            # replacement North looks identical, so kita may repeat (max 4).
            return Action('emergency:nuki', 'nuki', button='nuki'), None
        index = obs.drawn_index
        if kind != 'own':
            return None, 'fresh_draw_not_proven_including_post_pon'
        if pending_riichi:
            indices = obs.riichi_discard_indices
            if not indices:
                return None, 'riichi_selection_not_observed'
            # Any selectable tile keeps the declared riichi legal.
            index = index if index in indices else indices[0]
        return Action(f'emergency:discard:{index}:{obs.hand[index].tile}', 'discard',
                      obs.hand[index].tile, index), None

    @staticmethod
    def verification(action, before, after):
        if action.kind == 'discard':
            return assess_discard_hand_change(action, before, after)
        if action.kind == 'nuki':
            from .verification import VerificationResult
            changed = Counter(t.tile for t in before.hand) != Counter(t.tile for t in after.hand)
            gone = after.action_buttons is not None and 'nuki' not in after.action_buttons
            if changed or gone:
                return VerificationResult('verified', 'nuki_sent_hand_or_button_changed',
                                          {'hand_changed': changed, 'button_gone': gone})
            return VerificationResult('unknown', 'nuki_identical_hand_possible_replacement_north')
        # Hand unchanged plus PASS disappearance proves response resolution;
        # it deliberately does not claim client acknowledgement versus timeout.
        from .verification import VerificationResult
        if (after.action_buttons is not None and 'pass' not in after.action_buttons and
                not any(i.startswith('action_buttons') for i in after.issues) and
                Counter(t.tile for t in before.hand) == Counter(t.tile for t in after.hand)):
            return VerificationResult('verified', 'pass_window_closed', {'acknowledgement': 'window_resolution_only'})
        return VerificationResult('unknown', 'local_transition_not_yet_proven')
