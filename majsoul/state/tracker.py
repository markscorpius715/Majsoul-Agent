from __future__ import annotations

from collections import Counter
from dataclasses import replace
from pathlib import Path

from .schemas import (FrameObservation, GameEvent, GameEventType, MahjongState,
                      PlayerState, TrackedMeld, TrackingPhase, TrackingUpdate)
from .validator import StateValidator, _labels, _river_delta
from ..tiles import tile_index
from .meld_transition import prove_pon_transition


def reconcile_own_ankan(observed, previous, before_hand, after_hand, *, before_frame, after_frame):
    """Recover a meld's tile multiset only from already observed own-hand evidence.

    Back positions remain marked face-down. Their identities are historical
    deductions, never face-CNN predictions. No opponent concealed tiles enter.
    """
    labels = tuple(tile.tile for tile in observed.tiles)
    backs = tuple(tile.face_down for tile in observed.tiles)
    raw = TrackedMeld(labels, observed.kind, observed.opened, backs)
    if observed.kind != 'ankan' or all(label is not None for label in labels):
        return raw
    known = Counter(label for label in labels if label is not None)
    if (previous is not None and previous.kind == 'ankan'
            and all(label is not None for label in previous.tiles)
            and not known - Counter(previous.tiles)):
        return TrackedMeld(previous.tiles, 'ankan', False, backs, previous.evidence_frames)
    removed = Counter(before_hand) - Counter(after_hand)
    if (sum(removed.values()) != 4 or known - removed or
            len({tile_index(label) for label in removed}) != 1):
        return raw
    remaining = removed - known
    hidden = iter(sorted(remaining.elements()))
    # TrackedMeld records a multiset; no hidden physical position is inferred.
    resolved = tuple(label if label is not None else next(hidden) for label in labels)
    return TrackedMeld(resolved, 'ankan', False, backs, (before_frame, after_frame))


def _is_riichi_discard(orientation: str) -> bool:
    return orientation.startswith("horizontal")


def _observed_riichi(player) -> bool:
    # Live stick detection supplies explicit evidence even while retesting.
    # Legacy Phase 5 observations have no such detector; retain their river
    # evidence without allowing it to override a live negative/unknown stick.
    return (player.riichi is True or not player.riichi_evidence and
            any(_is_riichi_discard(t.orientation) for t in player.river))


def _round_origin(observation: FrameObservation) -> bool:
    return all(not player.river and not player.melds and player.kita_count == 0
               for player in observation.players)


class MahjongStateTracker:
    """Commit stable full-table snapshots and reconstruct only local events.

    A transition is accepted only when the same semantic observation is seen in
    ``stable_frames`` consecutive frames. Non-local changes never get expanded
    into guessed events: the stable snapshot becomes a new baseline and the
    history is marked incomplete.
    """

    def __init__(self, validator: StateValidator, *, stable_frames: int = 2,
                 mode: str = "sanma", self_slot: str = "bottom",
                 allow_chi: bool = False):
        if stable_frames < 1:
            raise ValueError("stable_frames must be at least one")
        if validator.self_slot != self_slot:
            raise ValueError("Tracker and validator self slots must match")
        self.validator = validator
        self.stable_frames = stable_frames
        self.mode = mode
        self.self_slot = self_slot
        self.allow_chi = allow_chi
        self.state: MahjongState | None = None
        self._committed: FrameObservation | None = None
        self._pending_fingerprint: str | None = None
        self._pending_count = 0
        self._last_seen_timestamp: float | None = None
        self._last_seen_frame: int | None = None

    @classmethod
    def load(cls, rules_path: str | Path, *, stable_frames: int = 2,
             self_slot: str = "bottom") -> "MahjongStateTracker":
        import json

        data = json.loads(Path(rules_path).read_text(encoding="utf-8-sig"))
        validator = StateValidator.load(rules_path)
        return cls(validator, stable_frames=stable_frames,
                   mode="sanma" if int(data["players"]) == 3 else "yonma",
                   self_slot=self_slot, allow_chi=bool(data.get("allow_chi", True)))

    def decision_snapshot(self, observation: FrameObservation) -> MahjongState:
        """Validate one current frame without committing events or history.

        Event reconstruction still requires consecutive stable observations.
        A fresh decision snapshot never claims that an unobserved interval is
        complete, nor mutates the tracker baseline.
        """
        issues = self.validator.validate_observation(observation, allow_ui_occlusion=True,
                                                     allow_public_uncertainty=True)
        if issues:
            raise ValueError("; ".join(issues))
        if (self.state is not None and self.state.decision_ready and
                self.state.observation_fingerprint == observation.fingerprint):
            return self.state
        snapshot = self._build_state(
            observation, (), phase=TrackingPhase.TRACKING, history_complete=False,
            reset_round=self.state is None or self.state.round_id != observation.round_id)
        # A validated full concealed count proves an outstanding discard,
        # including after pon when there is no separated drawn tile.
        count = snapshot.self_meld_count
        if count is not None and len(snapshot.hand) == 14 - 3 * count:
            snapshot = replace(snapshot, current_actor=self.self_slot)
        return snapshot

    @property
    def stability_diagnostics(self):
        return {"consecutive_frames": self._pending_count,
                "required_frames": self.stable_frames,
                "reason": ("stable_full_table" if self._pending_count >= self.stable_frames else
                           "awaiting_consecutive_matching_frames" if self._pending_count else
                           "validation_rejected_or_no_candidate")}

    def update(self, observation: FrameObservation) -> TrackingUpdate:
        self._check_order(observation)
        observation_issues = self.validator.validate_observation(observation, allow_ui_occlusion=True)
        if observation_issues:
            self._clear_pending()
            return TrackingUpdate(observation, stable=False, committed=False,
                                  state=self.state,
                                  validation_issues=observation_issues)

        if observation.fingerprint == self._pending_fingerprint:
            self._pending_count += 1
        else:
            self._pending_fingerprint = observation.fingerprint
            self._pending_count = 1
        if self._pending_count < self.stable_frames:
            return TrackingUpdate(observation, stable=False, committed=False,
                                  state=self.state)

        if (self._committed is not None and
                observation.fingerprint == self._committed.fingerprint):
            if self.state is not None and self.state.phase == TrackingPhase.RESYNC:
                # The missing transition stays missing, but freshly validated
                # stable snapshots can re-establish the current table baseline.
                # No DRAW/DISCARD or complete history is invented here.
                self.state = self._build_state(
                    observation, (), phase=TrackingPhase.TRACKING,
                    history_complete=False, reset_round=False)
                self._committed = observation
                return TrackingUpdate(observation, stable=True, committed=True,
                                      state=self.state)
            return TrackingUpdate(observation, stable=True, committed=False,
                                  state=self.state)

        if observation.unavailable_regions or (self._committed is not None and
                                               self._committed.unavailable_regions):
            # UI loss is a missing interval, not evidence of removal, zero kita,
            # a new meld, or an observed opening deal. Commit current known
            # fields for action-specific checks, keeping history incomplete.
            before = self._committed
            transition_issues = (self.validator.validate_available_transition(before, observation)
                                 if before is not None else ())
            if transition_issues:
                self._clear_pending()
                return TrackingUpdate(observation, stable=False, committed=False, state=self.state,
                                      validation_issues=transition_issues)
            first_gap = before is None or not before.unavailable_regions
            events = ((self._event(GameEventType.RESYNC, observation, details={
                "reasons": ["UI occlusion: public fields unavailable"],
                "unavailable_regions": list(observation.unavailable_regions),
                "previous_frame": before.frame_index if before else None,
                "history_incomplete": True}),) if first_gap else ())
            self.state = self._build_state(observation, events, phase=TrackingPhase.TRACKING,
                                          history_complete=False, reset_round=True)
            self._committed = observation
            return TrackingUpdate(observation, stable=True, committed=True, state=self.state, events=events)

        if self._committed is None:
            origin = _round_origin(observation)
            events = (self._event(GameEventType.ROUND_START, observation,
                                  certainty=("observed" if origin else "inferred"),
                                  details={"round_id": observation.round_id,
                                           "baseline": True,
                                           "joined_mid_round": not origin}),)
            complete = origin
            self.state = self._build_state(observation, events,
                                           phase=TrackingPhase.TRACKING,
                                           history_complete=complete,
                                           reset_round=True)
            self._committed = observation
            return TrackingUpdate(observation, stable=True, committed=True,
                                  state=self.state, events=events)

        transition_issues = self.validator.validate_transition(
            self._committed, observation)
        if transition_issues:
            event = self._event(
                GameEventType.RESYNC, observation, certainty="observed",
                details={"reasons": list(transition_issues),
                         "previous_frame": self._committed.frame_index})
            self.state = self._build_state(
                observation, (event,), phase=TrackingPhase.RESYNC,
                history_complete=False, reset_round=True)
            self._committed = observation
            self._clear_pending()
            return TrackingUpdate(
                observation, stable=True, committed=True, state=self.state,
                events=(event,), validation_issues=transition_issues)

        events, round_changed = self._derive_events(self._committed, observation)
        prior_complete = self.state.history_complete if self.state is not None else False
        # Completeness belongs to this hand, not to every earlier hand in the
        # match. A newly observed deal can recover from a previous RESYNC;
        # ordinary same-hand snapshots cannot fill a missing interval.
        complete = _round_origin(observation) if round_changed else prior_complete
        if any(event.details.get("source_discard_unobserved") for event in events):
            complete = False  # Local call proved, intervening source history unknown.
        self.state = self._build_state(
            observation, events, phase=TrackingPhase.TRACKING,
            history_complete=complete, reset_round=round_changed)
        self._committed = observation
        return TrackingUpdate(observation, stable=True, committed=True,
                              state=self.state, events=events)

    def _check_order(self, observation: FrameObservation) -> None:
        if (self._last_seen_timestamp is not None and
                (observation.timestamp <= self._last_seen_timestamp or
                 observation.frame_index <= self._last_seen_frame)):
            raise ValueError("Observations must have increasing timestamps and frame indices")
        self._last_seen_timestamp = observation.timestamp
        self._last_seen_frame = observation.frame_index

    def _clear_pending(self) -> None:
        self._pending_fingerprint = None
        self._pending_count = 0

    @staticmethod
    def _event(event_type: GameEventType, observation: FrameObservation, *,
               player: str | None = None, tile: str | None = None,
               certainty: str = "observed", details=None) -> GameEvent:
        return GameEvent(event_type, observation.timestamp,
                         observation.frame_index, player=player, tile=tile,
                         certainty=certainty, details=details or {})

    def _derive_events(self, before: FrameObservation,
                       after: FrameObservation) -> tuple[tuple[GameEvent, ...], bool]:
        round_changed = ((before.round_wind, before.hand_number, before.honba) !=
                         (after.round_wind, after.hand_number, after.honba))
        if round_changed:
            events = []
            origin = _round_origin(after)
            score_changes = self._score_changes(before, after)
            if score_changes:
                events.append(self._event(
                    GameEventType.SCORE_UPDATE, after,
                    details={"changes": score_changes,
                             "scores": {p.slot: p.score for p in after.players}}))
            events.append(self._event(
                GameEventType.ROUND_END, after, certainty="inferred",
                details={"round_id": before.round_id}))
            events.append(self._event(
                GameEventType.ROUND_START, after,
                certainty=("observed" if origin else "inferred"),
                details={"round_id": after.round_id, "baseline": False,
                         "joined_mid_round": not origin}))
            return tuple(events), True

        delta = _river_delta(_labels(before.player(self.self_slot).river),
                             _labels(after.player(self.self_slot).river))
        pon_proof = prove_pon_transition(before, after, self.self_slot, delta[0] if delta else ())
        removed = [tile for slot in before.active_slots
                   for tile in ((_river_delta(_labels(before.player(slot).river),
                                               _labels(after.player(slot).river)) or ((), ()))[1])]
        meld_events = []
        for slot, change, tiles in self.validator._meld_changes(before, after) or ():
            known_tiles = tuple(tile for tile in tiles if tile is not None)
            normalized = tuple(("5" + tile[1]) if tile[0] == "0" else tile
                               for tile in known_tiles)
            if len(tiles) == 4 or change == "extend":
                kind = GameEventType.KAN
            elif len(known_tiles) != len(tiles):
                kind = GameEventType.MELD
            elif len(set(normalized)) == 1:
                kind = GameEventType.PON
            else:
                kind = GameEventType.CHI
            meld_events.append(self._event(
                kind, after, player=slot,
                tile=(known_tiles[0] if known_tiles else None),
                details={"tiles": list(tiles), "change": change,
                         **({**pon_proof, "evidence": "own_hand_meld_conservation",
                             "source_discard_unobserved": not removed}
                            if kind == GameEventType.PON and slot == self.self_slot and pon_proof else {}),
                         **({"subtype": "unknown"}
                            if kind == GameEventType.MELD else {})}))

        kita_events = []
        for slot in after.active_slots:
            delta = after.player(slot).kita_count - before.player(slot).kita_count
            if delta:
                kita_events.append(self._event(GameEventType.KITA, after,
                                               player=slot, tile="4z"))

        discard_events = []
        newly_riichi = []
        for slot in after.active_slots:
            old_river = _labels(before.player(slot).river)
            new_river = _labels(after.player(slot).river)
            delta = _river_delta(old_river, new_river)
            if delta is None or not delta[0]:
                continue
            tile = delta[0][0]
            discard_events.append(self._event(GameEventType.DISCARD, after,
                                              player=slot, tile=tile))
            old_riichi = _observed_riichi(before.player(slot))
            new_riichi = _observed_riichi(after.player(slot))
            if new_riichi and not old_riichi:
                newly_riichi.append(slot)

        score_changes = self._score_changes(before, after)
        stick_delta = after.riichi_sticks - before.riichi_sticks
        if stick_delta == 1 and not newly_riichi:
            payers = [slot for slot, change in score_changes.items() if change == -1000]
            if len(payers) == 1:
                newly_riichi.append(payers[0])
        riichi_events = [self._event(GameEventType.RIICHI, after, player=slot,
                                     certainty=("observed" if any(
                                         event.player == slot for event in discard_events)
                                                else "inferred"))
                         for slot in newly_riichi]

        draw_events = []
        draw_after_discard = False
        draw_timing_ambiguous = False
        if before.remaining_tiles - after.remaining_tiles == 1:
            actor = None
            tile = None
            certainty = "inferred"
            draw_details = {}
            if after.drawn_index is not None:
                actor = self.self_slot
                tile = after.hand[after.drawn_index].tile
                certainty = "observed"
                if (len(discard_events) == 1 and
                        self._next_slot(after, discard_events[0].player) == actor):
                    draw_after_discard = True
            elif (inferred_tile := self._self_draw_from_hand(
                    before, after, discard_events, meld_events, kita_events)) is not None:
                actor = self.self_slot
                tile = inferred_tile
                draw_details["evidence"] = "self_hand_conservation"
                if (len(discard_events) == 1 and
                        self._next_slot(after, discard_events[0].player) == actor):
                    draw_after_discard = True
            elif len(discard_events) == 1:
                discarder = discard_events[0].player
                self_hand_had_extra_tile = len(before.hand) == len(after.hand) + 1
                if (discarder == self.self_slot and
                        (before.drawn_index is not None or self_hand_had_extra_tile)):
                    actor = self._next_slot(after, discarder)
                    draw_after_discard = True
                else:
                    draw_timing_ambiguous = True
            elif len(kita_events) + len(meld_events) == 1:
                actor = (kita_events + meld_events)[0].player
            draw_events.append(self._event(GameEventType.DRAW, after, player=actor,
                                           tile=tile, certainty=certainty,
                                           details=(
                                               {"relative_order":
                                                "ambiguous_with_discard"}
                                               if draw_timing_ambiguous
                                               else draw_details)))

        score_events = []
        if score_changes:
            score_events.append(self._event(
                GameEventType.SCORE_UPDATE, after,
                details={"changes": score_changes,
                         "scores": {p.slot: p.score for p in after.players}}))
        if draw_after_discard:
            ordered = (meld_events + kita_events + riichi_events + discard_events +
                       score_events + draw_events)
        else:
            ordered = (meld_events + kita_events + draw_events + riichi_events +
                       discard_events + score_events)
        return tuple(ordered), False

    def _self_draw_from_hand(self, before: FrameObservation,
                             after: FrameObservation, discard_events,
                             meld_events, kita_events) -> str | None:
        """Infer one self draw only when exact hand conservation proves its tile."""
        structural_events = (*meld_events, *kita_events)
        if any(event.player == self.self_slot for event in structural_events):
            return None
        self_discards = Counter(
            event.tile for event in discard_events
            if event.player == self.self_slot and event.tile is not None)
        explained = Counter(_labels(after.hand)) + self_discards
        prior = Counter(_labels(before.hand))
        if prior - explained:
            return None
        inferred = explained - prior
        if sum(inferred.values()) != 1:
            return None
        return next(inferred.elements())

    @staticmethod
    def _next_slot(observation: FrameObservation, slot: str) -> str | None:
        winds = ("1z", "2z", "3z", "4z")
        ordered = sorted(observation.players, key=lambda player: winds.index(player.seat))
        for index, player in enumerate(ordered):
            if player.slot == slot:
                return ordered[(index + 1) % len(ordered)].slot
        return None

    @staticmethod
    def _score_changes(before: FrameObservation,
                       after: FrameObservation) -> dict[str, int]:
        changes = {}
        for slot in before.active_slots:
            new, old = after.player(slot).score, before.player(slot).score
            if new is None or old is None:
                continue  # Score OCR unavailable: never diff unknown values.
            delta = new - old
            if delta:
                changes[slot] = delta
        return changes

    def _build_state(self, observation: FrameObservation,
                     events: tuple[GameEvent, ...], *, phase: TrackingPhase,
                     history_complete: bool, reset_round: bool) -> MahjongState:
        prior = self.state
        riichi_players = {event.player for event in events
                          if event.type == GameEventType.RIICHI and event.player}
        new_discards = {slot: [] for slot in observation.active_slots}
        for event in events:
            if event.type == GameEventType.DISCARD and event.player and event.tile:
                new_discards[event.player].append(event.tile)

        players = []
        for observed in observation.players:
            river = _labels(observed.river)
            observed_riichi = _observed_riichi(observed)
            if prior is None or reset_round or phase == TrackingPhase.RESYNC:
                discard_history = river
                prior_riichi = False
            else:
                previous = prior.player(observed.slot)
                discard_history = previous.discard_history + tuple(
                    new_discards[observed.slot])
                prior_riichi = previous.riichi
            if prior is not None and prior.round_id == observation.round_id and not history_complete:
                # Proven declarations remain true throughout the hand; losses
                # of UI visibility do not turn an established riichi into False.
                previous = prior.player(observed.slot)
                prior_riichi = previous.riichi is True
                if f"river_{observed.slot}" in observation.unavailable_regions:
                    discard_history = previous.discard_history
            tracked_melds = []
            for index, meld in enumerate(observed.melds):
                previous_meld = None
                if prior is not None and not reset_round and phase != TrackingPhase.RESYNC:
                    prior_melds = prior.player(observed.slot).melds
                    previous_meld = prior_melds[index] if index < len(prior_melds) else None
                if (observed.slot == self.self_slot and prior is not None
                        and not reset_round and phase != TrackingPhase.RESYNC):
                    tracked = reconcile_own_ankan(
                        meld, previous_meld, prior.hand, _labels(observation.hand),
                        before_frame=prior.source_frame, after_frame=observation.frame_index)
                else:
                    tracked = TrackedMeld(tuple(tile.tile for tile in meld.tiles),
                                          meld.kind, meld.opened,
                                          tuple(tile.face_down for tile in meld.tiles))
                tracked_melds.append(tracked)
            players.append(PlayerState(
                slot=observed.slot,
                seat=observed.seat,
                dealer=observed.dealer,
                score=observed.score,
                river=river,
                discard_history=discard_history,
                melds=tuple(tracked_melds),
                riichi=(True if prior_riichi is True or observed_riichi or observed.slot in riichi_players
                        else False if observed.riichi is False or history_complete or observation.riichi_sticks == 0
                        else None),
                kita_count=observed.kita_count,
                score_available=observed.score_available,
            ))

        drawn_tile = (observation.hand[observation.drawn_index].tile
                      if observation.drawn_index is not None else None)
        discard_events = [event for event in events
                          if event.type == GameEventType.DISCARD]
        if discard_events:
            last_discard = (discard_events[-1].player, discard_events[-1].tile)
        else:
            last_discard = None if reset_round else (
                prior.last_discard if prior is not None else None)
        turn_events = [event for event in events
                       if event.type in (GameEventType.DRAW,
                                         GameEventType.DISCARD,
                                         GameEventType.CHI, GameEventType.PON,
                                         GameEventType.KAN, GameEventType.KITA)]
        if turn_events:
            current_actor = (None if turn_events[-1].type == GameEventType.DISCARD
                             else turn_events[-1].player)
        else:
            current_actor = self.self_slot if observation.drawn_index is not None else None
        history = ((prior.history if prior is not None else ()) + events)
        dealer = next(player.slot for player in observation.players if player.dealer)
        return MahjongState(
            mode=self.mode,
            player_count=len(observation.active_slots),
            active_slots=observation.active_slots,
            self_slot=self.self_slot,
            round_wind=observation.round_wind,
            hand_number=observation.hand_number,
            honba=observation.honba,
            riichi_sticks=observation.riichi_sticks,
            remaining_tiles=observation.remaining_tiles,
            dealer=dealer,
            self_seat=observation.player(self.self_slot).seat,
            players=tuple(players),
            hand=_labels(observation.hand),
            drawn_tile=drawn_tile,
            dora_indicators=_labels(observation.dora_indicators),
            current_actor=current_actor,
            last_discard=last_discard,
            available_actions=observation.action_buttons or (),
            uncertain_fields=observation.unavailable_regions + tuple(f'melds_{player.slot}' for player in players
                                   if any(any(tile is None for tile in meld.tiles)
                                          for meld in player.melds)),
            phase=phase,
            history=history,
            history_complete=history_complete,
            source_frame=observation.frame_index,
            source_timestamp=observation.timestamp,
            unavailable_regions=observation.unavailable_regions,
            current_snapshot_validated=True,
            observation_fingerprint=observation.fingerprint,
        )
