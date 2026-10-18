from collections import Counter
import json
from pathlib import Path
import time

from .domain import Action, Observation, State
from .rules import RuleEngine
from .strategy import ChoicePolicy
from .tracker import StateTracker
from .vision import Layout, ObservationBuilder


class ControlError(RuntimeError):
    pass


def action_point(action: Action, obs: Observation) -> tuple[float, float]:
    if action.kind == "discard":
        if action.hand_index is None or not 0 <= action.hand_index < len(obs.hand):
            raise ControlError("Discard index missing or out of range")
        tile = obs.hand[action.hand_index]
        if tile.tile != action.tile:
            raise ControlError("Discard tile changed")
        return tile.box.center
    if action.button not in obs.buttons:
        raise ControlError("Action button disappeared")
    return obs.buttons[action.button].center


def preflight(action: Action, before: State, fresh: State, engine: RuleEngine, layout: Layout, now: float) -> tuple[float, float]:
    old, new = before.observation, fresh.observation
    if not fresh.stable or new.issues:
        raise ControlError("Fresh observation is uncertain")
    if old.fingerprint != new.fingerprint or old.window != new.window:
        raise ControlError("State or window changed while the model was choosing")
    if not 0 <= now - old.captured_at <= layout.max_decision_age or not 0 <= now - new.captured_at <= layout.max_decision_age:
        raise ControlError("Decision expired")
    if action not in engine.legal_actions(fresh):
        raise ControlError("Selected action is no longer legal")
    p, q = action_point(action, old), action_point(action, new)
    if max(abs(p[0]-q[0]) / new.window[2], abs(p[1]-q[1]) / new.window[3]) > layout.target_tolerance:
        raise ControlError("Click target moved beyond configured tolerance")
    return q


def action_verified(action: Action, before: Observation, after: Observation) -> bool:
    if after.issues or before.round_id != after.round_id or before.unavailable_fields or after.unavailable_fields:
        return False
    if action.kind == "discard":
        old, new = before.rivers[0], after.rivers[0]
        return len(new) == len(old) + 1 and new[:-1] == old and new[-1] == action.tile
    if action.kind == "nuki":
        return before.nuki[0] is not None and after.nuki[0] == before.nuki[0] + 1
    if action.kind == "pon":
        return len(after.melds[0]) == len(before.melds[0]) + 1
    if action.kind == "kan":
        return sum(len(m.tiles) == 4 for m in after.melds[0]) == sum(len(m.tiles) == 4 for m in before.melds[0]) + 1
    if action.kind == "riichi":
        return after.phase == "riichi_select" or (after.riichi[0] and not before.riichi[0])
    if action.kind in ("ron", "tsumo", "abort"):
        return after.phase == "result" and before.phase != "result"
    if action.kind == "pass":
        return after.phase in ("waiting", "discard") and action.button not in after.buttons
    return False


class AgentLoop:
    def __init__(self, window, detector, builder: ObservationBuilder, engine: RuleEngine,
                 policy: ChoicePolicy, layout: Layout, artifacts: Path):
        self.window, self.detector, self.builder = window, detector, builder
        self.engine, self.policy, self.layout = engine, policy, layout
        self.tracker = StateTracker(layout.stable_frames)
        self.artifacts = artifacts
        self.artifacts.mkdir(parents=True, exist_ok=True)
        self.last_decision: str | None = None
        self.sequence = 0
        self.actionable_since: float | None = None
        if layout.action_margin_seconds >= engine.rules.data["turn_seconds"]:
            raise ValueError("Action margin must be smaller than base turn time")

    def observe(self):
        frame = self.window.capture()
        detections = self.detector.detect(frame.image)
        record = {"captured_at": frame.captured_at, "window": frame.window,
                  "detections": [{"label": d.label, "confidence": d.confidence,
                                  "box": [d.box.x1,d.box.y1,d.box.x2,d.box.y2]} for d in detections]}
        with (self.artifacts / "observations.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        obs = self.builder.build(detections, frame.captured_at, frame.window)
        if not obs.issues:
            if obs.phase in ("discard", "response", "riichi_select"):
                if self.actionable_since is None:
                    self.actionable_since = obs.captured_at
            elif obs.phase in ("waiting", "result"):
                self.actionable_since = None
        return frame, self.tracker.update(obs)

    def log(self, event: str, **fields):
        entry = {"event": event, **fields}
        line = json.dumps(entry, ensure_ascii=False)
        print(line, flush=True)
        with (self.artifacts / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")

    def step(self, execute: bool = False) -> Action | None:
        frame, state = self.observe()
        if state.observation.issues:
            self.log("uncertain", issues=state.observation.issues)
            return None
        actions = self.engine.legal_actions(state)
        if not actions or state.observation.fingerprint == self.last_decision:
            return None
        turn_deadline = (self.actionable_since if self.actionable_since is not None else state.observation.captured_at) + self.engine.rules.data["turn_seconds"]
        model_deadline = min(turn_deadline - self.layout.action_margin_seconds,
                             state.observation.captured_at + self.layout.max_decision_age)
        if time.monotonic() >= model_deadline:
            raise ControlError("No model budget remains in the base turn time")
        try:
            action = self.policy.choose(state, actions, deadline=model_deadline)
        finally:
            self.log("model_timing", **getattr(self.policy, "last_timing", {}))
        if getattr(self.policy, "last_result", None) is not None:
            self.log("model_result", **self.policy.last_result)
        # Re-observe even in dry-run; model latency must be accounted for.
        _, fresh = self.observe()
        point = preflight(action, state, fresh, self.engine, self.layout, time.monotonic())
        if time.monotonic() >= turn_deadline:
            raise ControlError("Base turn time expired before click")
        self.last_decision = state.observation.fingerprint
        self.sequence += 1
        if execute:
            # Avoid slow PNG/log I/O between final validation and the actual click.
            self.window.click(point, fresh.observation.window)
            self.tracker.record_action(action, fresh)
        frame.image.save(self.artifacts / f"{self.sequence:05d}-before.png")
        self.log("decision", execute=execute, action=action.for_model(), state=state.for_model())
        if not execute:
            return action
        deadline = time.monotonic() + self.layout.verification_timeout
        while time.monotonic() < deadline:
            time.sleep(self.layout.poll_seconds)
            frame, after = self.observe()
            if after.stable and action_verified(action, state.observation, after.observation):
                frame.image.save(self.artifacts / f"{self.sequence:05d}-after.png")
                self.log("verified", action_id=action.id, phase=after.observation.phase)
                return action
        frame.image.save(self.artifacts / f"{self.sequence:05d}-unverified.png")
        raise ControlError("Action could not be verified; stopped without retrying the click")
