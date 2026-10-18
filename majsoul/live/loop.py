"""Live agent loop: menu navigation, layout lock, perceive-decide-click-verify."""

from __future__ import annotations

from hashlib import sha256
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import time
from typing import Callable
from uuid import uuid4

from ..calibration.debug import export_projection_trials
from ..calibration.frames import write_image
from ..calibration.roi_config import ROIConfig
from ..domain import Action, State
from ..jev import FallbackPolicy, JevPolicy, JevSettings
from ..perception import GlyphTemplateClassifier, SeatLayoutResolver, VisionConfig
from ..rules import RuleEngine, Rules
from ..policy_prompt import PROMPT_VERSION, rule_instructions
from ..state import (FrameObservation, FrameObservationReader, GameEventType,
                     MahjongStateTracker)
from ..state.schemas import MahjongState
from ..state.decision import decision_observation
from ..strategy import DeepSeekPolicy, LocalFallbackPolicy, ModelSettings, StrategyError
from ..window import WindowError
from .bridge import BridgeError, build_domain_state, infer_phase
from .buttons import ActionButtonReader
from .capture import LiveFrame, LiveWindow, roi_for_frame
from .classifier import build_tile_classifier
from .config import ROOT, LiveConfig
from concurrent.futures import Future
from .decisions import ChoiceResult, DecisionWorker
from .menu import MenuConfig, MenuNavigator
from .notify import FeishuNotifier
from .notifications import NotificationQueue
from .session import AgentSession
from .collector import CollectConfig
from .verification import VerificationResult, verify_action
from .execution_verification import assess_execution_action, apply_execution_context
from .countdown import CountdownReader, DecisionBudget
from .modals import ModalReader
from .turn_timer import MatchTimer, TurnClock
from .unknown_review import UnknownButtonReview
from .selection import RiichiSelection
from .riichi import RiichiDetector
from .emergency import TurnWatchdog
from .gates import input_issues, input_signature, same_input_window, fresh_input_domain

ACTIONABLE_PHASES = ("discard", "response", "riichi_select")


class LiveError(RuntimeError):
    pass


def action_point(action: Action, state: State) -> tuple[float, float]:
    """Click point in full-window coordinates for an action against a domain state."""
    observation = state.observation
    if action.kind == "discard" or (action.kind == "riichi" and action.hand_index is not None):
        if (action.hand_index is None or
                not 0 <= action.hand_index < len(observation.hand)):
            raise LiveError("Discard index missing or out of range")
        view = observation.hand[action.hand_index]
        if view.tile != action.tile:
            raise LiveError("Discard tile changed")
        return view.box.center
    if action.button not in observation.buttons:
        raise LiveError("Action button disappeared")
    # A compound riichi first clicks the riichi button; its tile comes later.
    return observation.buttons[action.button].center


def preflight(action: Action, before_obs: FrameObservation,
              fresh_obs: FrameObservation, fresh_domain: State,
              window_before: tuple, window_after: tuple,
              config: LiveConfig, engine: RuleEngine, *,
              point_before: tuple[float, float] | None = None,
              now: float | None = None) -> tuple[float, float]:
    """Revalidate a chosen action against a fresh observation; return the click point."""
    if input_issues(fresh_obs):
        raise LiveError("Fresh observation is uncertain")
    if not same_input_window(before_obs, fresh_obs):
        raise LiveError("Own hand or action window changed while the model was choosing")
    if tuple(window_before) != tuple(window_after):
        raise LiveError("Window geometry changed while the model was choosing")
    current = time.monotonic() if now is None else now
    if not 0 <= current - fresh_obs.timestamp <= config.decision.max_decision_age:
        raise LiveError("Observation expired before the click")
    if action not in engine.legal_actions(fresh_domain):
        raise LiveError("Selected action is no longer legal")
    point = action_point(action, fresh_domain)
    if point_before is not None:
        width, height = window_after[2], window_after[3]
        drift = max(abs(point[0] - point_before[0]) / width,
                    abs(point[1] - point_before[1]) / height)
        if drift > config.execution.target_tolerance:
            raise LiveError("Click target moved beyond configured tolerance")
    return point


def decision_deadline(actionable_since: float | None, captured_at: float, *,
                      turn_seconds: float, action_margin: float,
                      max_decision_age: float) -> float:
    base = captured_at if actionable_since is None else actionable_since
    return min(base + turn_seconds - action_margin, captured_at + max_decision_age)


def build_policy(config: LiveConfig, root: Path = ROOT):
    # The live configuration is authoritative, including explicit direct mode.
    # Mutating process environment makes a retired route survive later resets.
    jev_settings = replace(JevSettings.load(root), proxy=config.decision.jev_proxy)
    deepseek = lambda: DeepSeekPolicy(ModelSettings.load(root))
    jev = lambda: JevPolicy(jev_settings)
    if config.decision.policy == "jev":
        model = jev()
    elif config.decision.policy == "deepseek":
        model = deepseek()
    else:
        model = FallbackPolicy(jev, deepseek, jev_settings.timeout,
                               hedge_seconds=config.decision.hedge_seconds)
    if not config.decision.local_fallback_seconds:
        return model
    # The Agent always commits its own legal action before its deadline.
    return LocalFallbackPolicy(model, reserve_seconds=config.decision.local_fallback_seconds)


class LiveAgentLoop:
    def __init__(self, window, config: LiveConfig, *, execute: bool, skip_menu: bool,
                 once: bool = False, session: str | None = None,
                 sleep: Callable[[float], None] = time.sleep,
                 now: Callable[[], float] = time.monotonic,
                 roi: ROIConfig | None = None, reader=None, button_reader=None,
                 tracker=None, engine=None, policy=None, notifier=None,
                 join: bool = False, input_gate: Callable[[], bool] | None = None,
                 layout_hint=None, handback_frames=None, checkpoint: Path | None = None,
                 supervised: bool = False):
        if (execute and config.watchdog.enabled and isinstance(window,LiveWindow)
                and not getattr(window,'protected_owner',False)):
            raise LiveError('Execute runs under the watchdog; use python -m majsoul.live run --execute')
        self.window = window
        self.config = config
        self.execute = execute
        self.skip_menu = skip_menu
        self.once = once
        # Rejoin: enter the current match mid-hand without menu or start-of-game
        # deadlines. While ``input_gate`` is false the loop only observes and
        # reports handback readiness; the supervisor owns every input decision.
        self.join = join
        self._input_gate = input_gate
        self._input_was_allowed = None
        self._layout_hint = None
        self.set_layout_hint(layout_hint)
        self._handback_frames = handback_frames
        self._handback_since = None
        self._handback_count = 0
        self._handback_signalled_at = float('-inf')
        self._checkpoint = checkpoint
        # A supervised Agent is recovered by its supervisor, so local watchdogs
        # that merely lack evidence resynchronize instead of ending the process.
        self.supervised = supervised
        self._game_started = False
        self._recognizing_start = False
        self._recognition_deadline = None
        self.sleep = sleep
        self.now = now
        self.roi = roi if roi is not None else ROIConfig.load(config.roi)
        self.engine = engine if engine is not None else RuleEngine(Rules.load(config.rules))
        countdown_reader = CountdownReader.load(config.countdown, ROOT, self.engine.rules.data)
        self.classifier_manifest = None
        numeric_ocr = None
        if reader is None:
            if config.numeric_ocr is not None:
                from ..perception.numeric_ocr import NumericOCR, NumericOCRConfig
                numeric_ocr = NumericOCR(NumericOCRConfig.load(config.numeric_ocr, ROOT))
            vision = VisionConfig.load(config.vision)
            tile_classifier = build_tile_classifier(
                config.classifier, vision, cnn_checkpoint=config.cnn_checkpoint,
                template_root=config.tile_templates, expected_metadata=config.cnn_expected)
            self.classifier_manifest = getattr(tile_classifier, 'manifest', None)
            button_reader = ActionButtonReader(config.buttons, config.button_templates)
            reader = FrameObservationReader(
                vision, tile_classifier,
                GlyphTemplateClassifier(config.glyph_templates, vision),
                button_reader=button_reader, read_scores=config.read_scores,
                countdown_reader=countdown_reader, numeric_ocr=numeric_ocr)
        self.reader = reader
        self.riichi_detector = RiichiDetector(config.riichi_detection)
        rules_data = self.engine.rules.data
        # The client's 7+20 control: base per turn, one reserve per hand,
        # then time-out turns. Shared by decisions and the tsumogiri rung.
        self.match_timer = MatchTimer(base_seconds=rules_data['turn_seconds'],
                                      reserve_seconds=rules_data['reserve_seconds'],
                                      timeout_turn_seconds=rules_data.get('fallback_turn_seconds', 5))
        self.turn_watchdog = TurnWatchdog(config.emergency, timer=self.match_timer)
        self._own_riichi_round = None   # hand in which our own riichi was verified
        self._joined_round = None       # hand this Agent joined mid-way (state unknown)
        self._last_round_id = None
        self._riichi_plan = None        # (tile, index) chosen together with riichi
        self._kan_blocked = None        # input signature of an unresolved kan window
        self._actionable_phase = None
        self._last_emergency_blocker = None
        self.button_reader = (button_reader if button_reader is not None
                              else getattr(reader, "button_reader", None))
        if (self.button_reader is not None and
                hasattr(self.button_reader, "unknown_sink")):
            self.button_reader.unknown_sink = self._buffer_unknown_button
        self.tracker = (tracker if tracker is not None else
                        MahjongStateTracker.load(config.rules,
                                                 stable_frames=config.stable_frames))
        turn_seconds = self.engine.rules.data["turn_seconds"]
        if config.decision.action_margin_seconds >= turn_seconds:
            raise ValueError("action_margin_seconds must be smaller than rules turn_seconds")
        budget_options = ({"confirmations": countdown_reader.config.confirmations,
                           "quantization_margin": countdown_reader.config.quantization_margin_seconds,
                           "maximum_sample_gap": countdown_reader.config.maximum_sample_gap_seconds}
                          if countdown_reader is not None else {})
        self.decision_budget = DecisionBudget(
            turn_seconds=turn_seconds, reserve_seconds=self.engine.rules.data["reserve_seconds"],
            fallback_turn_seconds=self.engine.rules.data.get('fallback_turn_seconds'),
            action_margin=config.decision.action_margin_seconds, **budget_options)
        # The screen clock is read once per window and counted down locally.
        # Decisions must land before the Agent's own tsumogiri rung fires; no
        # other process plays for the Agent.
        self.operation_clock = (TurnClock(self.match_timer, config.emergency.timing(),
                                          action_margin=config.decision.action_margin_seconds)
                                if supervised else None)
        self._idle_at = None
        self._last_click_at = None
        self._policy_factory = (lambda: build_policy(config, ROOT)) if policy is None else None
        self.policy = policy if policy is not None else self._policy_factory()
        self.decision_worker = DecisionWorker(self.policy)
        self._pending_decision = None
        self._last_request_key = None
        self._decision_epoch = 0
        self._last_gate_status = None
        self._last_decision_blocker = None
        self._last_layout_status = None
        self.notifier = (notifier if notifier is not None else
                         FeishuNotifier(config.notify,
                                        logger=lambda message: self.log(
                                            "notify_error", error=message)))
        self._notified_path = config.artifacts_root / "unknown_buttons_notified.json"
        self._notified = self._load_notified()
        self._notify_attempted: set[str] = set()
        self._frame_unknowns: list = []
        self.menu_config = MenuConfig.load(config.menu)
        name = session or time.strftime("session-%Y%m%d-%H%M%S")
        self.artifacts = config.artifacts_root / name
        self.artifacts.mkdir(parents=True, exist_ok=True)
        card = rule_instructions(self.engine.rules.data)
        (self.artifacts / 'policy-rules.txt').write_text(card, encoding='utf-8')
        self._rule_card = {'version': PROMPT_VERSION, 'file': 'policy-rules.txt',
                           'sha256': sha256(card.encode('utf-8')).hexdigest()}
        if numeric_ocr is not None:
            self.artifacts.mkdir(parents=True, exist_ok=True)
            (self.artifacts / 'numeric-ocr.json').write_text(
                json.dumps(numeric_ocr.manifest, ensure_ascii=False, indent=2), encoding='utf-8')
        self.unknown_review = UnknownButtonReview(
            config.unknown_review, config.artifacts_root / 'unknown_buttons', ROOT / 'datasets', session=name,
            versions={"button_config": asdict(config.buttons), "classifier": self.classifier_manifest,
                      "roi_sha256": sha256(config.roi.read_bytes()).hexdigest(),
                      "continual_config_sha256": sha256(config.unknown_review.continual_config.read_bytes()).hexdigest(),
                      "button_templates": {str(p.relative_to(config.button_templates)): sha256(p.read_bytes()).hexdigest()
                                           for p in sorted(config.button_templates.glob('*/*')) if p.is_file()}})
        if countdown_reader is not None:
            self.artifacts.mkdir(parents=True, exist_ok=True)
            (self.artifacts / 'countdown.json').write_text(
                json.dumps(countdown_reader.manifest, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
        if self.classifier_manifest is not None:
            self.artifacts.mkdir(parents=True, exist_ok=True)
            (self.artifacts / 'classifier.json').write_text(json.dumps(
                self.classifier_manifest, ensure_ascii=False, indent=2), encoding='utf-8')
        self.notifications = (NotificationQueue(
            self.notifier, self.artifacts / "notifications",
            capacity=config.notify.queue_capacity, logger=self.log,
            max_attempts=config.notify.max_attempts, retry_seconds=config.notify.retry_seconds)
            if notifier is None else None)

        self._frame_index = 0
        self._roi_eff: ROIConfig | None = None
        self._roi_size: tuple[int, int] | None = None
        self._resolver: SeatLayoutResolver | None = None
        self._active_slots: tuple[str, ...] | None = None
        self._pending_frame: tuple[LiveFrame, int] | None = None
        self._pending_riichi = False
        self.riichi_selection = RiichiSelection(config.riichi_selection)
        if self.riichi_selection.cancel_marker.manifest is not None:
            self.artifacts.mkdir(parents=True, exist_ok=True)
            (self.artifacts / 'selection.json').write_text(json.dumps(
                {'riichi_selection': asdict(config.riichi_selection),
                 'cancel_marker': self.riichi_selection.cancel_marker.manifest},
                ensure_ascii=False, indent=2), encoding='utf-8')
        self._riichi_round_id: str | None = None
        self._actionable_since: float | None = None
        self._last_decision_fingerprint: str | None = None
        self._last_unknown_pass_key = None
        self._last_activity: float | None = None
        self._issue_since: float | None = None
        self._clean_streak = 0
        self._paused = False
        self._verified_operation = None
        self._decisions_made = 0
        self._last_settlement_restart = None
        self.session_ui = (AgentSession(
            window, CollectConfig.load(config.lifecycle_ui), execute=execute,
            capture=lambda: self._capture()[0], sleep=sleep, log=self.log,
            directory=self.artifacts, notify=self._notify_session,
            modals=ModalReader(config.modal_manifest) if config.modal_manifest else None)
            if config.lifecycle_ui is not None else None)

    def _notify_session(self, message, image=None, *, metadata=None):
        if self.notifications is not None:
            return self.notifications.submit('notify_text', message, image, metadata=metadata)
        if hasattr(self.notifier, 'notify_text'):
            return self.notifier.notify_text(message, image)
        return False

    def set_layout_hint(self, layout):
        self._layout_hint = tuple(layout) if layout else None

    def _input_allowed(self) -> bool:
        return self._input_gate is None or bool(self._input_gate())

    def _write_checkpoint(self, **fields):
        if self._checkpoint is None or not self._active_slots:
            return
        data = {'active_slots': list(self._active_slots), 'session': self.artifacts.name,
                'written_at': time.time(), **fields}
        temporary = self._checkpoint.with_name(f'.{self._checkpoint.name}.{uuid4().hex}.tmp')
        try:
            temporary.write_text(json.dumps(data, ensure_ascii=False, default=str), encoding='utf-8')
            temporary.replace(self._checkpoint)
        except OSError as exc:
            self.log('checkpoint_write_failed', error=str(exc))

    def _on_input_granted(self):
        # Observation history is kept; every decision clock starts fresh.
        self._cancel_decision('input granted after handback', window_closed=True)
        self._reset_window_clock()
        self.turn_watchdog.reset()
        self._actionable_since = None
        self._verified_operation = None
        self._issue_since = None
        self._paused = False
        self._handback_since = None
        self._handback_count = 0
        self._last_activity = self.now()
        self.log('input_granted', frame_index=self._frame_index, active_slots=self._active_slots)

    def _observe_handback(self, frame, observation, phase, state, issues, unknowns):
        """Report readiness from consecutive validated opponent-turn snapshots."""
        fresh = 0 <= self.now()-frame.captured_at <= self.config.decision.max_decision_age
        blockers = [name for name, failed in (
            ('stale_frame', not fresh), ('phase:' + phase, phase != 'waiting'),
            ('no_snapshot', state is None), ('validation', bool(issues)),
            ('unknown_buttons', bool(unknowns)), ('perception', bool(observation.actionable_issues)),
            ('no_layout', not self._active_slots)) if failed]
        if blockers != getattr(self, '_handback_blockers', None):
            # Logged on change only: the reason readiness cannot accumulate.
            self._handback_blockers = blockers
            self.log('handback_readiness', frame_index=observation.frame_index, blockers=blockers,
                     issues=list(issues)[:5], perception=list(observation.actionable_issues)[:5])
        if blockers:
            self._handback_since = None
            self._handback_count = 0
            return
        if self._handback_since is None:
            self._handback_since = frame.captured_at
        self._handback_count += 1
        need = self._handback_frames() if callable(self._handback_frames) else self._handback_frames
        need = max(1, int(need or self.config.protection.handback_stable_frames))
        span = frame.captured_at-self._handback_since
        if (self._handback_count >= need and span >= self.config.protection.handback_stable_seconds
                and frame.captured_at-self._handback_signalled_at >= .25):
            self._handback_signalled_at = frame.captured_at
            self.log('agent_ready_for_handback', frame_index=observation.frame_index,
                     frames=self._handback_count, span=round(span, 3), round_id=observation.round_id)

    def _session_frame(self, frame):
        if self.session_ui is None:
            return False
        # Result confirmations are input as well; an observe-only Agent never clicks.
        self.session_ui.execute = self.execute and self._input_allowed()
        suppress, events = self.session_ui.observe(frame, self.now())
        if events:
            self._last_activity = frame.captured_at
        if 'confirm_appeared' in events:
            self.turn_watchdog.reset()
            round_index = self.session_ui.lifecycle.round_index
            if self._last_settlement_restart != round_index:
                self.reset_runtime('hand settlement', restart_policy=True)
                self._last_settlement_restart = round_index
            self._verified_operation = None
            self._reset_window_clock(self.now())
            self._cancel_decision('result confirmation appeared', window_closed=True)
            self._pending_riichi = False
            self.riichi_selection.reset()
            self._riichi_round_id = None
            self._actionable_since = None
            self._last_decision_fingerprint = None
        if 'table_ready' in events and self.session_ui.lifecycle.round_index > 1:
            self.riichi_detector.new_hand()
            self.turn_watchdog.reset()
            self.reset_runtime('new hand ready', restart_policy=False)
        return suppress

    def reset_runtime(self, reason, *, restart_policy=True):
        """Rebuild per-hand state while retaining capture, layout and lifecycle.

        No UI input occurs here. The operator supervisor retains client takeover
        until recovery validation and verified OFF toggles finish.
        """
        self.log('agent_runtime_reset_started', reason=reason, frame_index=self._frame_index)
        self._cancel_decision(reason, window_closed=True)
        if restart_policy and self._policy_factory is not None:
            policy = self._policy_factory()
            self.decision_worker.close()
            self.policy = policy
            self.decision_worker = DecisionWorker(policy)
        self.tracker = MahjongStateTracker.load(
            self.config.rules, stable_frames=self.config.stable_frames)
        self._decision_epoch += 1
        self._reset_window_clock()
        self.riichi_selection.reset()
        self._pending_riichi = False
        self._riichi_round_id = None
        self._verified_operation = None
        self._actionable_since = None
        self._issue_since = None
        self._clean_streak = 0
        self._paused = False
        self._last_gate_status = None
        self._last_decision_blocker = None
        self._frame_unknowns = []
        self.unknown_review.reset()
        self._last_activity = self.now()
        self.log('agent_runtime_reset_done', reason=reason, frame_index=self._frame_index,
                 policy_rebuilt=restart_policy and self._policy_factory is not None,
                 active_slots=self._active_slots, hand_epoch=self._decision_epoch)

    # ------------------------------------------------------------------ logging

    def log(self, event: str, **fields) -> None:
        entry = {"event": event, **fields}
        # Comparable with the supervisor's and observer's monotonic times.
        entry.setdefault("monotonic", round(self.now(), 3))
        line = json.dumps(entry, ensure_ascii=False, default=str)
        print(line, flush=True)
        events = self.artifacts / "events.jsonl"
        events.parent.mkdir(parents=True, exist_ok=True)
        with events.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")

    def _log_decision(self, record: dict) -> None:
        path = self.artifacts / "decisions.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    # -------------------------------------------------------------- primitives

    def _capture(self) -> tuple[LiveFrame, int]:
        if (self.artifacts / "STOP").exists():
            raise KeyboardInterrupt
        frame = self.window.capture()
        self._frame_index += 1
        return frame, self._frame_index

    def _resolve_roi(self, frame: LiveFrame) -> ROIConfig:
        size = (int(frame.image.shape[1]), int(frame.image.shape[0]))
        if self._roi_size == size and self._roi_eff is not None:
            return self._roi_eff
        try:
            roi_eff = roi_for_frame(self.roi, size, self.config.projection)
        except ValueError as exc:
            output = self.artifacts / "projection"
            export_projection_trials(frame.image, self.roi, output)
            raise LiveError(
                f"{exc}; exported projection candidates to {output} — review the "
                "crops and set the projection field in the live config to the "
                "chosen projection name") from exc
        self._roi_eff = roi_eff
        self._roi_size = size
        self._resolver = None
        return roi_eff

    def _layout_resolver(self, roi_eff: ROIConfig) -> SeatLayoutResolver:
        if self._resolver is None:
            self._resolver = SeatLayoutResolver(
                roi_eff, self.reader.metadata_reader,
                starting_points=self.tracker.validator.starting_points,
                confirmations_required=self.config.layout_confirmations)
        return self._resolver

    def _track(self, image, roi_eff: ROIConfig, active_slots, timestamp: float,
               frame_index: int, frame: LiveFrame | None = None):
        # Preflight/layout/verification also read buttons. Their callbacks must
        # never accumulate as extra samples of the next main-loop frame.
        self._frame_unknowns = []
        try:
            observation, visible = self.reader.read_with_visible(
                image, roi_eff, active_slots,
                timestamp=timestamp, frame_index=frame_index)
            prior_baselines = set(self.riichi_detector.baselines)
            observation = self.riichi_detector.apply(image, roi_eff, observation)
            observation = self._apply_own_state(observation)
            for slot in set(self.riichi_detector.baselines) - prior_baselines:
                path = self.artifacts / 'riichi-baseline' / (slot + '.png')
                write_image(path, (self.riichi_detector.baselines[slot] * 255).astype('uint8'))
                self.log('riichi_baseline_frozen', slot=slot, frame_index=frame_index,
                         file=str(path.relative_to(self.artifacts)), round_id=observation.round_id)
            if self._pending_riichi:
                indices, evidence = ((None, {'status': 'unknown', 'reason': 'capture_geometry_missing'})
                                     if frame is None else self.riichi_selection.read(frame, observation, visible, roi=roi_eff))
                observation = replace(observation, riichi_discard_indices=indices,
                                      selection_evidence=evidence)
            update = self.tracker.update(observation)
        except (ValueError, RuntimeError) as exc:
            self.log("perception_error", frame_index=frame_index, error=str(exc))
            return None
        return observation, visible, update

    def _buttons(self) -> dict:
        if self.button_reader is None:
            return {}
        return dict(getattr(self.button_reader, "last_buttons", None) or {})

    def _buffer_unknown_button(self, label, candidate, box) -> None:
        self._frame_unknowns.append((label, candidate, box))

    def _load_notified(self) -> set:
        try:
            data = json.loads(self._notified_path.read_text(encoding="utf-8"))
            return {str(item) for item in data.get("notified", ())}
        except (OSError, ValueError, AttributeError):
            return set()

    def _save_notified(self) -> None:
        temporary = self._notified_path.with_name(
            f".{self._notified_path.name}.{uuid4().hex}.tmp")
        temporary.write_text(json.dumps(
            {"notified": sorted(self._notified)}, ensure_ascii=False, indent=2),
            encoding="utf-8")
        temporary.replace(self._notified_path)

    def _handle_unknown_buttons(self, frame: LiveFrame, index: int,
                                observation: FrameObservation, phase: str,
                                buttons: dict, unknowns: list, *, eligible: bool) -> None:
        note = ("Will attempt to click the recognized skip button; click acceptance is verified separately" if phase == "response"
                else "Match decision-making continues normally")
        events = self.unknown_review.observe(frame, index, observation, phase, unknowns, eligible=eligible,
            predictions=getattr(self.button_reader, 'last_predictions', {}), epoch=self._decision_epoch)
        for event in events:
            if event['state'] != 'HUMAN_REVIEW':
                self.log('unknown_button_resync', **event)
                if event.get('reason') == 'review_storage_failed':
                    if not self.supervised:
                        raise LiveError('Cannot archive persistent unknown button evidence')
                    # Review evidence is not needed to play; the response
                    # itself remains blocked from the model below.
                    self.log('unknown_button_archive_failed', digest=event.get('digest'))
                continue
            case = event['case']
            case_dir, digest = Path(case.path), case.case_id
            hint = event['hint']
            if digest not in self._notified and digest not in self._notify_attempted:
                self._notify_attempted.add(digest)
                message = (
                    f"[majsoul-live] unrecognized interaction button detected phase={phase} "
                    f"hint={hint or 'unknown'}? session={self.artifacts.name} "
                    f"case={case_dir}, {note}. consecutive {event['streak']} valid stable frames before entering human review, "
                    "Candidate hint is not a label; no automatic training or online template updates.\n"
                    f"local review: python -m majsoul.continual --cases \"{case_dir.parent}\" gui --cases-only --kind {event['kind']}\n"
                    + case.message)
                def record_delivery(delivered, digest=digest):
                    if not delivered:
                        return
                    self._notified.add(digest)
                    try:
                        self._save_notified()
                    except OSError as exc:
                        self.log("notify_store_error", error=str(exc))
                if self.notifications is not None:
                    future = self.notifications.submit("notify_unknown_button", case_dir, message,
                        metadata={'purpose':'human_review','case_id':digest,'session_id':self.artifacts.name})
                    future.add_done_callback(lambda result, callback=record_delivery:
                                             callback(result.result()))
                else:
                    record_delivery(self.notifier.notify_unknown_button(case_dir, message))
            self.log("unknown_button", digest=digest, hint=hint, phase=phase,
                     control_state='HUMAN_REVIEW', streak=event['streak'], created=case.created,
                     notified=digest in self._notified, case=str(case_dir))
        if unknowns and phase == "response":
            # An unreadable response window is never entrusted to the model.
            self._last_decision_fingerprint = observation.fingerprint
            if "pass" not in buttons:
                self.log("unknown_button_no_pass",
                         note="no recognized pass button; waiting for the "
                              "client timeout to skip")
            elif self.execute and not self._paused and self._input_allowed():
                self._pass_unknown_response(frame, buttons, observation)

    def _pass_unknown_response(self, frame: LiveFrame, buttons: dict,
                               observation: FrameObservation) -> None:
        key = (self._decision_epoch, observation.fingerprint, tuple(frame.window))
        if key == self._last_unknown_pass_key:
            return
        point_before = buttons["pass"].center
        try:
            fresh_frame, fresh_index = self._capture()
        except WindowError as exc:
            self.log("preflight_abort", reason=str(exc))
            return
        tracked = self._track(fresh_frame.image, self._resolve_roi(fresh_frame),
                              self._active_slots, fresh_frame.captured_at,
                              fresh_index)
        if tracked is None:
            self.log("preflight_abort", reason="fresh observation failed")
            return
        if tracked[0].fingerprint != observation.fingerprint:
            self.log("preflight_abort", reason="response changed before pass")
            return
        fresh_buttons = self._buttons()
        if "pass" not in fresh_buttons:
            self.log("preflight_abort", reason="pass button disappeared")
            return
        point = fresh_buttons["pass"].center
        if tuple(frame.window) != tuple(fresh_frame.window):
            self.log("preflight_abort", reason="window geometry changed")
            return
        width, height = fresh_frame.window[2], fresh_frame.window[3]
        drift = max(abs(point[0] - point_before[0]) / width,
                    abs(point[1] - point_before[1]) / height)
        if drift > self.config.execution.target_tolerance:
            self.log("preflight_abort",
                     reason="pass button moved beyond configured tolerance")
            return
        if not tracked[2].stable:
            self.log("preflight_abort", reason="pass observation is not stable")
            return
        deadline = self._window_deadline(tracked[0])
        if self.decision_budget.pending or self.now() >= deadline:
            self.log("preflight_abort", reason="response deadline passed before pass")
            return
        action = Action("button:pass", "pass", button="pass")
        sequence = f"unknown-pass-{uuid4().hex}"
        write_image(self.artifacts / "frames" / f"{sequence}-before.png", fresh_frame.image)
        # Input may take effect even if restoring focus subsequently fails.
        # Never resubmit merely because the response remains visible.
        self._last_unknown_pass_key = key
        record = {"reason": "unknown_button_response", "action": action.for_model(),
                  "timestamp": fresh_frame.captured_at, "frame_index": fresh_index,
                  "source_image": f"frames/{sequence}-before.png",
                  "state_version": tracked[0].fingerprint}
        try:
            self._last_click_at = self.now()
            self.window.click(point, expected=fresh_frame,
                              hold_seconds=self.config.execution.mouse_press_seconds)
            self.log("clicked", action="pass", reason="unknown_button_response",
                     point=[round(point[0], 1), round(point[1], 1)])
            record["mode"] = self._verify(action, tracked[0], sequence)
        except KeyboardInterrupt:
            record["mode"] = "cancelled"
            raise
        except WindowError as exc:
            record["mode"] = "execution_unknown"
            record["reason"] = str(exc)
            raise
        finally:
            self._log_decision(record)

    # -------------------------------------------------------------------- run

    def run(self) -> int:
        self.artifacts.mkdir(parents=True, exist_ok=True)
        self.log("session_start", execute=self.execute, skip_menu=self.skip_menu,
                 session=self.artifacts.name,
                 policy=self.config.decision.policy,
                 projection=self.config.projection)
        outcome, exit_code = 'exception', 1
        try:
            if not self.skip_menu and not self.join:
                self._navigate_menu()
            self._wait_for_table()
            self._lock_layout()
            exit_code = self._main_loop()
            outcome = ('match_complete' if self.session_ui is not None and
                       self.session_ui.lifecycle.complete else 'loop_returned')
            return exit_code
        except KeyboardInterrupt:
            outcome, exit_code = 'stop_requested', 130
            self.log("interrupted")
            return 130
        except (LiveError, WindowError) as exc:
            outcome = f'{type(exc).__name__}: {exc}'
            self.log("fatal_error", error=str(exc))
            raise
        except Exception as exc:
            outcome = f'{type(exc).__name__}: {exc}'
            raise
        finally:
            # Bypass observer hooks: an observer may itself be the exception
            # which stopped this loop. Always persist the terminal reason.
            pending = self._pending_decision
            LiveAgentLoop.log(self, 'agent_stopped', reason=outcome, exit_code=exit_code,
                frame_index=self._frame_index, paused=self._paused,
                pending_request_id=pending['request_id'] if pending else None,
                pending_phase=pending['phase'] if pending else None,
                pending_deadline=pending['deadline'] if pending else None,
                last_decision_blocker=self._last_decision_blocker,
                last_gate=self._last_gate_status,
                match_complete=bool(self.session_ui and self.session_ui.lifecycle.complete))
            try:
                self._cancel_decision('stopped', window_closed=True)
            finally:
                self.decision_worker.close()
                numeric = getattr(getattr(self.reader, 'metadata_reader', None), 'numeric_ocr', None)
                if numeric is not None and callable(getattr(numeric, 'close', None)):
                    numeric.close()
                if self.notifications is not None:
                    self.notifications.close(self.config.notify.shutdown_seconds)

    def _navigate_menu(self) -> None:
        if not self.execute:
            self.log("dry_run_menu_skipped", hint="Enter a table manually; dry-run never sends input")
            return
        if not self.menu_config.ready:
            self.log("menu_not_ready",
                     hint="Menu coordinates are not calibrated; enter a sanma "
                          "east match manually or run calibrate-menu")
            return
        if self.session_ui is not None:
            frame, _ = self._capture()
            self._resolve_roi(frame)
            lobby, kind, table = self.session_ui.classify(frame)
            if kind or table:
                self.log('menu_skipped', reason='table or result visible')
                return
            if not lobby:
                raise LiveError('Unknown screen; cannot verify menu navigation start')
        MenuNavigator(self.window, self.menu_config, sleep=self.sleep, now=self.now,
                      poll_seconds=min(self.config.poll_seconds,
                                       self.config.watchdog.heartbeat_seconds / 2)).run()
        self.log("menu_navigated")

    def _wait_for_table(self) -> None:
        self._recognizing_start = True
        # A rejoin observer sends no input; its liveness is supervised instead.
        seconds=(float('inf') if self.join else
                 min(self.menu_config.table_timeout_seconds,self.config.watchdog.recognition_seconds))
        self._recognition_deadline=self.now()+seconds
        self.log('table_wait_started', timeout_seconds=None if self.join else seconds,
                 source='agent_rejoin' if self.join else 'agent_recognition')
        deadline = self._recognition_deadline
        while True:
            frame, index = self._capture()
            suppressed=self._session_frame(frame)
            # Lifecycle still handles results and the lobby, but an unknown
            # UI template must not veto the Agent's own metadata recognition.
            blocked=False
            if suppressed and self.session_ui is not None:
                lobby,kind,_=self.session_ui.classify(frame)
                blocked=bool(lobby or kind)
            if blocked:
                if self.session_ui.lifecycle.complete:
                    raise LiveError('Returned to lobby before Agent reached a table')
                if self.now() >= deadline:
                    raise LiveError('Table UI did not become ready before timeout')
                self.sleep(self.config.poll_seconds)
                continue
            roi_eff = self._resolve_roi(frame)
            resolution = self._layout_resolver(roi_eff).observe(frame.image)
            self._log_layout_gate(resolution, index)
            if resolution.candidates:
                self.log("table_detected",
                         candidates=[list(item) for item in resolution.candidates])
                self._pending_frame = (frame, index)
                return
            if self.now() >= deadline:
                try:
                    write_image(self.artifacts / "table_timeout.png", frame.image)
                except (OSError, ValueError):
                    pass
                raise LiveError(
                    "Table screen was not detected within "
                    f"{seconds:g}s")
            self.sleep(self.config.poll_seconds)

    def _lock_layout(self) -> None:
        frame, index = self._pending_frame
        buffer: list[tuple[object, int, float]] = []
        deadline = self._recognition_deadline or (self.now()+self.config.watchdog.recognition_seconds)
        while True:
            if self.now() >= deadline:
                raise LiveError('Seat layout did not stabilize before timeout')
            roi_eff = self._resolve_roi(frame)
            locked = self._active_slots
            if locked is None:
                resolution = self._layout_resolver(roi_eff).observe(frame.image)
                self._log_layout_gate(resolution, index)
                locked = resolution.locked_layout
                if (locked is None and self._layout_hint is not None and
                        (self._layout_hint in resolution.candidates or
                         self._hint_blocked_only_by_counters(resolution))):
                    # The seat combination is match-level state from this run;
                    # one fresh frame must still accept it as a candidate.
                    locked = self._layout_hint
                    self.log('layout_hint_confirmed', active_slots=list(locked), frame_index=index)
            if locked is None:
                if resolution.pending_layout is not None:
                    buffer.append((frame.image.copy(), index, frame.captured_at))
                else:
                    buffer.clear()
            else:
                if self._active_slots is None:
                    self._active_slots = tuple(locked)
                    self.log("layout_locked", active_slots=list(self._active_slots))
                    self._write_checkpoint(source='layout_locked')
                    numeric = getattr(getattr(self.reader, 'metadata_reader', None), 'numeric_ocr', None)
                    if numeric is not None and callable(getattr(numeric, 'enable_async', None)):
                        numeric.enable_async()
                for image, buffered_index, buffered_ts in buffer:
                    self._track(image, roi_eff, self._active_slots,
                                buffered_ts, buffered_index)
                buffer.clear()
                tracked=self._track(frame.image, roi_eff, self._active_slots,
                                    frame.captured_at, index)
                if tracked is not None and self._signal_game_started(tracked[0],frame):
                    return
            frame, index = self._capture()
            self.sleep(self.config.poll_seconds)

    def _signal_game_started(self, observation, frame):
        if self._game_started:return True
        if not self._active_slots:return False
        if not 0 <= self.now()-frame.captured_at <= self.config.decision.max_decision_age:
            return False
        try:
            self.tracker.decision_snapshot(decision_observation(observation,self.tracker.self_slot))
        except ValueError as exc:
            self.log('game_start_recognition_pending',frame_index=observation.frame_index,reason=str(exc))
            return False
        self._game_started=True
        self._recognizing_start=False
        self.log('agent_game_started',frame_index=observation.frame_index,
                 active_slots=list(self._active_slots),hand_count=len(observation.hand),
                 round_id=observation.round_id,source='validated_agent_observation')
        return True

    COUNTER_ISSUES = ('honba:', 'riichi_sticks:', 'remaining_tiles:', 'Metadata reading is incomplete')

    def _hint_blocked_only_by_counters(self, resolution):
        """A rejoining Agent's saved layout is still valid when the only
        problems are the numeric counters (live 212426: an unreadable honba
        digit kept a rejoin off the table for 180 s). Seat glyphs must match."""
        for item in resolution.evaluations:
            if tuple(item.layout) == tuple(self._layout_hint):
                issues = list(item.issues)
                return bool(issues) and all(i.startswith(self.COUNTER_ISSUES) for i in issues) and                     any(not i.startswith('Metadata reading is incomplete') for i in issues)
        return False

    def _log_layout_gate(self, resolution, index):
        status = tuple((item.layout, item.issues) for item in resolution.evaluations)
        if status != self._last_layout_status:
            self._last_layout_status = status
            self.log('layout_gate', frame_index=index,
                     candidates=[asdict(item) for item in resolution.evaluations])

    def _main_loop(self) -> int:
        self._last_activity = self.now()
        while True:
            self.sleep(self.config.poll_seconds)
            try:
                frame, index = self._capture()
            except WindowError as exc:
                self._cancel_decision('window capture failed')
                self.log("window_error", error=str(exc))
                continue
            if self._process_frame(frame, index):
                return 0

    def _win_hint(self):
        """A button candidate resembling ron/tsumo, even if it was rejected
        (e.g. cut by the ROI): the fallback rung must never PASS over it."""
        reader = getattr(self.reader, 'button_reader', None) or self.button_reader
        predictions = getattr(reader, 'last_predictions', None) or {}
        return any(p.get('hint') in ('ron', 'tsumo') for p in predictions.values()
                   if isinstance(p, dict))

    @property
    def local_fallback_enabled(self):
        return (self.execute and self.config.emergency.enabled and
                not self._recognizing_start and self._input_allowed() and
                getattr(self.reader, 'supports_local_input', True) and
                callable(getattr(self.reader, 'read_local', None)))

    def _read_emergency(self, frame, index):
        roi = self._resolve_roi(frame)
        observation, visible, buttons = self.reader.read_local(
            frame.image, roi, self._active_slots or (roi.self_slot,),
            timestamp=frame.captured_at, frame_index=index)
        if self._pending_riichi:
            indices, evidence = self.riichi_selection.read(frame, observation, visible, roi=roi)
            observation = replace(observation, riichi_discard_indices=indices, selection_evidence=evidence)
        return observation, buttons

    def _emergency_tick(self, frame, index):
        """Runs before full-table recognition, using its own lightweight reader."""
        if not self.local_fallback_enabled:
            return False
        try:
            observation, buttons = self._read_emergency(frame, index)
        except (ValueError, RuntimeError) as exc:
            self.log('emergency_reader_error', frame_index=index, error=str(exc))
            return False
        # During a riichi tile selection the rung may wait for the hard end.
        self.turn_watchdog.extended = self._pending_riichi
        due = self.turn_watchdog.observe(observation)
        if (self.turn_watchdog.rebuild_pending and self.turn_watchdog.since is None and
                self._active_slots is not None):
            self.reset_runtime('rebuild during opponent turn after local fallback', restart_policy=True)
            self.turn_watchdog.rebuild_pending = False
            self.log('emergency_state_rebuild', frame_index=index, phase='opponent_turn',
                     global_state_ready=False, next_step='continue_full_table_recognition')
        if not due:
            if self.turn_watchdog.since is None:
                self._last_emergency_blocker = None
            return False
        action, reason = self.turn_watchdog.select(
            observation, buttons, removed_tiles=self.engine.rules.data.get('removed_tiles', ()),
            pending_riichi=self._pending_riichi, win_hint=self._win_hint())
        if action is None:
            if reason != self._last_emergency_blocker:
                self._last_emergency_blocker = reason
                self.log('emergency_unavailable', frame_index=index, reason=reason,
                         since=self.turn_watchdog.since, elapsed=self.now()-self.turn_watchdog.since,
                         countdown=observation.countdown, hand_count=len(observation.hand),
                         drawn_index=observation.drawn_index, buttons=sorted(buttons))
            return False
        key = (action.kind, input_signature(observation))
        if action.kind == 'nuki':
            # A replacement North keeps the signature; repeating kita is right.
            self.turn_watchdog.nuki_clicks += 1
        elif key in self.turn_watchdog.sent:
            return False
        # No late model result may race the emergency click.
        self._last_emergency_blocker = None
        self._cancel_decision('tsumogiri fallback rung', window_closed=True)
        self._decision_epoch += 1
        self.turn_watchdog.rebuild_pending = True
        self.log('emergency_started', frame_index=index, action=action.for_model(),
                 since=self.turn_watchdog.since, elapsed=self.now()-self.turn_watchdog.since)
        fresh, fresh_index = self._capture()
        if self._session_frame(fresh):
            return True
        after, fresh_buttons = self._read_emergency(fresh, fresh_index)
        self.turn_watchdog.observe(after)
        fresh_action, reason = self.turn_watchdog.select(
            after, fresh_buttons, removed_tiles=self.engine.rules.data.get('removed_tiles', ()),
            pending_riichi=self._pending_riichi, win_hint=self._win_hint())
        if (fresh_action != action or tuple(fresh.window) != tuple(frame.window) or
                not same_input_window(observation, after) or
                not 0 <= self.now()-fresh.captured_at <= self.config.emergency.maximum_input_age):
            self.log('emergency_preflight_abort', reason=reason or 'local_window_changed_or_expired')
            return True
        def point(obs, current_buttons):
            if action.button is not None:
                return current_buttons[action.button].center
            box = obs.hand[action.hand_index].box
            if box is None:
                raise ValueError('drawn tile geometry missing')
            return ((box[0]+box[2])/2, (box[1]+box[3])/2)
        try:
            old_point, target = point(observation, buttons), point(after, fresh_buttons)
            if any(abs(a-b)/size > self.config.execution.target_tolerance
                   for a, b, size in zip(old_point, target, fresh.window[2:])):
                raise ValueError('local target moved')
            options = {'hold_seconds': self.config.execution.mouse_press_seconds,
                       'max_age_seconds': self.config.emergency.maximum_input_age}
            if action.kind == 'discard':
                options.update(click_count=self.config.execution.hand_click_count,
                               inter_click_seconds=self.config.execution.inter_click_seconds)
                if not self._pending_riichi and self._riichi_mode_visible(fresh, self._resolve_roi(fresh)):
                    # The client already waits for a riichi tile (live 212426
                    # agent-004 f289): never blind double-click; select a tile.
                    self.turn_watchdog.sent.add(key)
                    self.log('client_in_riichi_selection', action=action.for_model(), source='fallback_rung')
                    self._pending_riichi = True
                    self._riichi_plan = (action.tile, action.hand_index)
                    self._click_planned_riichi_tile(f'emergency-{fresh_index:06d}')
                    return True
            # Commit before input: uncertain acknowledgement never causes a
            # repeated blind click into the next turn.
            self.turn_watchdog.sent.add(key)
            write_image(self.artifacts / 'frames' / f'emergency-{fresh_index:06d}-before.png', fresh.image)
            self._last_click_at = self.now()
            self.window.click(target, expected=fresh, **options)
            self.log('emergency_clicked', frame_index=fresh_index, action=action.for_model(),
                     point=list(target), input_timing=getattr(self.window, 'last_click_timing', None))
        except (ValueError, WindowError) as exc:
            self.log('emergency_failed', reason=str(exc), stage='input')
            return True
        # The rung acted: no deadline from this window may carry into the next.
        self._actionable_since = None
        self._actionable_phase = None
        self._reset_window_clock(self._last_click_at)
        deadline = min(self.now()+self.config.emergency.verification_seconds, self.turn_watchdog.expires)
        result = None
        while self.now() < deadline:
            self.sleep(self.config.poll_seconds)
            current, current_index = self._capture()
            if self._session_frame(current):
                self.log('emergency_resolved', reason='round_result_observed')
                self.turn_watchdog.reset()
                return True
            observed, _ = self._read_emergency(current, current_index)
            result = self.turn_watchdog.verification(action, after, observed)
            if result.status == 'verified':
                write_image(self.artifacts / 'frames' / f'emergency-{fresh_index:06d}-after.png', current.image)
                if action.kind == 'discard':
                    self.turn_watchdog.reset()
                    self._pending_riichi = False
                    self.riichi_selection.reset()
                    self._riichi_round_id = None
                elif action.kind == 'nuki':
                    self.turn_watchdog.reset()
                    self.turn_watchdog.idle_at = self._last_click_at
                self.turn_watchdog.rebuild_pending = True
                self._verified_operation = None
                self._last_activity = current.captured_at
                self.log('emergency_verified', action=action.for_model(), reason=result.reason,
                         evidence=result.evidence, rebuild='at_next_proven_opponent_turn')
                return True
            if result.status == 'failed':
                break
        self.log('emergency_failed', stage='verification', action=action.for_model(),
                 reason=result.reason if result else 'verification_budget_expired')
        self.turn_watchdog.rebuild_pending = True
        return True

    # ------------------------------------------------------------ frame handling

    def _process_frame(self, frame: LiveFrame, index: int) -> bool:
        allowed = self._input_allowed()
        if allowed != self._input_was_allowed:
            if allowed and self._input_was_allowed is False:
                self._on_input_granted()
            self._input_was_allowed = allowed
        if self._session_frame(frame):
            self._handback_since, self._handback_count = None, 0
            self.unknown_review.reset()
            self._frame_unknowns = []
            self._cancel_decision('lifecycle overlay or options setup', window_closed=True)
            if self.session_ui.lifecycle.complete:
                return True
            return self._silence_expired(frame.captured_at)
        if self._emergency_tick(frame, index):
            return False
        roi_eff = self._resolve_roi(frame)
        tracked = self._track(frame.image, roi_eff, self._active_slots,
                              frame.captured_at, index, frame)
        if tracked is None:
            self._handback_since, self._handback_count = None, 0
            self.unknown_review.reset()
            self._frame_unknowns = []
            self._cancel_decision('fresh perception failed')
            self._expire_riichi_selection()
            self._note_issue_frame(frame.captured_at, ["perception_error"])
            return self._silence_expired(frame.captured_at)
        observation, visible, update = tracked
        buttons = self._buttons()
        if update.events:
            self.log("events",
                     frame_index=index,
                     events=[event.to_dict() for event in update.events])
        if update.events or buttons:
            self._last_activity = frame.captured_at
        self._advance_decision_window(update)
        self._update_riichi(update)
        if self._pending_riichi:
            self._window_deadline(observation)
        self._expire_riichi_selection()
        pending = self._pending_decision
        # Once a valid snapshot was submitted, public animations cannot restart
        # or cancel that decision. Local hand/window changes still cancel it.
        raw_observation = observation
        observation = decision_observation(observation, self.tracker.self_slot)
        unavailable = frozenset(observation.unavailable_regions)
        previous = getattr(self, '_last_public_unavailable', frozenset())
        if unavailable != previous:
            self.log('public_fields_updated', frame_index=index,
                     degraded=sorted(unavailable - previous), restored=sorted(previous - unavailable),
                     unavailable=sorted(unavailable), recovery='fresh_frame_retry_without_blocking_model',
                     issues=[i for i in raw_observation.issues if i.split(':', 1)[0] in unavailable])
            self._last_public_unavailable = unavailable
        decision_issues = self.tracker.validator.validate_observation(
            observation, allow_ui_occlusion=True, allow_public_uncertainty=True)
        issues_for_pause = (list(input_issues(observation)) if pending is not None else
                            list(observation.actionable_issues) + list(decision_issues))
        self._note_issue_frame(observation.timestamp, issues_for_pause)
        state = self.tracker.state
        if pending is not None:
            state = pending["snapshot"]
        elif not decision_issues:
            state = self.tracker.decision_snapshot(observation)
        phase = "unknown"
        if state is not None and not (input_issues(observation) if pending else observation.actionable_issues):
            try:
                phase = infer_phase(observation, state,
                                    pending_riichi=self._pending_riichi)
            except (KeyError, ValueError):
                phase = "unknown"
        if phase in ACTIONABLE_PHASES:
            previous = self._actionable_phase
            if (previous is not None and previous != phase and
                    {previous, phase} != {'discard', 'riichi_select'}):
                # A call window and the following own draw are separate turns:
                # never carry a deadline from one into the other.
                self._cancel_decision('operation window changed', window_closed=True)
                self._actionable_since = None
                self._reset_window_clock(observation.timestamp)
            self._actionable_phase = phase
            if self._actionable_since is None:
                self._actionable_since = observation.timestamp
            self._window_deadline(observation)
        elif phase in ("waiting", "result"):
            self._actionable_since = None
            self._actionable_phase = None
            self._reset_window_clock(observation.timestamp)

        unknowns, self._frame_unknowns = self._frame_unknowns, []
        if unknowns:
            if phase == "response":
                self._cancel_decision('unknown response button')
        self._handle_unknown_buttons(frame, index, observation, phase,
            buttons, unknowns, eligible=(update.stable and not update.validation_issues and
                                        not observation.actionable_issues and
                                        0 <= self.now()-frame.captured_at <= self.config.decision.max_decision_age))

        blockers = []
        if pending is not None:
            if input_issues(observation):
                blockers.append('local_input_uncertain')
            if not same_input_window(pending['observation'], observation):
                blockers.append('own_hand_or_action_window_changed')
            if tuple(frame.window) != tuple(pending['frame'].window):
                blockers.append('window_geometry_changed')
        else:
            if decision_issues:
                blockers.append('current_frame_failed_validation')
            # The clock is read once per window and counted down locally
            # (_window_deadline); an unreadable clock frame blocks nothing.
            if state is None:
                blockers.append('no_decision_snapshot')
            elif not state.decision_ready:
                blockers.extend(state.decision_blockers)
        if phase not in ACTIONABLE_PHASES:
            blockers.append('phase_not_actionable:' + phase)
        if unknowns and phase == 'response':
            blockers.append('unknown_action_buttons')
        if not 0 <= self.now()-frame.captured_at <= self.config.decision.max_decision_age:
            blockers.append('observation_expired')
        if not allowed:
            blockers.append('input_not_granted')
            self._observe_handback(frame, observation, phase, state, decision_issues, unknowns)
        gate = (phase, update.stable, tuple(update.validation_issues),
                observation.actionable_issues, bool(state and state.decision_ready), tuple(blockers),
                bool(pending), tuple(sorted(buttons)))
        if gate != self._last_gate_status:
            self._last_gate_status = gate
            self.log('decision_gate', frame_index=index, phase=phase,
                     stable=update.stable, validation_issues=list(update.validation_issues),
                     perception_issues=list(raw_observation.actionable_issues),
                     public_blind_spots=list(observation.unavailable_regions),
                     decision_ready=bool(state and state.decision_ready),
                     ready_for_model=not blockers, blocking_reasons=blockers,
                     decision_validation_issues=list(decision_issues),
                     snapshot_source='submitted_snapshot' if pending else 'single_validated_frame',
                     stability_required_for_decision=False,
                     table_stability=self.tracker.stability_diagnostics,
                     tracker_decision_ready=bool(self.tracker.state and self.tracker.state.decision_ready),
                     tracker_state_frame=self.tracker.state.source_frame if self.tracker.state else None,
                     buttons={'labels': sorted(buttons), 'ready': bool(buttons) and not unknowns,
                              'blocking_reasons': ['unknown_candidates'] if unknowns else
                                                  [] if buttons else ['no_recognized_buttons']},
                     countdown=observation.countdown)
        if pending is not None and not blockers:
            pending['deadline'] = min(pending['deadline'], self._window_deadline(observation))
            if self.now() >= pending['deadline']:
                self._cancel_decision('deadline expired')
            elif pending['future'].done():
                self._pending_decision = None
                self._finish_decision(pending)
        elif not blockers:
            self._decide(frame, observation, visible, buttons, state, phase)
        else:
            self._cancel_decision(';'.join(blockers),
                                  window_closed=phase in ('waiting', 'result'))
        if not blockers and self.once and self._decisions_made:
            return True
        return self._silence_expired(frame.captured_at)

    def _advance_decision_window(self, update) -> None:
        # A resync can hide every intervening waiting frame. Only a proved own
        # draw/discard closes the previous clock; uncertainty alone cannot renew it.
        if (self._pending_decision is not None and
                getattr(update, 'observation', None) is not None and
                same_input_window(self._pending_decision['observation'], update.observation)):
            return  # This committed event only catches up to the submitted frame.
        boundaries = [event for event in update.events
                      if event.player == self.tracker.self_slot and
                      event.certainty == 'observed' and
                      event.type in (GameEventType.DRAW, GameEventType.DISCARD)]
        if not boundaries:
            return
        self._cancel_decision('observed own turn boundary', window_closed=True)
        self._actionable_since = None
        self._reset_window_clock(self._last_click_at)

    def _note_issue_frame(self, timestamp: float, issues: list) -> None:
        if issues:
            if self._issue_since is None:
                self._issue_since = timestamp
            self._clean_streak = 0
            if (not self._paused and
                    timestamp - self._issue_since > self.config.persistent_issue_seconds):
                self._paused = True
                self.log("safe_pause", issues=issues,
                         since_seconds=round(timestamp - self._issue_since, 3))
        else:
            self._issue_since = None
            self._clean_streak += 1
            if self._paused:
                # One validated current frame also suffices to resume decisions.
                self._paused = False
                self.log("resumed", clean_frames=self._clean_streak)

    def _expire_riichi_selection(self):
        """A missed selection is not fatal: the tsumogiri rung or the client's
        timeout discards, and the Agent keeps playing."""
        if not (self._pending_riichi and self.riichi_selection.deadline is not None and
                self.now() >= self.riichi_selection.deadline):
            return
        self._cancel_decision('riichi interaction deadline expired', window_closed=True)
        self.log('riichi_selection_expired', deadline=round(self.riichi_selection.deadline, 3),
                 next_step='tsumogiri_rung_or_client_timeout')
        self._pending_riichi = False
        self._riichi_plan = None
        self._riichi_round_id = None
        self.riichi_selection.reset()

    def _apply_own_state(self, observation):
        """Own riichi comes from our own verified declaration, never from the
        stick detector: an opponent's riichi must not make our own status
        unknown (that locked every decision to tsumogiri). A new hand also
        restores the shared time reserve."""
        round_id = observation.round_id
        if round_id is not None and round_id != self._last_round_id:
            if self._last_round_id is None and self.join:
                self._joined_round = round_id
                self.match_timer.unknown_hand(round_id)
            else:
                self.match_timer.new_hand(round_id)
            self._last_round_id = round_id
        current = round_id if round_id is not None else self._last_round_id
        slot = self.tracker.self_slot
        players = []
        for player in observation.players:
            if player.slot == slot:
                if self._own_riichi_round is not None and current == self._own_riichi_round:
                    own, source = True, 'own_verified_declaration'
                elif current is not None and current == self._joined_round:
                    # Joined mid-hand: our own sideways river tile is the evidence.
                    sideways = any(t.tile is not None and t.orientation.startswith('horizontal')
                                   for t in player.river)
                    own = player.riichi if player.riichi is not None else sideways
                    source = 'joined_hand_detector_or_own_river'
                else:
                    own, source = False, 'no_own_declaration_this_hand'
                player = replace(player, riichi=own,
                                 riichi_evidence={**player.riichi_evidence, 'own_status_source': source})
            players.append(player)
        return replace(observation, players=tuple(players))

    def _update_riichi(self, update) -> None:
        self_slot = self.tracker.self_slot
        for event in update.events:
            if event.type == GameEventType.RIICHI and event.player == self_slot:
                self._own_riichi_round = self._last_round_id
                self._pending_riichi = False
                self.riichi_selection.reset()
                self._riichi_round_id = None
        state = self.tracker.state
        if state is None:
            return
        try:
            if state.player(state.self_slot).riichi:
                self._pending_riichi = False
                self.riichi_selection.reset()
                self._riichi_round_id = None
        except KeyError:
            pass
        if (self._riichi_round_id is not None and
                state.round_id != self._riichi_round_id):
            self._pending_riichi = False
            self.riichi_selection.reset()
            self._riichi_round_id = None

    def _silence_expired(self, timestamp: float) -> bool:
        if (self._last_activity is not None and
                timestamp - self._last_activity > self.config.match_end_silence_seconds):
            self.log("match_end_suspected",
                     silence_seconds=round(timestamp - self._last_activity, 3))
            if not self.supervised:
                raise LiveError('Activity watchdog expired without verified match completion')
            # The independent observer certifies the lobby and every live
            # operation deadline; silence alone only warrants a resync.
            self.reset_runtime('activity watchdog without lifecycle progress', restart_policy=True)
            self._last_activity = timestamp
        return False

    # --------------------------------------------------------------- decisions

    def _decision_blocked(self, reason, observation):
        marker = (reason, input_signature(observation))
        if marker != self._last_decision_blocker:
            self._last_decision_blocker = marker
            self.log('decision_blocked', frame_index=observation.frame_index, reason=reason)

    def _reset_window_clock(self, idle_at=None):
        """Close the operation window; ``idle_at`` is the last moment it was
        proven closed, from which a later base-time fallback counts."""
        self.decision_budget.reset()
        if self.operation_clock is not None:
            self.operation_clock.reset()
        self._idle_at = idle_at

    def _window_deadline(self, observation):
        start = self._actionable_since if self._actionable_since is not None else observation.timestamp
        self.decision_budget.start(start)
        self.decision_budget.update(observation.countdown, observation.timestamp)
        deadline = self.decision_budget.deadline(observation.timestamp,
                                                 self.config.decision.max_decision_age)
        if self.operation_clock is not None:
            self.operation_clock.start(start, self._idle_at)
            self.operation_clock.observe(observation.timestamp, observation.countdown)
            if self._pending_riichi and self.riichi_selection.deadline is not None:
                # The tile selection may use the whole remaining reserve.
                return min(self.riichi_selection.deadline,
                           observation.timestamp+self.config.decision.max_decision_age)
            # The hand's 7+20 timer is the authority; the older per-window
            # budget stays only as a diagnostic.
            deadline = min(self.operation_clock.decision_due,
                           observation.timestamp+self.config.decision.max_decision_age)
        elif self._pending_riichi and self.riichi_selection.deadline is not None:
            deadline = min(deadline, self.riichi_selection.deadline)
            self.riichi_selection.deadline = deadline
        return deadline

    def _click_deadline(self, decision_deadline):
        """A ready answer may still be clicked after its decision deadline as
        long as the client's real deadline allows (live 204720: five model
        answers were discarded 0.3 s late with 20 s of reserve left). The
        loop is single-threaded, so the fallback cannot act in between."""
        clock = self.operation_clock
        if clock is None or not clock.open or clock.last_safe_input is None:
            return decision_deadline
        return max(decision_deadline, clock.last_safe_input)

    def _riichi_mode_visible(self, frame, roi):
        marker = getattr(self.riichi_selection, 'cancel_marker', None)
        if marker is None or getattr(marker, 'template', None) is None:
            return False
        try:
            return bool(marker.read(frame, roi).get('accepted'))
        except (KeyError, ValueError, TypeError, AttributeError):
            return False

    def _selection_deadline(self, fallback):
        clock = self.operation_clock
        return clock.selection_due if clock is not None and clock.open else fallback

    def _decision_record(self, pending, **fields):
        return {
            "request_id": pending["request_id"],
            "timestamp": pending["observation"].timestamp,
            "frame_index": pending["observation"].frame_index,
            "state_version": pending["observation"].fingerprint,
            "source_image": pending.get("source_image"),
            "window": list(pending["frame"].window),
            "phase": pending["phase"], "deadline": pending["deadline"],
            "countdown": pending["observation"].countdown,
            "decision_budget": pending["budget"],
            "rule_card": self._rule_card,
            "hand_epoch": pending["key"][0],
            "state": pending["domain"].for_model(),
            "actions": [action.for_model() for action in pending["actions"]],
            **fields,
        }

    def _cancel_decision(self, reason, *, window_closed=False):
        pending, self._pending_decision = self._pending_decision, None
        if pending is not None:
            self.decision_worker.cancel(pending["future"])
            self.log("decision_cancelled", request_id=pending["request_id"], reason=reason)
            self._log_decision(self._decision_record(
                pending, mode="cancelled", reason=reason, action=None))
        if window_closed:
            self._last_request_key = None
            self._last_decision_fingerprint = None
            self._last_unknown_pass_key = None

    def _model_candidates(self, actions, domain_state):
        """Efficiency guardrail (live 212426 f604: a triplet was broken for a
        worse shanten). Without a visible riichi threat, discards that raise
        shanten are not offered; defence stays available under threat."""
        if not self.config.decision.shanten_guard:
            return actions
        o = domain_state.observation
        threat = any(r is not False for r in o.riichi[1:]) and (o.riichi_sticks is None or o.riichi_sticks > 0
                                                                 or any(r is True for r in o.riichi[1:]))
        discards = [a for a in actions if a.kind == 'discard' and type(a.features.get('shanten')) is int]
        if threat or not discards:
            return actions
        best = min(a.features['shanten'] for a in discards)
        kept = [a for a in actions if a.kind != 'discard' or a.features.get('shanten') == best
                or type(a.features.get('shanten')) is not int]
        return kept

    def _immediate_choice(self, actions, phase):
        """Choices that never wait for the model: a legal win is always taken,
        and a riichi tile chosen together with the declaration is selected."""
        if self.config.decision.always_win:
            win = next((a for a in actions if a.kind in ('tsumo', 'ron')), None)
            if win is not None:
                return win, 'always_win'
        if phase == 'riichi_select' and self._riichi_plan is not None:
            tile, index = self._riichi_plan
            options = [a for a in actions if a.kind == 'riichi' and a.hand_index is not None]
            planned = (next((a for a in options if a.hand_index == index and a.tile == tile), None)
                       or next((a for a in options if a.tile == tile), None)
                       or (options[0] if options else None))
            if planned is not None:
                return planned, 'riichi_plan'
        return None

    def _decide(self, frame: LiveFrame, observation: FrameObservation, visible,
                buttons: dict, state: MahjongState, phase: str) -> None:
        self._last_activity = observation.timestamp
        current_deadline = self._window_deadline(observation)
        if self._pending_riichi:
            if self.riichi_selection.discard_sent or self.riichi_selection.deadline is None:
                self._cancel_decision('riichi discard already submitted or baseline missing')
                self._decision_blocked('riichi_discard_already_sent_or_baseline_missing', observation)
                return
            current_deadline = min(current_deadline, self.riichi_selection.deadline)
        try:
            domain_state = build_domain_state(
                state, observation, visible, buttons,
                pending_riichi=self._pending_riichi, window=frame.window)
        except BridgeError as exc:
            self._cancel_decision('fresh bridge failed')
            self.log("bridge_error", error=str(exc))
            self._decision_blocked('bridge_failed:' + str(exc), observation)
            return
        if (self._verified_operation is not None and observation.timestamp -
                self._verified_operation['after'].timestamp > self.config.decision.max_decision_age):
            self._verified_operation = None
        domain_state = apply_execution_context(
            domain_state, observation, self._verified_operation, self.tracker.self_slot)
        domain_state.rules_context = dict(self.engine.rules.data)
        actions = self.engine.legal_actions(domain_state)
        if not actions:
            self._cancel_decision('no legal actions', window_closed=True)
            self._decision_blocked('no_legal_actions', observation)
            return
        key = (self._decision_epoch, phase, observation.round_id, input_signature(observation), tuple(frame.window),
               tuple(action.id for action in actions))
        if self._kan_blocked is not None:
            if input_signature(observation) == self._kan_blocked:
                self._decision_blocked('kan_unresolved_wait_for_hand_change', observation)
                return
            self._kan_blocked = None
        immediate = self._immediate_choice(actions, phase)
        if immediate is not None and self._pending_decision is None and key != self._last_request_key:
            action, rule = immediate
            self._last_request_key = key
            future = Future()
            future.set_result(ChoiceResult(action, {'rung': rule}, {'backend': rule, 'action_id': action.id}))
            pending = {"request_id": uuid4().hex, "key": key, "future": future,
                       "frame": frame, "observation": observation, "domain": domain_state,
                       "actions": actions, "phase": phase, "deadline": current_deadline, "snapshot": state,
                       "budget": self.decision_budget.to_dict(), "source_image": None}
            self.log("decision_submitted", request_id=pending["request_id"], phase=phase,
                     frame_index=observation.frame_index, deadline=current_deadline, rule=rule)
            self._finish_decision(pending)
            return
        pending = self._pending_decision
        if pending is not None:
            pending["deadline"] = min(pending["deadline"], current_deadline)
            if key != pending["key"]:
                self._cancel_decision('state, geometry, phase or legal actions changed')
            elif self.now() >= pending["deadline"]:
                self._cancel_decision('deadline expired')
            elif pending["future"].done():
                self._pending_decision = None
                self._finish_decision(pending)
            return
        if key == self._last_request_key or observation.fingerprint == self._last_decision_fingerprint:
            self._decision_blocked('already_requested_in_this_window', observation)
            return
        if self.decision_budget.pending:
            self._decision_blocked('countdown_confirmation_pending', observation)
            return
        deadline = current_deadline
        if self.now() >= deadline:
            if self._last_decision_blocker != ('decision_deadline_expired', input_signature(observation)):
                self.log("decision_expired", deadline=round(deadline, 3))
            self._decision_blocked('decision_deadline_expired', observation)
            return
        future = self.decision_worker.submit(domain_state, self._model_candidates(actions, domain_state),
                                             deadline=deadline)
        if future is None:
            self._decision_blocked('previous_request_still_unwinding', observation)
            return  # A cancelled request is still unwinding; never queue another.
        self._last_request_key = key
        self._last_decision_blocker = None
        self._pending_decision = {
            "request_id": uuid4().hex, "key": key, "future": future,
            "frame": frame, "observation": observation, "domain": domain_state,
            "actions": actions, "phase": phase, "deadline": deadline, "snapshot": state,
            "budget": self.decision_budget.to_dict(),
        }
        evidence = self.artifacts / "frames" / f'{self._pending_decision["request_id"]}-request.png'
        try:
            write_image(evidence, frame.image)
        except (OSError, ValueError) as exc:
            if not self.supervised:
                self._cancel_decision('request image could not be saved')
                raise LiveError('Cannot save decision evidence') from exc
            # Evidence is diagnostic; losing it must not forfeit the turn.
            self.log('evidence_write_failed', request_id=self._pending_decision["request_id"], error=str(exc))
            self._pending_decision["source_image"] = None
        else:
            self._pending_decision["source_image"] = str(evidence.relative_to(self.artifacts))
        self.log("decision_submitted", request_id=self._pending_decision["request_id"],
                 phase=phase, frame_index=observation.frame_index, deadline=deadline)

    def _finish_decision(self, pending):
        frame, observation = pending["frame"], pending["observation"]
        domain_state, actions = pending["domain"], pending["actions"]
        deadline, phase = pending["deadline"], pending["phase"]
        try:
            choice = pending["future"].result()
        except Exception as exc:
            error = str(exc) if isinstance(exc, StrategyError) else type(exc).__name__
            if self.local_fallback_enabled:
                self.turn_watchdog.rebuild_pending = True
            self._log_decision(self._decision_record(
                pending, mode="failed", error=error, action=None,
                timing=getattr(exc, "timing", None), backend=getattr(exc, "backend", None)))
            self.log("decision_failed", request_id=pending["request_id"], error=error)
            return
        action, timing, result = choice.action, choice.timing, choice.backend
        if timing:
            self.log("model_timing", **timing)
        if result:
            self.log("model_result", **result)
        if self.now() >= self._click_deadline(deadline):
            self.log('decision_expired', reason='model returned after deadline')
            self._log_decision(self._decision_record(pending, mode="expired", action=None))
            return
        if action not in actions:
            self._log_decision(self._decision_record(pending, mode="invalid_action", action=None))
            self.log('decision_failed', error='model selected an action outside the legal set')
            return
        record = self._decision_record(pending, action=action.for_model(), backend=result, timing=timing)
        effective_execute = self.execute and not self._paused and self._input_allowed()
        if not effective_execute:
            record["mode"] = "paused_dry_run" if self._paused else "dry_run"
            self._log_decision(record)
            self._last_decision_fingerprint = observation.fingerprint
            self._decisions_made += 1
            self.log("decision_dry_run", action=action.for_model(), phase=phase,
                     paused=self._paused)
            return
        self._decisions_made += 1
        try:
            outcome = self._execute_action(frame, observation, action, domain_state,
                                           deadline=deadline)
        except KeyboardInterrupt:
            record["mode"] = "cancelled"
            record["reason"] = "stop_during_execution_or_verification"
            self._log_decision(record)
            raise
        except WindowError as exc:
            # Not fatal (live 212426: a minimised / background window killed
            # the Agent): keep observing; input resumes once the window is usable.
            record["mode"] = "execution_unknown"
            record["reason"] = str(exc)
            self._log_decision(record)
            self.log('input_unavailable', reason=str(exc), action=action.for_model(),
                     next_step='keep_observing_fallback_rung_covers_this_turn')
            return
        record["mode"] = outcome
        self._log_decision(record)
        if ((outcome == 'verified' and action.kind in ('discard', 'pon', 'kan', 'nuki', 'riichi',
                                                       'riichi_declare')) or
                outcome == 'window_resolved'):
            # A local completion releases this clock even if event tracking lags.
            self._cancel_decision('local operation window completed', window_closed=True)
            self._actionable_since = None
            self._reset_window_clock(self._last_click_at)
            if action.kind == 'riichi' or outcome == 'window_resolved':
                self._pending_riichi = False
                self._riichi_round_id = None
                self.riichi_selection.reset()
            if action.kind in ('discard', 'riichi') or outcome == 'window_resolved':
                self._verified_operation = None
            if outcome == 'verified' and action.kind in ('discard', 'riichi'):
                self.turn_watchdog.reset()
            elif outcome == 'verified' and action.kind in ('nuki', 'kan'):
                # The replacement draw is a fresh turn with a fresh base: the
                # fallback must not keep counting from before the kita
                # (live 204720/212426: 7 replacement tiles auto-discarded).
                self.turn_watchdog.reset()
                self.turn_watchdog.idle_at = self._last_click_at
        elif self.local_fallback_enabled:
            self.turn_watchdog.rebuild_pending = True
        if action.kind == 'nuki' and outcome == 'verification_unknown':
            # A replacement North leaves the hand identical; deciding again is
            # right either way (retry a lost click or declare the next North).
            self._last_request_key = None
            self._last_decision_fingerprint = None
        if action.kind == 'kan' and outcome in ('verification_unknown', 'verification_failed'):
            # A surviving candidate overlay can retain the original hand. Do
            # not reinterpret the missing button as an ordinary discard turn.
            # Not fatal, but never click into a possibly surviving kan overlay:
            # the model gets no new input until the own hand visibly changes.
            self._kan_blocked = input_signature(observation)
            self.log('kan_unresolved', outcome=outcome,
                     next_step='no_model_input_until_hand_changes')

    def _execute_action(self, frame: LiveFrame, observation: FrameObservation,
                        action: Action, domain_state: State, *, deadline=None) -> str:
        sequence = self._decisions_made
        write_image(self.artifacts / "frames" / f"{sequence:05d}-before.png",
                    frame.image)
        try:
            point_before = action_point(action, domain_state)
            fresh_frame, fresh_index = self._capture()
        except (LiveError, WindowError) as exc:
            self.log("preflight_abort", reason=str(exc))
            return "preflight_abort"
        roi_eff = self._resolve_roi(fresh_frame)
        tracked = self._track(fresh_frame.image, roi_eff, self._active_slots,
                              fresh_frame.captured_at, fresh_index, fresh_frame)
        if tracked is None:
            self.log("preflight_abort", reason="fresh observation failed")
            return "preflight_abort"
        fresh_observation, fresh_visible, _ = tracked
        try:
            if self._frame_unknowns and domain_state.observation.phase == 'response':
                raise LiveError('Fresh action buttons contain unknown candidates')
            fresh_domain = fresh_input_domain(
                domain_state, fresh_observation, fresh_visible, self._buttons(), fresh_frame.window)
            point = preflight(action, observation, fresh_observation, fresh_domain,
                              frame.window, fresh_frame.window, self.config,
                              self.engine, point_before=point_before,
                              now=self.now())
        except (BridgeError, LiveError, ValueError) as exc:
            self.log("preflight_abort", reason=str(exc))
            return "preflight_abort"
        fresh_deadline = self._window_deadline(fresh_observation)
        if self.now() >= self._click_deadline(min(deadline if deadline is not None else fresh_deadline,
                                                  fresh_deadline)):
            self.log('preflight_abort', reason='decision deadline passed before click')
            return 'preflight_abort'
        if action.kind == 'discard' and not self._pending_riichi and self._riichi_mode_visible(fresh_frame, roi_eff):
            # The client is already selecting a riichi tile (e.g. a manual
            # riichi click): a plain double-click would act on the wrong tile
            # (live 212426 agent-004 f289/f299). Finish it with this tile.
            self.log('client_in_riichi_selection', action=action.for_model())
            self._pending_riichi = True
            self._riichi_plan = (action.tile, action.hand_index)
            return self._click_planned_riichi_tile(sequence)
        # Persist the exact input baseline before sending any input.
        write_image(self.artifacts / "frames" / f"{sequence:05d}-before.png", fresh_frame.image)
        if action.kind == "riichi_declare" or (action.kind == "riichi" and action.hand_index is None):
            try:
                # Choosing the tile may use the hand's whole remaining reserve.
                self.riichi_selection.begin(
                    fresh_frame, fresh_observation, fresh_visible,
                    self._selection_deadline(min(deadline if deadline is not None else fresh_deadline,
                                                 fresh_deadline)), roi=roi_eff)
            except ValueError as exc:
                self.log('preflight_abort', reason=str(exc))
                return 'preflight_abort'
        elif action.kind == "riichi":
            # Sending a selection is terminal for this interaction even if its
            # acknowledgement is lost. A changed mask cannot authorize a retry.
            if self.riichi_selection.discard_sent:
                self.log('preflight_abort', reason='riichi selection already sent')
                return 'preflight_abort'
            self.riichi_selection.discard_sent = True
        click_options = {'hold_seconds': self.config.execution.mouse_press_seconds}
        if (action.hand_index is not None and action.kind in ('discard', 'riichi')
                and action.kind != 'riichi_declare' and self.config.execution.hand_click_count == 2):
            click_options.update(click_count=2,
                                 inter_click_seconds=self.config.execution.inter_click_seconds)
        self._last_click_at = self.now()
        self.window.click(point, expected=fresh_frame, **click_options)
        self._last_decision_fingerprint = observation.fingerprint
        if action.kind == "riichi_declare" or (action.kind == "riichi" and action.hand_index is None):
            self._pending_riichi = True
            self._riichi_plan = ((action.tile, action.hand_index)
                                 if action.kind == "riichi_declare" else None)
            self._riichi_round_id = (self.tracker.state.round_id
                                     if self.tracker.state is not None else None)
        self.log("clicked", action=action.for_model(),
                 input_timing=getattr(self.window, 'last_click_timing', None),
                 point=[round(point[0], 1), round(point[1], 1)])
        if action.kind == "riichi_declare":
            # Verify the button part; the planned tile is selected next frame.
            action = Action('button:riichi', 'riichi', button='riichi',
                            features={'legality_source': 'enabled_client_button',
                                      'planned_tile': self._riichi_plan})
            outcome = self._verify(action, fresh_observation, sequence)
            if outcome != 'selection_open' and self._pending_riichi and self._riichi_plan is not None:
                # live 212426 f60: dimmed tiles made the selection unprovable
                # and the declaration was never finished.
                return self._click_planned_riichi_tile(sequence)
            return outcome
        return self._verify(action, fresh_observation, sequence)

    def _click_planned_riichi_tile(self, sequence):
        """Finish a sent riichi when the selection mask cannot be proven: the
        riichi button is gone, so the client waits for a tile. The planned
        tile was tenpai-legal when chosen; the client ignores a non-selectable
        tile, so one click cannot discard illegally."""
        tile, index = self._riichi_plan
        if not callable(getattr(self.reader, 'read_local', None)) or not getattr(self.reader, 'supports_local_input', True):
            return 'verification_unknown'
        try:
            frame, frame_index = self._capture()
            observation, buttons = self._read_emergency(frame, frame_index)
        except (ValueError, RuntimeError, WindowError) as exc:
            self.log('riichi_planned_tile_unavailable', reason=str(exc))
            return 'verification_unknown'
        hand = [t.tile for t in observation.hand]
        if 'riichi' in buttons or (observation.action_buttons and 'riichi' in observation.action_buttons):
            self.log('riichi_planned_tile_unavailable', reason='riichi button still visible')
            return 'verification_unknown'
        if not (0 <= index < len(hand) and hand[index] == tile):
            index = next((i for i, t in enumerate(hand) if t == tile), None)
        if index is None or observation.hand[index].box is None:
            self.log('riichi_planned_tile_unavailable', reason='planned tile not found in hand')
            return 'verification_unknown'
        box = observation.hand[index].box
        point = ((box[0]+box[2])/2, (box[1]+box[3])/2)
        options = {'hold_seconds': self.config.execution.mouse_press_seconds}
        if self.config.execution.hand_click_count == 2:
            options.update(click_count=2, inter_click_seconds=self.config.execution.inter_click_seconds)
        self.riichi_selection.discard_sent = True
        self._last_click_at = self.now()
        try:
            self.window.click(point, expected=frame, **options)
        except WindowError as exc:
            self.log('riichi_planned_tile_failed', reason=str(exc))
            return 'verification_unknown'
        self.log('riichi_planned_tile_clicked', tile=tile, hand_index=index, frame_index=frame_index,
                 reason='selection_mask_unproven')
        before = sum(t == tile for t in hand)
        deadline = self.now()+self.config.execution.verification_timeout
        while self.now() < deadline:
            self.sleep(self.config.poll_seconds)
            try:
                current, current_index = self._capture()
                after, _ = self._read_emergency(current, current_index)
            except (ValueError, RuntimeError, WindowError):
                continue
            tiles = [t.tile for t in after.hand]
            if len(tiles) == len(hand)-1 and sum(t == tile for t in tiles) == before-1:
                self._own_riichi_round = after.round_id or self._last_round_id
                self._pending_riichi = False
                self._riichi_plan = None
                self.riichi_selection.reset()
                self.log('verified', action_id=f'riichi_discard:{index}:{tile}', status='verified',
                         reason='planned_riichi_tile_left_hand', evidence={'mask': 'unproven'})
                return 'verified'
        return 'verification_unknown'

    def _verify(self, action: Action, before: FrameObservation,
                sequence: int | str) -> str:
        deadline = self.now() + self.config.execution.verification_timeout
        if action.kind == 'riichi' and self.riichi_selection.deadline is not None:
            deadline = min(deadline, self.riichi_selection.deadline)
        stem = f"{sequence:05d}" if isinstance(sequence, int) else sequence
        last_frame = None
        last_after = None
        result = None
        assessed_frame = None
        def record(status, reason, evidence=None):
            if status == 'resolved':
                # Also covers the unknown-button PASS path, which does not
                # return through _finish_decision to release its old clock.
                self._cancel_decision('response or result window resolved', window_closed=True)
                self._actionable_since = None
                self._reset_window_clock(self._last_click_at)
                self._verified_operation = None
            if (status == 'verified' and action.kind in ('pon', 'kan', 'nuki') and
                    last_after is not None):
                self._verified_operation = {'kind': action.kind, 'after': last_after,
                                            'evidence': evidence or {}}
            path = self.artifacts / "frames" / f"{stem}-verification.json"
            payload = {"action": action.for_model(), "status": status, "reason": reason,
                       "evidence": evidence or {}, "before": before.to_dict(),
                       "after": last_after.to_dict() if last_after is not None else None,
                       "assessed_frame_index": assessed_frame,
                       "before_image": f"frames/{stem}-before.png",
                       "after_image": f"frames/{stem}-after.png" if last_frame is not None else None}
            if last_frame is not None:
                write_image(self.artifacts / "frames" / f"{stem}-after.png", last_frame.image)
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            event = {"verified": "verified", "resolved": "window_resolved", "unknown": "verification_unknown",
                     "failed": "verification_failed", "cancelled": "verification_cancelled", "pending": "verification_pending"}[status]
            self.log(event,
                     action_id=action.id, status=status, reason=reason,
                     evidence=evidence or {}, verification=str(path.relative_to(self.artifacts)))
            return {"verified": "verified", "resolved": "window_resolved", "unknown": "verification_unknown",
                    "failed": "verification_failed", "cancelled": "cancelled", "pending": "selection_open"}[status]
        def observe_after_click():
            nonlocal last_frame, last_after, result, assessed_frame
            while self.now() < deadline:
                self.sleep(self.config.poll_seconds)
                try:
                    frame, index = self._capture()
                except WindowError:
                    continue
                last_frame = frame
                last_after = None
                if self.session_ui is not None and self.session_ui.ui.confirm(frame.image):
                    write_image(self.artifacts / 'frames' / f'{stem}-result.png', frame.image)
                    self.log('result_overlay_observed', action_id=action.id,
                             outcome='resolution visible; winner/action attribution unverified')
                    return record('resolved', 'round_resolution_observed',
                                  {'winner_attribution': 'unknown', 'action_attribution': 'unknown'})
                roi_eff = self._resolve_roi(frame)
                tracked = self._track(frame.image, roi_eff, self._active_slots,
                                      frame.captured_at, index, frame)
                if tracked is None:
                    continue
                after = tracked[0]
                last_after = after
                if action.kind == 'riichi':
                    if self.now() >= self._window_deadline(after):
                        return record('unknown', 'riichi_interaction_deadline_expired')
                if (action.kind == 'riichi' and action.hand_index is None and not input_issues(after) and
                        after.riichi_discard_indices is not None):
                    self.log('selection_open', action_id=action.id)
                    return record('pending', 'riichi_discard_selection_open')
                result = (VerificationResult('unknown', 'response_buttons_unrecognized')
                          if action.kind == 'pass' and self._frame_unknowns else
                          assess_execution_action(action, before, after, self.tracker.self_slot))
                assessed_frame = after.frame_index
                if result.status in ("verified", "failed", "resolved"):
                    if result.status == 'verified' and action.kind == 'riichi':
                        if action.hand_index is not None:
                            # The selected tile left the hand: we are in riichi.
                            self._own_riichi_round = after.round_id or self._last_round_id
                            self._riichi_plan = None
                        self._update_riichi(tracked[2])
                    return record(result.status, result.reason, result.evidence)
            if last_frame is not None:
                write_image(self.artifacts / "frames" / f"{stem}-unverified.png",
                            last_frame.image)
            return record("unknown", result.reason if result is not None else "no_usable_post_action_observation",
                          result.evidence if result is not None else None)

        try:
            return observe_after_click()
        except KeyboardInterrupt:
            record("cancelled", "stop_requested_after_click")
            raise
