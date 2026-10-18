"""Live Steam-client CLI: doctor / calibrate-menu / observe / run / collect."""

from __future__ import annotations

import argparse
from importlib.metadata import PackageNotFoundError, version
import json
import math
from pathlib import Path
import re
import time

from ..calibration.debug import export_debug, export_projection_trials
from ..calibration.frames import write_image
from ..calibration.roi_config import ROIConfig
from ..domain import Action, Observation, State
from ..jev import JevSettings
from ..rules import Rules
from ..state import MahjongStateTracker
from ..strategy import ModelSettings
from ..window import WindowError, list_windows
from .buttons import ActionButtonReader
from .capture import LiveWindow, roi_for_frame
from .config import ROOT, LiveConfig
from .loop import LiveAgentLoop, LiveError, build_policy
from .menu import MenuConfig, NavigationError, calibrate_menu

DEFAULT_CONFIG = "configs/live.steam.yaml"


def _dump(value) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def _classifier_report(config: LiveConfig) -> dict:
    report = {"requested": config.classifier}
    try:
        import torch
        report["torch"] = version("torch")
        report["cuda_available"] = bool(torch.cuda.is_available())
    except (ImportError, PackageNotFoundError):
        report["torch"] = "missing"
        report["cuda_available"] = False
    try:
        version("torchvision")
        report["torchvision"] = version("torchvision")
    except PackageNotFoundError:
        report["torchvision"] = "missing"
    requested = config.classifier
    report["resolved_backend"] = (
        requested if requested != "auto"
        else ("cnn" if report["torch"] != "missing" else "template"))
    report["cnn_checkpoint"] = str(config.cnn_checkpoint)
    report["cnn_checkpoint_exists"] = config.cnn_checkpoint.is_file()
    report["tile_templates_exist"] = config.tile_templates.is_dir()
    report["glyph_templates_exist"] = config.glyph_templates.is_dir()
    return report


def cmd_doctor(args) -> int:
    config = LiveConfig.load(args.config)
    result: dict = {"config": str(args.config)}
    window = None
    rect = None
    try:
        hwnd = LiveWindow.find(config.window_title_regex, config.window_class_regex)
        candidate = LiveWindow(hwnd)
        rect = candidate.window_rect()
        window = candidate
        result["window"] = {"hwnd": hwnd, "rect": list(rect)}
    except (WindowError, OSError) as exc:
        result["window"] = {"error": str(exc)}
        try:
            pattern = re.compile(config.window_title_regex)
            result["window"]["candidates"] = [
                item["title"] for item in list_windows() if pattern.search(item["title"])]
        except (WindowError, OSError) as list_exc:
            result["window"]["list_error"] = str(list_exc)
    roi = ROIConfig.load(config.roi)
    if rect is not None:
        result["roi_assessment"] = roi.assess_target_size((rect[2], rect[3])).to_dict()
    else:
        result["roi_assessment"] = {"error": "window unavailable"}
    result["classifier"] = _classifier_report(config)
    reader = ActionButtonReader(config.buttons, config.button_templates)
    result["buttons"] = {"templates": str(config.button_templates),
                         "known_labels": list(reader.known_labels)}
    menu = MenuConfig.load(config.menu)
    result["menu"] = {"ready": menu.ready,
                      "steps": [step.name for step in menu.steps],
                      "table_timeout_seconds": menu.table_timeout_seconds}
    jev = JevSettings.load(ROOT)
    deepseek = ModelSettings.load(ROOT)
    result["decision"] = {
        "policy": config.decision.policy,
        "jev_key_configured": bool(jev.api_key),
        "deepseek_key_configured": bool(deepseek.api_key),
        "jev_proxy_configured": config.decision.jev_proxy is not None,
    }
    if args.check_api:
        policy = build_policy(config, ROOT)
        try:
            probe = State(Observation(time.monotonic(), (0, 0, 1, 1)),
                          True, ((), (), ()))
            diagnostic = Action("diagnostic", "diagnostic")
            started = time.monotonic()
            result["api_choice"] = policy.choose(probe, [diagnostic]).id
            result["latency_seconds"] = round(time.monotonic() - started, 3)
            result["timing"] = getattr(policy, "last_timing", None)
            result["decision_result"] = getattr(policy, "last_result", None)
        except RuntimeError as exc:
            result["api_error"] = str(exc)
        finally:
            if hasattr(policy, "close"):
                policy.close()
    _dump(result)
    return 0


def cmd_calibrate_menu(args) -> int:
    config = LiveConfig.load(args.config)
    window = LiveWindow(LiveWindow.find(config.window_title_regex, config.window_class_regex))
    window.activate()
    saved = calibrate_menu(window, args.output)
    for path in saved:
        print(path)
    print(f"Saved {len(saved)} frames; fill configs/menu.steam.yaml click positions.")
    return 0


def cmd_observe(args) -> int:
    config = LiveConfig.load(args.config)
    window = LiveWindow(LiveWindow.find(config.window_title_regex, config.window_class_regex))
    window.restore_if_minimized()
    frame = window.capture()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    write_image(output / "frame.png", frame.image)
    size = (int(frame.image.shape[1]), int(frame.image.shape[0]))
    roi = ROIConfig.load(config.roi)
    try:
        roi_eff = roi_for_frame(roi, size, config.projection)
    except ValueError as exc:
        export_projection_trials(frame.image, roi, output / "projection")
        raise LiveError(
            f"{exc}; exported projection candidates to {output / 'projection'} — "
            "review the crops and set projection in the live config") from exc

    from ..perception import GlyphTemplateClassifier, SeatLayoutResolver, VisionConfig
    from ..state import FrameObservationReader
    from .classifier import build_tile_classifier
    from .countdown import CountdownReader

    vision = VisionConfig.load(config.vision)
    countdown_reader = CountdownReader.load(config.countdown, ROOT, Rules.load(config.rules).data)
    tile_classifier = build_tile_classifier(
        config.classifier, vision, cnn_checkpoint=config.cnn_checkpoint,
        template_root=config.tile_templates, expected_metadata=config.cnn_expected)
    button_reader = ActionButtonReader(config.buttons, config.button_templates)
    reader = FrameObservationReader(
        vision, tile_classifier,
        GlyphTemplateClassifier(config.glyph_templates, vision),
        button_reader=button_reader, read_scores=config.read_scores, countdown_reader=countdown_reader)
    tracker = MahjongStateTracker.load(config.rules,
                                       stable_frames=config.stable_frames)
    resolver = SeatLayoutResolver(
        roi_eff, reader.metadata_reader,
        starting_points=tracker.validator.starting_points,
        confirmations_required=config.layout_confirmations)
    resolution = resolver.observe(frame.image)
    layout = resolution.locked_layout or (
        resolution.candidates[0] if resolution.candidates else None)
    report: dict = {
        "frame_size": list(size),
        "classifier_manifest": getattr(tile_classifier, "manifest", None),
        "countdown": countdown_reader.read(frame.image, roi_eff).to_dict() if countdown_reader else None,
        "countdown_manifest": countdown_reader.manifest if countdown_reader else None,
        "layout_locked": list(resolution.locked_layout) if resolution.locked_layout else None,
        "layout_candidates": [list(item) for item in resolution.candidates],
        "layout_issues": list(resolution.issues),
        "observation": None,
        "update": None,
    }
    if layout is not None:
        observation, _ = reader.read_with_visible(
            frame.image, roi_eff, layout, timestamp=frame.captured_at, frame_index=1)
        update = tracker.update(observation)
        report["observation"] = observation.to_dict()
        report["update"] = {"stable": update.stable, "committed": update.committed,
                            "validation_issues": list(update.validation_issues)}
        export_debug(frame.image, roi_eff, output / "roi", layout)
    (output / "observation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")
    print(f"Observation report written to {output / 'observation.json'}")
    return 0


def cmd_run(args) -> int:
    config = LiveConfig.load(args.config)
    window = LiveWindow(LiveWindow.find(config.window_title_regex, config.window_class_regex))
    if args.execute and config.watchdog.enabled:
        # The Agent owns all input; the watchdog only restarts it. The retired
        # takeover supervisor (protection.run_protected) is not used.
        from .watchdog import run_watchdog
        return run_watchdog(config,window.hwnd,skip_menu=args.skip_menu,once=args.once,session=args.session)
    loop = LiveAgentLoop(window, config, execute=args.execute,
                         skip_menu=args.skip_menu, once=args.once,
                         session=args.session)
    return loop.run()


def cmd_notify_retry(args) -> int:
    """Inspect/retry one explicit result ledger, without touching the game."""
    from .notifications import NotificationQueue
    from .notify import FeishuNotifier
    config = LiveConfig.load(args.config)
    directory = Path(args.directory)
    if not directory.is_dir():
        raise ValueError("Notification ledger directory does not exist")
    if not math.isfinite(args.wait_seconds) or args.wait_seconds <= 0:
        raise ValueError("wait-seconds must be finite and positive")
    queue = NotificationQueue(
        FeishuNotifier(config.notify), directory, capacity=config.notify.queue_capacity,
        max_attempts=config.notify.max_attempts, retry_seconds=config.notify.retry_seconds)
    if not args.execute:
        _dump({"execute": False, "entries": queue.recovery_report()})
        return 0
    futures = []
    try:
        futures = queue.recover_results(limit=args.limit)
        deadline = time.monotonic() + args.wait_seconds
        for future in futures:
            try:
                future.result(timeout=max(0, deadline-time.monotonic()))
            except TimeoutError:
                break
    finally:
        pending = queue.close(timeout=0.1)
    entries = queue.recovery_report()
    _dump({"execute": True, "retried": len(futures), "pending": pending, "entries": entries})
    return 0 if not pending and all(f.done() and f.result() for f in futures) else 1


def cmd_collect(args) -> int:
    from .collector import CollectConfig, MatchCollector

    config = CollectConfig.load(args.config)
    live = LiveConfig.load(config.live)
    window = LiveWindow(LiveWindow.find(live.window_title_regex, live.window_class_regex))
    collector = MatchCollector(window, config, live,
                               max_matches=args.matches, session=args.session)
    from .protection import WindowLease
    with WindowLease(live.artifacts_root/'control-leases',window.hwnd):
        return collector.run()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="majsoul.live",
        description="Live Steam-client pipeline: menu navigation, perception, decision, verified clicks")
    sub = parser.add_subparsers(dest="command", required=True)
    doctor = sub.add_parser("doctor", help="Check window, ROI, classifier and model configuration")
    doctor.add_argument("--config", default=DEFAULT_CONFIG)
    doctor.add_argument("--check-api", action="store_true",
                        help="Run one real decision request against the configured backend")
    calibrate = sub.add_parser("calibrate-menu",
                               help="Capture menu screenshots for manual waypoint calibration")
    calibrate.add_argument("--config", default=DEFAULT_CONFIG)
    calibrate.add_argument("--output", default="artifacts/live/menu-calib")
    observe = sub.add_parser("observe", help="Capture one frame and dump perception diagnostics; never clicks")
    observe.add_argument("--config", default=DEFAULT_CONFIG)
    observe.add_argument("--output", default="artifacts/live/observe")
    run = sub.add_parser("run", help="Run the live agent loop")
    run.add_argument("--config", default=DEFAULT_CONFIG)
    run.add_argument("--execute", action="store_true",
                     help="Click in-game actions after preflight; default is perception + decision logging only")
    run.add_argument("--skip-menu", action="store_true",
                     help="Assume the match screen is already being entered")
    run.add_argument("--once", action="store_true",
                     help="Exit after the first decision (dry-run included)")
    run.add_argument("--session", default=None, help="Session directory name under artifacts_root")
    collect = sub.add_parser(
        "collect",
        help="Auto-play data collection: client auto toggles play; background capture + key frames")
    collect.add_argument("--config", default="configs/collect.steam.yaml")
    collect.add_argument("--matches", type=int, default=0,
                         help="Stop after N matches; default 0 = run until interrupted")
    collect.add_argument("--session", default=None,
                         help="Session directory name under dataset_root")
    retry = sub.add_parser("notify-retry", help="Inspect or retry saved final-scorecard notifications; no game input")
    retry.add_argument("--config", default=DEFAULT_CONFIG)
    retry.add_argument("--directory", required=True, help="One explicit session notifications directory")
    retry.add_argument("--execute", action="store_true", help="Send eligible result jobs; default only inspects")
    retry.add_argument("--limit", type=int, default=8, help="Maximum result jobs in this bounded retry")
    retry.add_argument("--wait-seconds", type=float, default=120, help="Bounded total delivery wait")
    args = parser.parse_args(argv)
    try:
        if args.command == "doctor":
            return cmd_doctor(args)
        if args.command == "calibrate-menu":
            return cmd_calibrate_menu(args)
        if args.command == "observe":
            return cmd_observe(args)
        if args.command == "collect":
            return cmd_collect(args)
        if args.command == "notify-retry":
            return cmd_notify_retry(args)
        return cmd_run(args)
    except KeyboardInterrupt:
        print("Stopped.")
        return 130
    except (NavigationError, LiveError, WindowError, ValueError, OSError) as exc:
        print(f"Stopped: {exc}")
        return 1
    except RuntimeError as exc:
        print(f"Stopped: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
