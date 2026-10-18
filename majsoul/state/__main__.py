from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re

import cv2

from ..calibration.frames import read_image
from ..calibration.roi_config import ROIConfig
from ..continual import (ContinualConfig, ReviewCaseStore,
                         UnknownVideoCollector, build_known_index)
from ..continual.schemas import PerceptionLoopState
from ..perception import (GlyphTemplateClassifier, TileTemplateClassifier,
                          SeatLayoutResolver, VisionConfig)
from .observation import FrameObservationReader
from .schemas import GameEventType
from .tracker import MahjongStateTracker


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--live-config", help="Replay with the live ROI, models, buttons, rules, score and stability settings; overrides their individual offline options. Never sends input or notifications")
    parser.add_argument("--export-roi", action="store_true",
                        help="Export ROI overlays and crops alongside the report")
    parser.add_argument("--roi-config", default="configs/roi.yaml")
    parser.add_argument("--vision-config", default="configs/vision.yaml")
    parser.add_argument("--rules", default="configs/sanma.json")
    parser.add_argument("--tile-templates", default="artifacts/templates/tiles")
    parser.add_argument("--glyph-templates", default="artifacts/templates/glyphs")
    parser.add_argument(
        "--active-slots", nargs="+",
        help="Occupied physical slots; omit to resolve and lock them from metadata")
    parser.add_argument(
        "--layout-confirmations", type=int, default=2,
        help="Consecutive unique metadata readings required for automatic layout lock")
    parser.add_argument("--stable-frames", type=int, default=2)
    parser.add_argument(
        "--review-unknowns", action="store_true",
        help="Create a review case and stop after a persistent unknown")
    parser.add_argument("--continual-config", default="configs/continual.json")
    parser.add_argument("--review-cases", default="artifacts/unknown_cases")
    parser.add_argument("--datasets", default="datasets")
    parser.add_argument("--output", required=True)


def _image_frame_index(path: Path, fallback: int) -> int:
    match = re.search(r"(?:frame[_-]?)?(\d+)$", path.stem, re.IGNORECASE)
    return int(match.group(1)) if match else fallback


def _images(paths, fps: float):
    for fallback, item in enumerate(paths):
        path = Path(item)
        frame_index = _image_frame_index(path, fallback)
        yield read_image(path), frame_index, frame_index / fps, str(path)


def _video(path: str, start: int, end: int | None, stride: int):
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise OSError(f"Unable to open video: {path}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if fps <= 0:
            raise ValueError("Video does not report a positive frame rate")
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        stop = total if end is None else min(end, total)
        if start >= stop:
            raise ValueError("Video frame range is empty")
        capture.set(cv2.CAP_PROP_POS_FRAMES, start)
        for frame_index in range(start, stop):
            ok, image = capture.read()
            if not ok:
                raise OSError(f"Failed to decode video frame {frame_index}")
            if (frame_index - start) % stride == 0:
                yield image, frame_index, frame_index / fps, path
    finally:
        capture.release()


def _run(args, frames) -> dict:
    live = None
    button_reader = None
    countdown_reader = None
    if args.live_config:
        from ..live.config import ROOT, LiveConfig
        from ..live.buttons import ActionButtonReader
        from ..live.countdown import CountdownReader
        live = LiveConfig.load(args.live_config)
        args.roi_config, args.vision_config, args.rules = live.roi, live.vision, live.rules
        args.tile_templates, args.glyph_templates = live.tile_templates, live.glyph_templates
        args.stable_frames, args.layout_confirmations = live.stable_frames, live.layout_confirmations
        button_reader = ActionButtonReader(live.buttons, live.button_templates)
        countdown_reader = CountdownReader.load(
            live.countdown, ROOT, json.loads(Path(args.rules).read_text(encoding="utf-8-sig")))
    roi = ROIConfig.load(args.roi_config)
    config = VisionConfig.load(args.vision_config)
    if live is None:
        tile_classifier = TileTemplateClassifier(args.tile_templates, config)
    else:
        from ..live.classifier import build_tile_classifier
        tile_classifier = build_tile_classifier(
            live.classifier, config, cnn_checkpoint=live.cnn_checkpoint,
            template_root=live.tile_templates, expected_metadata=live.cnn_expected)
    reader = FrameObservationReader(
        config,
        tile_classifier,
        GlyphTemplateClassifier(args.glyph_templates, config),
        button_reader=button_reader, read_scores=live.read_scores if live else True,
        countdown_reader=countdown_reader,
    )
    tracker = MahjongStateTracker.load(
        args.rules, stable_frames=args.stable_frames)
    active_slots = (tuple(roi.layout_for_active_slots(args.active_slots))
                    if args.active_slots is not None else None)
    layout_resolver = None
    if active_slots is None:
        layout_resolver = SeatLayoutResolver(
            roi, reader.metadata_reader,
            starting_points=tracker.validator.starting_points,
            confirmations_required=args.layout_confirmations)
    review_collector = None

    def ensure_review_collector():
        nonlocal review_collector
        if args.review_unknowns and review_collector is None and active_slots is not None:
            review_collector = UnknownVideoCollector(
                roi, config, tile_classifier, active_slots,
                ContinualConfig.load(args.continual_config),
                ReviewCaseStore(args.review_cases, args.datasets),
                build_known_index(args.tile_templates))

    if active_slots is not None:
        ensure_review_collector()
    updates = []
    event_counts = {event.value: 0 for event in GameEventType}
    complete_count = 0

    def frame_roi(image):
        if live is None:
            return roi
        from ..live.capture import roi_for_frame
        return roi_for_frame(roi, (image.shape[1], image.shape[0]), live.projection)

    def process_state_frame(image, frame_index, timestamp, source,
                            frame_layout, layout_candidates):
        nonlocal complete_count
        effective_roi = frame_roi(image)
        observation = reader.read(
            image, effective_roi, frame_layout,
            timestamp=timestamp, frame_index=frame_index)
        observation_validation = tracker.validator.validate_observation(observation)
        update = tracker.update(observation)
        complete_count += int(observation.complete)
        for event in update.events:
            event_counts[event.type.value] += 1
        record = {
            "source": source,
            "frame_size": [image.shape[1], image.shape[0]],
            "observation": observation.to_dict(),
            "frame_index": frame_index,
            "timestamp": timestamp,
            "layout_locked": list(frame_layout),
            "layout_candidates": [list(item) for item in layout_candidates],
            "observation_complete": observation.complete,
            "round_id": observation.round_id,
            "stable": update.stable,
            "committed": update.committed,
            "issues": list(observation.issues),
            "observation_validation": list(observation_validation),
            "resync_requested": bool(observation_validation),
            "validation_issues": list(update.validation_issues),
            "confidence": observation.confidence,
            "events": [event.to_dict() for event in update.events],
            "committed_state": (update.state.to_dict()
                                if update.committed and update.state is not None else None),
        }
        if args.export_roi:
            from ..calibration.debug import export_debug
            target = Path(args.output).with_suffix("") / "roi" / str(frame_index)
            export_debug(image, effective_roi, target, frame_layout)
            record["roi_review"] = str(target)
        return observation, update, record

    review_waiting = []
    review_cases = []
    previous_image = None
    last_image = None
    safe_pause = False
    paused_after_frame = None
    next_context_frame = None
    layout_buffer = []
    for image, frame_index, timestamp, source in frames:
        last_image = image
        if review_waiting:
            for evidence in review_waiting:
                case = review_collector.finalize(evidence, image)
                review_cases.append({
                    "case_id": case.case_id,
                    "path": case.path,
                    "status": case.status.value,
                    "created": case.created,
                    "message": case.message,
                })
            safe_pause = True
            paused_after_frame = min(item.frame_index for item in review_waiting)
            next_context_frame = frame_index
            break
        layout_candidates = [active_slots] if active_slots is not None else []
        if layout_resolver is not None and active_slots is None:
            layout_resolver.roi = frame_roi(image)
            resolution = layout_resolver.observe(image)
            layout_candidates = list(resolution.candidates)
            if resolution.locked_layout is None:
                if resolution.pending_count == 1:
                    layout_buffer = []
                elif resolution.pending_count == 0:
                    layout_buffer = []
                layout_issues = list(resolution.issues)
                if not layout_issues:
                    layout_issues.append(
                        "Seat layout is waiting for consecutive confirmation")
                pending_index = len(updates)
                updates.append({
                    "source": source,
                    "frame_index": frame_index,
                    "timestamp": timestamp,
                    "layout_locked": None,
                    "layout_candidates": [list(item) for item in layout_candidates],
                    "layout_pending": (list(resolution.pending_layout)
                                       if resolution.pending_layout is not None else None),
                    "layout_pending_count": resolution.pending_count,
                    "layout_evaluations": [
                        {"layout": list(item.layout), "accepted": item.accepted,
                         "issues": list(item.issues)}
                        for item in resolution.evaluations
                    ],
                    "observation_complete": False,
                    "round_id": None,
                    "stable": False,
                    "committed": False,
                    "issues": layout_issues,
                    "observation_validation": [],
                    "resync_requested": False,
                    "validation_issues": [],
                    "confidence": {},
                    "events": [],
                    "committed_state": None,
                })
                if resolution.pending_layout is not None:
                    layout_buffer.append((
                        image.copy(), frame_index, timestamp, source,
                        tuple(resolution.pending_layout), pending_index))
                if args.export_roi:
                    from ..calibration.debug import export_debug
                    target = Path(args.output).with_suffix("") / "roi" / str(frame_index)
                    export_debug(image, frame_roi(image), target)
                    updates[-1]["roi_review"] = str(target)
                previous_image = image.copy()
                continue
            active_slots = resolution.locked_layout
            for (buffered_image, buffered_index, buffered_timestamp,
                 buffered_source, buffered_layout, update_index) in layout_buffer:
                _, _, buffered_record = process_state_frame(
                    buffered_image, buffered_index, buffered_timestamp,
                    buffered_source, buffered_layout, (buffered_layout,))
                updates[update_index] = buffered_record
            if layout_buffer:
                previous_image = layout_buffer[-1][0].copy()
            layout_buffer = []
            ensure_review_collector()
        observation, update, record = process_state_frame(
            image, frame_index, timestamp, source,
            active_slots, layout_candidates)
        updates.append(record)
        observation_validation = tuple(record["observation_validation"])
        if review_collector is not None:
            review_waiting = review_collector.observe(
                image, timestamp=timestamp, frame_index=frame_index,
                previous_frame=previous_image,
                state=(update.state.to_dict() if update.state is not None else None),
                observation=observation,
                state_issues=observation_validation)
        previous_image = image.copy()
    if review_waiting and not safe_pause and last_image is not None:
        for evidence in review_waiting:
            case = review_collector.finalize(evidence, last_image)
            review_cases.append({
                "case_id": case.case_id,
                "path": case.path,
                "status": case.status.value,
                "created": case.created,
                "message": case.message,
            })
        safe_pause = True
        paused_after_frame = min(item.frame_index for item in review_waiting)
    event_counts = {key: value for key, value in event_counts.items() if value}
    return {
        "pipeline": {
            "live_config": str(args.live_config) if args.live_config else None,
            "classifier": getattr(tile_classifier, "backend_name", "template"),
            "classifier_manifest": getattr(tile_classifier, "manifest", None),
            "countdown_manifest": countdown_reader.manifest if countdown_reader else None,
            "read_scores": live.read_scores if live else True,
            "stable_frames": args.stable_frames,
            "layout_confirmations": args.layout_confirmations,
            "config_sources": {
                name: {"path": str(Path(path).resolve()),
                       "sha256": sha256(Path(path).read_bytes()).hexdigest()}
                for name, path in (("roi", args.roi_config),
                                   ("vision", args.vision_config), ("rules", args.rules))
            },
        },
        "sample_count": len(updates),
        "complete_observation_count": complete_count,
        "committed_count": sum(item["committed"] for item in updates),
        "resync_request_count": sum(item["resync_requested"] for item in updates),
        "event_counts": event_counts,
        "updates": updates,
        "active_slots": list(active_slots) if active_slots is not None else None,
        "layout_source": "manual" if args.active_slots is not None else "metadata",
        "final_state": tracker.state.to_dict() if tracker.state is not None else None,
        "safe_pause": safe_pause,
        "control_state": (PerceptionLoopState.SAFE_PAUSE.value
                          if safe_pause else PerceptionLoopState.OBSERVE.value),
        "paused_after_frame": paused_after_frame,
        "next_context_frame": next_context_frame,
        "review_cases": review_cases,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Offline fixed-ROI state reconstruction; never clicks the game")
    sub = parser.add_subparsers(dest="command", required=True)
    images = sub.add_parser("images", help="Track an ordered screenshot sequence")
    images.add_argument("images", nargs="+")
    images.add_argument("--fps", type=float, default=30.0)
    _common(images)
    video = sub.add_parser("video", help="Track a frame range from one video")
    video.add_argument("video")
    video.add_argument("--start", type=int, default=0)
    video.add_argument("--end", type=int)
    video.add_argument("--stride", type=int, default=1)
    _common(video)
    args = parser.parse_args(argv)
    try:
        if args.stable_frames < 1:
            raise ValueError("stable-frames must be at least one")
        if args.layout_confirmations < 2:
            raise ValueError("layout-confirmations must be at least two")
        if args.command == "images":
            if args.fps <= 0:
                raise ValueError("fps must be positive")
            frames = _images(args.images, args.fps)
        else:
            if args.start < 0 or args.end is not None and args.end <= 0:
                raise ValueError("video frame bounds must be non-negative")
            if args.stride < 1:
                raise ValueError("stride must be at least one")
            frames = _video(args.video, args.start, args.end, args.stride)
        report = _run(args, frames)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                          encoding="utf-8")
        summary = {key: report[key] for key in (
            "sample_count", "complete_observation_count", "committed_count",
            "resync_request_count", "event_counts", "safe_pause",
            "paused_after_frame", "active_slots", "layout_source")}
        summary["review_case_ids"] = [
            item["case_id"] for item in report["review_cases"]]
        summary["history_complete"] = (
            report["final_state"]["history_complete"]
            if report["final_state"] is not None else None)
        summary["output"] = str(output)
        print(json.dumps(summary, ensure_ascii=False))
    except (OSError, ValueError, KeyError) as exc:
        print(f"Stopped: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
