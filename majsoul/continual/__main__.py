from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from ..calibration.roi_config import ROIConfig
from ..perception import (GlyphTemplateClassifier, TileTemplateClassifier,
                          VisionConfig)
from ..state.observation import FrameObservationReader
from .collector import UnknownVideoCollector, build_known_index, iter_video
from .commands import parse_review_command
from .config import ContinualConfig
from .dataset_audit import audit_parquet_splits, parse_split_specs
from .crop_corpus import scan_captured_video
from .notifier import notify_review_case
from .schemas import PerceptionLoopState
from .store import ReviewCaseStore, format_review_message
from .video_source import register_captured_video
from .review_sheet import create_crop_corpus_sheet, create_review_sheet


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Review persistent perception unknowns; never trains or clicks")
    parser.add_argument("--cases", default="artifacts/unknown_cases")
    parser.add_argument("--datasets", default="datasets")
    sub = parser.add_subparsers(dest="command", required=True)
    show = sub.add_parser("show", help="Print one review case status")
    show.add_argument("case_id")
    gui = sub.add_parser("gui", help="Open the desktop human-labeling interface")
    gui.add_argument("--reviewer", default="human_gui")
    gui.add_argument(
        "--corpus",
        help="Raw crop corpus manifest; auto-discovers one when omitted")
    gui.add_argument("--cases-only", action="store_true")
    gui.add_argument(
        "--kind", default="tile",
        choices=("all", "tile", "action", "meld", "ui_state", "metadata"))
    label = sub.add_parser("label", help="Verify a case into the dataset/training queue")
    label.add_argument("case_id")
    label.add_argument("label")
    label.add_argument("--reviewer", required=True)
    reject = sub.add_parser("reject", help="Reject an unusable review crop")
    reject.add_argument("case_id")
    reject.add_argument("reason")
    reject.add_argument("--reviewer", required=True)
    suggest = sub.add_parser(
        "suggest", help="Add ordered reviewer candidates without verifying a label")
    suggest.add_argument("case_id")
    suggest.add_argument("labels", nargs="+")
    suggest.add_argument("--reviewer", required=True)
    suggest.add_argument("--note")
    sheet = sub.add_parser("review-sheet", help="Build a labeled contact sheet for cases")
    sheet.add_argument("case_ids", nargs="+")
    sheet.add_argument("--output", required=True)
    sheet.add_argument("--columns", type=int, default=4)
    crop_sheet = sub.add_parser(
        "crop-sheet", help="Build a time-spread audit sheet for a crop corpus")
    crop_sheet.add_argument("manifest")
    crop_sheet.add_argument("--output", required=True)
    crop_sheet.add_argument("--max-per-region", type=int, default=8)
    notify = sub.add_parser(
        "notify-case", help="Send one review package through the configured Feishu helper")
    notify.add_argument("case_id")
    notify.add_argument(
        "--helper", default=os.environ.get("MAJSOUL_FEISHU_HELPER"),
        help="Path to the Feishu skill send helper")
    notify.add_argument(
        "--chat-id", default=os.environ.get("MAJSOUL_FEISHU_CHAT_ID"),
        help="Explicit destination; omit only when the helper has a default target")
    backfill = sub.add_parser(
        "backfill-context",
        help="Add explicit game-mode provenance to legacy review cases")
    backfill.add_argument("--source", required=True)
    backfill.add_argument("--players", type=int, choices=(3, 4), required=True)
    backfill.add_argument("--mode", required=True)
    ingest = sub.add_parser(
        "ingest-command", help="Apply one explicit /label command from human review")
    ingest.add_argument("text")
    ingest.add_argument("--reviewer", required=True)
    audit = sub.add_parser(
        "audit-parquet", help="Audit image Parquet splits without ingesting them")
    audit.add_argument("--split", action="append", required=True, metavar="NAME=PATH")
    audit.add_argument("--report", required=True)
    audit.add_argument("--contact-sheet")
    audit.add_argument("--near-hamming", type=int, default=1)
    audit.add_argument("--max-examples", type=int, default=25)
    collect = sub.add_parser(
        "collect-video", help="Survey a video and create deduplicated review packages")
    collect.add_argument("video")
    collect.add_argument("--start", type=int, default=0)
    collect.add_argument("--end", type=int)
    collect.add_argument("--stride", type=int, default=1)
    collect.add_argument("--max-cases", type=int, default=20)
    collect.add_argument("--active-slots", nargs="+", required=True)
    collect.add_argument("--players", type=int, choices=(3, 4),
                         help="Explicit game-mode override for shared ROI geometry")
    collect.add_argument("--kita", action=argparse.BooleanOptionalAction, default=None)
    collect.add_argument("--source-id", default="runtime_unknown")
    collect.add_argument("--roi-config", default="configs/roi.yaml")
    collect.add_argument("--vision-config", default="configs/vision.yaml")
    collect.add_argument("--continual-config", default="configs/continual.json")
    collect.add_argument("--tile-templates", default="artifacts/templates/tiles")
    collect.add_argument("--glyph-templates", default="artifacts/templates/glyphs")
    collect.add_argument(
        "--include-metadata", action="store_true",
        help="Also collect persistent numeric/seat/round unknown regions")
    collect.add_argument(
        "--notify-feishu", action="store_true",
        help="Send the first persistent review case and enter SAFE_PAUSE")
    collect.add_argument(
        "--feishu-helper", default=os.environ.get("MAJSOUL_FEISHU_HELPER"))
    collect.add_argument(
        "--feishu-chat-id", default=os.environ.get("MAJSOUL_FEISHU_CHAT_ID"))
    collect.add_argument("--report", required=True)
    crop = sub.add_parser(
        "crop-video",
        help="Build a resumable, deduplicated, unverified tile-crop corpus")
    crop.add_argument("video")
    crop.add_argument("--source-manifest", required=True)
    crop.add_argument("--output-root", required=True)
    crop.add_argument("--start", type=int, default=0)
    crop.add_argument("--end", type=int)
    crop.add_argument("--stride", type=int, default=15)
    crop.add_argument("--resume", action="store_true")
    crop.add_argument("--active-slots", nargs="+", required=True)
    crop.add_argument("--players", type=int, choices=(3, 4))
    crop.add_argument("--kita", action=argparse.BooleanOptionalAction, default=None)
    crop.add_argument("--roi-config", default="configs/roi.yaml")
    crop.add_argument("--vision-config", default="configs/vision.yaml")
    crop.add_argument("--continual-config", default="configs/continual.json")
    register = sub.add_parser(
        "register-video", help="Register a captured video as review-only source data")
    register.add_argument("video")
    register.add_argument("--source-id", required=True)
    register.add_argument("--manifest", required=True)
    register.add_argument("--players", type=int, choices=(3, 4), required=True)
    register.add_argument("--mode")
    register.add_argument("--skip-sha256", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "gui":
            from .corpus_label_gui import (
                discover_crop_manifests, run_corpus_label_gui)
            manifests = (() if args.cases_only else
                         ((Path(args.corpus),) if args.corpus else
                          discover_crop_manifests(args.datasets)))
            if manifests:
                run_corpus_label_gui(
                    manifests[0], args.datasets,
                    cases_root=args.cases, reviewer=args.reviewer)
            else:
                from .label_gui import run_label_gui
                run_label_gui(
                    args.cases, args.datasets,
                    reviewer=args.reviewer, initial_kind=args.kind)
        elif args.command == "audit-parquet":
            report = audit_parquet_splits(
                parse_split_specs(args.split), report_path=args.report,
                contact_sheet_path=args.contact_sheet,
                near_hamming=args.near_hamming, max_examples=args.max_examples)
            print(json.dumps({
                "status": report["status"],
                "total_rows": report["total_rows"],
                "class_count": report["class_count"],
                "report": args.report,
                "contact_sheet": args.contact_sheet,
                "training_allowed": report["admission"]["training_allowed"],
            }, ensure_ascii=False))
        elif args.command == "register-video":
            manifest = register_captured_video(
                args.video, source_id=args.source_id,
                manifest_path=args.manifest, player_count=args.players,
                mode=args.mode, sha256=not args.skip_sha256)
            print(json.dumps({
                "source_id": manifest["source_id"],
                "status": manifest["status"],
                "manifest": args.manifest,
                "training_allowed": manifest["admission"]["training_allowed"],
            }, ensure_ascii=False))
        elif args.command == "review-sheet":
            report = create_review_sheet(
                args.cases, args.case_ids, args.output, columns=args.columns)
            print(json.dumps(report, ensure_ascii=False))
        elif args.command == "crop-sheet":
            report = create_crop_corpus_sheet(
                args.manifest, args.output,
                max_per_region=args.max_per_region)
            print(json.dumps({
                "output": report["output"],
                "region_count": report["region_count"],
                "sample_count": len(report["entries"]),
            }, ensure_ascii=False))
        elif args.command == "notify-case":
            if not args.helper:
                raise ValueError(
                    "--helper or MAJSOUL_FEISHU_HELPER is required")
            result = notify_review_case(
                args.cases, args.case_id, helper=args.helper,
                chat_id=args.chat_id)
            print(json.dumps(result, ensure_ascii=False))
        elif args.command == "backfill-context":
            result = ReviewCaseStore(args.cases, args.datasets).backfill_source_context(
                args.source, player_count=args.players, mode=args.mode)
            print(json.dumps(result, ensure_ascii=False))
        elif args.command == "show":
            store = ReviewCaseStore(args.cases, args.datasets)
            metadata = json.loads((Path(args.cases) / args.case_id / "metadata.json")
                                  .read_text(encoding="utf-8-sig"))
            print(format_review_message(metadata))
        elif args.command == "reject":
            store = ReviewCaseStore(args.cases, args.datasets)
            result = store.reject(
                args.case_id, rejected_by=args.reviewer, reason=args.reason)
            print(json.dumps({key: result[key] for key in (
                "case_id", "status", "kind", "rejection_reason")},
                ensure_ascii=False))
        elif args.command == "suggest":
            store = ReviewCaseStore(args.cases, args.datasets)
            result = store.suggest(
                args.case_id, args.labels, suggested_by=args.reviewer,
                note=args.note)
            latest = result["reviewer_suggestions"][-1]
            print(json.dumps({
                "case_id": result["case_id"],
                "status": result["status"],
                "suggested_labels": latest["labels"],
                "suggestion_status": latest["status"],
            }, ensure_ascii=False))
        elif args.command in ("label", "ingest-command"):
            store = ReviewCaseStore(args.cases, args.datasets)
            if args.command == "label":
                action, case_id, value = "label", args.case_id, args.label
            else:
                action, case_id, value = parse_review_command(args.text)
            if action == "reject":
                result = store.reject(
                    case_id, rejected_by=args.reviewer, reason=value)
                keys = ("case_id", "status", "kind", "rejection_reason")
            else:
                result = store.label(case_id, value, verified_by=args.reviewer)
                keys = ("case_id", "status", "kind", "label", "dataset_path")
            print(json.dumps({key: result[key] for key in keys}, ensure_ascii=False))
        elif args.command == "crop-video":
            roi = ROIConfig.load(args.roi_config)
            if args.players is not None or args.kita is not None:
                roi = roi.for_game_mode(
                    args.players or roi.player_count, kita=args.kita)
            manifest = scan_captured_video(
                args.video, source_manifest=args.source_manifest,
                output_root=args.output_root, roi=roi,
                vision=VisionConfig.load(args.vision_config),
                active_slots=args.active_slots,
                config=ContinualConfig.load(args.continual_config),
                start=args.start, end=args.end, stride=args.stride,
                resume=args.resume)
            progress = manifest["progress"]
            print(json.dumps({
                "corpus_id": manifest["corpus_id"],
                "complete": progress["complete"],
                "sampled_frames": progress["sampled_frames"],
                "candidate_crops": progress["candidate_crops"],
                "unique_crops": progress["unique_crops"],
                "duplicate_crops": progress["duplicate_crops"],
                "manifest": str(Path(args.output_root) / "manifest.json"),
                "training_allowed": manifest["training_allowed"],
            }, ensure_ascii=False))
        else:
            store = ReviewCaseStore(args.cases, args.datasets)
            if args.max_cases < 1:
                raise ValueError("max-cases must be positive")
            vision = VisionConfig.load(args.vision_config)
            classifier = TileTemplateClassifier(args.tile_templates, vision)
            if not classifier.labels:
                raise ValueError("No tile templates are available")
            roi = ROIConfig.load(args.roi_config)
            if args.players is not None or args.kita is not None:
                roi = roi.for_game_mode(
                    args.players or roi.player_count, kita=args.kita)
            collector = UnknownVideoCollector(
                roi, vision, classifier,
                args.active_slots, ContinualConfig.load(args.continual_config),
                store, build_known_index(args.tile_templates), source=args.source_id)
            observation_reader = None
            if args.include_metadata:
                glyph_classifier = GlyphTemplateClassifier(
                    args.glyph_templates, vision)
                if not glyph_classifier.labels:
                    raise ValueError("No glyph templates are available")
                observation_reader = FrameObservationReader(
                    vision, classifier, glyph_classifier)
            previous = None
            waiting = []
            created = []
            reviewed = []
            notifications = []
            notification_errors = []
            safe_pause = False
            sampled = 0
            if args.notify_feishu and not args.feishu_helper:
                raise ValueError(
                    "--feishu-helper or MAJSOUL_FEISHU_HELPER is required "
                    "with --notify-feishu")
            for frame, frame_index, timestamp in iter_video(
                    args.video, start=args.start, end=args.end, stride=args.stride):
                sampled += 1
                for evidence in waiting:
                    case = collector.finalize(evidence, frame)
                    entry = {"case_id": case.case_id, "path": case.path,
                             "status": case.status.value,
                             "created": case.created,
                             "message": case.message}
                    if not any(item["case_id"] == case.case_id
                               for item in reviewed):
                        reviewed.append(entry)
                    if case.created:
                        created.append(entry)
                        print(case.message + "\n")
                    if args.notify_feishu:
                        try:
                            notifications.append(notify_review_case(
                                args.cases, case.case_id,
                                helper=args.feishu_helper,
                                chat_id=args.feishu_chat_id))
                        except RuntimeError as exc:
                            notification_errors.append({
                                "case_id": case.case_id, "error": str(exc)})
                        safe_pause = True
                        break
                    if len(created) >= args.max_cases:
                        break
                if safe_pause or len(created) >= args.max_cases:
                    break
                observation = (observation_reader.read(
                    frame, roi, args.active_slots,
                    timestamp=timestamp, frame_index=frame_index)
                    if observation_reader is not None else None)
                waiting = collector.observe(
                    frame, timestamp=timestamp, frame_index=frame_index,
                    previous_frame=previous, observation=observation)
                previous = frame.copy()
            if waiting and not safe_pause and len(created) < args.max_cases:
                for evidence in waiting[:args.max_cases - len(created)]:
                    case = collector.finalize(evidence, evidence.full_frame)
                    entry = {"case_id": case.case_id, "path": case.path,
                             "status": case.status.value,
                             "created": case.created,
                             "message": case.message}
                    if not any(item["case_id"] == case.case_id
                               for item in reviewed):
                        reviewed.append(entry)
                    if case.created:
                        created.append(entry)
                    if args.notify_feishu:
                        try:
                            notifications.append(notify_review_case(
                                args.cases, case.case_id,
                                helper=args.feishu_helper,
                                chat_id=args.feishu_chat_id))
                        except RuntimeError as exc:
                            notification_errors.append({
                                "case_id": case.case_id, "error": str(exc)})
                        safe_pause = True
                        break
            report = {
                "source_id": args.source_id,
                "player_count": roi.player_count,
                "active_slots": list(roi.layout_for_active_slots(args.active_slots)),
                "include_metadata": args.include_metadata,
                "sampled_frames": sampled,
                "created_count": len(created),
                "cases": created,
                "review_cases": reviewed,
                "safe_pause": safe_pause,
                "control_state": (PerceptionLoopState.SAFE_PAUSE.value
                                  if safe_pause else PerceptionLoopState.OBSERVE.value),
                "notifications": notifications,
                "notification_errors": notification_errors,
                "training_started": False,
            }
            report_path = Path(args.report)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                   encoding="utf-8")
            print(json.dumps({"sampled_frames": sampled,
                              "created_count": len(created),
                              "safe_pause": safe_pause,
                              "notification_error_count": len(notification_errors),
                              "report": str(report_path)}, ensure_ascii=False))
            if notification_errors:
                return 2
    except (OSError, RuntimeError, ValueError, KeyError,
            json.JSONDecodeError) as exc:
        print(f"Stopped: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
