import argparse
import json
from pathlib import Path

from .debug import export_debug, export_projection_trials
from .frames import extract_frame, read_image, write_image
from .roi_config import ROIConfig, default_layout
from .roi_editor import edit_rois


def main(argv=None):
    parser = argparse.ArgumentParser(description="Offline ROI tools; no model requests or game clicks")
    sub = parser.add_subparsers(dest="command", required=True)
    frame = sub.add_parser("extract", help="Extract an exact zero-based video frame")
    frame.add_argument("video")
    frame.add_argument("--frame", type=int, required=True)
    frame.add_argument("--output", required=True)
    edit = sub.add_parser("edit", help="Select named normalized ROIs with the mouse")
    edit.add_argument("image")
    edit.add_argument("--config", default="configs/roi.yaml")
    edit.add_argument("--players", type=int, default=3)
    edit.add_argument("--slots", nargs="+", help="Physical screen slots; defaults from --players")
    edit.add_argument("--kita", action="store_true")
    edit.add_argument("--names", nargs="+", help="Edit only these ROI names")
    edit.add_argument("--preview-size", nargs=2, type=int, default=[1280, 800],
                      metavar=("WIDTH", "HEIGHT"), help="Maximum interactive preview size in pixels")
    gui = sub.add_parser("gui", help="Open the persistent ROI annotation window")
    gui.add_argument("image", help="Reference screenshot")
    gui.add_argument("--config", default="configs/roi.yaml")
    gui.add_argument("--players", type=int, default=3)
    gui.add_argument("--slots", nargs="+", help="Physical screen slots; defaults from --players")
    gui.add_argument("--kita", action="store_true")
    gui.add_argument("--names", nargs="+", help="Show/edit only these ROI names")
    debug = sub.add_parser("debug", help="Reload ROIs onto a different frame and export every crop")
    debug.add_argument("image")
    debug.add_argument("--config", default="configs/roi.yaml")
    debug.add_argument("--output", required=True)
    debug.add_argument("--require-complete", action="store_true")
    debug.add_argument("--active-slots", nargs="+",
                       help="Locked match layout, for example: bottom top left")
    debug.add_argument(
        "--projection", default="auto",
        help="ROI projection name, or 'all' to export multiple crop hypotheses")
    args = parser.parse_args(argv)
    try:
        if args.command == "extract":
            image, metadata = extract_frame(args.video, args.frame)
            write_image(args.output, image)
            Path(args.output).with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            print(json.dumps(metadata))
        elif args.command == "edit":
            existing = ROIConfig.load(args.config) if Path(args.config).exists() else None
            slots, variants, self_slot = default_layout(args.players)
            config = edit_rois(read_image(args.image),
                               player_count=existing.player_count if existing else args.players,
                               screen_slots=existing.screen_slots if existing else args.slots or slots,
                               layout_variants=existing.layout_variants if existing else variants,
                               self_slot=existing.self_slot if existing else self_slot,
                               kita=existing.kita if existing else args.kita,
                               existing=existing, names=args.names, preview_size=args.preview_size)
            config.save(args.config)
            print(json.dumps({"saved": args.config, "uncalibrated": config.validate()}))
        elif args.command == "gui":
            from .roi_gui import run_roi_gui
            slots, variants, self_slot = default_layout(args.players)
            run_roi_gui(args.image, args.config, player_count=args.players,
                        screen_slots=args.slots or slots, layout_variants=variants,
                        self_slot=self_slot, kita=args.kita, names=args.names)
        else:
            config = ROIConfig.load(args.config)
            config.validate(complete=args.require_complete)
            image = read_image(args.image)
            report = (export_projection_trials(
                image, config, args.output, active_slots=args.active_slots)
                if args.projection == "all" else
                export_debug(image, config, args.output,
                             active_slots=args.active_slots,
                             projection=args.projection))
            print(json.dumps(report))
    except (ValueError, OSError) as exc:
        parser.exit(1, f"ROI error: {exc}\n")
    except KeyboardInterrupt:
        parser.exit(130, "Cancelled; configuration was not saved.\n")


if __name__ == "__main__":
    main()
