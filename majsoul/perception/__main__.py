import argparse
import json

from ..calibration.frames import read_image
from ..calibration.roi_config import ROIConfig
from .config import VisionConfig
from .debug import export_metadata, export_recognition, export_segmentation
from .glyph_classifier import GlyphTemplateClassifier
from .hand_reader import HandReader
from .metadata_reader import GameMetadataReader
from .river_reader import RiverReader
from .roi_extractor import extract_roi
from .tile_classifier import TileTemplateClassifier
from .tile_segmenter import HandSegmenter, RiverSegmenter
from .visible_tiles import VisibleTilesReader


def _add_common_arguments(command):
    command.add_argument("image")
    command.add_argument("--roi-config", default="configs/roi.yaml")
    command.add_argument("--vision-config", default="configs/vision.yaml")
    command.add_argument("--hand-region", default="self_hand")
    command.add_argument("--river-region", default="river_bottom")
    command.add_argument("--output", required=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Fixed-ROI tile perception tools; never clicks the game")
    sub = parser.add_subparsers(dest="command", required=True)
    segment = sub.add_parser("segment", help="Segment self hand and one river on a screenshot")
    _add_common_arguments(segment)
    recognize = sub.add_parser(
        "recognize", help="Segment and classify self hand and one river on a screenshot")
    _add_common_arguments(recognize)
    recognize.add_argument("--templates", default="artifacts/templates/tiles")
    visible = sub.add_parser(
        "visible", help="Recognize all visible tiles for a locked seat layout")
    visible.add_argument("image")
    visible.add_argument("--roi-config", default="configs/roi.yaml")
    visible.add_argument("--vision-config", default="configs/vision.yaml")
    visible.add_argument("--templates", default="artifacts/templates/tiles")
    visible.add_argument("--active-slots", nargs="+", required=True)
    visible.add_argument("--players", type=int, choices=(3, 4),
                         help="Explicit game-mode override for shared ROI geometry")
    visible.add_argument("--kita", action=argparse.BooleanOptionalAction, default=None)
    visible.add_argument("--output", required=True)
    metadata = sub.add_parser(
        "metadata", help="Recognize numeric metadata for a locked seat layout")
    metadata.add_argument("image")
    metadata.add_argument("--roi-config", default="configs/roi.yaml")
    metadata.add_argument("--vision-config", default="configs/vision.yaml")
    metadata.add_argument("--templates", default="artifacts/templates/glyphs")
    metadata.add_argument("--active-slots", nargs="+", required=True)
    metadata.add_argument("--players", type=int, choices=(3, 4),
                          help="Explicit game-mode override for shared ROI geometry")
    metadata.add_argument("--kita", action=argparse.BooleanOptionalAction, default=None)
    metadata.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        image = read_image(args.image)
        roi = ROIConfig.load(args.roi_config)
        if (getattr(args, "players", None) is not None or
                getattr(args, "kita", None) is not None):
            roi = roi.for_game_mode(
                getattr(args, "players", None) or roi.player_count,
                kita=getattr(args, "kita", None))
        config = VisionConfig.load(args.vision_config)
        if args.command == "segment":
            results = {
                args.hand_region: HandSegmenter(config).segment(
                    extract_roi(image, roi, args.hand_region)),
                args.river_region: RiverSegmenter(config).segment(
                    extract_roi(image, roi, args.river_region)),
            }
            print(json.dumps(export_segmentation(image, results, args.output), ensure_ascii=False))
        elif args.command == "recognize":
            classifier = TileTemplateClassifier(args.templates, config)
            if not classifier.labels:
                raise ValueError(f"No labeled tile templates found in {args.templates}")
            readings = {
                args.hand_region: HandReader(config, classifier).read(
                    extract_roi(image, roi, args.hand_region)),
                args.river_region: RiverReader(config, classifier).read(
                    extract_roi(image, roi, args.river_region)),
            }
            print(json.dumps(export_recognition(image, readings, args.output), ensure_ascii=False))
        elif args.command == "visible":
            classifier = TileTemplateClassifier(args.templates, config)
            if not classifier.labels:
                raise ValueError(f"No labeled tile templates found in {args.templates}")
            visible_tiles = VisibleTilesReader(config, classifier).read(
                image, roi, args.active_slots)
            readings = {"self_hand": visible_tiles.hand,
                        "dora_indicators": visible_tiles.dora}
            readings.update({f"river_{slot}": reading
                             for slot, reading in visible_tiles.rivers.items()})
            readings.update({f"melds_{slot}": reading
                             for slot, reading in visible_tiles.melds.items()})
            readings.update({f"kita_{slot}": reading
                             for slot, reading in visible_tiles.kita.items()})
            print(json.dumps(export_recognition(
                image, readings, args.output, active_slots=visible_tiles.active_slots),
                ensure_ascii=False))
        elif args.command == "metadata":
            classifier = GlyphTemplateClassifier(args.templates, config)
            if not classifier.labels:
                raise ValueError(f"No labeled glyph templates found in {args.templates}")
            reading = GameMetadataReader(config, classifier).read(
                image, roi, args.active_slots)
            print(json.dumps(export_metadata(image, reading, args.output), ensure_ascii=False))
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"Perception error: {exc}\n")


if __name__ == "__main__":
    main()
