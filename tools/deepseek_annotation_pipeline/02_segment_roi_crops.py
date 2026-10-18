"""Step 2: segment and deduplicate Mahjong tile crops from extracted frames."""

from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path
import sys


def locate_project_root() -> Path:
    """Find the checkout when this script is run from tools/ or an extracted ZIP."""
    for start in (Path(__file__).resolve().parent, Path.cwd().resolve()):
        for candidate in (start, *start.parents):
            if ((candidate / "pyproject.toml").is_file() and
                    (candidate / "majsoul" / "__init__.py").is_file()):
                return candidate
    raise RuntimeError(
        "Cannot find the majsoul project root. Keep this package inside the "
        "majsoul checkout or run it from that checkout.")


PROJECT_ROOT = locate_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from majsoul.calibration.frames import read_image
from majsoul.calibration.roi_config import ROIConfig
from majsoul.continual.config import ContinualConfig
from majsoul.continual.crop_corpus import CapturedTileCorpus, TileCropExtractor
from majsoul.perception import VisionConfig


def main() -> int:
    parser = ArgumentParser(description="Step 2/4: segment ROI tile crops")
    parser.add_argument("frames_manifest")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--active-slots", nargs="+", required=True)
    parser.add_argument("--players", type=int, choices=(3, 4))
    parser.add_argument("--kita", action="store_true")
    parser.add_argument("--no-kita", action="store_true")
    parser.add_argument(
        "--roi-config", default=str(PROJECT_ROOT / "configs" / "roi.yaml"))
    parser.add_argument(
        "--vision-config", default=str(PROJECT_ROOT / "configs" / "vision.yaml"))
    parser.add_argument(
        "--continual-config",
        default=str(PROJECT_ROOT / "configs" / "continual.json"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.kita and args.no_kita:
        raise SystemExit("Choose at most one of --kita and --no-kita")

    frames_manifest_path = Path(args.frames_manifest)
    frames = json.loads(frames_manifest_path.read_text(encoding="utf-8-sig"))
    if not frames.get("complete"):
        raise SystemExit("Frame extraction is incomplete; resume step 1 first")
    signature = frames["extraction_signature"]
    source_manifest = Path(frames["source_manifest"])
    roi = ROIConfig.load(args.roi_config)
    kita = True if args.kita else False if args.no_kita else None
    if args.players is not None or kita is not None:
        roi = roi.for_game_mode(args.players or roi.player_count, kita=kita)
    config = ContinualConfig.load(args.continual_config)
    corpus = CapturedTileCorpus(
        args.output_root, source_manifest,
        roi=roi, active_slots=args.active_slots, config=config,
        start=int(signature["start"]), end=int(signature["end"]),
        stride=int(signature["stride"]), resume=args.resume)
    if corpus.complete:
        print(json.dumps({"manifest": str(corpus.manifest_path),
                          "complete": True,
                          "unique_crops": corpus.manifest["progress"]["unique_crops"]},
                         ensure_ascii=False))
        return 0
    extractor = TileCropExtractor(
        roi, VisionConfig.load(args.vision_config), args.active_slots)
    frame_root = frames_manifest_path.parent
    processed = 0
    try:
        for entry in frames["frames"]:
            frame_index = int(entry["frame_index"])
            if frame_index < corpus.next_frame:
                continue
            relative = Path(entry["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Unsafe extracted frame path: {relative}")
            image = read_image(frame_root / relative)
            corpus.add_frame(
                extractor.extract(image), frame_index=frame_index,
                timestamp=float(entry["timestamp"]))
            processed += 1
            if processed % 100 == 0:
                progress = corpus.manifest["progress"]
                print(f"segmented {processed}; unique crops={progress['unique_crops']}")
    except BaseException:
        corpus.checkpoint(complete=False)
        raise
    corpus.checkpoint(complete=True)
    progress = corpus.manifest["progress"]
    print(json.dumps({"manifest": str(corpus.manifest_path),
                      "complete": progress["complete"],
                      "sampled_frames": progress["sampled_frames"],
                      "unique_crops": progress["unique_crops"],
                      "training_allowed": corpus.manifest["training_allowed"]},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
