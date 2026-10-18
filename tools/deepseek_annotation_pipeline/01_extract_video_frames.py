"""Step 1: extract a resumable, sparse frame corpus from one video."""

from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from uuid import uuid4

import cv2


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

from majsoul.continual.collector import iter_video


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def write_frame(path: Path, image, image_format: str, jpeg_quality: int) -> None:
    suffix = ".png" if image_format == "png" else ".jpg"
    parameters = ([] if image_format == "png" else
                  [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    ok, encoded = cv2.imencode(suffix, image, parameters)
    if not ok:
        raise ValueError(f"Cannot encode frame: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded.tofile(str(path))


def main() -> int:
    parser = ArgumentParser(description="Step 1/4: extract video frames")
    parser.add_argument("video")
    parser.add_argument("--source-manifest", required=True,
                        help="Registered captured-video source manifest")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int)
    parser.add_argument("--stride", type=int, default=15)
    parser.add_argument("--format", choices=("jpg", "png"), default="jpg")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    video = Path(args.video)
    source_manifest_path = Path(args.source_manifest)
    output_root = Path(args.output_root)
    manifest_path = output_root / "manifest.json"
    if not video.is_file() or not source_manifest_path.is_file():
        raise SystemExit("Video or source manifest does not exist")
    if (args.start < 0 or args.stride < 1 or
            not 1 <= args.jpeg_quality <= 100 or args.checkpoint_every < 1):
        raise SystemExit("Invalid frame extraction parameters")
    source = json.loads(source_manifest_path.read_text(encoding="utf-8-sig"))
    if source.get("source_type") != "captured_video":
        raise SystemExit("Source manifest is not a captured video")
    total_frames = int(source["video"]["frame_count"])
    stop = total_frames if args.end is None else min(args.end, total_frames)
    if args.start >= stop:
        raise SystemExit("Empty frame range")
    signature = {
        "video": str(video.resolve()),
        "source_manifest": str(source_manifest_path.resolve()),
        "start": args.start,
        "end": stop,
        "stride": args.stride,
        "format": args.format,
        "jpeg_quality": args.jpeg_quality if args.format == "jpg" else None,
    }

    if manifest_path.exists():
        if not args.resume:
            raise SystemExit("Output exists; pass --resume to continue")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        if manifest.get("extraction_signature") != signature:
            raise SystemExit("Existing frame corpus uses different parameters")
    else:
        output_root.mkdir(parents=True, exist_ok=True)
        manifest = {
            "schema_version": 1,
            "source_id": source["source_id"],
            "source_manifest": str(source_manifest_path.resolve()),
            "player_count": source["video"]["player_count"],
            "mode": source["video"].get("mode"),
            "extraction_signature": signature,
            "complete": False,
            "frames": [],
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        atomic_json(manifest_path, manifest)

    completed = {int(item["frame_index"]) for item in manifest["frames"]}
    expected = list(range(args.start, stop, args.stride))
    next_frames = [index for index in expected if index not in completed]
    if not next_frames:
        manifest["complete"] = True
        atomic_json(manifest_path, manifest)
        print(json.dumps({"manifest": str(manifest_path),
                          "frames": len(manifest["frames"]),
                          "complete": True}, ensure_ascii=False))
        return 0

    first_missing = next_frames[0]
    extension = ".png" if args.format == "png" else ".jpg"
    try:
        for image, frame_index, timestamp in iter_video(
                video, start=first_missing, end=stop, stride=args.stride):
            if frame_index in completed:
                continue
            relative = Path("frames") / f"frame_{frame_index:09d}{extension}"
            target = output_root / relative
            write_frame(target, image, args.format, args.jpeg_quality)
            manifest["frames"].append({
                "frame_index": frame_index,
                "timestamp": timestamp,
                "path": relative.as_posix(),
                "resolution": [image.shape[1], image.shape[0]],
            })
            if len(manifest["frames"]) % args.checkpoint_every == 0:
                manifest["frames"].sort(key=lambda item: item["frame_index"])
                manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
                atomic_json(manifest_path, manifest)
                print(f"extracted {len(manifest['frames'])}/{len(expected)}")
    except BaseException:
        manifest["frames"].sort(key=lambda item: item["frame_index"])
        manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
        atomic_json(manifest_path, manifest)
        raise
    manifest["frames"].sort(key=lambda item: item["frame_index"])
    manifest["complete"] = len(manifest["frames"]) == len(expected)
    manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
    atomic_json(manifest_path, manifest)
    print(json.dumps({"manifest": str(manifest_path),
                      "frames": len(manifest["frames"]),
                      "complete": manifest["complete"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
