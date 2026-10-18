from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from typing import Iterable, Mapping

from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError


def _load_pyarrow():
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError(
            "Parquet audit requires the optional data dependency: pip install -e .[data]"
        ) from exc
    return parquet


def _class_names(schema) -> tuple[str, ...]:
    metadata = schema.metadata or {}
    raw = metadata.get(b"huggingface")
    if raw is None:
        raise ValueError("Parquet schema has no Hugging Face feature metadata")
    try:
        feature_metadata = json.loads(raw.decode("utf-8"))
        names = feature_metadata["info"]["features"]["label"]["names"]
    except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError("Parquet label metadata is invalid") from exc
    if not isinstance(names, list) or not names or not all(
            isinstance(name, str) and name for name in names):
        raise ValueError("Parquet label names must be a non-empty string list")
    if len(names) != len(set(names)):
        raise ValueError("Parquet label names must be unique")
    return tuple(names)


def _image_bytes(value: object, *, split: str, row_index: int) -> bytes:
    if not isinstance(value, dict):
        raise ValueError(f"{split}[{row_index}] image is not a struct")
    payload = value.get("bytes")
    if not isinstance(payload, bytes) or not payload:
        raise ValueError(f"{split}[{row_index}] has no embedded image bytes")
    return payload


def _pixel_hash(image: Image.Image) -> str:
    normalized = image.convert("RGB")
    header = f"{normalized.width}x{normalized.height}:RGB:".encode("ascii")
    return sha256(header + normalized.tobytes()).hexdigest()


def _difference_hash(image: Image.Image) -> int:
    grayscale = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    pixels = grayscale.tobytes()
    result = 0
    for row in range(8):
        offset = row * 9
        for column in range(8):
            result = (result << 1) | int(
                pixels[offset + column] > pixels[offset + column + 1])
    return result


def _file_record(path: Path) -> dict[str, object]:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": digest.hexdigest()}


def _duplicate_summary(
    records: Mapping[str, list[dict[str, object]]], *, max_examples: int,
) -> dict[str, object]:
    groups = [items for items in records.values() if len(items) > 1]
    cross_split = [items for items in groups if len({item["split"] for item in items}) > 1]
    cross_label = [items for items in groups if len({item["label"] for item in items}) > 1]
    return {
        "group_count": len(groups),
        "duplicate_sample_count": sum(len(items) - 1 for items in groups),
        "cross_split_group_count": len(cross_split),
        "cross_label_group_count": len(cross_label),
        "examples": groups[:max_examples],
    }


def _near_cross_split(
    rows: list[dict[str, object]], threshold: int, max_examples: int,
) -> dict[str, object]:
    by_split: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_split[str(row["split"])].append(row)
    split_names = sorted(by_split)
    count = 0
    cross_label_count = 0
    examples: list[dict[str, object]] = []
    for left_index, left_split in enumerate(split_names):
        for right_split in split_names[left_index + 1:]:
            for left in by_split[left_split]:
                for right in by_split[right_split]:
                    if left["pixel_sha256"] == right["pixel_sha256"]:
                        continue
                    distance = (int(left["dhash64"]) ^ int(right["dhash64"])).bit_count()
                    if distance > threshold:
                        continue
                    count += 1
                    if left["label"] != right["label"]:
                        cross_label_count += 1
                    if len(examples) < max_examples:
                        examples.append({
                            "distance": distance,
                            "left": {key: left[key] for key in ("split", "row", "label")},
                            "right": {key: right[key] for key in ("split", "row", "label")},
                        })
    return {"pair_count": count, "cross_label_pair_count": cross_label_count,
            "examples": examples}


def _contact_sheet(
    representatives: Mapping[str, Image.Image], labels: Iterable[str], path: Path,
) -> None:
    labels = tuple(labels)
    columns = min(7, max(1, len(labels)))
    rows = (len(labels) + columns - 1) // columns
    cell_width, cell_height = 150, 190
    sheet = Image.new("RGB", (columns * cell_width, rows * cell_height), "#eeeeee")
    draw = ImageDraw.Draw(sheet)
    for index, label in enumerate(labels):
        x = (index % columns) * cell_width
        y = (index // columns) * cell_height
        tile = ImageOps.contain(representatives[label].convert("RGB"), (130, 150))
        sheet.paste(tile, (x + (cell_width - tile.width) // 2, y + 8))
        draw.text((x + 8, y + 166), label, fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    sheet.save(temporary, format="PNG")
    temporary.replace(path)


def audit_parquet_splits(
    split_paths: Mapping[str, str | Path], *, report_path: str | Path,
    contact_sheet_path: str | Path | None = None, near_hamming: int = 1,
    max_examples: int = 25,
) -> dict[str, object]:
    """Audit embedded-image Parquet splits without admitting samples to training."""
    if not split_paths:
        raise ValueError("At least one split is required")
    if near_hamming < 0 or near_hamming > 64:
        raise ValueError("near_hamming must be between 0 and 64")
    if max_examples < 1:
        raise ValueError("max_examples must be positive")
    parquet = _load_pyarrow()
    class_names: tuple[str, ...] | None = None
    representatives: dict[str, Image.Image] = {}
    raw_hashes: dict[str, list[dict[str, object]]] = defaultdict(list)
    pixel_hashes: dict[str, list[dict[str, object]]] = defaultdict(list)
    all_rows: list[dict[str, object]] = []
    split_reports: dict[str, object] = {}
    files: list[dict[str, object]] = []

    for split, raw_path in sorted(split_paths.items()):
        if not split or any(character in split for character in "/\\"):
            raise ValueError(f"Invalid split name: {split!r}")
        path = Path(raw_path)
        if not path.is_file():
            raise ValueError(f"Parquet file does not exist: {path}")
        table = parquet.read_table(path, columns=["image", "label"])
        names = _class_names(table.schema)
        if class_names is None:
            class_names = names
        elif names != class_names:
            raise ValueError(f"Label metadata differs in split {split}")
        files.append({"split": split, **_file_record(path)})
        label_counts: Counter[str] = Counter()
        dimensions: Counter[str] = Counter()
        formats: Counter[str] = Counter()
        invalid: list[dict[str, object]] = []
        images = table.column("image").to_pylist()
        labels = table.column("label").to_pylist()
        for row_index, (image_value, label_index) in enumerate(zip(images, labels)):
            if not isinstance(label_index, int) or not 0 <= label_index < len(names):
                raise ValueError(f"{split}[{row_index}] has invalid label index {label_index!r}")
            label = names[label_index]
            payload = _image_bytes(image_value, split=split, row_index=row_index)
            try:
                with Image.open(BytesIO(payload)) as opened:
                    opened.load()
                    decoded = opened.copy()
                    image_format = opened.format or "unknown"
            except (OSError, UnidentifiedImageError) as exc:
                invalid.append({"row": row_index, "label": label,
                                "error": type(exc).__name__})
                continue
            label_counts[label] += 1
            dimensions[f"{decoded.width}x{decoded.height}"] += 1
            formats[image_format] += 1
            representatives.setdefault(label, decoded)
            record: dict[str, object] = {
                "split": split,
                "row": row_index,
                "label": label,
                "raw_sha256": sha256(payload).hexdigest(),
                "pixel_sha256": _pixel_hash(decoded),
                "dhash64": _difference_hash(decoded),
            }
            all_rows.append(record)
            raw_hashes[str(record["raw_sha256"])].append(record)
            pixel_hashes[str(record["pixel_sha256"])].append(record)
        split_reports[split] = {
            "rows": table.num_rows,
            "decoded_rows": sum(label_counts.values()),
            "invalid_rows": invalid,
            "label_counts": dict(sorted(label_counts.items())),
            "dimensions": dict(sorted(dimensions.items())),
            "formats": dict(sorted(formats.items())),
        }

    assert class_names is not None
    missing_representatives = sorted(set(class_names) - set(representatives))
    report: dict[str, object] = {
        "schema_version": 1,
        "status": "audited_not_ingested",
        "files": files,
        "class_names": list(class_names),
        "class_count": len(class_names),
        "total_rows": sum(int(value["rows"]) for value in split_reports.values()),
        "splits": split_reports,
        "missing_representatives": missing_representatives,
        "duplicates": {
            "raw_bytes": _duplicate_summary(raw_hashes, max_examples=max_examples),
            "decoded_pixels": _duplicate_summary(pixel_hashes, max_examples=max_examples),
            "near_cross_split": _near_cross_split(
                all_rows, near_hamming, max_examples),
        },
        "admission": {
            "training_allowed": False,
            "reason": "Audit does not assign internal labels or protect a golden split",
        },
    }
    destination = Path(report_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    temporary.replace(destination)
    if contact_sheet_path is not None:
        if missing_representatives:
            raise ValueError(
                f"Cannot build contact sheet; missing labels: {missing_representatives}")
        _contact_sheet(representatives, class_names, Path(contact_sheet_path))
    return report


def parse_split_specs(values: Iterable[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("Split must use NAME=PATH")
        name, raw_path = value.split("=", 1)
        name, raw_path = name.strip(), raw_path.strip()
        if not name or not raw_path or name in result:
            raise ValueError(f"Invalid or duplicate split: {value!r}")
        result[name] = Path(raw_path)
    return result
