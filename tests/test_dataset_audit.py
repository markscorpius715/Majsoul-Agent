from __future__ import annotations

from io import BytesIO
import json

from PIL import Image
import pytest

from majsoul.continual.dataset_audit import (
    audit_parquet_splits,
    parse_split_specs,
)


pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")


def _png(color: tuple[int, int, int], size: tuple[int, int] = (24, 32)) -> bytes:
    stream = BytesIO()
    Image.new("RGB", size, color).save(stream, format="PNG")
    return stream.getvalue()


def _write_parquet(path, rows, names=("alpha", "beta")):
    image_type = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
    metadata = {
        b"huggingface": json.dumps({
            "info": {"features": {
                "image": {"_type": "Image"},
                "label": {"names": list(names), "_type": "ClassLabel"},
            }}
        }).encode("utf-8")
    }
    schema = pa.schema([
        pa.field("image", image_type),
        pa.field("label", pa.int64()),
    ], metadata=metadata)
    table = pa.Table.from_arrays([
        pa.array([{"bytes": image, "path": f"{index}.png"}
                  for index, (image, _) in enumerate(rows)], type=image_type),
        pa.array([label for _, label in rows], type=pa.int64()),
    ], schema=schema)
    pq.write_table(table, path)


def test_audit_detects_cross_split_duplicates_and_stays_non_admitted(tmp_path):
    red, blue = _png((255, 0, 0)), _png((0, 0, 255))
    train, test = tmp_path / "train.parquet", tmp_path / "test.parquet"
    _write_parquet(train, [(red, 0), (blue, 1)])
    _write_parquet(test, [(red, 0), (_png((0, 255, 0)), 1)])
    report_path, sheet_path = tmp_path / "audit.json", tmp_path / "sheet.png"

    report = audit_parquet_splits(
        {"train": train, "test": test}, report_path=report_path,
        contact_sheet_path=sheet_path, near_hamming=0)

    assert report["total_rows"] == 4
    assert report["class_names"] == ["alpha", "beta"]
    assert report["duplicates"]["raw_bytes"]["cross_split_group_count"] == 1
    assert report["duplicates"]["decoded_pixels"]["cross_split_group_count"] == 1
    assert report["admission"]["training_allowed"] is False
    assert report_path.is_file()
    assert sheet_path.is_file()


def test_audit_rejects_different_label_schemas(tmp_path):
    train, test = tmp_path / "train.parquet", tmp_path / "test.parquet"
    _write_parquet(train, [(_png((1, 2, 3)), 0)])
    _write_parquet(test, [(_png((1, 2, 3)), 0)], names=("other",))

    with pytest.raises(ValueError, match="Label metadata differs"):
        audit_parquet_splits(
            {"train": train, "test": test}, report_path=tmp_path / "audit.json")


@pytest.mark.parametrize("value", ["train", "=file.parquet", "train="])
def test_parse_split_specs_rejects_invalid_values(value):
    with pytest.raises(ValueError):
        parse_split_specs([value])


def test_parse_split_specs_rejects_duplicates():
    with pytest.raises(ValueError, match="duplicate"):
        parse_split_specs(["train=a.parquet", "train=b.parquet"])
