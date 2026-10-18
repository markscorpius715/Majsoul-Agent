import json

import numpy as np

from majsoul.calibration.frames import write_image
from majsoul.dataset_browser import attach_context, build_catalog, load_records, reference_tags


def test_reference_ui_presence_does_not_create_game_event():
    tags = reference_tags({"reason": "visual_change", "observation": {
        "action_buttons": ["ron"], "players": [{"melds": [{"kind": "ankan"}]}]}})
    assert "button:ron" in tags
    assert "meld:ankan" in tags
    assert "event:RON" not in tags


def test_context_excludes_other_rounds_and_unsaved_time_gaps():
    entries = [{"id": str(i), "session": "s", "match_index": 1,
                "round_index": r, "time": t} for i, (r, t) in enumerate(
                    [(1, 1), (1, 2), (2, 2), (1, 10)])]
    attach_context(entries, 3)
    assert entries[0]["context"] == ["1"]
    assert entries[2]["context"] == []
    assert entries[3]["context"] == []


def test_catalog_reads_old_manifest_preserves_original_and_recovers_orphans(tmp_path):
    source = tmp_path / "source"
    session = source / "session"
    image = np.zeros((80, 100, 3), dtype=np.uint8)
    write_image(session / "frames" / "1.jpg", image)
    write_image(session / "frames" / "2.jpg", image)
    manifest = session / "observations.jsonl"
    manifest.write_text(json.dumps({"file": "frames/1.jpg", "seq": 1,
                                   "reason": "confirm_appeared", "captured_at": 1})
                        + '\n{"file": "../../outside.jpg"}\n{broken\n', encoding="utf-8")
    original = manifest.read_bytes()
    records, warnings = load_records(source)
    assert len(records) == 2
    assert len(warnings) == 2
    report = build_catalog(source, tmp_path / "catalog")
    assert report["entries"] == 2
    assert manifest.read_bytes() == original
    data = json.loads((tmp_path / "catalog/catalog.json").read_text(encoding="utf-8"))
    assert all(not entry["labels_verified"] for entry in data["entries"])
    assert "__CATALOG__" not in (tmp_path / "catalog/index.html").read_text(encoding="utf-8")
