import json
from pathlib import Path

import cv2
import pytest

from majsoul.calibration.frames import read_image, write_image
from majsoul.live.config import LiveConfig
from majsoul.state.__main__ import main


ROOT = Path(__file__).resolve().parents[1]


def test_live_profile_replays_gold_model_and_preserves_real_occlusion(tmp_path):
    torch = pytest.importorskip("torch")
    config = LiveConfig.load(ROOT / "configs/live.steam.yaml")
    source = read_image(ROOT / "artifacts/live/workflow-audit/steam-right-called-pon.jpg")
    paths = []
    for index, scale in enumerate((1., .75)):
        path = tmp_path / f"frame_{index:04d}.png"
        write_image(path, cv2.resize(source, None, fx=scale, fy=scale))
        paths.append(str(path))
    output = tmp_path / "replay.json"
    threads = torch.get_num_threads()
    torch.set_num_threads(4)
    try:
        assert main(["images", *paths, "--live-config", str(ROOT / "configs/live.steam.yaml"),
                     "--active-slots", "bottom", "left", "right", "--export-roi",
                     "--output", str(output)]) == 0
    finally:
        torch.set_num_threads(threads)
    report = json.loads(output.read_text(encoding="utf-8"))
    pipeline = report["pipeline"]
    assert pipeline["classifier"] == "cnn"
    assert pipeline["classifier_manifest"]["sha256"] == config.cnn_expected["sha256"]
    assert pipeline["classifier_manifest"]["experiment"] == "D"
    assert pipeline["read_scores"] is False
    assert pipeline["countdown_manifest"]["rule_bounds"] == {"base": 7, "reserve": 20}
    assert pipeline["countdown_manifest"]["display_bounds"] == {"base": 8, "reserve": 20}
    for record in report["updates"]:
        players = {p["slot"]: p for p in record["observation"]["players"]}
        assert all(p["score"] is None and not p["score_available"] for p in players.values())
        assert [t["tile"] for t in players["right"]["melds"][0]["tiles"]] == ["9s"] * 3
        # This source really has a blank left seat marker during animation.
        # A correct tile classifier must not turn missing metadata into fact.
        assert players["left"]["seat"] is None
        assert not record["observation_complete"]
        assert record["resync_requested"]
        assert (Path(record["roi_review"]) / "roi_debug.png").is_file()


def test_live_profile_rejects_aspect_drift_before_reading_tiles(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import numpy as np
    from majsoul.live import classifier

    monkeypatch.setattr(classifier, "build_tile_classifier", lambda *a, **k: SimpleNamespace(
        classify_many=lambda tiles: pytest.fail("Must reject geometry before classification")))
    path = tmp_path / "frame_0.png"
    write_image(path, np.zeros((100, 100, 3), dtype=np.uint8))
    output = tmp_path / "bad.json"
    assert main(["images", str(path), "--live-config", str(ROOT / "configs/live.steam.yaml"),
                 "--active-slots", "bottom", "left", "right", "--output", str(output)]) == 1
    assert not output.exists()
