import json

import numpy as np

from majsoul.live.capture import LiveFrame
from majsoul.live.reference_archive import ReferenceArchive


def frame(timestamp):
    return LiveFrame(np.zeros((60, 100, 3), dtype=np.uint8), timestamp,
                     (0, 0, 100, 60), (0, 0))


def test_reference_archive_saves_actual_before_after_without_fake_event(tmp_path):
    archive = ReferenceArchive(tmp_path, before_seconds=2, after_seconds=2)
    for index in range(4):
        archive.observe(frame(index), index, 1, 1)
    archive.trigger(["button:ron"], frame(3), 3, 1, 1)
    archive.observe(frame(4), 4, 1, 1)
    archive.observe(frame(6), 6, 1, 1)
    case = json.loads((tmp_path / "000001/clip.json").read_text())
    assert [f["frame_index"] for f in case["frames"]] == [1, 2, 3, 4]
    assert case["triggers"][0]["tags"] == ["button:ron"]
    assert case["complete"]
    assert case["available_before_seconds"] == 2
    assert case["available_after_seconds"] == 1
    assert "events" not in case


def test_reference_archive_never_carries_context_to_another_round(tmp_path):
    archive = ReferenceArchive(tmp_path)
    archive.observe(frame(0), 1, 1, 0)
    archive.trigger(["capture:table_ready"], frame(0), 1, 1, 1)
    archive.observe(frame(1), 2, 1, 1)
    archive.trigger(["button:kan"], frame(1), 2, 1, 1)
    archive.observe(frame(2), 3, 1, 2)
    first = json.loads((tmp_path / "000001/clip.json").read_text())
    assert {f["round_index"] for f in first["frames"]} == {1}
    assert first["status"] == "scope_changed"
    assert not first["complete"]
    assert len(first["triggers"]) == 2
    archive.trigger(["button:ron"], frame(2), 3, 1, 2)
    archive.close()
    second = json.loads((tmp_path / "000002/clip.json").read_text())
    assert {f["round_index"] for f in second["frames"]} == {2}


def test_reference_archive_frame_capacity_is_enforced_even_for_trigger_bursts(tmp_path):
    archive = ReferenceArchive(tmp_path, maximum_frames=3)
    archive.trigger(["ui:confirm"], frame(0), 0, 1, 1)
    for index in range(1, 10):
        archive.trigger(["button:pass"], frame(index / 10), index, 1, 1)
    archive.close()
    case = json.loads((tmp_path / "000001/clip.json").read_text())
    assert len(case["frames"]) == 3
    assert case["frames_limited"]
    assert not case["complete"]
