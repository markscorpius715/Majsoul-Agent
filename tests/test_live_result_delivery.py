from concurrent.futures import Future
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
import time
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest

from majsoul.calibration.frames import read_image, write_image
from majsoul.live.capture import LiveFrame
from majsoul.live.collector import CollectConfig, MatchCollector
from majsoul.live.config import LiveConfig, NotifyConfig
from majsoul.live.notifications import NotificationQueue
from majsoul.live.notify import FeishuNotifier
from majsoul.live.results import ResultDelivery
from majsoul.live.session import AgentSession

ROOT = Path(__file__).resolve().parents[1]


def metadata(image):
    return {"purpose": "match_result", "session_id": "acceptance", "match_index": 1,
            "mode": "agent", "image_sha256": sha256(image.read_bytes()).hexdigest()}


def image_file(tmp_path):
    image = tmp_path / "scorecard.png"
    write_image(image, np.full((32, 64, 3), 180, np.uint8))
    return image


def runner(calls, *, fail_image=False):
    def run(command, **kwargs):
        calls.append(command[2])
        failure = fail_image and command[2] == "image"
        return subprocess.CompletedProcess(command, int(failure),
                                           json.dumps({"ready": True}), "TLS" if failure else "")
    return run


def notifier(calls, *, fail_image=False, target="test"):
    return FeishuNotifier(NotifyConfig(True, Path("helper.mjs"), target),
                          runner=runner(calls, fail_image=fail_image))


def read_job(directory):
    return json.loads(next(directory.glob("*.json")).read_text(encoding="utf-8"))


def test_partial_image_failure_survives_new_notifier_and_queue(tmp_path):
    image = image_file(tmp_path)
    directory = tmp_path / "notifications"
    calls = []
    first = NotificationQueue(notifier(calls, fail_image=True), directory,
                              max_attempts=2, retry_seconds=.01)
    assert first.submit("notify_text", "final", image, metadata=metadata(image)).result(2) is False
    assert first.close(1) == 0
    failed = read_job(directory)
    assert failed["attempts"] == 2 and failed["status"] == "failed"
    assert calls.count("text") == 1 and calls.count("image") == 2
    assert {p["status"] for p in failed["delivery"].values()} == {"delivered", "failed"}
    # A new process has no in-memory dedupe set; only persisted acknowledgments.
    resumed_calls = []
    resumed = NotificationQueue(notifier(resumed_calls), directory)
    assert resumed.recovery_report()[0]["eligible"]
    futures = resumed.recover_results(limit=1)
    assert len(futures) == 1 and futures[0].result(2)
    assert resumed.close(1) == 0
    assert resumed_calls == ["check-config", "image"]
    delivered = read_job(directory)
    assert delivered["status"] == "delivered" and delivered["attempts"] == 3
    assert delivered["recovery_count"] == 1 and "error" not in delivered
    assert all(p["status"] == "delivered" and p["platform_message_id"] is None
               for p in delivered["delivery"].values())
    third_calls = []
    third = NotificationQueue(notifier(third_calls), directory)
    assert third.submit("notify_text", "final", image, metadata=metadata(image)).result(1)
    assert third.recover_results(limit=1) == []
    assert third.close(1) == 0 and third_calls == []


def test_canonical_review_queue_binds_both_images_and_delivery_receipts(tmp_path):
    case=tmp_path/'case_000001'
    write_image(case/'crop.png',np.full((8,8,3),100,np.uint8))
    write_image(case/'full_frame.png',np.full((16,16,3),120,np.uint8))
    (case/'metadata.json').write_text(json.dumps({'status':'pending'}))
    calls=[]
    directory=tmp_path/'notifications'
    queue=NotificationQueue(notifier(calls),directory)
    try:
        assert queue.submit('notify_unknown_button',case,'human review',
            metadata={'purpose':'human_review','case_id':'case_000001'}).result(2)
    finally:
        assert queue.close(1)==0
    job=read_job(directory)
    assert {Path(a['path']).name for a in job['attachments']}=={'crop.png','full_frame.png'}
    assert all(a['sha256']==sha256(Path(a['path']).read_bytes()).hexdigest() for a in job['attachments'])
    assert calls==['check-config','text','image','image']
    assert len(job['delivery'])==3 and all(p['status']=='delivered' for p in job['delivery'].values())


def test_canonical_review_missing_full_frame_cannot_be_queued(tmp_path):
    case=tmp_path/'case_000001'
    write_image(case/'crop.png',np.zeros((8,8,3),np.uint8))
    (case/'metadata.json').write_text(json.dumps({'status':'pending'}))
    calls=[]
    queue=NotificationQueue(notifier(calls),tmp_path/'notifications')
    with pytest.raises(ValueError,match='lacks crop or full frame'):
        queue.submit('notify_unknown_button',case,'human review')
    assert queue.close(1)==0 and calls==[]


@pytest.mark.parametrize("change", ["replace", "missing", "destination"])
def test_recovery_rejects_changed_image_or_destination(tmp_path, change):
    image = image_file(tmp_path)
    directory = tmp_path / "notifications"
    first = NotificationQueue(notifier([], fail_image=True), directory)
    assert not first.submit("notify_text", "final", image, metadata=metadata(image)).result(2)
    first.close(1)
    if change == "replace":
        write_image(image, np.zeros((32, 64, 3), np.uint8))
    elif change == "missing":
        image.unlink()
    calls = []
    resumed = NotificationQueue(notifier(calls, target="other" if change == "destination" else "test"), directory)
    report = resumed.recovery_report()
    assert len(report) == 1 and not report[0]["eligible"] and report[0]["error"]
    assert resumed.recover_results(limit=1) == []
    assert resumed.close(1) == 0 and calls == []


@pytest.mark.parametrize("status,part_status", [("sending", "sending"), ("failed", "sending")])
def test_interrupted_send_is_not_automatically_replayed(tmp_path, status, part_status):
    image = image_file(tmp_path)
    directory = tmp_path / "notifications"
    first = NotificationQueue(notifier([], fail_image=True), directory)
    assert not first.submit("notify_text", "final", image, metadata=metadata(image)).result(2)
    first.close(1)
    path = next(directory.glob("*.json"))
    job = read_job(directory)
    job["status"] = status
    next(p for p in job["delivery"].values() if p["kind"] == "image")["status"] = part_status
    path.write_text(json.dumps(job), encoding="utf-8")
    calls = []
    resumed = NotificationQueue(notifier(calls), directory)
    assert not resumed.recovery_report()[0]["eligible"]
    assert resumed.recover_results(limit=1) == []
    resumed.close(1)
    assert calls == []


def test_recovery_scope_excludes_unknown_images_legacy_and_broken_records(tmp_path):
    image = image_file(tmp_path)
    directory = tmp_path / "notifications"
    first = NotificationQueue(notifier([], fail_image=True), directory)
    assert not first.submit("notify_text", "preview", image).result(2)
    first.close(1)
    (directory / "legacy.json").write_text(json.dumps({"id": "legacy", "status": "failed"}))
    (directory / "broken.json").write_text("{")
    calls = []
    resumed = NotificationQueue(notifier(calls), directory)
    assert len(resumed.recovery_report()) == 3
    assert not any(row["eligible"] for row in resumed.recovery_report())
    assert resumed.recover_results(limit=1) == []
    resumed.close(1)
    assert calls == []


def test_queue_lease_prevents_concurrent_delivery_even_after_bounded_close(tmp_path):
    entered, release = Event(), Event()
    def send(*args):
        entered.set()
        assert release.wait(3)
        return True
    first = NotificationQueue(SimpleNamespace(notify_text=send), tmp_path)
    future = first.submit("notify_text", "one")
    assert entered.wait(1)
    assert first.close(.01) == 1
    second = NotificationQueue(SimpleNamespace(notify_text=send), tmp_path)
    try:
        with pytest.raises(RuntimeError, match="live worker"):
            second.submit("notify_text", "two")
    finally:
        release.set()
        assert future.result(2)
        first.close(1)
    assert second.submit("notify_text", "two").result(2)
    assert second.close(1) == 0


def test_image_change_between_retries_is_detected(tmp_path):
    image = image_file(tmp_path)
    calls = []
    def send(*args):
        calls.append(args)
        write_image(image, np.zeros((32, 64, 3), np.uint8))
        return False
    queue = NotificationQueue(SimpleNamespace(notify_text=send), tmp_path / "notifications",
                              max_attempts=3, retry_seconds=.01)
    assert not queue.submit("notify_text", "final", image, metadata=metadata(image)).result(2)
    assert queue.close(1) == 0 and len(calls) == 1
    assert "identity changed" in read_job(queue.directory)["error"]


def test_result_priority_and_deferred_result_survive_full_queue(tmp_path):
    entered, release = Event(), Event()
    calls = []
    def send(message, *args):
        calls.append(message)
        if message == "blocking":
            entered.set()
            assert release.wait(3)
        return True
    queue = NotificationQueue(SimpleNamespace(notify_text=send), tmp_path / "notifications", capacity=2)
    image = image_file(tmp_path)
    blocking = queue.submit("notify_text", "blocking")
    assert entered.wait(1)
    routine = queue.submit("notify_text", "routine")
    result = queue.submit("notify_text", "result", image, metadata=metadata(image))
    deferred = queue.submit("notify_text", "result-two", image, metadata=metadata(image))
    assert deferred.result(1) is False
    release.set()
    assert blocking.result(2) and routine.result(2) and result.result(2)
    assert queue.close(1) == 0
    assert calls == ["blocking", "result", "routine"]
    resumed = NotificationQueue(SimpleNamespace(notify_text=send), queue.directory, capacity=2)
    assert len(resumed.recover_results(limit=1)) == 1
    assert resumed.close(1) == 0
    assert calls[-1] == "result-two"


@pytest.mark.parametrize("execute", [False, True])
def test_agent_queues_scorecard_before_any_confirm_and_lobby_is_separate(tmp_path, execute):
    config = CollectConfig.load(ROOT / "configs/collect.steam.yaml")
    # A real historical scorecard; this test does not send it externally.
    pixels = read_image(ROOT / "artifacts/live/async-readonly-observe-20261008/frame.png")
    frame = LiveFrame(pixels, 10, (0, 0, pixels.shape[1], pixels.shape[0]), (0, 0))
    delivered = Future()
    steps = []
    def notify(message, image=None, **kwargs):
        if image is not None:
            assert read_image(image).shape == pixels.shape
            assert kwargs["metadata"]["image_sha256"] == sha256(image.read_bytes()).hexdigest()
            steps.append("queue-image")
            return delivered
        steps.append("lobby-text")
        return True
    def click(*args, **kwargs):
        assert steps[0] == "queue-image" and not delivered.done()
        steps.append("confirm")
    session = AgentSession(SimpleNamespace(click=click), config, execute=execute,
                           capture=lambda: frame, sleep=lambda _: None, log=lambda *a, **k: None,
                           directory=tmp_path, notify=notify)
    session.classify = lambda f: (False, "gold", False)
    session.ui = SimpleNamespace(match_result=lambda im: True, confirm=lambda im: "gold")
    session.observe(frame, 10)
    session.observe(frame, 10 + config.table_ready_seconds)
    assert steps == ["queue-image", "confirm"] if execute else steps == ["queue-image"]
    record_path = tmp_path / "match-0001-result-delivery.json"
    assert json.loads(record_path.read_text(encoding="utf-8"))["notification_status_snapshot"] == "queued"
    session.classify = lambda f: (True, None, False)
    session.ui.match_result = lambda im: False
    session.observe(frame, 20)
    session.observe(frame, 20 + config.table_ready_seconds)
    assert steps[-1] == "lobby-text" and steps.count("queue-image") == 1
    assert json.loads(record_path.read_text(encoding="utf-8"))["returned_lobby"] is True
    delivered.set_result(True)
    assert json.loads(record_path.read_text(encoding="utf-8"))["notification_status_snapshot"] == "delivered"


def test_result_enqueue_failure_prevents_confirm_without_overwriting_scorecard(tmp_path):
    config = CollectConfig.load(ROOT / "configs/collect.steam.yaml")
    frame = LiveFrame(np.zeros((200, 400, 3), np.uint8), 0, (0, 0, 400, 200), (0, 0))
    def notify(*args, **kwargs):
        raise OSError("disk full")
    clicks = []
    session = AgentSession(SimpleNamespace(click=lambda *a, **k: clicks.append(a)), config,
                           execute=True, capture=lambda: frame, sleep=lambda _: None,
                           log=lambda *a, **k: None, directory=tmp_path, notify=notify)
    session.classify = lambda f: (False, "gold", False)
    session.ui = SimpleNamespace(match_result=lambda im: True, confirm=lambda im: "gold")
    with pytest.raises(OSError, match="disk full"):
        session.observe(frame, 0)
    assert not clicks and (tmp_path / "match_result.png").is_file()
    assert not session.result_delivery.submitted


def test_collector_uses_same_save_queue_contract_and_does_not_resend(tmp_path):
    live = LiveConfig.load(ROOT / "configs/live.steam.yaml")
    config = replace(CollectConfig.load(ROOT / "configs/collect.steam.yaml"), dataset_root=tmp_path)
    calls = []
    transport = SimpleNamespace(notify_text=lambda *args: calls.append(args) or True)
    collector = MatchCollector(SimpleNamespace(), config, live, notifier=transport, session="test")
    collector._match_index = 2
    frame = LiveFrame(np.zeros((200, 400, 3), np.uint8), 1, (0, 0, 400, 200), (0, 0))
    collector.ui = SimpleNamespace(match_result=lambda im: True)
    collector._capture_result(frame)
    collector._capture_result(frame)
    assert len(calls) == 1 and collector._result_image.is_file()
    assert isinstance(collector._result_delivery, ResultDelivery)
    record = json.loads((collector.artifacts / "match-0002-result-delivery.json").read_text(encoding="utf-8"))
    assert record["mode"] == "collect" and record["match_index"] == 2
    assert record["notification_status_snapshot"] == "delivered"
    collector.references.close()


def test_retry_cli_defaults_to_read_only_without_model_window_or_send(tmp_path, capsys, monkeypatch):
    import majsoul.live.__main__ as cli
    import majsoul.live.notify as notify_module
    directory = tmp_path / "notifications"
    directory.mkdir()
    (directory / "legacy.json").write_text(json.dumps({"id": "legacy", "status": "failed"}))
    def forbidden(*args, **kwargs):
        raise AssertionError("No external operation in preview")
    monkeypatch.setattr(notify_module.FeishuNotifier, "_run", forbidden)
    monkeypatch.setattr(cli.LiveWindow, "find", forbidden)
    assert cli.main(["notify-retry", "--directory", str(directory)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["execute"] is False and not report["entries"][0]["eligible"]
    assert list(directory.iterdir()) == [directory / "legacy.json"]


@pytest.mark.parametrize("stop", ["close", "timeout"])
def test_real_helper_process_is_reaped_and_unacknowledged_send_is_unknown(tmp_path, stop):
    entered = tmp_path / "entered"
    helper = tmp_path / "helper.py"
    helper.write_text(
        "import sys,json,time\nfrom pathlib import Path\n"
        "if sys.argv[1]=='check-config': print(json.dumps({'ready':True}))\n"
        f"else:\n Path({str(entered)!r}).touch()\n time.sleep(10)\n", encoding="utf-8")
    transport = FeishuNotifier(NotifyConfig(True, helper, "test", sys.executable,
                                          timeout_seconds=.5 if stop == "timeout" else 10))
    queue = NotificationQueue(transport, tmp_path / "notifications", max_attempts=3)
    future = queue.submit("notify_text", "phase")
    deadline = time.monotonic() + 2
    while not entered.exists() and time.monotonic() < deadline:
        time.sleep(.01)
    assert entered.exists()
    process = transport._process
    assert process is not None and process.poll() is None
    if stop == "close":
        started = time.monotonic()
        queue.close(.02)
        assert time.monotonic()-started < .5
    assert future.result(2) is False
    assert process.wait(timeout=2) != 0
    assert queue.close(1) == 0
    receipt = read_job(queue.directory)
    assert receipt["status"] == "delivery_unknown" and receipt["attempts"] == 1
    assert next(iter(receipt["delivery"].values()))["status"] == "unknown"
    assert queue.recovery_report()[0]["reason"] == "interrupted_send_requires_review"


def test_result_intent_links_to_canonical_notification_ledger(tmp_path):
    queue = NotificationQueue(SimpleNamespace(notify_text=lambda *a: True), tmp_path / "notifications")
    def notify(message, image, **kwargs):
        return queue.submit("notify_text", message, image, **kwargs)
    delivery = ResultDelivery(tmp_path, mode="agent", match_index=1, notify=notify,
                              log=lambda *a, **k: None)
    frame = LiveFrame(np.zeros((32,64,3),np.uint8), 1, (0,0,64,32), (0,0))
    path = image_file(tmp_path)
    delivery.capture(frame, lambda _: path)
    assert queue.close(1) == 0
    record = json.loads((tmp_path / "match-0001-result-delivery.json").read_text(encoding="utf-8"))
    ledger = Path(record["delivery_ledger"])
    assert ledger.is_file() and record["notification_id"] == ledger.stem
    assert json.loads(ledger.read_text(encoding="utf-8"))["status"] == "delivered"


def test_identical_image_bytes_in_distinct_matches_are_each_delivered(tmp_path):
    calls = []
    queue = NotificationQueue(notifier(calls), tmp_path / "notifications")
    image = image_file(tmp_path)
    for match in (1, 2):
        context = {**metadata(image), "match_index": match}
        assert queue.submit("notify_text", "final", image, metadata=context).result(2)
    assert queue.close(1) == 0
    assert calls == ["check-config", "text", "image", "check-config", "text", "image"]
