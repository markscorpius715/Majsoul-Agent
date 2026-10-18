import asyncio
import json
import time

import httpx
import pytest

from majsoul.strategy import DeepSeekPolicy, ModelSettings, StrategyError
from majsoul.benchmark import scenarios, summarize


def event(content=None, reason=None, **extra):
    value = {"choices": [{"index": 0, "delta": {"content": content}, "finish_reason": reason}], **extra}
    return ("data: " + json.dumps(value) + "\n\n").encode()


class Chunks(httpx.AsyncByteStream):
    def __init__(self, parts, delay=0):
        self.parts, self.delay, self.closed = parts, delay, False

    async def __aiter__(self):
        for part in self.parts:
            if self.delay:
                await asyncio.sleep(self.delay)
            yield part

    async def aclose(self):
        self.closed = True


def make_policy(parts, *, timeout=8, delay=0):
    chunks = Chunks(parts, delay)
    def respond(request):
        payload = json.loads(request.content)
        assert payload["stream"] is True
        assert payload["thinking"] == {"type": "disabled"}
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=chunks)
    return DeepSeekPolicy(ModelSettings("test-only", timeout=timeout), httpx.AsyncClient(transport=httpx.MockTransport(respond))), chunks


def test_sse_fragmented_bytes_json_and_comments(engine, state):
    action = engine.legal_actions(state)[0]
    answer = json.dumps({"action_id": action.id})
    payload = b": keepalive\n\n" + event() + event(answer[:5]) + event(answer[5:]) + event(reason="stop", usage={"completion_tokens": 12}) + b"data: [DONE]\n\n"
    policy, chunks = make_policy([payload[i:i+7] for i in range(0,len(payload),7)])
    try:
        assert policy.choose(state, [action]) == action
        timing = policy.last_timing
        assert timing["first_event_seconds"] <= timing["first_content_seconds"] <= timing["first_valid_json_seconds"] <= timing["decision_seconds"] <= timing["total_seconds"]
        assert timing["usage"]["completion_tokens"] == 12
        assert timing["content_chunks"] == 2
        assert chunks.closed
    finally:
        policy.close()


@pytest.mark.parametrize("suffix", [
    [], [event(reason="length"), b"data: [DONE]\n\n"],
    [b"data: [DONE]\n\n"],
    [event(' garbage'), event(reason="stop"), b"data: [DONE]\n\n"],
    [event(reason="stop"), event(' garbage'), b"data: [DONE]\n\n"],
    [event(reason="stop")],  # Missing stream terminator is not success.
])
def test_valid_prefix_not_enough_to_execute(engine, state, suffix):
    action = engine.legal_actions(state)[0]
    policy, chunks = make_policy([event(json.dumps({"action_id": action.id})), *suffix])
    try:
        with pytest.raises(StrategyError):
            policy.choose(state, [action])
        assert policy.last_timing["status"] == "invalid_response"
        assert chunks.closed
    finally:
        policy.close()


def test_wall_deadline_interrupts_continuous_keepalive(engine, state):
    policy, chunks = make_policy([b": heartbeat\n\n"] * 100, timeout=.04, delay=.01)
    started = time.monotonic()
    try:
        with pytest.raises(StrategyError, match="deadline"):
            policy.choose(state, engine.legal_actions(state))
        assert time.monotonic()-started < .5
        assert policy.last_timing["status"] == "timeout"
        assert chunks.closed
    finally:
        policy.close()


def test_external_deadline_overrides_longer_api_timeout(engine, state):
    policy, chunks = make_policy([b": heartbeat\n\n"] * 100, timeout=8, delay=.01)
    try:
        with pytest.raises(StrategyError, match="deadline"):
            policy.choose(state, engine.legal_actions(state), deadline=time.monotonic()+.04)
        assert policy.last_timing["total_seconds"] < .5
    finally:
        policy.close()


def test_duplicate_json_action_key_rejected(engine, state):
    action = engine.legal_actions(state)[0]
    content = '{"action_id":"' + action.id + '","action_id":"' + action.id + '"}'
    with pytest.raises(StrategyError):
        DeepSeekPolicy._select(content, [action])


def test_benchmark_snapshots_respect_tile_inventory(engine):
    for seed in range(3):
        cases = list(scenarios(engine, seed))
        assert len(cases) == 8
        for _, state in cases:
            engine.rules.validate_tiles(engine.visible(state))
            assert engine.legal_actions(state)


def test_failure_denominator_is_not_discarded():
    rows = [{"mode":"stream", "status":"ok", "total_seconds":1, "decision_seconds":1,
             "first_content_seconds":.5, "features_seconds":.1},
            {"mode":"stream", "status":"timeout"}]
    summary = summarize(rows)["stream"]
    assert summary["attempts"] == 2
    assert summary["successes"] == summary["valid_within_5s"] == 1
