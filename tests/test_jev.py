import asyncio
import json
import time

import httpx
import pytest

from majsoul.jev import JevPolicy, JevSettings, FallbackPolicy
from majsoul.strategy import StrategyError


def response(actions):
    return {"model": "jev-test", "answers": {"action": {"type": "choice",
        "choice": actions[-1].id, "confidence": 1.0,
        "probabilities": {a.id: float(a == actions[-1]) for a in actions}}}}


def test_official_contract(engine, state):
    actions = engine.legal_actions(state)
    def respond(request):
        assert str(request.url) == "https://api.typesafe.ai/v1/systemone"
        assert request.headers["Authorization"] == "Bearer test-secret"
        body = json.loads(request.content)
        assert body["state"] == json.loads(json.dumps(state.for_model()))
        assert body["questions"]["action"]["criteria"] == {a.id: a.for_model() for a in actions}
        return httpx.Response(200, json=response(actions))
    policy = JevPolicy(JevSettings("test-secret"), httpx.AsyncClient(transport=httpx.MockTransport(respond)))
    try:
        assert policy.choose(state, actions) == actions[-1]
        assert policy.last_result["probabilities"][actions[-1].id] == 1
    finally:
        policy.close()


@pytest.mark.parametrize("mutation", ["unknown", "missing", "nan", "bool", "sum", "type", "confidence", "wrong_winner"])
def test_rejects_invalid_distribution(engine, state, mutation):
    actions = engine.legal_actions(state)
    body = response(actions)
    answer = body["answers"]["action"]
    if mutation == "unknown": answer["choice"] = "invented"
    if mutation == "missing": answer["probabilities"].pop(actions[0].id)
    if mutation == "nan": answer["probabilities"][actions[0].id] = float("nan")
    if mutation == "bool": answer["probabilities"][actions[0].id] = False
    if mutation == "sum": answer["probabilities"][actions[-1].id] = 0.5
    if mutation == "type": answer["type"] = "score"
    if mutation == "confidence": answer["confidence"] = 2
    if mutation == "wrong_winner": answer["choice"] = actions[0].id
    with pytest.raises(StrategyError):
        JevPolicy.parse(body, actions)


def test_deadline_and_secret_safe_http_error(engine, state):
    calls = []
    def respond(request):
        calls.append(request)
        return httpx.Response(401, text="test-secret")
    policy = JevPolicy(JevSettings("test-secret"), httpx.AsyncClient(transport=httpx.MockTransport(respond)))
    try:
        with pytest.raises(StrategyError, match="deadline"):
            policy.choose(state, engine.legal_actions(state), deadline=time.monotonic() - 1)
        assert not calls
        with pytest.raises(StrategyError, match="HTTP 401") as error:
            policy.choose(state, engine.legal_actions(state))
        assert "test-secret" not in str(error.value)
        assert policy.last_result is None
    finally:
        policy.close()


def test_inflight_timeout(engine, state):
    async def slow(request):
        await asyncio.sleep(1)
        return httpx.Response(200, json={})
    policy = JevPolicy(JevSettings("test", timeout=.02), httpx.AsyncClient(transport=httpx.MockTransport(slow)))
    try:
        with pytest.raises(StrategyError, match="deadline"):
            policy.choose(state, engine.legal_actions(state))
    finally:
        policy.close()


def test_fallback_shares_deadline(engine, state):
    deadlines = []
    class Backend:
        def __init__(self, fail): self.fail = fail
        def choose(self, state, actions, *, deadline):
            deadlines.append(deadline)
            if self.fail: raise StrategyError("Unavailable")
            return actions[0]
        def close(self): pass
    policy = FallbackPolicy(lambda: Backend(True), lambda: Backend(False))
    end = time.monotonic() + 2
    actions = engine.legal_actions(state)
    assert policy.choose(state, actions, deadline=end) == actions[0]
    assert deadlines == [end, end, end]
    assert policy.last_timing["selected_backend"] == "deepseek"
    policy.close()


def test_missing_primary_credentials_falls_back(engine, state):
    class Backend:
        def choose(self, state, actions, *, deadline): return actions[0]
        def close(self): pass
    policy = FallbackPolicy(lambda: JevPolicy(JevSettings("")), Backend)
    actions = engine.legal_actions(state)
    assert policy.choose(state, actions) == actions[0]
    policy.close()


def test_settings_local_then_environment(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("TYPESAFE_API_KEY=local\nJEV_PROXY=http://127.0.0.1:7890\n", encoding="utf-8")
    monkeypatch.setenv("TYPESAFE_API_KEY", "process-secret")
    monkeypatch.delenv("JEV_PROXY", raising=False)
    settings = JevSettings.load(tmp_path)
    assert settings.api_key == "process-secret"
    assert settings.proxy == "http://127.0.0.1:7890"
    assert "process-secret" not in repr(settings)


def test_primary_success_does_not_start_recovery(engine, state):
    actions = engine.legal_actions(state)
    def unused():
        pytest.fail("Recovery must remain lazy after primary success")
    class Backend:
        async def achoose(self, state, actions, *, deadline):
            return actions[0]
    policy = FallbackPolicy(Backend, unused)
    try:
        assert policy.choose(state, actions) == actions[0]
        assert len(policy.last_timing["attempts"]) == 1
        assert policy.last_timing["selected_backend"] == "jev"
    finally:
        policy.close()


@pytest.mark.parametrize("winner", ["jev", "deepseek"])
def test_recovery_starts_both_backends_and_cancels_loser(engine, state, winner):
    actions = engine.legal_actions(state)
    instances, calls, cancelled = [], [], []
    started = {"jev": asyncio.Event(), "deepseek": asyncio.Event()}
    end = time.monotonic() + 2

    class Backend:
        def __init__(self, name, fail=False):
            self.name, self.fail, self.closed = name, fail, False
            self.last_result = {"backend": name}
        async def achoose(self, state, candidates, *, deadline):
            calls.append((self.name, deadline))
            if self.fail:
                raise StrategyError("Initial connection failed")
            started[self.name].set()
            await started["deepseek" if self.name == "jev" else "jev"].wait()
            if self.name == winner:
                return candidates[0 if winner == "jev" else 1]
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.append(self.name)
        async def aclose(self):
            self.closed = True

    def primary():
        backend = Backend("jev", fail=not instances)
        instances.append(backend)
        return backend
    fallback = Backend("deepseek")
    policy = FallbackPolicy(primary, lambda: fallback)
    try:
        assert policy.choose(state, actions, deadline=end) == actions[0 if winner == "jev" else 1]
        assert len(instances) == 2 and instances[0].closed
        assert calls == [("jev", end), ("jev", end), ("deepseek", end)]
        assert cancelled == ["deepseek" if winner == "jev" else "jev"]
        assert policy.last_timing["selected_backend"] == winner
        assert policy.last_result == {"backend": winner}
        assert [r["status"] for r in policy.last_timing["attempts"]] == (
            ["failed", "ok", "cancelled"] if winner == "jev" else ["failed", "cancelled", "ok"])
        # The next decision again starts with the newly built Jev client.
        instances[1].achoose = successful_choice
        assert policy.choose(state, actions, deadline=end) == actions[0]
        assert len(policy.last_timing["attempts"]) == 1
        assert policy.last_timing["selected_backend"] == "jev"
    finally:
        policy.close()
    assert all(p.closed for p in instances) and fallback.closed


async def successful_choice(state, actions, *, deadline):
    return actions[0]


def test_illegal_recovery_result_does_not_cancel_valid_backend(engine, state):
    from majsoul.domain import Action
    actions = engine.legal_actions(state)
    created = []
    class Primary:
        async def achoose(self, state, actions, *, deadline):
            if len(created) == 1:
                raise StrategyError("Transport failed")
            return Action("invented", "pass")
    def primary():
        created.append(Primary())
        return created[-1]
    class Fallback:
        async def achoose(self, state, actions, *, deadline):
            await asyncio.sleep(0)
            return actions[0]
    policy = FallbackPolicy(primary, Fallback)
    try:
        assert policy.choose(state, actions) == actions[0]
        assert policy.last_timing["attempts"][1]["status"] == "invalid_action"
        assert policy.last_timing["selected_backend"] == "deepseek"
    finally:
        policy.close()


@pytest.mark.parametrize("cancel", [False, True])
def test_shared_deadline_or_stop_cancels_both_recovery_requests(engine, state, cancel):
    entered, interrupted = set(), set()
    created = []
    class Backend:
        def __init__(self, name): self.name = name
        async def achoose(self, state, actions, *, deadline):
            if self.name == "initial":
                raise StrategyError("Unavailable")
            entered.add(self.name)
            try:
                await asyncio.Event().wait()
            finally:
                interrupted.add(self.name)
    def primary():
        created.append(Backend("initial" if not created else "retry"))
        return created[-1]
    policy = FallbackPolicy(primary, lambda: Backend("fallback"))
    async def run():
        task = asyncio.create_task(policy.achoose(state, engine.legal_actions(state),
                                                  deadline=time.monotonic() + .05))
        if cancel:
            while len(entered) < 2:
                await asyncio.sleep(0)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            with pytest.raises(StrategyError, match="deadline"):
                await task
        assert entered == interrupted == {"retry", "fallback"}
        assert all(r["status"] == "cancelled" for r in policy.last_timing["attempts"][1:])
    try:
        policy.runner.run(run())
    finally:
        policy.close()


def test_both_recovery_failures_are_retained(engine, state):
    class Backend:
        async def achoose(self, state, actions, *, deadline):
            raise StrategyError("Unavailable")
    policy = FallbackPolicy(Backend, Backend)
    try:
        with pytest.raises(StrategyError, match="both failed"):
            policy.choose(state, engine.legal_actions(state))
        assert [(r["backend"], r["attempt"], r["status"]) for r in policy.last_timing["attempts"]] == [
            ("jev", 1, "failed"), ("jev", 2, "failed"), ("deepseek", 1, "failed")]
        assert policy.last_result is None
        assert "selected_backend" not in policy.last_timing
    finally:
        policy.close()


@pytest.mark.parametrize("backend", ["jev", "deepseek"])
def test_real_http_routing_ignores_process_proxy_settings(engine, state, monkeypatch, backend):
    from majsoul.strategy import DeepSeekPolicy, ModelSettings
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy",
                 "DEEPSEEK_PROXY"):
        monkeypatch.setenv(name, "http://127.0.0.1:1")
    for name in ("NO_PROXY", "no_proxy"):
        monkeypatch.setenv(name, "")
    actions = engine.legal_actions(state)
    received = []
    async def run():
        async def serve(reader, writer, route):
            headers = await reader.readuntil(b"\r\n\r\n")
            received.append((route, headers.split(b"\r\n")[0].decode()))
            length = next(int(h.split(b":", 1)[1]) for h in headers.split(b"\r\n")
                          if h.lower().startswith(b"content-length:"))
            await reader.readexactly(length)
            body = response(actions) if backend == "jev" else {"choices": [{
                "finish_reason": "stop", "message": {"content": json.dumps({"action_id": actions[-1].id})}}]}
            data = json.dumps(body).encode()
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nConnection: close\r\n"
                         + f"Content-Length: {len(data)}\r\n\r\n".encode() + data)
            await writer.drain()
            writer.close()
            await writer.wait_closed()
        async with await asyncio.start_server(lambda r, w: serve(r, w, "direct"), "127.0.0.1", 0) as direct:
            async with await asyncio.start_server(lambda r, w: serve(r, w, "proxy"), "127.0.0.1", 0) as proxy:
                base = f"http://127.0.0.1:{direct.sockets[0].getsockname()[1]}"
                proxy_url = f"http://127.0.0.1:{proxy.sockets[0].getsockname()[1]}"
                policy = (JevPolicy(JevSettings("test", base_url=base, proxy=proxy_url)) if backend == "jev"
                          else DeepSeekPolicy(ModelSettings("test", base_url=base)))
                try:
                    kwargs = {} if backend == "jev" else {"stream": False}
                    assert await policy.achoose(state, actions, **kwargs) == actions[-1]
                finally:
                    await policy.aclose()
                    policy.runner.close()
        assert len(received) == 1
        assert received[0][0] == ("proxy" if backend == "jev" else "direct")
        assert policy.last_timing["route"] == received[0][0]
    asyncio.run(run())


@pytest.mark.parametrize("mutation,reason", [
    ("sum", "do not sum to 1"), ("missing", "do not cover exactly"),
    ("unknown", "not a legal candidate"), ("wrong_winner", "not the most probable")])
def test_invalid_response_names_the_failed_check(engine, state, mutation, reason):
    actions = engine.legal_actions(state)
    body = response(actions)
    answer = body["answers"]["action"]
    if mutation == "sum": answer["probabilities"][actions[-1].id] = 0.5
    if mutation == "missing": answer["probabilities"].pop(actions[0].id)
    if mutation == "unknown": answer["choice"] = "invented"
    if mutation == "wrong_winner": answer["choice"] = actions[0].id
    assert reason in JevPolicy.problem(body, actions)
    assert JevPolicy.problem(response(actions), actions) is None


def test_http_error_is_not_reported_as_invalid_response(engine, state):
    def respond(request):
        return httpx.Response(402, text="test-secret")
    policy = JevPolicy(JevSettings("test-secret"), httpx.AsyncClient(transport=httpx.MockTransport(respond)))
    try:
        with pytest.raises(StrategyError, match="HTTP 402"):
            policy.choose(state, engine.legal_actions(state))
        assert policy.last_timing["status"] == "http_error"
        assert policy.last_timing["http_status"] == 402
        assert "test-secret" not in json.dumps(policy.last_timing)
    finally:
        policy.close()


def test_hedge_races_deepseek_when_jev_is_slow(engine, state):
    actions = engine.legal_actions(state)
    cancelled = []

    class Slow:
        last_result = {"backend": "jev"}
        async def achoose(self, state, candidates, *, deadline):
            try:
                await asyncio.sleep(5)
            finally:
                cancelled.append("jev")
            return candidates[0]
    class Fast:
        last_result = {"backend": "deepseek"}
        async def achoose(self, state, candidates, *, deadline):
            return candidates[1]
    policy = FallbackPolicy(Slow, Fast, hedge_seconds=0.05)
    try:
        started = time.monotonic()
        assert policy.choose(state, actions, deadline=time.monotonic()+3) == actions[1]
        assert time.monotonic()-started < 1
        assert policy.last_timing["recovery"] == "hedged_deepseek"
        assert policy.last_timing["selected_backend"] == "deepseek"
        assert cancelled == ["jev"]
    finally:
        policy.close()


def test_hedge_is_not_started_when_jev_answers_in_time(engine, state):
    actions = engine.legal_actions(state)
    def unused():
        pytest.fail("DeepSeek must stay idle while Jev answers within the hedge delay")
    class Quick:
        async def achoose(self, state, candidates, *, deadline):
            return candidates[0]
    policy = FallbackPolicy(Quick, unused, hedge_seconds=1)
    try:
        assert policy.choose(state, actions) == actions[0]
        assert len(policy.last_timing["attempts"]) == 1
    finally:
        policy.close()
