import json

import httpx
import pytest

from majsoul.strategy import DeepSeekPolicy, ModelSettings, StrategyError, read_model_env


def test_config_precedence_and_no_secret_repr(tmp_path, monkeypatch):
    ref = tmp_path / "reference.env"
    ref.write_text("API_KEY:reference\n#model-name:deepseek-flash\nUNRELATED_PASSWORD=no\n", encoding="utf-8")
    (tmp_path / ".env").write_text("API_KEY=local\n", encoding="utf-8")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setenv("API_KEY", "process-secret")
    s = ModelSettings.load(tmp_path, ref)
    assert s.api_key == "process-secret"
    assert "process-secret" not in repr(s)
    assert "UNRELATED_PASSWORD" not in read_model_env(ref)


@pytest.mark.parametrize("content", ['{"action_id":"invented"}', '```json\n{}\n```', '{"action_id":3}', '{}', 'null', '[]'])
def test_bad_model_responses_never_choose(engine, state, content):
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, json={"choices":[{"finish_reason":"stop","message":{"content":content}}]})))
    policy = DeepSeekPolicy(ModelSettings("test-only"), client)
    with pytest.raises(StrategyError):
        policy.choose(state, engine.legal_actions(state), stream=False)
    policy.close()


def test_exact_candidate_selection(engine, state):
    actions = engine.legal_actions(state)
    def respond(request):
        body = json.loads(request.content)
        assert body["model"] == "deepseek-flash"
        sent = json.loads(body["messages"][1]["content"])
        assert sent["actions"][0]["id"] == actions[0].id
        assert "box" not in json.dumps(sent)
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": json.dumps({"action_id": actions[-1].id})}}]})
    policy = DeepSeekPolicy(ModelSettings("test-only"), httpx.AsyncClient(transport=httpx.MockTransport(respond)))
    assert policy.choose(state, actions, stream=False) == actions[-1]
    policy.close()


def test_errors_do_not_echo_provider_body_or_key(engine, state):
    policy = DeepSeekPolicy(ModelSettings("secret-value"), httpx.AsyncClient(transport=httpx.MockTransport(
        lambda _: httpx.Response(401, text="secret-value"))))
    with pytest.raises(StrategyError, match="HTTP 401") as error:
        policy.choose(state, engine.legal_actions(state), stream=False)
    assert "secret-value" not in str(error.value)
    policy.close()


class _Model:
    def __init__(self, behaviour):
        self.behaviour = behaviour
        self.last_timing = {"backend": "fake"}
        self.last_result = {"backend": "fake"}
        self.deadlines = []

    async def achoose(self, state, actions, *, deadline=None):
        self.deadlines.append(deadline)
        if self.behaviour == "fail":
            raise StrategyError("backend down")
        if self.behaviour == "illegal":
            return "not-an-action"
        return actions[-1]


def _efficient(actions):
    discards = [a for a in actions if a.kind == "discard"]
    return min(discards, key=lambda a: (a.features["shanten"], -a.features["ukeire"],
                                        -a.features["retained_dora"], a.id))


@pytest.mark.parametrize("behaviour", ["fail", "illegal"])
def test_local_rung_answers_when_the_model_cannot(engine, state, behaviour):
    from majsoul.strategy import LocalFallbackPolicy
    import time
    actions = engine.legal_actions(state)
    policy = LocalFallbackPolicy(_Model(behaviour), reserve_seconds=0.5)
    deadline = time.monotonic()+5
    assert policy.choose(state, actions, deadline=deadline) == _efficient(actions)
    assert policy.last_timing["rung"] == "local"
    assert policy.last_result["backend"] == "local_efficiency"


def test_model_gets_deadline_minus_reserve_and_wins_when_in_time(engine, state):
    from majsoul.strategy import LocalFallbackPolicy
    import time
    actions = engine.legal_actions(state)
    model = _Model("ok")
    policy = LocalFallbackPolicy(model, reserve_seconds=0.5)
    deadline = time.monotonic()+5
    assert policy.choose(state, actions, deadline=deadline) == actions[-1]
    assert model.deadlines == [pytest.approx(deadline-0.5)]
    assert policy.last_timing["rung"] == "model"


def test_no_model_call_once_only_the_reserve_remains(engine, state):
    from majsoul.strategy import LocalFallbackPolicy
    import time
    actions = engine.legal_actions(state)
    model = _Model("ok")
    policy = LocalFallbackPolicy(model, reserve_seconds=0.5)
    assert policy.choose(state, actions, deadline=time.monotonic()+0.2) == _efficient(actions)
    assert model.deadlines == []


def test_shipped_live_policy_is_deepseek_with_local_rung(monkeypatch):
    from dataclasses import replace
    from majsoul.jev import FallbackPolicy
    from majsoul.live.config import LiveConfig
    import majsoul.live.loop as module
    from majsoul.strategy import LocalFallbackPolicy
    monkeypatch.setattr(module.JevSettings, 'load', lambda _: module.JevSettings('test'))
    monkeypatch.setattr(module.ModelSettings, 'load', lambda *_: ModelSettings('test'))
    config = LiveConfig.load('configs/live.steam.yaml')
    policy = module.build_policy(config)
    try:
        assert isinstance(policy, LocalFallbackPolicy) and isinstance(policy.model, DeepSeekPolicy)
        assert policy.reserve_seconds == config.decision.local_fallback_seconds
    finally:
        policy.close()
    auto = replace(config, decision=replace(config.decision, policy='auto'))
    hedged = module.build_policy(auto)
    assert isinstance(hedged.model, FallbackPolicy)
    assert hedged.model.hedge_seconds == config.decision.hedge_seconds
    off = replace(auto, decision=replace(auto.decision, local_fallback_seconds=0))
    assert isinstance(module.build_policy(off), FallbackPolicy)
