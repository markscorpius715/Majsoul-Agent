"""The DeepSeek latency script, exercised offline against a mock transport."""
import json

import httpx

import tools.deepseek_ping as ping
from majsoul.strategy import DeepSeekPolicy, ModelSettings


def test_replays_a_real_recorded_decision_through_the_agent_policy(monkeypatch):
    path, record = ping.latest_decision()
    assert record['state'] and record['actions']
    chosen = record['actions'][0]['id']
    seen = []

    def respond(request):
        body = json.loads(request.content)
        seen.append(body)
        content = json.dumps({'action_id': chosen})
        return httpx.Response(200, json={'choices': [{'finish_reason': 'stop',
                                                      'message': {'content': content}}]})

    real = DeepSeekPolicy
    monkeypatch.setattr(ping, 'DeepSeekPolicy', lambda settings: real(
        settings, httpx.AsyncClient(transport=httpx.MockTransport(respond))))
    monkeypatch.setattr(real, 'choose', lambda self, state, actions, *, deadline=None:
                        self.runner.run(self.achoose(state, actions, deadline=deadline, stream=False)))
    settings = ModelSettings('test-key')
    rules = json.loads((ping.ROOT/'configs/sanma.json').read_text(encoding='utf-8'))
    report = ping.decide(settings, record, rules, 2)
    assert report['choices'] == [chosen, chosen]
    assert report['decision_seconds']['n'] == 2
    user = json.loads(seen[0]['messages'][1]['content'])
    assert user['state'] == record['state']
    assert [a['id'] for a in user['actions']] == [a['id'] for a in record['actions']]
    assert 'test-key' not in json.dumps(report)


def test_summary_percentiles():
    assert ping.summary([]) is None
    assert ping.summary([3, 1, 2]) == {'n': 3, 'min': 1, 'p50': 2, 'p95': 3, 'max': 3}
