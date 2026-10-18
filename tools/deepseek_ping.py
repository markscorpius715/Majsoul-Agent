"""Measure DeepSeek latency with the configured key and model.

Two measurements, both direct (no proxy), never printing the key:

1. ``ping``: a minimal chat completion, i.e. network plus model start-up.
2. ``decision``: a real recorded live decision (full state, legal actions and
   rule card) replayed through the same ``DeepSeekPolicy`` the Agent uses.
   This is the number that has to fit inside a turn.

    python -m tools.deepseek_ping
    python -m tools.deepseek_ping --runs 20 --model deepseek-flash --output report.json

Each run is one billed API call. The verdict compares decision p95 with the
stability target from GOAL_PROMPT.stability.md (1.5 s by default).
"""
import argparse
import asyncio
import json
from pathlib import Path
from statistics import median
import time

import httpx

from majsoul.strategy import DeepSeekPolicy, ModelSettings, StrategyError

ROOT = Path(__file__).resolve().parents[1]


class RecordedState:
    """The two members DeepSeekPolicy uses, taken from a logged decision."""
    def __init__(self, record, rules):
        self._state = record['state']
        self.rules_context = rules

    def for_model(self):
        return self._state


class RecordedAction:
    def __init__(self, data):
        self.id = data['id']
        self._data = data

    def for_model(self):
        return self._data


def latest_decision(root=ROOT):
    """Newest logged decision that carries a full state and its legal actions."""
    files = sorted((root/'artifacts/live').glob('protected-*/agent*/decisions.jsonl'),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    for path in files:
        for line in reversed(path.read_text(encoding='utf-8').splitlines()):
            record = json.loads(line)
            if record.get('state') and record.get('actions'):
                return path, record
    raise FileNotFoundError('no logged decision with state and actions under artifacts/live')


def summary(values):
    if not values:
        return None
    ordered = sorted(values)
    return {'n': len(ordered), 'min': round(ordered[0], 3), 'p50': round(median(ordered), 3),
            'p95': round(ordered[min(len(ordered)-1, round(.95*(len(ordered)-1)))], 3),
            'max': round(ordered[-1], 3)}


async def ping(settings, runs):
    timings, statuses = [], []
    async with httpx.AsyncClient(timeout=settings.timeout, proxy=None, trust_env=False) as client:
        for _ in range(runs):
            started = time.monotonic()
            try:
                response = await client.post(
                    settings.base_url.rstrip('/')+'/chat/completions',
                    headers={'Authorization': f'Bearer {settings.api_key}'},
                    json={'model': settings.model, 'stream': False, 'max_tokens': 5,
                          'thinking': {'type': settings.thinking},
                          'messages': [{'role': 'user', 'content': 'Reply with the word OK.'}]})
                statuses.append(response.status_code)
                if response.status_code == 200:
                    timings.append(time.monotonic()-started)
            except httpx.HTTPError as exc:
                statuses.append(type(exc).__name__)
    return {'seconds': summary(timings), 'statuses': statuses}


def decide(settings, record, rules, runs):
    state = RecordedState(record, rules)
    actions = [RecordedAction(a) for a in record['actions']]
    policy = DeepSeekPolicy(settings)
    decisions, first_content, results = [], [], []
    try:
        for _ in range(runs):
            try:
                action = policy.choose(state, actions, deadline=time.monotonic()+settings.timeout)
                results.append(action.id)
                timing = policy.last_timing
                decisions.append(timing['decision_seconds'])
                if timing.get('first_content_seconds') is not None:
                    first_content.append(timing['first_content_seconds'])
            except StrategyError as error:
                results.append(f"failed: {error} ({policy.last_timing.get('status')})")
    finally:
        policy.close()
    return {'decision_seconds': summary(decisions), 'first_content_seconds': summary(first_content),
            'request_bytes': policy.last_timing.get('request_bytes'), 'choices': results}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--model', help='override the configured model name')
    parser.add_argument('--decision', type=Path, help='decisions.jsonl to replay (default: newest live run)')
    parser.add_argument('--target', type=float, default=1.5, help='decision p95 target in seconds')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if not 1 <= args.runs <= 100:
        parser.error('runs must be between 1 and 100')
    settings = ModelSettings.load(ROOT)
    if args.model:
        from dataclasses import replace
        settings = replace(settings, model=args.model)
    if not settings.api_key:
        parser.error('no DeepSeek key found in .env, ../PuwuQuiz/.env or the environment')
    if args.decision:
        record = next(json.loads(line) for line in reversed(args.decision.read_text(encoding='utf-8').splitlines())
                      if json.loads(line).get('state'))
        source = args.decision
    else:
        source, record = latest_decision()
    # The live loop sends the rule file itself as the prompt's rules context.
    rules = json.loads((ROOT/'configs/sanma.json').read_text(encoding='utf-8'))
    print(f'model {settings.model} at {settings.base_url} (thinking {settings.thinking}); key set; '
          f'{args.runs} runs each')
    report = {'model': settings.model, 'base_url': settings.base_url, 'thinking': settings.thinking,
              'decision_source': str(source), 'runs': args.runs,
              'ping': asyncio.run(ping(settings, args.runs))}
    print('ping     :', json.dumps(report['ping']['seconds']), 'statuses', report['ping']['statuses'])
    report['decision'] = decide(settings, record, rules, args.runs)
    print('decision :', json.dumps(report['decision']['decision_seconds']),
          f"({report['decision']['request_bytes']} request bytes)")
    print('first content:', json.dumps(report['decision']['first_content_seconds']))
    failures = [c for c in report['decision']['choices'] if c.startswith('failed')]
    p95 = (report['decision']['decision_seconds'] or {}).get('p95')
    fast = p95 is not None and p95 <= args.target and not failures
    report['verdict'] = {'target_p95_seconds': args.target, 'decision_p95_seconds': p95,
                         'failures': len(failures), 'fast_enough': fast}
    print(f"verdict  : decision p95 {p95} s vs target {args.target} s, {len(failures)} failures -> "
          f"{'FAST ENOUGH for DeepSeek-only' if fast else 'NOT fast enough'}")
    if failures:
        print('failures :', failures[:3])
    if args.output:
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return 0 if fast else 1


if __name__ == '__main__':
    raise SystemExit(main())
