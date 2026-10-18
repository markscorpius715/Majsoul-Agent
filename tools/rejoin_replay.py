"""Replay recorded frames through an observe-only rejoin Agent, offline.

The real readers (tile CNN, OCR, buttons, clock, lifecycle) run on frames saved
during a live takeover, and the report shows whether and when the Agent would
have reported ``agent_ready_for_handback``. Input is impossible: the replay
window raises on any click. No model or notification service is called.

    python -m tools.rejoin_replay artifacts/live/<run>/client-002 --output <dir>
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path

from majsoul.calibration.frames import read_image
from majsoul.live.capture import LiveFrame
from majsoul.live.config import LiveConfig
from majsoul.live.loop import LiveAgentLoop


class ReplayWindow:
    """Serves recorded frames in order; refuses every input."""
    def __init__(self, frames, window):
        self.frames = list(frames)
        self.window = tuple(window)
        self.position = -1
        self.current = None

    def capture(self):
        self.position += 1
        if self.position >= len(self.frames):
            raise KeyboardInterrupt('replay finished')
        path, captured_at = self.frames[self.position]
        self.current = captured_at
        return LiveFrame(read_image(path), captured_at, self.window, (0, 0))

    def click(self, *args, **kwargs):
        raise AssertionError('rejoin replay must never send input')

    def activate(self):
        raise AssertionError('rejoin replay must never take focus')

    def window_rect(self):
        return self.window


class NoPolicy:
    """The observe-only Agent never decides; any call is a defect."""
    last_timing = last_result = None

    def choose(self, *args, **kwargs):
        raise AssertionError('rejoin replay must never call a policy')

    def close(self):
        pass


def recorded_frames(source: Path, start=None, end=None):
    """(path, captured_at, window) from a collector session's observations."""
    rows = [json.loads(line) for line in
            (source/'observations.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    frames = []
    for row in rows:
        at = row['captured_at']
        if (start is not None and at < start) or (end is not None and at > end):
            continue
        frames.append((source/row['file'], at, row['window']))
    return frames


def replay(source: Path, output: Path, *, config_path='configs/live.steam.yaml',
           layout=('bottom', 'left', 'right'), start=None, end=None, latency=.3):
    base = LiveConfig.load(config_path)
    config = replace(base, artifacts_root=output, notify=replace(base.notify, enabled=False))
    frames = recorded_frames(source, start, end)
    if not frames:
        raise ValueError(f'no recorded frames in {source}')
    window = ReplayWindow([(path, at) for path, at, _ in frames], frames[0][2])
    # The Agent sees each frame `latency` seconds after capture, like live.
    now = lambda: (window.current if window.current is not None else frames[0][1])+latency
    loop = LiveAgentLoop(window, config, execute=True, skip_menu=True, join=True,
                         session='rejoin-replay', sleep=lambda _seconds: None, now=now,
                         policy=NoPolicy(), input_gate=lambda: False, supervised=True,
                         layout_hint=list(layout))
    try:
        loop.run()
    except KeyboardInterrupt:
        pass
    finally:
        loop.decision_worker.close()
    events = [json.loads(line) for line in
              (loop.artifacts/'events.jsonl').read_text(encoding='utf-8').splitlines()]
    ready = [e for e in events if e['event'] == 'agent_ready_for_handback']
    readiness = [e for e in events if e['event'] == 'handback_readiness']
    return {'source': str(source), 'frames': len(frames), 'ready_signals': len(ready),
            'first_ready': ready[0] if ready else None, 'readiness_changes': readiness,
            'clicks': 0, 'events': str(loop.artifacts/'events.jsonl')}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('source', type=Path, help='collector session directory with observations.jsonl')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', default='configs/live.steam.yaml')
    parser.add_argument('--layout', default='bottom,left,right')
    parser.add_argument('--start', type=float)
    parser.add_argument('--end', type=float)
    args = parser.parse_args(argv)
    report = replay(args.source, args.output, config_path=args.config,
                    layout=tuple(args.layout.split(',')), start=args.start, end=args.end)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'rejoin-replay.json').write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                                   encoding='utf-8')
    summary = {k: v for k, v in report.items() if k != 'readiness_changes'}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    for change in report['readiness_changes']:
        print(change.get('frame_index'), change.get('blockers'), change.get('perception'), change.get('issues'))
    return 0 if report['ready_signals'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
