"""Measure the real independent UI observer without starting any input owner.

Run from the project root with ``python -m tools.protection_observer_probe``.
The observer may restore a minimized window for capture; it never clicks.
"""
import argparse
import json
import math
import multiprocessing as mp
from pathlib import Path
from queue import Empty
import time

from majsoul.live.capture import LiveWindow
from majsoul.live.config import LiveConfig
from majsoul.live.protection import _observer, stop_input_owner


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/live.steam.yaml')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seconds',type=float,default=5)
    args=parser.parse_args(argv)
    if not math.isfinite(args.seconds) or not 1<=args.seconds<=30:
        parser.error('seconds must be between 1 and 30')
    config=LiveConfig.load(args.config)
    hwnd=LiveWindow.find(config.window_title_regex,config.window_class_regex)
    args.output.mkdir(parents=True,exist_ok=True)
    ctx=mp.get_context('spawn')
    output=ctx.Queue(maxsize=32)
    stop=ctx.Event()
    child=ctx.Process(target=_observer,args=(config,hwnd,output,stop),daemon=True)
    started=time.monotonic()
    child.start()
    samples=[]
    first=None
    try:
        while True:
            now=time.monotonic()
            if now-(first if first is not None else started)> (
                    args.seconds if first is not None else config.protection.startup_seconds):
                break
            try:
                item=output.get(timeout=.1)
            except Empty:
                if not child.is_alive():break
                continue
            received=time.monotonic()
            if first is None:first=received
            samples.append(dict(item['sample'],received_at=received,
                latency_seconds=received-item['sample']['captured_at']))
    finally:
        stop.set()
        stop_input_owner(child,config.protection.stop_grace_seconds)
    capture_healthy=bool(samples) and all(not x.get('error') for x in samples)
    gaps=[b['captured_at']-a['captured_at'] for a,b in zip(samples,samples[1:])]
    recognized=sum(bool(x.get('lobby') or x.get('confirm') or x.get('table') or x.get('modal'))
                   for x in samples)
    timely=capture_healthy and all(0<=x['latency_seconds']<=config.protection.observation_seconds
                                  for x in samples) and all(
        0<gap<=config.protection.observation_seconds for gap in gaps)
    report=dict(read_only=True,input_sent=False,hwnd=hwnd,pid=child.pid,
        capture_healthy=capture_healthy,timely=timely,recognized_samples=recognized,
        sample_count=len(samples),
        startup_seconds=first-started if first else None,
        maximum_latency_seconds=max((x['latency_seconds'] for x in samples),default=None),
        maximum_sample_gap_seconds=max(gaps,default=None),samples=samples)
    path=args.output/'observer-probe.json'
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='samples'},ensure_ascii=True))
    return 0 if timely else 1


if __name__=='__main__':
    raise SystemExit(main())
