"""Verify the independent observer against the current window; never send input."""
from datetime import datetime
import json
import multiprocessing as mp
from pathlib import Path
from queue import Empty
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from majsoul.live.config import LiveConfig
from majsoul.live.capture import LiveWindow
from majsoul.live.protection import _observer,stop_input_owner

def main():
    cfg=LiveConfig.load(ROOT/'configs/live.steam.yaml')
    out=ROOT/'artifacts/live/workflow-audit/protection-20261011'
    out.mkdir(parents=True,exist_ok=True)
    try:hwnd=LiveWindow.find(cfg.window_title_regex,cfg.window_class_regex)
    except Exception as exc:
        report={'recorded_at':datetime.now().astimezone().isoformat(),'game_inputs_sent':0,
                'samples':[],'status':'window_unavailable','error':str(exc)}
        (out/'readonly-observer.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(report,ensure_ascii=False));return 1
    ctx=mp.get_context('spawn');queue=ctx.Queue(maxsize=32);stop=ctx.Event()
    process=ctx.Process(target=_observer,args=(cfg,hwnd,queue,stop),daemon=True)
    started=time.monotonic();samples=[]
    process.start()
    try:
        while len(samples)<10 and time.monotonic()-started<15:
            try:samples.append(queue.get(timeout=.5)['sample'])
            except Empty:
                if not process.is_alive():break
    finally:
        stop.set();process.join(1);stop_input_owner(process,.1)
    report={'recorded_at':datetime.now().astimezone().isoformat(),'game_inputs_sent':0,
            'samples':samples,'exit_code':process.exitcode,'elapsed_seconds':time.monotonic()-started}
    (out/'readonly-observer.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False))
    return 0 if len(samples)==10 and not any(s.get('error') for s in samples) else 1

if __name__=='__main__':raise SystemExit(main())
