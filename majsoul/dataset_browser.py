"""Build an offline, searchable reference-image catalogue without relabelling data.

Usage: python -m majsoul.dataset_browser --source E:/majsoul_dataset
       --output artifacts/dataset-browser --scan-ui
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path

import cv2

from .calibration.frames import read_image, write_image


def reference_tags(record: dict) -> list[str]:
    """Keep observed UI, visible state, and proven events in separate namespaces."""
    tags = {"capture:" + str(record.get("reason", "unindexed")),
            *(record.get("reference_tags") or [])}
    for event in record.get("events") or []:
        kind = event.get("type")
        if kind:
            tags.add("event:" + str(kind))
    observation = record.get("observation") or {}
    for button in observation.get("action_buttons") or []:
        tags.add("button:" + str(button))
    for player in observation.get("players") or []:
        for meld in player.get("melds") or []:
            if isinstance(meld, dict):
                tags.add("meld:" + str(meld.get("kind", "unknown")))
    if record.get("validation_issues"):
        tags.add("quality:validation_failed")
    if observation:
        tags.add("quality:" + ("complete" if observation.get("complete") else "incomplete"))
    else:
        tags.add("quality:no_semantics")
    return sorted(tags)


def load_records(source: Path) -> tuple[list[dict], list[str]]:
    source = source.resolve()
    if not source.is_dir():
        raise ValueError(f"Dataset directory does not exist: {source}")
    sessions = [source] if (source / "observations.jsonl").exists() else sorted(
        p for p in source.iterdir() if p.is_dir())
    records, warnings = [], []
    for session in sessions:
        manifest = session / "observations.jsonl"
        indexed = set()
        if manifest.exists():
            for number, line in enumerate(manifest.read_text(encoding="utf-8-sig").splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    image = (session / record["file"]).resolve()
                    image.relative_to(session.resolve())
                    if not image.is_file():
                        raise ValueError("missing image")
                except (ValueError, TypeError, KeyError) as exc:
                    warnings.append(f"{manifest}:{number}: {exc}")
                    continue
                indexed.add(image)
                records.append({**record, "session": session.name, "image_path": str(image)})
        # Early sessions may have images without a manifest. Keep them discoverable.
        for image in sorted((session / "frames").glob("*")):
            if image.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
                continue
            if image.resolve() not in indexed:
                records.append({"session": session.name, "image_path": str(image.resolve()),
                                "file": str(image.relative_to(session)), "reason": "unindexed",
                                "wall_time": image.stat().st_mtime})
        for manifest in sorted((session / "references").glob("*/clip.json")):
            try:
                case = json.loads(manifest.read_text(encoding="utf-8"))
                tags = sorted({tag for trigger in case["triggers"] for tag in trigger["tags"]})
                for frame in case["frames"]:
                    image = (manifest.parent / frame["file"]).resolve()
                    image.relative_to(session.resolve())
                    if image.is_file():
                        records.append({**frame, "session": session.name, "image_path": str(image),
                                        "file": image.relative_to(session.resolve()).as_posix(),
                                        "reason": "reference_context",
                                        "reference_tags": ["context:" + tag for tag in tags]})
            except (KeyError, ValueError, TypeError) as exc:
                warnings.append(f"{manifest}: {exc}")
    return records, warnings


def attach_context(entries: list[dict], seconds: float, maximum: int = 12) -> None:
    """Link only actually saved nearby frames from the same session/match/round."""
    groups = defaultdict(list)
    for entry in entries:
        groups[(entry["session"], entry.get("match_index"), entry.get("round_index"))].append(entry)
    for group in groups.values():
        group.sort(key=lambda item: (item.get("time") or 0, item["id"]))
        for index, entry in enumerate(group):
            neighbours = group[max(0, index-maximum):index] + group[index+1:index+maximum+1]
            neighbours = [item for item in neighbours if entry.get("time") is not None
                          and item.get("time") is not None
                          and abs(item["time"]-entry["time"]) <= seconds]
            neighbours.sort(key=lambda item: abs(item["time"]-entry["time"]))
            entry["context"] = [item["id"] for item in sorted(
                neighbours[:maximum], key=lambda item: item["time"])]


class UIReferenceScanner:
    def __init__(self, config_path):
        from .live.buttons import ActionButtonReader
        from .live.collector import CollectConfig, CollectionUI
        from .live.config import LiveConfig
        from .calibration.roi_config import ROIConfig
        self.config = CollectConfig.load(config_path)
        self.live = LiveConfig.load(self.config.live)
        self.roi = ROIConfig.load(self.live.roi)
        self.ui = CollectionUI(self.config)
        self.buttons = ActionButtonReader(self.live.buttons, self.live.button_templates)

    def scan(self, image):
        from .live.capture import roi_for_frame
        tags, issues = [], []
        try:
            roi = roi_for_frame(self.roi, (image.shape[1], image.shape[0]), self.live.projection)
            tags += ["button:" + name for name in self.buttons.read(image, roi)]
        except ValueError as exc:
            issues.append(str(exc))
        confirm = self.ui.confirm(image)
        if confirm:
            tags.append("ui:confirm_" + confirm)
        if self.ui.match_result(image):
            tags.append("ui:match_result")
        if self.ui.visible(image, "drawer"):
            tags.append("ui:drawer")
        return tags, issues


def build_catalog(source: Path, output: Path, *, scanner=None, context_seconds=3.0,
                  thumbnail_width=400) -> dict:
    records, warnings = load_records(source)
    output.mkdir(parents=True, exist_ok=True)
    entries = []
    for number, record in enumerate(records, 1):
        path = Path(record["image_path"])
        identity = sha256(str(path).encode("utf-8")).hexdigest()[:24]
        thumbnail = output / "thumbs" / f"{identity}.jpg"
        tags = reference_tags(record)
        scan_issues = []
        try:
            image = read_image(path)
            if scanner:
                detected, scan_issues = scanner.scan(image)
                tags = sorted(set(tags + detected))
            width = min(thumbnail_width, image.shape[1])
            height = max(1, round(image.shape[0]*width/image.shape[1]))
            write_image(thumbnail, cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA))
        except (ValueError, OSError, RuntimeError, cv2.error) as exc:
            warnings.append(f"{path}: {exc}")
            continue
        observation = record.get("observation") or {}
        entries.append({
            "id": identity, "session": record["session"], "seq": record.get("seq"),
            "file": record["file"], "original": path.as_uri(),
            "thumbnail": thumbnail.relative_to(output).as_posix(),
            "match_index": record.get("match_index"), "round_index": record.get("round_index"),
            "time": record.get("captured_at", record.get("wall_time")),
            "wall_time": record.get("wall_time"), "frame_index": record.get("frame_index"),
            "round_id": observation.get("round_id"), "tags": tags,
            "events": record.get("events") or [],
            "issues": list(record.get("issues") or []) + list(record.get("validation_issues") or []),
            "scan_issues": scan_issues, "labels_verified": False,
            "tag_sources": {"stored": reference_tags(record),
                            "offline_ui": sorted(set(tags)-set(reference_tags(record)))},
        })
        if number % 100 == 0:
            print(json.dumps({"processed": number, "total": len(records)}, ensure_ascii=False), flush=True)
    attach_context(entries, context_seconds)
    counts = Counter(tag for entry in entries for tag in entry["tags"])
    payload = {"schema_version": 1, "source": str(source.resolve()), "entries": entries,
               "counts": dict(sorted(counts.items())), "warnings": warnings,
               "context_seconds": context_seconds,
               "notice": "UI/meld labels are automatically detected reference candidates, not manual gold labels; adjacent frames are limited to saved screens only."}
    (output / "catalog.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    data = json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c")
    (output / "index.html").write_text(HTML.replace("__CATALOG__", data), encoding="utf-8")
    return {"entries": len(entries), "counts": dict(counts), "warnings": len(warnings),
            "index": str((output / "index.html").resolve())}


HTML = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>Mahjong Soul Reference Gallery</title><style>
body{font:15px system-ui;background:#101924;color:#e6edf5;margin:24px}h1{font-size:25px}
input,select,button{font:inherit;padding:8px;margin:4px;background:#213247;color:inherit;border:1px solid #52708c;border-radius:6px}
#filters{position:sticky;top:0;background:#101924;padding:10px 0;z-index:1}#grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:15px}
article{background:#213247;border-radius:8px;overflow:hidden}article img{width:100%;cursor:pointer}article div{padding:10px;font-size:13px;overflow-wrap:anywhere}
small{color:#abc2d8}dialog{background:#101924;color:inherit;width:90vw;border:1px solid #52708c}dialog img{max-width:100%;max-height:72vh}#near img{width:180px;height:auto;cursor:pointer;margin:4px}a{color:#7dc9ff}
</style><h1>Mahjong Soul Reference Gallery</h1><p id="notice"></p><div id="filters">
<input id="query" placeholder="Search: ron, concealed kan, riichi, nuki/North extraction, nine terminals and honors, error…" size="48">
<select id="tag"><option value="">All tags</option></select>
<select id="session"><option value="">All capture sessions</option></select>
<label>Match <input id="match" type="number" min="1" style="width:60px"></label>
<label>Hand <input id="round" type="number" min="0" style="width:60px"></label>
<button id="export">Export current filter JSON</button><p id="count"></p></div>
<div id="grid"></div><button id="more">Load more</button>
<dialog id="detail"><button id="close">Close</button><a id="original" target="_blank">Open original image</a><p id="info"></p><img id="large"><p>Adjacent saved frames (continuity not guaranteed; see tooltip for time difference)</p><div id="near"></div><pre id="evidence"></pre></dialog>
<script>const catalog=__CATALOG__;
const aliases={'button:ron':'ron win discard','button:tsumo':'tsumo','button:riichi':'riichi','button:nuki':'nuki/North extraction','button:kan':'kan','button:pon':'pon','button:abort':'nine terminals and honors draw','button:pass':'pass 跳過','meld:ankan':'concealed kan','meld:kakan':'added kan','meld:daiminkan':'open kan','event:KAN':'kan event','event:KITA':'nuki/North extraction event','event:RIICHI':'riichi event','ui:confirm_gold':'match end confirmation','ui:confirm_blue':'in-hand confirmation','ui:match_result':'final results match end','quality:incomplete':'incomplete recognition','quality:validation_failed':'error validation failed','quality:no_semantics':'no semantics','capture:toggle_panel_incomplete':'auto option failed'};
const $=id=>document.getElementById(id), byId=new Map(catalog.entries.map(e=>[e.id,e]));
function option(value,label){let o=document.createElement('option');o.value=value;o.textContent=label;return o}
Object.entries(catalog.counts).forEach(([k,v])=>$('tag').append(option(k,`${aliases[k]||k} (${v})`)));
[...new Set(catalog.entries.map(e=>e.session))].forEach(s=>$('session').append(option(s,s)));
$('notice').textContent=catalog.notice;let selected=[],shown=0;
function searchable(e){return [e.session,e.file,e.round_id,...e.tags,...e.tags.map(t=>aliases[t]||''),...e.issues,...e.scan_issues].join(' ').toLowerCase()}
function filter(){let q=$('query').value.trim().toLowerCase().split(/\\s+/).filter(Boolean);selected=catalog.entries.filter(e=>(!$('tag').value||e.tags.includes($('tag').value))&&(!$('session').value||e.session===$('session').value)&&(!$('match').value||e.match_index===Number($('match').value))&&(!$('round').value||e.round_index===Number($('round').value))&&q.every(t=>searchable(e).includes(t)));shown=0;$('grid').replaceChildren();$('count').textContent=`Matched ${selected.length} / ${catalog.entries.length} images · automatic candidates, not manually labeled · read warnings ${catalog.warnings.length}`;more()}
function more(){for(const e of selected.slice(shown,shown+60)){let card=document.createElement('article'),im=document.createElement('img'),info=document.createElement('div');im.src=e.thumbnail;im.loading='lazy';im.onclick=()=>open(e);info.textContent=`${e.session} · Match ${e.match_index??'?'} / Hand ${e.round_index??'?'} · ${e.file} | ${e.tags.map(t=>aliases[t]||t).join(' · ')}`;card.append(im,info);$('grid').append(card)}shown+=60;$('more').hidden=shown>=selected.length}
function open(e){$('large').src=e.original;$('original').href=e.original;$('info').textContent=`${e.session} / ${e.file} · Match ${e.match_index??'?'} / Hand ${e.round_index??'?'} · ${e.tags.join(', ')}`;$('evidence').textContent=JSON.stringify({sources:e.tag_sources,events:e.events,issues:e.issues,scan_issues:e.scan_issues},null,2);$('near').replaceChildren();for(const id of e.context){let n=byId.get(id),im=document.createElement('img');im.src=n.thumbnail;im.title=`${n.file} · ${(n.time-e.time).toFixed(2)}s`;im.onclick=()=>open(n);$('near').append(im)}if(!$('detail').open)$('detail').showModal()}
['query','tag','session','match','round'].forEach(id=>$(id).addEventListener('input',filter));$('more').onclick=more;$('close').onclick=()=>$('detail').close();
$('export').onclick=()=>{let u=URL.createObjectURL(new Blob([JSON.stringify(selected,null,2)],{type:'application/json'})),a=document.createElement('a');a.href=u;a.download='reference-selection.json';a.click();setTimeout(()=>URL.revokeObjectURL(u),1000)};filter();
</script></html>'''


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scan-ui", action="store_true")
    parser.add_argument("--config", default="configs/collect.steam.yaml")
    parser.add_argument("--context-seconds", type=float, default=3.0)
    args = parser.parse_args(argv)
    if args.context_seconds <= 0:
        parser.error("--context-seconds must be positive")
    report = build_catalog(args.source, args.output,
                           scanner=UIReferenceScanner(args.config) if args.scan_ui else None,
                           context_seconds=args.context_seconds)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
