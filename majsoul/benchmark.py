"""Reproducible synthetic decision snapshots; no window capture or clicks."""
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import statistics
import time

from .domain import Box, MeldView, Observation, State, TileView
from .rules import RuleEngine
from .strategy import DeepSeekPolicy, StrategyError
from .tiles import parse_hand, tile_name


def scenarios(engine: RuleEngine, seed: int):
    specs = [
        ("early_hand", "1m13469p12578s234z", (), {}, False, 1),
        ("tenpai_efficiency", "19m123456p123s111z", (), {}, False, 5),
        ("opponent_riichi", "119m24578p236s247z", (), {}, False, 10),
        ("seven_pairs", "1199m1122p3355s24z", (), {}, False, 7),
        ("open_yakuhai", "123456p123s11z", (MeldView(("5z",)*3),), {}, False, 8),
        ("ron_response", "123456p123s5551z", (), {"ron": Box(0,0,1,1), "pass": Box(2,0,3,1)}, False, 9),
        ("nuki_choice", "19m123456p123s114z", (), {"nuki": Box(0,0,1,1)}, False, 3),
        ("riichi_tsumogiri", "123456789p123s12z", (), {}, True, 10),
    ]
    rng = random.Random(seed)
    for name, text, melds, buttons, riichi, turns in specs:
        hand = parse_hand(text)
        deck = [tile_name(i) for i in engine.rules.available for _ in range(4)]
        for red in engine.rules.data["red_tiles"]:
            deck.remove("5" + red[1])
            deck.append(red)
        for tile in hand + [t for m in melds for t in m.tiles]:
            deck.remove(tile)
        if name == "ron_response":
            deck.remove("1z")
        rng.shuffle(deck)
        indicator = deck.pop()
        rivers = [[deck.pop() for _ in range(turns)] for _ in range(3)]
        if name == "ron_response":
            rivers[1].append("1z")
            # Preserve an actual non-furiten ron snapshot.
            rivers[0] = [t for t in rivers[0] if t != "1z"]
        obs = Observation(
            time.monotonic(), (0,0,1,1),
            hand=tuple(TileView(t, Box(0,0,1,1)) for t in hand),
            draw_index=None if name == "ron_response" else len(hand)-1,
            rivers=tuple(tuple(r) for r in rivers), melds=(melds,(),()),
            dora_indicators=(indicator,), riichi=(riichi, name == "opponent_riichi", False),
            buttons=buttons, phase="response" if name == "ron_response" else "discard",
            round_id="1z:1:0", seat_wind="2z", round_wind="1z",
        )
        yield name, State(obs, True, obs.rivers, history_complete=True)


def distribution(values):
    if not values:
        return None
    ordered = sorted(values)
    return {"n": len(values), "min": min(values), "mean": statistics.mean(values),
            "p50": statistics.median(values), "p95": ordered[math.ceil(.95*len(values))-1], "max": max(values)}


def summarize(rows):
    result = {}
    for mode in sorted({r["mode"] for r in rows}):
        subset = [r for r in rows if r["mode"] == mode]
        successes = [r for r in subset if r["status"] == "ok"]
        result[mode] = {
            "attempts": len(subset), "successes": len(successes),
            "failures": dict(Counter(r["status"] for r in subset if r["status"] != "ok")),
            "request_seconds": distribution([r["total_seconds"] for r in successes]),
            "first_content_seconds": distribution([r["first_content_seconds"] for r in successes if r["first_content_seconds"] is not None]),
            "decision_seconds": distribution([r["decision_seconds"] for r in successes]),
            "features_plus_request_seconds": distribution([r["features_seconds"]+r["total_seconds"] for r in successes]),
            "valid_within_4s": sum(r["features_seconds"]+r["total_seconds"] <= 4 for r in successes),
            "valid_within_5s": sum(r["features_seconds"]+r["total_seconds"] <= 5 for r in successes),
        }
    return result


def run_benchmark(engine: RuleEngine, policy: DeepSeekPolicy, *, rounds: int,
                  modes: list[str], seed: int, output: Path):
    if rounds < 1 or not modes or any(m not in ("stream", "buffered") for m in modes):
        raise ValueError("Positive rounds and stream/buffered modes required")
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    # Exclusive creation prevents accidentally mixing or overwriting prior runs.
    with (output / "requests.jsonl").open("x", encoding="utf-8") as stream:
        for round_index in range(rounds):
            for scenario_index, (name, state) in enumerate(scenarios(engine, seed+round_index)):
                start = time.monotonic()
                actions = engine.legal_actions(state)
                feature_seconds = time.monotonic()-start
                if not actions:
                    raise ValueError(f"No legal actions in benchmark scenario {name}")
                order = modes if (round_index+scenario_index) % 2 == 0 else list(reversed(modes))
                for mode in order:
                    row = {"round": round_index+1, "scenario": name, "mode": mode,
                           "candidate_count": len(actions), "features_seconds": feature_seconds,
                           "state": state.for_model(), "candidate_ids": [a.id for a in actions]}
                    try:
                        row["action_id"] = policy.choose(state, actions, stream=mode == "stream").id
                    except StrategyError as exc:
                        row["error"] = str(exc)
                    row.update(policy.last_timing)
                    rows.append(row)
                    stream.write(json.dumps(row, ensure_ascii=False)+"\n")
                    stream.flush()
                    print(json.dumps({k: row.get(k) for k in (
                        "round", "scenario", "mode", "status", "action_id", "first_content_seconds", "total_seconds")}), flush=True)
    report = {
        "utc": datetime.now(timezone.utc).isoformat(), "model": policy.settings.model,
        "thinking": policy.settings.thinking, "timeout_seconds": policy.settings.timeout,
        "rounds": rounds, "seed": seed, "rule_seconds": engine.rules.data["turn_seconds"],
        "reserve_seconds": engine.rules.data["reserve_seconds"], "summary": summarize(rows),
        "limitations": ["Synthetic snapshots, not a full game simulator or strategy-strength evaluation.",
                         "No screenshot, YOLO, mouse or live game timer latency included.",
                         "Sequential requests reuse HTTP connections; provider caching may affect results.",
                         "Small sample p95 is descriptive, not a latency guarantee; failures remain in denominators."],
    }
    (output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
