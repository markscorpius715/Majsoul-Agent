from dataclasses import replace

import pytest

from majsoul.domain import Box, MeldView, State
from majsoul.tiles import counts34, parse_hand, physical_ids, tile_index
from conftest import observation


@pytest.mark.parametrize("hand,expected", [
    ("123456789p123s11z", -1), ("123456789p123s1z", 0),
    ("1199m112233p11s22z", -1), ("19m19p19s1234567z", 0),
    ("119m19p19s1234567z", -1),
])
def test_shanten_families(engine, hand, expected):
    assert engine.shanten(parse_hand(hand)) == expected


def test_open_shanten(engine):
    assert engine.shanten(parse_hand("123456p11s22z"), 1) == 0


@pytest.mark.parametrize("tile", ["2m", "5m", "0m", "8m", "0z", "8z"])
def test_forbidden_tiles(engine, tile):
    with pytest.raises(ValueError):
        engine.rules.validate_tiles([tile])


def test_red_counts_and_ids(engine):
    tiles = ["5p", "0p", "5p", "5p"]
    engine.rules.validate_tiles(tiles)
    assert len(set(physical_ids([tiles])[0])) == 4
    for invalid in (["5p"]*4, ["0p"]*2, ["4z"]*5):
        with pytest.raises(ValueError):
            engine.rules.validate_tiles(invalid)


def test_dora_cycles(engine):
    for indicator, expected in [("1m", "9m"), ("9m", "1m"), ("9p", "1p"), ("4z", "1z"), ("7z", "5z")]:
        assert engine.rules.dora(indicator) == tile_index(expected)


def test_ukeire_excludes_exhausted_tiles_but_furiten_does_not(engine):
    hand = parse_hand("123456789p123s1z")
    visible = hand + ["1z"]*3
    assert engine.improvements(hand, 0, visible) == {}
    assert tile_index("1z") in engine.waits(hand)
    obs = observation("123456789p123s1z", draw_index=None, phase="response", buttons={"ron": Box(1, 1, 2, 2)})
    s = State(obs, True, (("1z",), (), ()))
    assert engine.permanent_furiten(s)
    assert not any(a.kind == "ron" for a in engine.legal_actions(s))


def test_riichi_tsumogiri_and_kuikae(engine, state):
    state.observation.riichi = (True, False, False)
    assert [a.hand_index for a in engine.legal_actions(state)] == [13]
    state.observation.draw_index = None
    assert engine.legal_actions(state) == []
    state.observation.riichi = (False, False, False)
    state.kuikae_tile = "1z"
    assert all(a.tile != "1z" for a in engine.legal_actions(state))


def test_uncertain_and_chi_never_produce_actions(engine, state):
    state.stable = False
    assert engine.legal_actions(state) == []
    state.stable = True
    state.observation.buttons = {"chi": Box(1, 1, 2, 2)}
    assert all(a.kind != "chi" for a in engine.legal_actions(state))
    state.observation.issues = ["missing detection"]
    assert engine.legal_actions(state) == []


def test_riichi_selection_keeps_only_tenpai(engine, state):
    state.observation.phase = "riichi_select"
    state.observation.riichi_discard_indices = tuple(range(len(state.observation.hand)))
    actions = engine.legal_actions(state)
    assert actions
    assert all(a.features["shanten"] == 0 for a in actions)


def test_sanma_score_tsumo_loss_and_nuki(engine):
    hand = parse_hand("123456789p123s11z")
    normal = engine.score(hand, 13, seat="2z", round_wind="1z", tsumo=True, riichi=True)
    assert "error" not in normal
    assert len(normal["payments"]) == 2
    assert normal["total"] == sum(normal["payments"])
    with_nuki = engine.score(hand, 13, seat="2z", round_wind="1z", tsumo=True, riichi=True, nuki=1)
    assert with_nuki["han"] == normal["han"] + 1
    assert with_nuki["total"] >= normal["total"]
    dealer = engine.score(hand, 13, seat="1z", round_wind="1z", tsumo=True, riichi=True)
    assert dealer["payments"][0] == dealer["payments"][1]


def test_nuki_and_rob_nuki_cannot_supply_yaku(engine):
    # Closed ron with an honor tanki, mixed sequences, no yaku.
    hand = parse_hand("123456p234678s44z")
    result = engine.score(hand, 13, seat="2z", round_wind="1z", tsumo=False, nuki=1, rob_nuki=True)
    assert "error" in result


def test_sanma_man_dora_scoring(engine):
    hand = parse_hand("999m123456p123s11z")
    base = engine.score(hand, 13, seat="2z", round_wind="1z", tsumo=True)
    with_dora = engine.score(hand, 13, seat="2z", round_wind="1z", tsumo=True, indicators=("1m",))
    assert with_dora["han"] == base["han"] + 3


def test_open_hand_scoring(engine):
    melds = (MeldView(("5z", "5z", "5z")),)
    result = engine.score(parse_hand("123456p123s11z"), 10, melds, seat="2z", round_wind="1z", tsumo=False)
    assert "error" not in result
    assert result["han"] >= 1
