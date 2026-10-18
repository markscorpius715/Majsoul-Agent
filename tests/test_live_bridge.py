from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from majsoul.calibration.roi_config import ROIConfig
from majsoul.domain import Box
from majsoul.live.bridge import (BridgeError, build_domain_state, infer_phase,
                                 normalize_buttons)
from majsoul.perception.schemas import (MeldReading, RecognizedTile,
                                        SegmentedTile, TilePrediction,
                                        TileSequenceReading, VisibleTilesReading)
from majsoul.state import (FrameObservation, FrameObservationReader,
                           MahjongState, ObservedTile, PlayerObservation,
                           PlayerState, TrackingPhase)

from majsoul.state.schemas import GameEvent, GameEventType, TrackedMeld
from majsoul.tiles import parse_hand

ROOT = Path(__file__).resolve().parents[1]
SLOTS = ("bottom", "top", "left")
SEATS = ("1z", "2z", "3z")
HAND13 = ("1p", "2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p",
          "1s", "1s", "2s", "3s")
HAND14 = HAND13 + ("9s",)
WINDOW = (100, 100, 1000, 800)
BUTTON_BOX = Box(800, 600, 900, 640)


def recognized(tile, index):
    return RecognizedTile(
        SegmentedTile(Box(20 + index * 40, 600, 50 + index * 40, 660),
                      np.zeros((4, 4, 3), dtype=np.uint8)),
        TilePrediction(tile, 0.99, 0.0, 0.1, "upright"))


def visible_reading(hand):
    return VisibleTilesReading(
        active_slots=SLOTS,
        hand=TileSequenceReading(
            "hand", tuple(recognized(tile, i) for i, tile in enumerate(hand)),
            drawn_index=len(hand) - 1 if len(hand) == 14 else None),
        rivers={slot: TileSequenceReading(f"river_{slot}", ()) for slot in SLOTS},
        melds={slot: MeldReading(f"melds_{slot}", ()) for slot in SLOTS},
        dora=TileSequenceReading("dora", (recognized("9s", 0),)),
        kita={slot: TileSequenceReading(f"kita_{slot}", ()) for slot in SLOTS})


def make_player(slot, seat, *, dealer=False, river=(), history=(), melds=(),
                riichi=False, kita=0):
    return PlayerState(slot, seat, dealer, 35000, tuple(river), tuple(history),
                       tuple(meld if isinstance(meld, TrackedMeld) else tuple(meld)
                             for meld in melds), riichi, kita)


def make_state(hand=HAND14, *, self_slot="bottom", current_actor="bottom",
               phase=TrackingPhase.TRACKING, history_complete=True,
               uncertain_fields=(), players=None, drawn_tile="9s"):
    if players is None:
        players = tuple(make_player(
            slot, seat, dealer=index == 0,
            river=(("7p",) if index == 1 else ()),
            history=(("7p",) if index == 1 else ()),
        ) for index, (slot, seat) in enumerate(zip(SLOTS, SEATS)))
    return MahjongState(
        mode="sanma", player_count=3, active_slots=SLOTS, self_slot=self_slot,
        round_wind="1z", hand_number=1, honba=0, riichi_sticks=0,
        remaining_tiles=55, dealer="bottom", self_seat="1z",
        players=players, hand=tuple(hand),
        drawn_tile=drawn_tile if len(hand) == 14 else None,
        dora_indicators=("9s",), current_actor=current_actor, last_discard=None,
        available_actions=(), uncertain_fields=tuple(uncertain_fields),
        phase=phase, history=(), history_complete=history_complete,
        source_frame=100, source_timestamp=10.0)


def make_observation(hand=HAND14, *, action_buttons=None, drawn_index=13,
                     timestamp=10.0):
    players = tuple(PlayerObservation(
        slot, seat, index == 0, 35000, river=(), melds=(), kita_count=0,
    ) for index, (slot, seat) in enumerate(zip(SLOTS, SEATS)))
    return FrameObservation(
        timestamp, 100, SLOTS, "1z", 1, 0, 0, 55, players,
        tuple(ObservedTile(tile) for tile in hand), drawn_index,
        (ObservedTile("9s"),), action_buttons=action_buttons)


def test_build_domain_state_maps_every_field():
    state = make_state()
    observation = make_observation(action_buttons=("riichi",))
    visible = visible_reading(state.hand)

    domain = build_domain_state(state, observation, visible,
                                {"riichi": BUTTON_BOX},
                                pending_riichi=False, window=WINDOW)

    obs = domain.observation
    assert obs.captured_at == 10.0
    assert obs.window == WINDOW
    assert [view.tile for view in obs.hand] == list(HAND14)
    assert obs.hand[3].box == visible.hand.tiles[3].detection.box
    assert obs.draw_index == 13
    assert obs.rivers == ((), ("7p",), ())
    assert obs.melds == ((), (), ())
    assert obs.nuki == (0, 0, 0)
    assert obs.dora_indicators == ("9s",)
    assert obs.riichi == (False, False, False)
    assert obs.buttons == {"riichi": BUTTON_BOX}
    assert obs.phase == "discard"
    assert obs.round_id == "1z:1:0"
    assert obs.seat_wind == "1z"
    assert obs.round_wind == "1z"
    assert obs.issues == []
    assert domain.stable
    assert domain.discard_history == ((), ("7p",), ())
    assert domain.temporary_furiten is None
    assert domain.riichi_furiten is False
    assert domain.history_complete
    assert domain.kuikae_tile is None
    assert domain.replacement_draw is None


def test_player_order_starts_at_self_and_follows_active_slots():
    players = (make_player("bottom", "2z", river=("3p",)),
               make_player("top", "1z", river=("5s",)),
               make_player("left", "3z", river=("9p",)))
    state = make_state(self_slot="top", players=players)
    domain = build_domain_state(state, make_observation(), visible_reading(HAND14),
                                {}, pending_riichi=False, window=WINDOW)

    assert domain.observation.rivers == (("5s",), ("3p",), ("9p",))
    assert domain.discard_history == ((), (), ())


@pytest.mark.parametrize("riichi,ron_enabled", [(False, False), (True, False), (True, True)])
def test_unobserved_furiten_is_unknown_unless_client_proves_eligibility(riichi, ron_enabled):
    players = (make_player("bottom", "1z", dealer=True, riichi=riichi),
               make_player("top", "2z"), make_player("left", "3z"))
    tracked = make_state(HAND13, players=players, current_actor="top")
    buttons = {"ron": BUTTON_BOX} if ron_enabled else {}
    obs = make_observation(HAND13, action_buttons=tuple(buttons), drawn_index=None)
    domain = build_domain_state(tracked, obs, visible_reading(HAND13), buttons,
                                pending_riichi=False, window=WINDOW)
    assert domain.temporary_furiten is (False if ron_enabled else None)
    assert domain.riichi_furiten is (False if ron_enabled or not riichi else None)
    payload = domain.for_model()
    assert payload["temporary_furiten"] is domain.temporary_furiten
    assert payload["replacement_draw"] is None
    if ron_enabled:
        assert payload["context_sources"]["temporary_furiten"] == "enabled_client_ron"


@pytest.mark.parametrize("replacement_event", [GameEventType.KAN, GameEventType.KITA, None])
def test_draw_context_follows_current_hand_events_and_expires_on_discard(replacement_event):
    history = [GameEvent(GameEventType.ROUND_START, 1., 1)]
    if replacement_event:
        history.append(GameEvent(replacement_event, 2., 2, player="bottom"))
    history.append(GameEvent(GameEventType.DRAW, 3., 3, player="bottom", tile="9s"))
    tracked = replace(make_state(), history=tuple(history))
    domain = build_domain_state(tracked, make_observation(), visible_reading(HAND14), {},
                                pending_riichi=False, window=WINDOW)
    assert domain.replacement_draw is bool(replacement_event)
    discarded = GameEvent(GameEventType.DISCARD, 4., 4, player="bottom", tile="9s")
    domain = build_domain_state(replace(tracked, history=(*history, discarded)),
                                make_observation(), visible_reading(HAND14), {},
                                pending_riichi=False, window=WINDOW)
    assert domain.replacement_draw is False


def test_pon_kuikae_reaches_rules_and_clears_after_discard(engine):
    pon = GameEvent(GameEventType.PON, 2., 2, player="bottom",
                    details={"tiles": ("5z",) * 3})
    hand = tuple(parse_hand("123456p123s55z"))
    players = (make_player("bottom", "1z", dealer=True,
                           melds=(TrackedMeld(("5z",) * 3, "pon", True),)),
               make_player("top", "2z"), make_player("left", "3z"))
    # A legal tile population: the called type still has one concealed copy.
    hand = hand[:-1] + ("6z",)
    tracked = replace(make_state(hand, players=players), history=(pon,))
    obs = make_observation(hand, drawn_index=None)
    domain = build_domain_state(tracked, obs, visible_reading(hand), {},
                                pending_riichi=False, window=WINDOW)
    assert domain.kuikae_tile == "5z"
    assert domain.for_model()["kuikae_tile"] == "5z"
    assert all(a.tile != "5z" for a in engine.legal_actions(domain))
    discarded = GameEvent(GameEventType.DISCARD, 3., 3, player="bottom", tile="6z")
    domain = build_domain_state(replace(tracked, history=(pon, discarded)),
                                obs, visible_reading(hand), {},
                                pending_riichi=False, window=WINDOW)
    assert domain.kuikae_tile is None


def test_resync_or_new_hand_does_not_reuse_old_replacement_context():
    history = (GameEvent(GameEventType.KAN, 1., 1, player="bottom"),
               GameEvent(GameEventType.ROUND_START, 2., 2, certainty="inferred"),
               GameEvent(GameEventType.DRAW, 3., 3, player="bottom", tile="9s",
                         details={"relative_order": "ambiguous_with_discard"}))
    domain = build_domain_state(replace(make_state(), history=history), make_observation(),
                                visible_reading(HAND14), {}, pending_riichi=False, window=WINDOW)
    assert domain.replacement_draw is None


def test_melds_and_kita_map_per_ordered_player():
    players = (make_player("bottom", "1z", dealer=True, melds=(("5p", "5p", "5p"),),
                           kita=2),
               make_player("top", "2z", riichi=True),
               make_player("left", "3z"))
    hand = ("1p", "2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p", "1s", "2s")
    state = make_state(hand, players=players)
    observation = make_observation(hand, action_buttons=("pass",),
                                   drawn_index=None)
    domain = build_domain_state(state, observation, visible_reading(hand), {},
                                pending_riichi=False, window=WINDOW)

    assert domain.observation.melds[0][0].tiles == ("5p", "5p", "5p")
    assert domain.observation.melds[0][0].opened
    assert domain.observation.nuki == (2, 0, 0)
    assert domain.observation.riichi == (False, True, False)


@pytest.mark.parametrize("tracking_phase,expected", [
    (TrackingPhase.ROUND_END, "result"),
    (TrackingPhase.GAME_END, "result"),
    (TrackingPhase.WAIT_GAME, "unknown"),
    (TrackingPhase.RESYNC, "unknown"),
])
def test_infer_phase_terminal_and_inactive_phases(tracking_phase, expected):
    assert infer_phase(make_observation(), make_state(phase=tracking_phase),
                       pending_riichi=False) == expected


@pytest.mark.parametrize("buttons,hand,actor,pending_riichi,melds,expected", [
    (("tsumo",), HAND14, "bottom", False, (), "discard"),
    (("pass", "ron"), HAND13, "top", False, (), "response"),
    (None, HAND14, "bottom", False, (), "discard"),
    (None, HAND14, "bottom", True, (), "unknown"),
    (None, HAND14, "top", False, (), "waiting"),
    (None, HAND13, "bottom", False, (), "waiting"),
    (("kan",), HAND13[:-2], "bottom", False, (("5p", "5p", "5p"),), "discard"),
    (None, HAND13[:-2], "bottom", False, (("5p", "5p", "5p"),), "discard"),
])
def test_infer_phase_tracking_matrix(buttons, hand, actor, pending_riichi,
                                     melds, expected):
    players = tuple(make_player(
        slot, seat, dealer=index == 0,
        melds=melds if index == 0 else (),
    ) for index, (slot, seat) in enumerate(zip(SLOTS, SEATS)))
    state = make_state(hand, current_actor=actor, players=players)
    observation = make_observation(hand, action_buttons=buttons,
                                   drawn_index=len(hand) - 1 if len(hand) % 3 == 2 else None)
    assert infer_phase(observation, state, pending_riichi=pending_riichi) == expected


def test_normalize_buttons_resolves_he_by_phase():
    he = Box(700, 600, 780, 640)
    assert normalize_buttons({"he": he}, "discard") == {"tsumo": he}
    assert normalize_buttons({"he": he}, "response") == {"ron": he}
    assert normalize_buttons({"he": he}, "waiting") == {}
    assert normalize_buttons({"he": he}, "unknown") == {}


def test_normalize_buttons_keeps_original_key_on_conflict():
    he = Box(700, 600, 780, 640)
    ron = Box(800, 600, 880, 640)
    assert normalize_buttons({"he": he, "ron": ron}, "response") == {"ron": ron}


def test_normalize_buttons_passes_logical_names_and_drops_unknown():
    box = Box(0, 0, 10, 10)
    raw = {"pon": box, "pass": box, "nuki": box, "chi": box, "cancel": box}
    assert normalize_buttons(raw, "response") == {"pon": box, "pass": box,
                                                  "nuki": box}


def test_bridge_rejects_not_decision_ready_state():
    with pytest.raises(BridgeError, match="decision-ready"):
        build_domain_state(make_state(history_complete=False),
                           make_observation(), visible_reading(HAND14), {},
                           pending_riichi=False, window=WINDOW)


@pytest.mark.parametrize("kind,opened,riichi_allowed", [
    ("ankan", False, True), ("daiminkan", True, False),
    ("kakan", True, False), ("unknown", None, False),
])
def test_quad_type_survives_bridge_and_controls_riichi(engine, kind, opened,
                                                     riichi_allowed):
    hand = tuple(parse_hand("123456789p12s"))
    players = (make_player("bottom", "1z", dealer=True,
                           melds=(TrackedMeld(("5z",) * 4, kind, opened),)),
               make_player("top", "2z"), make_player("left", "3z"))
    state = make_state(hand, players=players)
    domain = build_domain_state(
        state, make_observation(hand, drawn_index=10), visible_reading(hand),
        {"riichi": BUTTON_BOX}, pending_riichi=False, window=WINDOW)
    assert domain.observation.melds[0][0].kind == kind
    assert domain.observation.melds[0][0].opened is opened
    actions = engine.legal_actions(domain)
    # Riichi is offered as compound riichi:<index>:<tile> choices.
    assert any(a.kind == "riichi_declare" for a in actions) == riichi_allowed
    assert any(a.kind == "discard" for a in actions)


def test_bridge_rejects_hand_length_mismatch():
    with pytest.raises(BridgeError, match="14"):
        build_domain_state(make_state(), make_observation(),
                           visible_reading(HAND13), {},
                           pending_riichi=False, window=WINDOW)


def test_bridge_rejects_positional_tile_mismatch():
    visible = visible_reading(("2p",) + HAND14[1:])
    with pytest.raises(BridgeError, match="index 0"):
        build_domain_state(make_state(), make_observation(), visible, {},
                           pending_riichi=False, window=WINDOW)


def test_bridge_rejects_unknown_meld_tiles():
    players = (make_player("bottom", "1z", dealer=True),
               make_player("top", "2z", melds=(("5p", None, "5p"),)),
               make_player("left", "3z"))
    with pytest.raises(BridgeError, match="unknown tiles"):
        build_domain_state(make_state(players=players), make_observation(),
                           visible_reading(HAND14), {},
                           pending_riichi=False, window=WINDOW)


def test_bridged_state_feeds_rule_engine(engine):
    state = make_state()
    observation = make_observation(action_buttons=("riichi",))
    domain = build_domain_state(state, observation, visible_reading(HAND14),
                                {"riichi": BUTTON_BOX},
                                pending_riichi=False, window=WINDOW)

    actions = engine.legal_actions(domain)

    ids = {action.id for action in actions}
    assert any(i.startswith("riichi:") for i in ids) and "button:riichi" not in ids
    discards = [action for action in actions if action.kind == "discard"]
    assert len(discards) == 14
    assert "discard:13:9s" in ids


def _mock_reader(button_reader=None, slots=SLOTS):
    sequence = SimpleNamespace(tiles=(), drawn_index=None, issues=(), complete=True)
    meld = SimpleNamespace(groups=(), issues=(), complete=True)
    visible = SimpleNamespace(
        active_slots=slots, hand=sequence, dora=sequence,
        rivers={slot: sequence for slot in slots},
        melds={slot: meld for slot in slots}, kita={}, issues=())
    number = lambda value: SimpleNamespace(
        value=value, glyphs=(), issues=(), complete=True)
    metadata = SimpleNamespace(
        active_slots=slots,
        round=SimpleNamespace(round_wind="1z", hand_number=1, glyphs=(),
                              issues=(), complete=True),
        remaining_tiles=number(55), honba=number(0), riichi_sticks=number(0),
        scores={slot: number(35000) for slot in slots},
        seats={slot: SimpleNamespace(
            wind=seat, dealer=index == 0, glyph=None, issues=(), complete=True)
            for index, (slot, seat) in enumerate(zip(slots, SEATS))},
        issues=())
    reader = object.__new__(FrameObservationReader)
    reader.visible_reader = SimpleNamespace(read=lambda *_: visible)
    reader.metadata_reader = SimpleNamespace(read=lambda *_: metadata)
    if button_reader is not None:
        reader.button_reader = button_reader
    return reader


def _mock_roi():
    return ROIConfig.load(ROOT / "configs/roi.yaml")


def test_reader_without_button_reader_keeps_action_buttons_none():
    reader = _mock_reader()
    observation = reader.read(np.zeros((1156, 1920, 3), dtype=np.uint8),
                              _mock_roi(), SLOTS, timestamp=1.0, frame_index=30)
    assert observation.action_buttons is None


def test_reader_with_button_reader_collects_sorted_labels():
    reader = _mock_reader(button_reader=SimpleNamespace(
        read=lambda *_: {"pass": BUTTON_BOX, "he": BUTTON_BOX}))
    observation, visible = reader.read_with_visible(
        np.zeros((1156, 1920, 3), dtype=np.uint8), _mock_roi(), SLOTS,
        timestamp=1.0, frame_index=30)
    assert observation.action_buttons == ("he", "pass")
    assert visible.hand.tiles == ()


def test_reader_button_failure_downgrades_to_issue():
    def fail(*_):
        raise RuntimeError("boom")

    reader = _mock_reader(button_reader=SimpleNamespace(read=fail))
    observation = reader.read(np.zeros((1156, 1920, 3), dtype=np.uint8),
                              _mock_roi(), SLOTS, timestamp=1.0, frame_index=30)
    assert observation.action_buttons is None
    assert "action_buttons: reader failed" in observation.issues
