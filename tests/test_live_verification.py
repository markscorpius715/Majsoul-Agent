"""Action evidence must match the target, own hand, source and replacement."""
from dataclasses import replace

import pytest

from majsoul.domain import Action
from majsoul.live.verification import assess_action, confirm_anchored_discard, verify_action
from majsoul.state.schemas import FrameObservation, ObservedMeld, ObservedTile, PlayerObservation


FILLER = ("1m", "9m", "1p", "2p", "3p", "4p", "6p", "7p", "8p", "9p")


def tiles(labels):
    return tuple(ObservedTile(label, confidence=.95) for label in labels)


def meld(labels, kind="pon"):
    return ObservedMeld(tiles(labels), kind)


def snapshot(hand, *, own_melds=(), own_river=(), kita=0, source_river=(), buttons=()):
    players = (PlayerObservation("bottom", "1z", True, None, river=tiles(own_river),
                                 melds=own_melds, kita_count=kita, score_available=False),
               PlayerObservation("left", "2z", False, None, river=tiles(source_river),
                                 kita_count=0, score_available=False),
               PlayerObservation("right", "3z", False, None, kita_count=0, score_available=False))
    return FrameObservation(10., 1, ("bottom", "left", "right"), "1z", 1, 0, 0, 50,
                            players, tiles(hand), None, tiles(("3s",)), buttons)


def advance(before, *, hand=None, own_melds=None, own_river=None, source_river=None,
            kita=None, wall_draw=0, drawn_index=None, **changes):
    own, source, other = before.players
    own = replace(own, melds=own.melds if own_melds is None else own_melds,
                  river=own.river if own_river is None else own_river,
                  kita_count=own.kita_count if kita is None else kita)
    if source_river is not None:
        source = replace(source, river=source_river)
    return replace(before, timestamp=before.timestamp+1, frame_index=before.frame_index+1,
                   players=(own, source, other), hand=before.hand if hand is None else tiles(hand),
                   remaining_tiles=before.remaining_tiles-wall_draw, drawn_index=drawn_index, **changes)


def check(action, before, after):
    return assess_action(action, before, after, "bottom")


def discard_case():
    before = snapshot(FILLER+("1s", "2s", "3s", "9s"), own_river=("7s",))
    action = Action("discard:13:9s", "discard", "9s", 13)
    after = advance(before, hand=FILLER+("1s", "2s", "3s"), own_river=tiles(("7s", "9s")))
    return action, before, after


def test_verify_discard_appends_exact_tile_and_removes_selected_hand_tile():
    action, before, after = discard_case()
    assert verify_action(action, before, after, "bottom")
    assert not verify_action(action, before, advance(before, own_river=after.player("bottom").river), "bottom")
    assert not verify_action(replace(action, hand_index=0), before, after, "bottom")
    wrong = advance(before, hand=FILLER+("1s", "2s", "3s"), own_river=tiles(("7s", "8s")))
    assert check(action, before, wrong).reason == "discard_target_mismatch"
    reordered = advance(before, hand=FILLER+("1s", "2s", "3s"), own_river=tiles(("9s", "7s")))
    assert check(action, before, reordered).status == "unknown"


def test_discard_can_include_next_opponent_wall_draw_but_not_multiple_draws():
    action, before, after = discard_case()
    assert check(action, before, replace(after, remaining_tiles=49)).status == "verified"
    assert check(action, before, replace(after, remaining_tiles=48)).status == "unknown"


def test_early_conserved_discard_can_be_confirmed_after_opponent_draws():
    action, before, anchor = discard_case()
    anchor = replace(anchor, remaining_tiles=49)
    later = replace(anchor, timestamp=12., frame_index=3, remaining_tiles=48)
    assert check(action, before, later).status == 'unknown'
    result = confirm_anchored_discard(action, before, anchor, later, 'bottom')
    assert result.status == 'verified'
    assert result.evidence['anchor_wall_delta'] == 1
    assert result.evidence['confirmation_wall_delta'] == 2


@pytest.mark.parametrize('change', ['absent', 'unproved_anchor', 'hand', 'river', 'round', 'unordered', 'wall_increase', 'uncertain'])
def test_earlier_discard_proof_cannot_hide_changed_or_missing_evidence(change):
    action, before, anchor = discard_case()
    anchor = replace(anchor, remaining_tiles=49)
    later = replace(anchor, timestamp=12., frame_index=3, remaining_tiles=48)
    if change == 'absent':
        anchor = None
    elif change == 'unproved_anchor':
        anchor = replace(anchor, remaining_tiles=48)
    elif change == 'hand':
        later = replace(later, hand=before.hand)
    elif change == 'river':
        own = replace(later.players[0], river=tiles(('7s', '9s', '8s')))
        later = replace(later, players=(own, *later.players[1:]))
    elif change == 'round':
        later = replace(later, hand_number=2)
    elif change == 'unordered':
        later = replace(later, timestamp=anchor.timestamp, frame_index=anchor.frame_index)
    elif change == 'wall_increase':
        later = replace(later, remaining_tiles=51)
    elif change == 'uncertain':
        later = replace(later, issues=('hand: unknown',))
    assert confirm_anchored_discard(action, before, anchor, later, 'bottom').status == 'unknown'


def test_a_different_conserved_discard_is_failed_rather_than_success_or_timeout():
    action, before, _ = discard_case()
    labels = [tile.tile for tile in before.hand]
    labels.remove("8p")
    after = advance(before, hand=labels, own_river=tiles(("7s", "8p")))
    result = check(action, before, after)
    assert result.status == "failed"
    assert result.reason == "different_discard_observed"
    assert result.evidence == {"requested_tile": "9s", "discarded_tile": "8p"}


def test_discard_immediately_called_is_proved_without_transient_river_frame():
    action, before, after = discard_case()
    own, source, other = after.players
    after = replace(after, players=(replace(own, river=before.players[0].river),
                                    replace(source, melds=(meld(("9s",)*3),)), other))
    result = check(action, before, after)
    assert result.status == "verified"
    assert result.evidence["called_by"] == "left"
    wrong = replace(after, players=(after.players[0], replace(after.players[1], melds=(meld(("8s",)*3),)), other))
    assert check(action, before, wrong).status == "unknown"


@pytest.mark.parametrize("replacement_tile", ["6s", "4z"])
def test_verify_nuki_requires_north_removal_display_and_replacement(replacement_tile):
    action = Action("button:nuki", "nuki", button="nuki")
    before = snapshot(FILLER+("1s", "2s", "3s", "4z"), kita=1)
    after = advance(before, hand=FILLER+("1s", "2s", "3s", replacement_tile),
                    kita=2, wall_draw=1, drawn_index=13)
    result = check(action, before, after)
    assert result.status == "verified"
    assert result.evidence["replacement_tile"] == replacement_tile
    assert check(action, before, advance(before, kita=2)).status == "unknown"
    assert check(action, before, replace(after, remaining_tiles=50)).status == "unknown"
    assert check(action, before, advance(before, kita=3)).status == "unknown"
    # The north was removed but replacement has not yet been observed.
    pending = advance(before, hand=FILLER+("1s", "2s", "3s"), kita=2)
    assert check(action, before, pending).status == "unknown"


def pon_case():
    action = Action("button:pon", "pon", button="pon")
    before = snapshot(FILLER+("5p", "0p", "2s"), source_river=("8s", "5p"), buttons=("pon", "pass"))
    after = advance(before, hand=FILLER+("2s",), own_melds=(meld(("5p", "0p", "5p")),),
                    source_river=tiles(("8s",)), action_buttons=())
    return action, before, after


def test_verify_pon_requires_hand_and_exact_called_tile_source_with_red_identity():
    action, before, after = pon_case()
    result = check(action, before, after)
    assert result.status == "verified"
    assert result.evidence["consumed_tiles"] == ["0p", "5p"]
    assert result.evidence["source_slot"] == "left"
    assert check(action, before, replace(after, hand=before.hand)).status == "unknown"
    no_source = replace(after, players=(after.players[0], before.players[1], after.players[2]))
    assert check(action, before, no_source).status == "unknown"
    lost_red = replace(after, hand=tiles(FILLER+("0p",)))
    assert check(action, before, lost_red).status == "unknown"
    wrong_source = replace(before, players=(before.players[0], replace(before.players[1], river=tiles(("8s", "0p"))), before.players[2]))
    assert check(action, wrong_source, after).status == "unknown"


def test_kan_cannot_be_verified_by_an_unrelated_pon_count_increment():
    _, before, after = pon_case()
    action = Action("button:kan", "kan", button="kan")
    assert check(action, before, after).status == "unknown"


@pytest.mark.parametrize("kind", ["ankan", "daiminkan", "kakan"])
@pytest.mark.parametrize("replacement_tile", ["6s", "1m"])
def test_verify_kan_distinguishes_types_and_requires_replacement(kind, replacement_tile):
    action = Action("button:kan", "kan", button="kan")
    group = meld(("5p", "0p", "5p", "5p"), kind)
    if kind == "ankan":
        before = snapshot(FILLER+("5p", "0p", "5p", "5p"))
    elif kind == "daiminkan":
        before = snapshot(FILLER+("5p", "0p", "5p"), source_river=("5p",))
    else:
        before = snapshot(FILLER+("5p",), own_melds=(meld(("5p", "0p", "5p")),))
    after = advance(before, hand=FILLER+(replacement_tile,), own_melds=(group,),
                    source_river=(), wall_draw=1, drawn_index=len(FILLER))
    result = check(action, before, after)
    assert result.status == "verified"
    assert result.evidence["meld_kind"] == kind
    assert result.evidence["replacement_tile"] == replacement_tile
    assert check(replace(action, tile="9s"), before, after).status == "unknown"
    assert check(action, before, replace(after, remaining_tiles=50)).status == "unknown"
    assert check(action, before, replace(after, hand=before.hand)).status == "unknown"
    # Unknown type cannot turn a new four-face group into a proved closed kan.
    if kind == "ankan":
        ambiguous = advance(before, hand=FILLER+(replacement_tile,), own_melds=(meld(("5p", "0p", "5p", "5p"), "unknown"),), wall_draw=1)
        assert check(action, before, ambiguous).status == "unknown"


def test_own_ankan_hidden_faces_use_only_preclick_own_hand_and_replacement_marker():
    action = Action("button:kan", "kan", button="kan")
    before = snapshot(FILLER+("5p", "0p", "5p", "5p"))
    hidden = ObservedMeld((ObservedTile(None, face_down=True), ObservedTile("5p"),
                           ObservedTile("5p"), ObservedTile(None, face_down=True)), "ankan")
    after = advance(before, hand=FILLER+("6s",), own_melds=(hidden,), wall_draw=1, drawn_index=10)
    result = check(action, before, after)
    assert result.status == "verified"
    assert result.evidence["consumed_tiles"] == ["0p", "5p", "5p", "5p"]
    # Exact conservation can also prove a draw after auto-sort removed the gap.
    assert check(action, before, replace(after, drawn_index=None)).status == "verified"
    incomplete = replace(before, hand=tiles(FILLER+("5p", None, "5p", "5p")))
    assert check(action, incomplete, after).status == "unknown"


def test_riichi_requires_a_new_horizontal_discard_and_matching_hand_removal():
    _, before, after = discard_case()
    action = Action("riichi_discard:13:9s", "riichi", "9s", 13)
    river = (*after.players[0].river[:-1], replace(after.players[0].river[-1], orientation="horizontal"))
    after = replace(after, players=(replace(after.players[0], river=river), *after.players[1:]))
    assert check(action, before, after).status == "verified"
    assert check(action, before, replace(after, hand=before.hand)).status == "unknown"
    assert check(action, before, advance(before)).status == "unknown"


@pytest.mark.parametrize("kind", ["ron", "tsumo", "abort", "pass"])
def test_verify_resolution_does_not_invent_winner_or_active_pass_attribution(kind):
    before = snapshot(FILLER+("1s", "2s", "3s"), buttons=(kind,))
    action = Action(f"button:{kind}", kind, button=kind)
    after = advance(before, action_buttons=())
    assert not verify_action(action, before, after, "bottom")
    if kind == "pass":
        assert check(action, before, after).reason == "response_closed_without_acceptance_evidence"
        assert check(action, before, after).evidence["client_timeout_possible"] is True
    assert check(action, before, advance(before)).status == "unknown"


@pytest.mark.parametrize("changes", [dict(issues=("hand: unknown",)), dict(hand_number=2),
                                     dict(timestamp=9.), dict(frame_index=1), dict(remaining_tiles=None)])
def test_verify_rejects_uncertain_unordered_cross_hand_or_missing_observation(changes):
    action, before, after = discard_case()
    assert check(action, before, replace(after, **changes)).status == "unknown"


def test_verify_rejects_unknown_kind_and_missing_self_slot():
    action, before, after = discard_case()
    assert check(replace(action, kind="chi"), before, after).status == "unknown"
    assert assess_action(action, before, after, "top").status == "unknown"


def ui_hidden(observation, fields=("river_bottom", "melds_bottom", "kita_bottom")):
    players = tuple(replace(p, kita_count=None) if f"kita_{p.slot}" in fields else p
                    for p in observation.players)
    return replace(observation, players=players, unavailable_regions=fields,
                   ui_occlusions=((100, 100, 200, 140),),
                   issues=tuple(f"{name}: UI occlusion; region unavailable" for name in fields))


def test_partial_discard_requires_new_visible_river_position_and_exact_hand_removal():
    action, before, after = discard_case()
    river_before = (replace(before.players[0].river[0], box=(10, 10, 20, 30)),)
    river_after = (*river_before, replace(after.players[0].river[-1], box=(30, 10, 40, 30)))
    before = ui_hidden(replace(before, players=(replace(before.players[0], river=river_before), *before.players[1:])))
    after = ui_hidden(replace(after, players=(replace(after.players[0], river=river_after), *after.players[1:])))
    assert check(action, before, after).status == "verified"
    assert check(action, before, replace(after, hand=before.hand)).status == "unknown"
    # A tile emerging from the former UI band may have already existed.
    exposed_old = (*river_before, replace(river_after[-1], box=(110, 110, 120, 130)))
    after_revealed = replace(after, players=(replace(after.players[0], river=exposed_old), *after.players[1:]),
                             unavailable_regions=(), issues=(), ui_occlusions=())
    assert check(action, before, after_revealed).reason == "discard_may_have_been_hidden_before_action"
    no_geometry = replace(after, players=(replace(after.players[0], river=tiles(("7s", "9s"))), *after.players[1:]))
    assert check(action, before, no_geometry).status == "unknown"
    shifted_prefix = (replace(river_before[0], box=(11, 10, 21, 30)), river_after[-1])
    shifted = replace(after, players=(replace(after.players[0], river=shifted_prefix), *after.players[1:]))
    assert check(action, before, shifted).reason == "river_positions_unproved_across_occlusion"


def test_unrelated_ui_blind_spot_does_not_erase_complete_discard_proof():
    action, before, after = discard_case()
    before, after = ui_hidden(before, ("river_left",)), ui_hidden(after, ("river_left",))
    assert check(action, before, after).status == "verified"
    assert check(action, before, replace(after, issues=(*after.issues, "hand: unknown"))).status == "unknown"
    assert check(action, before, replace(after, ui_occlusions=())).status == "unknown"


@pytest.mark.parametrize("kind", ["nuki", "kan", "pon", "riichi"])
def test_partial_table_does_not_prove_structure_or_declaration_actions(kind):
    action, before, after = discard_case()
    requested = (Action('button:riichi', 'riichi', button='riichi') if kind == 'riichi'
                 else replace(action, kind=kind))
    result = check(requested, ui_hidden(before), ui_hidden(after))
    assert result.status == "unknown"
    assert result.reason == "action_dependencies_unavailable"


def test_positive_hidden_meld_count_does_not_prove_unchanged_discard_structure():
    action, before, after = discard_case()
    before = ui_hidden(replace(before, hand=before.hand[3:]))
    after = ui_hidden(replace(after, hand=after.hand[3:]))
    # Even a plausible visible append is insufficient when the own melds are hidden.
    result = check(replace(action, hand_index=10), before, after)
    assert result.status == "unknown"
    assert result.reason == "self_meld_structure_unavailable"


def test_revealed_zero_nuki_does_not_erase_visible_discard_proof():
    action, before, after = discard_case()
    prefix = (replace(before.players[0].river[0], box=(10, 10, 20, 30)),)
    before = ui_hidden(replace(before, players=(replace(before.players[0], river=prefix),
                                               *before.players[1:])))
    appended = (*prefix, replace(after.players[0].river[-1], box=(30, 10, 40, 30)))
    after = replace(after, players=(replace(after.players[0], river=appended), *after.players[1:]))
    assert before.player('bottom').kita_count is None
    assert after.player('bottom').kita_count == 0
    assert check(action, before, after).status == 'verified'
    # Revealed positive counts, unmarked missing fields and changed known counts
    # do not prove unchanged nuki structure.
    positive = replace(after, players=(replace(after.players[0], kita_count=1), *after.players[1:]))
    assert check(action, before, positive).reason == 'self_structure_changed_during_discard'
    unmarked = replace(before, unavailable_regions=(), issues=(), ui_occlusions=())
    assert check(action, unmarked, after).status == 'unknown'
    known = replace(before, players=(replace(before.players[0], kita_count=1), *before.players[1:]))
    assert check(action, known, after).status == 'unknown'
