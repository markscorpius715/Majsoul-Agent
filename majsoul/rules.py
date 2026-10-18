from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path

from mahjong.hand_calculating.hand import HandCalculator
from mahjong.hand_calculating.hand_config import HandConfig, OptionalRules
from mahjong.hand_calculating.scores import ScoresCalculator
from mahjong.meld import Meld
from mahjong.shanten import Shanten

from .domain import Action, MeldView, State
from .tiles import counts34, physical_ids, tile_index, tile_name
from .tile_inventory import MeldKnowledge, TileInventory


@dataclass(frozen=True)
class Rules:
    data: dict

    @classmethod
    def load(cls, path: str | Path) -> "Rules":
        data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        if data["players"] != 3 or data["allow_chi"] or not data["tsumo_loss"]:
            raise ValueError("This engine currently supports no-chi sanma with tsumo loss")
        return cls(data)

    @property
    def available(self) -> tuple[int, ...]:
        removed = {tile_index(t) for t in self.data["removed_tiles"]}
        return tuple(i for i in range(34) if i not in removed)

    def validate_tiles(self, tiles: list[str]) -> None:
        counts34(tiles)
        if any(tile_index(t) not in self.available for t in tiles):
            raise ValueError("Tile not in the configured sanma tile set")
        reds = Counter(t for t in tiles if t[0] == "0")
        allowed = Counter(self.data["red_tiles"])
        if any(n > allowed[t] for t, n in reds.items()):
            raise ValueError("Invalid red tile multiplicity")
        for red, red_count in allowed.items():
            if tiles.count("5" + red[1]) > 4 - red_count:
                raise ValueError("Too many non-red fives")

    def dora(self, indicator: str) -> int:
        index = tile_index(indicator)
        if index < 27:
            suit = [i for i in self.available if i // 9 == index // 9]
        elif index < 31:
            suit = list(range(27, 31))
        else:
            suit = list(range(31, 34))
        return suit[(suit.index(index) + 1) % len(suit)]


class RuleEngine:
    def __init__(self, rules: Rules):
        self.rules = rules

    def shanten(self, hand: list[str], meld_count: int = 0) -> int:
        self.rules.validate_tiles(hand)
        if not 0 <= meld_count <= 4 or len(hand) not in (13 - 3 * meld_count, 14 - 3 * meld_count):
            raise ValueError("Concealed tile count does not match meld count")
        return Shanten.calculate_shanten(counts34(hand), use_chiitoitsu=not meld_count, use_kokushi=not meld_count)

    def improvements(self, hand: list[str], meld_count: int, visible: list[str], *,
                     family_counts=None) -> dict[str, int]:
        baseline = self.shanten(hand, meld_count)
        seen = counts34(visible) if family_counts is None else family_counts
        if len(seen) != 34 or any(type(n) is not int or not 0 <= n <= 4 for n in seen):
            raise ValueError("Invalid visible tile-family inventory")
        own = counts34(hand)
        result = {}
        for index in self.rules.available:
            remaining = 4 - seen[index]
            if remaining <= 0 or own[index] >= 4:
                continue
            # Numerical shanten calculation treats red and ordinary five equally.
            array = own.copy()
            array[index] += 1
            value = Shanten.calculate_shanten(array, use_chiitoitsu=not meld_count, use_kokushi=not meld_count)
            if value < baseline:
                result[tile_name(index)] = remaining
        return result

    def waits(self, hand: list[str], meld_count: int = 0) -> set[int]:
        # Include exhausted waits for furiten; availability must NOT remove a wait.
        self.shanten(hand, meld_count)
        own = counts34(hand)
        result = set()
        for index in self.rules.available:
            if own[index] == 4:
                continue
            array = own.copy()
            array[index] += 1
            if Shanten.calculate_shanten(array, use_chiitoitsu=not meld_count, use_kokushi=not meld_count) == -1:
                result.add(index)
        return result

    def permanent_furiten(self, state: State) -> bool:
        obs = state.observation
        hand = [t.tile for t in obs.hand]
        if len(hand) != 13 - 3 * obs.meld_count:
            return False
        return bool(self.waits(hand, obs.meld_count) & {tile_index(t) for t in state.discard_history[0]})

    def visible(self, state: State) -> list[str]:
        """Actually known physical labels, without inventing concealed red IDs."""
        return list(self.inventory(state).known)

    def inventory(self, state: State) -> TileInventory:
        o = state.observation
        known = ([t.tile for t in o.hand] + [t for river in o.rivers for t in river]
                 + list(o.dora_indicators) + [self.rules.data["nuki_tile"]] * sum(n for n in o.nuki if n is not None))
        melds = (MeldKnowledge(m.tiles, m.kind, m.opened, m.face_down)
                 for groups in o.melds for m in groups)
        return TileInventory.build(known, melds)

    DRAGONS = ('5z', '6z', '7z')

    def yakuhai_han(self, tile, o) -> int:
        """Han from a triplet of ``tile``: dragons, seat and round wind.
        North is a guest wind in this sanma ruleset (it is extracted as kita)."""
        name = tile_name(tile_index(tile))
        if name in self.DRAGONS:
            return 1
        if name == '4z' and self.rules.data.get('north_is_guest_wind', True):
            return 0
        return int(name == o.seat_wind) + int(name == o.round_wind)

    @staticmethod
    def _terminal_or_honor(tile) -> bool:
        index = tile_index(tile)
        return index >= 27 or index % 9 in (0, 8)

    def shanten_forms(self, hand, meld_count) -> dict:
        """Shanten per hand form; the combined minimum alone hides structure
        (a seven-pairs count makes breaking a real triplet look free)."""
        counts = counts34(hand)
        forms = {'regular': Shanten.calculate_shanten_for_regular_hand(counts)}
        if not meld_count:
            forms['seven_pairs'] = Shanten.calculate_shanten_for_chiitoitsu_hand(counts)
            forms['thirteen_orphans'] = Shanten.calculate_shanten_for_kokushi_hand(counts)
        return forms

    def yaku_hints(self, hand, o) -> dict:
        """Visible yaku paths for a 13-tile (post-discard) hand plus own melds."""
        counts = Counter(tile_name(tile_index(t)) for t in hand)
        melds = [m for m in o.melds[0]]
        closed = not any(m.opened for m in melds)
        yakuhai = {}
        for tile, n in counts.items():
            han = self.yakuhai_han(tile, o)
            if han and n >= 2:
                yakuhai[tile] = {'copies': n, 'han_if_triplet': han}
        meld_yakuhai = [m.tiles[0] for m in melds if m.tiles and m.tiles[0] and len(set(
            tile_name(tile_index(t)) for t in m.tiles)) == 1 and self.yakuhai_han(m.tiles[0], o)]
        all_tiles = list(hand)+[t for m in melds for t in m.tiles if t]
        non_simple = sum(self._terminal_or_honor(t) for t in all_tiles)
        return {
            'closed_hand_riichi_available': closed,
            'yakuhai_pairs_or_sets': yakuhai,
            'declared_yakuhai_melds': meld_yakuhai,
            'terminal_or_honor_tiles': non_simple,
            'tanyao_possible': non_simple == 0 if not closed else non_simple <= 4,
            'has_confirmed_yaku': bool(meld_yakuhai) or any(v['copies'] >= 3 for v in yakuhai.values()),
        }

    def call_features(self, state: State, button: str) -> dict:
        """What a pon (or open kan) of the current call tile would do."""
        o = state.observation
        tile = o.call_tile
        features = {'called_tile': tile, 'called_from': o.call_from, 'call_tile_known': tile is not None}
        if tile is None:
            return features
        hand = [t.tile for t in o.hand]
        target = tile_index(tile)
        need = 3 if button == 'kan' else 2
        rest, taken = [], 0
        for t in hand:
            if taken < need and tile_index(t) == target:
                taken += 1
            else:
                rest.append(t)
        if taken < need:
            return {**features, 'call_tile_in_hand': False}
        before = self.shanten(hand, o.meld_count)
        han = self.yakuhai_han(tile, o)
        closed = not any(m.opened for m in o.melds[0])
        features.update({
            'call_tile_in_hand': True,
            'yakuhai_han_from_call': han,
            'gives_yaku': han > 0,
            'opens_closed_hand': closed,
            'shanten_before_call': before,
            'existing_open_yakuhai': [m.tiles[0] for m in o.melds[0] if m.tiles and m.tiles[0]
                                      and self.yakuhai_han(m.tiles[0], o)],
        })
        if button == 'pon':
            after = {}
            for i in range(len(rest)):
                value = self.shanten(rest[:i]+rest[i+1:], o.meld_count+1)
                after.setdefault(value, set()).add(tile_name(tile_index(rest[i])))
            best = min(after)
            remaining = rest+[tile]*3
            features.update({
                'shanten_after_call_best_discard': best,
                'best_discards_after_call': sorted(after[best]),
                'speeds_up': best < before,
                'tanyao_after_call_possible': (not self._terminal_or_honor(tile) and
                    sum(self._terminal_or_honor(t) for t in remaining) <= 1),
            })
        return features

    def discard_features(self, state: State, index: int) -> dict:
        o = state.observation
        hand = [t.tile for t in o.hand]
        discarded = hand.pop(index)
        inventory = self.inventory(state)
        visible = list(inventory.known)  # Proposed discard stays publicly visible.
        improving = self.improvements(hand, o.meld_count, visible,
                                      family_counts=inventory.family_counts)
        dora = [self.rules.dora(t) for t in o.dora_indicators]
        retained = hand + [t for m in o.melds[0] for t in m.tiles]
        danger = []
        target = tile_index(discarded)
        for opponent in (1, 2):
            river = {tile_index(t) for t in state.discard_history[opponent]}
            danger.append({
                "opponent": opponent, "riichi": o.riichi[opponent],
                "genbutsu": True if target in river else False if state.history_complete else None,
                "suji_discards": [tile_name(i) for i in sorted(river)
                                  if target < 27 and i // 9 == target // 9 and abs(i - target) == 3],
                "deal_in_probability": None,
            })
        shanten = self.shanten(hand, o.meld_count)
        forms = self.shanten_forms(hand, o.meld_count)
        hints = self.yaku_hints(hand, o)
        winning_values = {}
        own_melds_known = "melds_0" not in o.unavailable_fields or o.meld_count == 0
        dora_known = "dora_indicators" not in o.unavailable_fields
        if (shanten == 0 and o.seat_wind and o.round_wind and own_melds_known and
                dora_known and
                o.nuki[0] is not None and o.riichi[0] is not None and
                all(m.opened is not None for m in o.melds[0])):
            for tile, remaining in improving.items():
                variants = [tile]
                red = "0" + tile[1]
                if tile[0] == "5" and red in self.rules.data["red_tiles"]:
                    variants = []
                    if visible.count(tile) < 3:
                        variants.append(tile)
                    if red not in visible:
                        variants.append(red)
                for variant in variants:
                    winning_values[variant] = self.score(
                        hand + [variant], len(hand), o.melds[0], seat=o.seat_wind,
                        round_wind=o.round_wind, tsumo=True, riichi=o.riichi[0],
                        indicators=o.dora_indicators, nuki=o.nuki[0])
        return {
            "shanten": shanten, "shanten_by_form": forms,
            "breaks_triplet": sum(tile_index(t) == target for t in hand) >= 2,
            "discards_yakuhai": self.yakuhai_han(discarded, o) > 0,
            "yaku_hints_after": hints,
            "effective_tiles": improving,
            "ukeire": sum(improving.values()), "tile_danger": danger,
            "retained_dora": (sum(dora.count(tile_index(t)) + (t[0] == "0") for t in retained) + o.nuki[0]
                              if own_melds_known and dora_known and o.nuki[0] is not None else None),
            "expected_value": None,
            "hand_value_if_next_tsumo": winning_values,
            "hand_value_context": "Current declared riichi/dora/nuki only; no future ura, ippatsu, honba or deposits.",
            "value_note": "No calibrated win/deal-in probability model; dora alone is not yaku.",
            "effective_tile_count_basis": ("known_faces_and_structural_ankan_constraints"
                                           if inventory.constraints else "known_faces"),
            "effective_tile_count_is_upper_bound": bool(o.unavailable_fields),
            "unavailable_fields": list(o.unavailable_fields),
        }

    def legal_actions(self, state: State) -> list[Action]:
        o = state.observation
        if not state.stable or o.issues or o.phase in ("unknown", "result", "waiting"):
            return []
        inventory = self.inventory(state)
        if inventory.unresolved or any(t is None for m in o.melds[0] for t in m.tiles):
            return []
        self.rules.validate_tiles(list(inventory.known))
        inventory.validate_family_counts()
        actions = []
        riichi_declarable = False
        hand = [t.tile for t in o.hand]
        if o.phase == 'riichi_select':
            mask = o.riichi_discard_indices
            if (not isinstance(mask, tuple) or not mask or
                    any(type(i) is not int or not 0 <= i < len(hand) for i in mask) or
                    len(set(mask)) != len(mask) or o.riichi[0] is True or
                    ('melds_0' in o.unavailable_fields and o.meld_count != 0) or
                    any(m.opened is not False for m in o.melds[0])):
                return []
        # UI detections represent ENABLED buttons. The client adjudicates yaku,
        # time-dependent restrictions and riichi-kan wait preservation.
        for button in sorted(o.buttons):
            phases = {
                "ron": {"response"}, "pon": {"response"}, "pass": {"response"},
                "tsumo": {"discard"}, "nuki": {"discard"}, "riichi": {"discard"},
                "abort": {"discard"}, "kan": {"discard", "response"},
            }
            if o.phase not in phases.get(button, set()):
                continue
            if button == "ron" and (state.temporary_furiten or state.riichi_furiten or self.permanent_furiten(state)):
                continue
            if button == "nuki" and self.rules.data["nuki_tile"] not in hand:
                continue
            if button == "tsumo" and self.shanten(hand, o.meld_count) != -1:
                continue
            if button == "riichi":
                if (o.riichi[0] is True or
                        ("melds_0" in o.unavailable_fields and o.meld_count != 0) or
                        any(m.opened is not False for m in o.melds[0]) or len(hand) != 14 - 3 * o.meld_count):
                    continue
                if not any(self.shanten(hand[:i] + hand[i + 1:], o.meld_count) == 0 for i in range(len(hand))):
                    continue
                # Declared together with its discard: one decision, two clicks.
                riichi_declarable = True
                continue
            if button in {"ron", "tsumo", "nuki", "pon", "kan", "pass", "abort"}:
                effects = ({"consumes_tile": "4z", "replacement_draw": True,
                            "bonus_han_on_legal_win": 1, "is_yaku": False,
                            "breaks_menzen": self.rules.data.get("nuki_breaks_menzen", False),
                            "cancels": self.rules.data.get("nuki_cancels", [])}
                           if button == "nuki" else {})
                if button == "pon" or (button == "kan" and o.phase == "response"):
                    effects = {**effects, **self.call_features(state, button)}
                actions.append(Action(f"button:{button}", button, button=button,
                                      features={"legality_source": "enabled_client_button", **effects}))
        if o.phase in ("discard", "riichi_select") and len(hand) == 14 - 3 * o.meld_count:
            if o.phase == "riichi_select" and o.riichi_discard_indices is None:
                return actions
            for index, tile in enumerate(hand):
                if o.phase == "riichi_select" and index not in o.riichi_discard_indices:
                    continue
                if o.phase != "riichi_select" and o.riichi[0] is not False and index != o.draw_index:
                    continue
                local_call_proven = (state.context_sources.get('kuikae_tile') == 'verified_local_pon'
                                     and state.kuikae_tile is not None)
                local_replacement_proven = state.context_sources.get('replacement_draw') in (
                    'verified_local_kan', 'verified_local_nuki')
                if (o.meld_count and not state.history_complete and o.draw_index is None and
                        not local_call_proven and not local_replacement_proven):
                    # A call can impose kuikae. No draw marker/history means we
                    # cannot classify this as an ordinary draw-and-discard turn.
                    # A locally verified call carries its exact kuikae evidence.
                    continue
                if state.kuikae_tile and tile_index(tile) == tile_index(state.kuikae_tile):
                    continue
                features = self.discard_features(state, index)
                if o.phase == "riichi_select" and features["shanten"] != 0:
                    continue
                if o.phase == "riichi_select":
                    actions.append(Action(f"riichi_discard:{index}:{tile}", "riichi", tile, index,
                                          features={**features, "legality_source": "client_selection_and_tenpai"}))
                else:
                    actions.append(Action(f"discard:{index}:{tile}", "discard", tile, index, features=features))
                    if riichi_declarable and features["shanten"] == 0:
                        actions.append(Action(
                            f"riichi:{index}:{tile}", "riichi_declare", tile, index, button="riichi",
                            features={**features, "declares_riichi": True,
                                      "legality_source": "enabled_client_button_and_tenpai"}))
        return actions

    def score(self, hand: list[str], win_index: int, melds: tuple[MeldView, ...] = (), *,
              seat: str, round_wind: str, tsumo: bool, riichi: bool = False,
              indicators: tuple[str, ...] = (), nuki: int = 0, rinshan: bool = False,
              rob_nuki: bool = False, honba: int = 0, deposits: int = 0) -> dict:
        """Score a known winning hand; missing situational yaku must be supplied upstream.

        Score assumptions not present in the screenshot are identified in the output.
        Robbing north is intentionally NOT passed as chankan (it grants no yaku).
        """
        if seat not in ("1z", "2z", "3z") or round_wind not in ("1z", "2z", "3z"):
            raise ValueError("North must remain a guest wind in this sanma ruleset")
        if min(nuki, honba, deposits) < 0 or nuki > 4 or not 0 <= win_index < len(hand):
            raise ValueError("Invalid scoring context")
        if rob_nuki and tsumo:
            raise ValueError("Robbing north is ron")
        all_tiles = hand + [t for m in melds for t in m.tiles]
        self.rules.validate_tiles(all_tiles + ["4z"] * nuki + list(indicators))
        groups = physical_ids([hand] + [list(m.tiles) for m in melds])
        declared = [Meld(Meld.KAN if len(m.tiles) == 4 else Meld.PON, ids, opened=m.opened)
                    for m, ids in zip(melds, groups[1:])]
        # Convert the sanma dora successor into the library's four-player indicator.
        mapped = []
        for indicator in indicators:
            dora = self.rules.dora(indicator)
            if dora < 27:
                predecessor = (dora // 9) * 9 + (dora % 9 - 1) % 9
            elif dora < 31:
                predecessor = 27 + (dora - 28) % 4
            else:
                predecessor = 31 + (dora - 32) % 3
            mapped.append(predecessor * 4)
        options = OptionalRules(has_open_tanyao=self.rules.data["open_tanyao"], has_aka_dora=True)
        config = HandConfig(is_tsumo=tsumo, is_riichi=riichi, is_rinshan=rinshan,
                            player_wind=tile_index(seat), round_wind=tile_index(round_wind), options=options)
        rules = self.rules
        # North extraction is a bonus, never a yaku. The calculator first enforces yaku.
        north_bonus = nuki * (1 + sum(rules.dora(i) == tile_index("4z") for i in indicators))

        class SanmaScores(ScoresCalculator):
            @staticmethod
            def calculate_scores(han, fu, config, is_yakuman=False):
                return ScoresCalculator.calculate_scores(han + (0 if is_yakuman else north_bonus), fu, config, is_yakuman)

        response = HandCalculator.estimate_hand_value(
            [i for group in groups for i in group], groups[0][win_index], melds=declared,
            dora_indicators=mapped, config=config, scores_calculator_factory=SanmaScores)
        if response.error:
            return {"error": response.error}
        yakuman = any(y.is_yakuman for y in response.yaku)
        cost = response.cost
        payments = ([cost["main"], cost["additional"]] if tsumo else [cost["main"]])
        bonus = rules.data["honba_tsumo_per_payer"] if tsumo else rules.data["honba_ron"]
        payments = [p + honba * bonus for p in payments]
        return {"han": response.han + (0 if yakuman else north_bonus), "fu": response.fu,
                "yaku": [str(y) for y in response.yaku], "payments": payments,
                "total": sum(payments) + deposits * 1000,
                "assumptions": rules.data["unconfirmed"]}
