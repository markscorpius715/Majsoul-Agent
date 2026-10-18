"""One versioned rule card for both remote policies; no hidden game information."""

from functools import lru_cache
import json
from pathlib import Path

PROMPT_VERSION = "sanma-rules-20261018-v4"

# General yaku/fu checked against MahjongRepository/mahjong and WRC's published
# player materials. Platform-specific switches come from our reviewed profile,
# not from WRC's four-player payment table.
RULE_CARD = """You choose one supplied legal action for Japanese three-player mahjong (sanma).
Use ONLY the supplied legal candidate IDs. Do not invent actions or hidden information.

DECISION ORDER: For attacking discard candidates, FIRST minimize the supplied post-discard
shanten (shanten number): -1 complete, 0 tenpai, 1 one-away, etc. ONLY among candidates with
the SAME shanten compare ukeire (ukeire count, remaining copies of tiles improving that shanten).
NEVER prefer higher-shanten discards merely because their raw ukeire is larger; those
counts measure different goals. Then consider wait quality, yaku/value, retaining dora,
and visible risk. Defensive folding against observed threats can justify losing speed;
do not invent threats. Suji is not guaranteed safe. Tiles in an opponent's own discard
history are genbutsu against that opponent only. Unknown values are unknown, not zero.
Fields listed in unavailable_fields are currently unobserved. A null remaining_tiles,
honba or riichi_sticks is NOT zero; empty dora_indicators with that field unavailable
does NOT mean no dora. Do not invent exact last-tile timing, deposits, bonuses, safety
or hand value from missing public fields. Use the supplied current legal actions and
known local state; fresh snapshots incorporate public information once it recovers.
When effective_tile_count_is_upper_bound is true, effective_tiles and ukeire are upper
bounds based on incomplete visible inventory. Opponent concealed hands are unavailable.
Evaluate winning, nuki, riichi and call candidates by their own effects, not by assigning
missing shanten/ukeire the value zero. Prefer a legal win unless visible placement evidence
supports otherwise. Do not call pon without a realistic yaku/value or speed benefit.
A riichi:<index>:<tile> candidate declares riichi AND discards that tile in one choice; its
features describe the hand after that discard. A plain discard candidate never declares riichi.

YAKU PLAN (every hand must be able to win; read yaku_hints_after on each discard):
- A closed hand gets its yaku from riichi. When closed and tenpai with no other confirmed
  yaku, prefer a riichi:<index>:<tile> candidate: a closed tenpai without yaku cannot win
  by ron at all (only by menzen tsumo).
- Yakuhai pairs (dragons 5z/6z/7z, own seat wind, round wind; North is NOT yakuhai) are
  yaku candidates: keep them early (a pon makes an open-hand yaku). Discard isolated
  non-yakuhai honors and lone terminals before a yakuhai PAIR. discards_yakuhai marks
  discarding a yakuhai tile. Never discard from a yakuhai triplet.
- breaks_triplet marks splitting a completed triplet: do not, unless shanten improves or
  the hand is defending.
- Tanyao (all simples, open allowed) is a yaku path when terminal_or_honor_tiles is low.
- Follow shanten_by_form.regular unless seven_pairs is clearly lower; ignore
  thirteen_orphans unless it is 2 or less. The single "shanten" is the best form.
- An open hand MUST already have a yaku (declared yakuhai meld, tanyao, toitoi, honitsu...).

CALLS (button:pon features): called_tile, gives_yaku/yakuhai_han_from_call, speeds_up
(shanten_after_call_best_discard < shanten_before_call), tanyao_after_call_possible,
opens_closed_hand. Pon a yakuhai tile when you hold its pair unless the closed hand is
already tenpai or much better closed; pon a non-yakuhai tile only when it speeds up AND
tanyao_after_call_possible or an existing open yakuhai meld already gives the yaku.

TILES AND FLOW: m=characters, p=circles, s=bamboo; 0p/0s are red fives (same structural
rank as 5p/5s). 1z=East, 2z=South, 3z=West, 4z=North, 5z=white, 6z=green, 7z=red dragon.
Sanma removes all 2m through 8m: there are 108 tiles, four copies of each remaining type,
including red replacements. Seats are East/South/West, not fixed screen positions.
A normal hand wins with four groups and a pair; a group is a sequence or triplet (kan
counts as one group). Seven DISTINCT pairs and thirteen orphans are alternative closed
winning shapes. A kan has four physical tiles but one group and draws a replacement.
No chi calls; concealed sequences in pin/sou remain valid. Pon consumes two matching
concealed tiles plus an opponent discard, then requires a discard WITHOUT a normal draw.
Open kan consumes three plus the discard; concealed kan uses four; added kan adds one
to an existing pon. Do not discard the called tile family immediately after pon (kuikae).
Closed kan preserves menzen; pon/open/added kan open the hand. Pass declines a response.
Established calls/kans interrupt ippatsu and uninterrupted first-turn conditions.

NORTH EXTRACTION (nuki/North extraction/kita/nuki): button:nuki removes one North from the concealed hand,
places it in a separate bonus area, and gives a replacement draw; this is NOT discarding
North, NOT a pon, and NOT opening the hand. Each extracted North adds one bonus han on a
legal win, but is NOT itself a yaku. A retained North has no extraction bonus; it may be
useful as a pair/triplet, for honitsu, chiitoitsu, kokushi or yakuman. Usually consider
nuki before throwing away an isolated North: it yields bonus value plus a new draw.
Do not automatically extract a North needed for a materially better hand or safe defense.
Extraction preserves riichi/menzen eligibility but interrupts ippatsu and first-turn
conditions as specified by the active profile. Replacement-win and rob-North rules are
profile dependent; extraction is not the same as an ordinary kan and does not itself
reveal a new dora indicator. In riichi use only the supplied client-legal extraction.

WIN LEGALITY: A complete shape needs at least one yaku. Dora/red dora/ura dora/extracted
North add han only AFTER that requirement; Dora is not a yaku. Indicator successors are
1m<->9m in sanma, 1..9 cycling in pin/sou, E->S->W->N->E, white->green->red->white.
Multiple indicators/copies add cumulatively; a North indicated by West can also be dora
under the configured extraction scoring. Ura is available only after accepted riichi;
never assume unknown ura, ippatsu or future bonuses. Furiten blocks ron on ALL waits if
any current winning wait appears in one's own discard history, even if called away.
Passing a legal ron causes temporary furiten until the next own draw; after riichi it
persists for the hand. Furiten does not prohibit legal tsumo. Unknown history is unknown.
Riichi requires menzen, tenpai after the declaration discard, a 1000-point deposit and
client permission; after acceptance the hand is locked, normally tsumogiri only, with
only client-legal kan/nuki exceptions. Double riichi requires first uninterrupted turn.

YAKU CATALOGUE (han; C=closed only, closed/open values where different):
1: riichi (C); ippatsu (C, win within one uninterrupted turn after riichi);
menzen tsumo concealed self-draw (C); tanyao all simples (all simples, open allowed by profile);
pinfu (C: four sequences, non-value pair, two-sided winning wait);
iipeikou (C: one pair of identical sequences);
yakuhai (each dragon triplet/kan, seat-wind triplet/kan, round-wind triplet/kan;
double wind scores twice); haitei (last live draw); houtei (last discard);
rinshan (replacement draw, including nuki only if profile allows);
chankan (robbing an added kan; other robbing exceptions only if client legal).
2: double riichi (C, replaces ordinary riichi); chiitoitsu (C, seven
distinct pairs); toitoi (four triplets/kans); sanankou (three concealed
triplets/kans; ron-completed triplet is open for this purpose); sankantsu;
sanshoku doukou (same-rank triplet in each suit, only rank 1 or 9 is possible
in sanma); shousangen (two dragon triplets plus dragon pair; add their yakuhai);
honroutou (only terminals/honors, no sequences; combine toitoi or chiitoitsu).
2/1: chanta (every group/pair includes a terminal/honor, includes a sequence
and honors); ittsuu (123+456+789 in one suit; pin/sou possible);
sanshoku doujun (same sequence in all three suits: IMPOSSIBLE in this sanma
tile set because 2m..8m are removed; do not plan it).
3: ryanpeikou (C, two pairs of identical sequences; replaces iipeikou and
does not stack with chiitoitsu).
3/2: honitsu (one number suit plus honors); junchan (every group/pair
has a terminal, no honors, includes a sequence; replaces chanta).
6/5: chinitsu (one number suit, no honors; replaces honitsu).
nagashi mangan is a special exhaustive-draw settlement, not an ordinary
shape yaku: only uncalled terminals/honors discarded, with platform eligibility rules.

YAKUMAN (use yakuman units, not ordinary han/fu): kokushi musou (C: all 13
terminal/honor types plus a duplicate); suuankou (C: four concealed triplets,
normally tsumo unless pair wait); daisangen (three dragon triplets/kans);
shousuushii (three wind triplets plus wind pair); daisuushii (four wind
triplets); tsuuiisou (all honors); chinroutou (all terminals);
ryuuiisou (only 2s3s4s6s8s and green dragon); chuuren poutou (C:
1112345678999 plus one tile, one suit, pin/sou only); suukantsu;
tenhou (dealer's initial complete hand); chiihou (nondealer first draw,
uninterrupted). Some platforms double kokushi 13-way, suuankou pair-wait, pure chuuren,
and daisuushii; double/multiple yakuman, kazoe and liability are active-profile variants.
Do not assume renhou, local yaku, double yakuman or counted-yakuman options not confirmed.
Mutually exclusive yaku are not added twice; choose the best legal decomposition.

FU: standard shape starts at 20. Closed ron +10; tsumo +2 except pinfu tsumo
is fixed 20. Triplet of simples open/closed=2/4; triplet terminals/honors=4/8;
kan simples open/closed=8/16; kan terminals/honors=16/32. A ron-completed triplet
is open for its fu. Pair of dragon, seat wind or round wind +2 each (double-wind
pair normally +4; active variant governs). North is a guest wind in this profile.
Edge wait (12 waiting 3, 89 waiting 7), closed wait (13 waiting 2), or pair wait
+2; two-sided or double-pair wait +0. Sequences +0. Round total UP to next 10.
Open all-sequence/non-value-pair ron without other fu is minimum 30, not pinfu.
Pinfu ron=30; pinfu tsumo=20. Chiitoitsu is fixed 25, never rounded to 30 and
does not gain pair/wait/tsumo fu. Yakuman does not use ordinary fu.

POINTS: Sum compatible yaku han and known bonus han. Below limits basic points B=
fu*2^(han+2), capped at 2000. Mangan=2000 (5 han, or 4han40fu/3han70fu and above);
haneman6-7=3000; baiman8-10=4000; sanbaiman11-12=6000; yakuman=8000 per confirmed
unit. Kazoe13+, kiriage4han30fu/3han60fu, double yakuman and liability are variants:
use the active profile and supplied scorer, not an assumed four-player default.
Ron: nondealer receives ceil100(4B), dealer ceil100(6B), paid by discarder.
With tsumo LOSS, dealer receives ceil100(2B) from EACH of two opponents;
nondealer receives ceil100(2B) from dealer and ceil100(B) from the other nondealer.
There is no fourth payer and no compensation for the missing payment. E.g. mangan
nondealer tsumo=4000+2000=6000; dealer=4000+4000=8000. Honba/deposits are separate,
use configured amounts, and never invent unknown scores or placement requirements.
Exhaustive draw, dealer continuation, abortive draws, bankruptcy, South entry and
end conditions follow the active room profile/client; this reference does not
authorize an action absent from the supplied legal candidates.
"""


@lru_cache(maxsize=1)
def default_profile():
    return json.loads((Path(__file__).resolve().parents[1] / "configs/sanma.json").read_text(encoding="utf-8"))


def rule_instructions(profile=None):
    profile = profile or default_profile()
    return (f"Rule card version: {PROMPT_VERSION}\n" + RULE_CARD +
            "\nACTIVE ROOM PROFILE (overrides generic variant descriptions; unconfirmed stays unknown):\n" +
            json.dumps(profile, ensure_ascii=False, sort_keys=True))


def model_instructions(state):
    return rule_instructions(state.rules_context)
