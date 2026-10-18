# Phase 5: Continuous state tracking

## Current state

Phase 5's state structure, stable-frame commit, local event diffing, deterministic validation, and resynchronization path have been implemented.
On 2026-09-22 the user confirmed that this phase is only responsible for events that table-state diffing can prove, and does not depend on Phase 6 action buttons:
`ROUND_START/DRAW/DISCARD/RIICHI/MELD/CHI/PON/KAN/KITA/SCORE_UPDATE/ROUND_END/RESYNC`。
Result events such as `RON/TSUMO/GAME_END` are left to Phase 6; `PASS` belongs only to AgentAction.

The current old recording still cannot verify long-duration continuous state, so Phase 5 has not yet reached real-recording acceptance, and cannot on that basis
proceed to automatic decision-making or clicking.

The implementation is located in `majsoul/state/`:

- `observation.py` combines Phase 3 visible tiles and Phase 4 metadata into a `FrameObservation`;
- `schemas.py` defines observations, player states, `MahjongState`, events, and tracking phases;
- `validator.py` checks tile counts, tile-type/red-tile upper limits, score conservation, local changes in discards/melds,
  remaining tiles, honba, dora, and round transitions;
- `tracker.py` commits only continuously consistent complete observations, and infers events from adjacent committed snapshots;
- `python -m majsoul.state` provides offline entry points for screenshot sequences and recording intervals, and never clicks the game.

All players are keyed by the match-locked physical slots `bottom/top/left/right`; changing rounds only updates seat winds,
dealer, and round number, without reassigning slots. `action_buttons` and `turn_indicator` remain unknown, belonging to
Phase 6。

## Match-level layout automatic locking

The state CLI by default no longer requires manually passing `--active-slots`. When unlocked, it reads metadata separately for each
legal layout in `roi.yaml`; a candidate must simultaneously satisfy: complete fields, correct sanma seat-wind set, exactly one dealer,
legal round-number range, scores in whole hundreds, and conservation of "player scores + honba". Only when exactly one candidate passes is it recorded as one
reliable piece of evidence; by default, it locks after two consecutive frames support the same candidate. Zero candidates or multiple candidates both clear the uncommitted confirmation streak,
so animation frames cannot stitch two non-adjacent pieces of evidence into a lock.

The lock belongs to the whole-match state and is not cleared on a new hand; only a new match or an explicit resync is allowed to reset. During automatic confirmation,
candidate frames are temporarily stored, and after locking they are re-fed into the state tracker in frame-number order, so opening events are not sacrificed. The report writes
`layout_source`, `active_slots`, candidates, and rejection reasons. If you need to reproduce the experiment, you can also explicitly pass in already manually
confirmed `--active-slots bottom top left`.

## Submission and Resync Semantics

By default, two frames with semantically consistent complete observations are required before submission. Confidence does not participate in the semantic fingerprint, avoiding the situation where the same screen, merely due to
small score fluctuations, can never stabilize; however, the confidence of each field and tile is still saved in the report. Incomplete frames will
clear the pending-submission count and do not modify already submitted state. The offline run report records this re-observation request as
`resync_requested=true`; since there is no new table state that can be submitted, it is not a `GameEvent.RESYNC`.

Meld is a controlled exception: if the group count and the 3/4-tile structure are reliable, but individual tile faces are still unknown, as long as the known
tile faces still admit at least one legal pon/chi/kan completion, the snapshot can continue to be tracked and output
`MELD(details.subtype="unknown")`. In this case `uncertain_fields` marks the corresponding meld and
`decision_ready=false`, so it cannot enter decision-making; subsequently merely filling unknown into known tile faces is perceptual
refinement and does not generate a meld event again. If the known tile faces contradict each other and cannot be completed into a legal meld, the snapshot is still rejected.

Adjacent snapshots may only undergo one local step. If the remaining tiles decrease by multiple tiles at once, the discard pond changes non-locally, multiple players' melds change
simultaneously, score changes cannot be explained by a single riichi, or the layout conflicts, the tracker will not fabricate intermediate events, but instead:

1. wait for the current complete snapshot itself to be continuously stable;
2. submit a `RESYNC` event and rebuild the visible state from that snapshot;
3. set `phase=RESYNC`, `history_complete=false`;
4. subsequent locally legal changes can restore `TRACKING`, but missing history will not be re-marked as complete.

When first joining mid-hand, even if the snapshot is stable, `history_complete` is false. Existing tiles in the discard pond can
serve as the current baseline, but cannot prove discards that were called away earlier or unobserved actions. At this time, the
`ROUND_START` used to establish the baseline is written as `certainty=inferred`, `joined_mid_round=true`; only a hand-start baseline with an empty discard pond, no melds,
and no nuki/North extraction marks the start as observed, avoiding writing "currently seeing that this hand exists" as "saw the opening".

Continuity validation not only checks the tile count of each frame, but also verifies the conservation of the concealed hand's multiset: a decrease in wall tiles can explain at most one
addition, additions to the discard pond must explain the corresponding removal from the concealed hand, nuki/North extraction must consume North, and known concealed melds must consume concealed hand tiles
consistent with the meld. Three new meld tiles must correspond exactly to one discard-pond tile that was removed and can be contained in that meld; four new
meld tiles are only then allowed for a concealed kan without discard-pond removal. Unexplainable same-count "whole-hand tile swaps", mismatched-tile pons, or sourceless melds
will all enter `RESYNC` and will not silently rewrite state.

## Real Video Evidence

Reproducible local sequence:

```powershell
.venv\Scripts\python -m majsoul.state video artifacts/input/QQ20260920-202822.mp4 `
  --start 405 --end 466 --stride 15 `
  --stable-frames 2 `
  --output artifacts/state/phase5/video405-465-auto-layout.json
```

Result:

- At frame 420, submit a mid-hand baseline for East 2: remaining tiles 45, concealed hand 14 tiles with the drawn tile recognized; this baseline's
  `ROUND_START` is inferred and does not pretend to be a real opening frame;
- At frame 465, the concealed hand's discard pond gains `1s`, the hand returns to 13 tiles, and then the remaining tiles change from 45 to 44;
- The event order is `DISCARD(bottom, 1s)`, then `DRAW(top)` is inferred according to seat-wind order;
- `last_discard=(bottom, 1s)`, `current_actor=top`; because of joining mid-hand, history is incomplete.
- The layout is automatically locked to `bottom/top/left` by continuous metadata evidence around 405/420; automatic and manual layout runs
  have consistent sample counts, submission counts, and event counts.

When an opponent's discard and the decrease in remaining tiles appear only in the same pair of sparse snapshots, it cannot be proven whether it is "that player drew first then discarded" or
"after their discard, the next player has already drawn". Such `DRAW` retains `player=null` and writes
`relative_order=ambiguous_with_discard`, no longer forcibly guessing the drawer to be the discarder. When the concealed hand has an
explicit draw marker in the previous frame, it can still be determined, as in the above frame 465, that after the discard the next player draws.

The concealed hand also has an evidence path that does not depend on UI interval markers: when there is no structural change in the concealed hand's melds/nuki/North extraction, use the before-and-after
hand's Counter plus the newly added concealed-hand discard for conservation. If the wall tiles decrease by exactly one and the difference set is unique, it can be proven that the concealed hand
Drawn tile; for example, 13 tiles → after draw-and-discard it is still the same hand, and `4s` is added to the river, so it can be recovered
`DRAW(bottom, 4s) → DISCARD(bottom, 4s)`, and with
`details.evidence=self_hand_conservation` mark it as inferred. If hand conservation cannot uniquely prove it, still keep
`player=null`, and will not use turn order guesses to replace evidence.

Sparse audit command for the entire recording:

```powershell
.venv\Scripts\python -m majsoul.state video artifacts/input/QQ20260920-202822.mp4 `
  --stride 15 --active-slots bottom top left --stable-frames 2 `
  --output artifacts/state/phase5/full-video-stride15-report.json
```

Among 629 sampled frames, 22 frames passed the perception layer's field completeness, and 17 frames simultaneously passed state determinism validation; finally
only 3 stable commits: the 420-frame baseline, the 465-frame above-mentioned discard/draw, and the 1605-frame resync. From 420 to 1605
too many unusable frames are crossed, so the 1605 frame only produces `RESYNC`, without fabricating draws, discards, or melds.

`--stride 15` is a coverage audit, not complete hand-history sampling. The main failures come from the wide river/meld/nuki-North areas
animations or large connected components, as well as unknown digits/seat winds; the recording itself also contains recording distortion already confirmed by the user, and the tile templates also
do not cover all 37 classes. Manually verified templates/normal continuous recordings must be added and complete-observation recall improved before
the "full-match event sequence" acceptance can be completed.

After enabling continuous unknown review, state replay will save a case when similar anomalies reach the configured threshold consecutively,
additionally read one frame as `next_frame.png`, then stop offline processing:

```powershell
.venv\Scripts\python -m majsoul.state video artifacts/input/QQ20260920-202822.mp4 `
  --start 180 --end 286 --stride 15 --active-slots bottom top left `
  --review-unknowns --output artifacts/state/phase5/review-smoke.json
```

The isolated smoke test processes 7 sampled frames, confirms at frame 270 that `river_left[0]` remains unknown, uses frame 285 as
the next-frame evidence, and then returns `safe_pause=true`. All 7 invalid observations are control-layer resync requests, and no table
events are generated. The unknown streak only allows accumulation across consecutive frames; any normal frame in between resets it to zero, avoiding false positives from non-consecutive anomalies.

## Tests

State tests cover: stable baseline, draw, discard, riichi and riichi sticks, pon, nuki-North replacement draw, round change, non-local jump
resync, generic MELD, pon/kan and replacement draw causal order, river call matching, concealed hand multiset conservation, sanma
no chi, out-of-order frame rejection, layout ambiguity interruption confirmation, whole-match locking, and the real Phase 3+4 synthetic
observation at frame 1590. Current full suite:

```powershell
.venv\Scripts\python -m pytest -q
# 172 passed
```

The next step is not to relax validation, but to use the normal continuous recording the user is about to provide to validate the table event sequence, and through
the Continual Perception Learning Loop collect persistent unknowns. After Phase 5 passes table event acceptance,
only then does Phase 6 add buttons, result overlays, and decision-window detection.
