# GOAL_PROMPT: Complete the Steam Mahjong Soul visual Agent and automatic collection workflow

Version: 2026-10-07. Working directory: `D:\pScript\majsoul`.

This file is the task specification for subsequent implementation; planning does not mean the functionality has been implemented or accepted. The old complete version is preserved at `docs/archive/GOAL_PROMPT.pre-workflow-20261007.md`.

## 1. Overall Goal and the User's Latest Decisions

On the user's current Windows computer and Steam Mahjong Soul client, complete a pure visual workflow of continuous observation, legal decision-making, mouse execution, action verification, continuation across hands, and whole-match end handling, while retaining an independent automatic collection loop.

Two paths must be implemented and accepted separately:

1. **Model autonomous play `run`**: recognize lobby → ranked room → Bronze Room → three-player East → match → recognize tiles and track state → generate legal actions → Jev primary decision-making, DeepSeek fallback → click and visually verify → continue across hands → save whole-match result screenshot and send to Feishu → confirm return to lobby. The first real-machine milestone completes one whole match, then exits or continues according to configuration.
2. **Client automatic play collection `collect`**: enter the same hand → check client automatic options for each new hand → save keyframes and provable events in the background → handle any number of settlement confirmations → save whole-match result screenshot and send to Feishu → return to lobby → continue matching according to the match-count limit.

The user's latest agreements override conflicting content in the old document:

- **2026-10-12 anti-burn update**: the user requires independently implementing “no matter which stage it gets stuck at, it can be caught in time,” without being bound to the previous suggestions of a ten-second fallback and repeatedly recovering the Agent within the same match. Steam real-machine run is by default supervised by an independent process; before the deadline, revoke the old input control and wait for exit, then reuse collect to protect hand by hand until the lobby; do not continue burning the timer on grounds of unknown clock, blocked main loop, or next-turn reset. After the first takeover, this match remains under client control, and the Agent is accepted in another match. External obstacles such as lock screen, network disconnection/client inoperability, or supervision being force-killed must be explicitly reported; do not claim to have enabled an invisible switch or guarantee no ban. dry-run does not click, and passing offline fault injection does not equal real whole-match acceptance.

- **2026-10-10 post-hoc update**: the user explicitly requires reasonably relaxing the post-hoc verification of each action. For pon, use the newly added open meld and the corresponding two hand tiles decreasing; for kan, use the newly added/upgraded meld, exact hand consumption, and one replacement draw; for North extraction, use North tile removal and replacement draw, and when the display count is occluded, allow proof by binding the button plus hand tile replacement. The above actions do not depend on whole-table stability, unrelated public areas, or wall readings. Riichi tile selection is complete once the target is removed for input interaction, and declaration attribution is recorded separately. When the PASS window closes or a settlement such as a win appears, the recording window has ended; continue the flow without stopping due to unknown attribution. Protection handling is still retained for unknown hand, wrong target, structural mismatch, missing replacement draw, etc. After pon, the known meld count and complete hand tile count can prove the pending discard, without requiring a marker in the draw gap.

- **2026-10-10 decision-flow update**: continuously recognize; one frame with a complete and legal hand, public information, legal operations, and countdown is enough to freeze a snapshot and asynchronously call Jev/DeepSeek; subsequent opponent animations do not cancel that decision, and consecutive multi-frame whole-table stability is not required. Before execution, only recheck one's own hand, operation opportunity, target, window, and original deadline. Ordinary discard post-hoc verification only checks that the hand has exactly lost the tile specified by the model, preserving red-five identity, and does not depend on a new discard appearing or the discard not being called; on success, return to listening. Event tracking and human-review stability thresholds are retained independently.

- Tile face recognition uses **ConvNeXt-Tiny trained on the 2026-10-02 human gold-label dataset, full augmentation configuration D**, and the September 29 mixed-review model must not continue to be used as the real-time default.
- **Real-time score recognition is removed from this main flow**. Reading each player's score is not required, and unknown scores must not block seat locking, state tracking, or normal decision-making. The whole-match end screenshot is sent via Feishu; automatic parsing of final scores or placements is not required.
- Recalibrate ROI using the current Steam diagnostic screen, and complete the local closed loop first. Generalization to other devices, themes, DPI, and aspect ratios is Extra and must not block the local milestone.
- Retain normalized coordinates, size checks, and the basic scaling mechanism; deferring cross-device generalization does not mean silently applying wrong coordinates is allowed.
- The default wait time is **7+20**, covering the historical 5+20. This must not be understood as every decision having a full 27 seconds.
- Capture in the background; clicking may briefly take the foreground, with a target of about 0.2–0.5 seconds; the computer foreground can still complete other tasks. Record the actual occupied duration; it cannot be guaranteed to always fall exactly within that range.
- Data uses keyframes + events, stored by default in `E:\majsoul_dataset`; stage progress and final results continue to use the already-confirmed Feishu session.
- Reasonable refactoring is allowed to solve structural problems; prioritize reusing existing modules, and do not write special patches for specific screenshots, hand patterns, or tests.

Terminology: a "hand" (small round) is from one deal to a win/draw; a "match" (whole game) is from matchmaking into the table to the whole-match result settlement and return to the lobby. Renchan, draws, and South round entry are based on the actual UI and state, and you cannot exit after a fixed count of three hands.

## 2. Implementation Boundaries and Execution Principles

- First read the project's `AGENTS.md` and `MEMORY.md`, and take the user's latest agreement as authoritative. The old stage arrangement is updated by this task document; do not redevelop existing modules from scratch.
- The round that generated this file on 2026-10-07 delivers only documentation, and does not implement business changes or live-machine operations; when the user later explicitly requests execution of this task document, proceed according to their authorization, and do not ask again whether to start. The following commands are references for the implementation stage, not immediate operations for the documentation-generation round.
- Use fixed ROI, image preprocessing, glyph templates/necessary OCR, a single-tile classifier, and temporal tracking. YOLO or direct screenshot upload to a decision-making large model is not the primary approach.
- Clearly distinguish Observation, State, LegalActions, Decision, Execution, and Verification; the UI lifecycle is independent of mahjong semantic recognition.
- Use visible information; do not read game memory, do not intercept private protocols, and do not assume knowledge of opponents' concealed hands. Four-player mahjong tile assets may be used across modes, but the source and mode must be preserved, and four-player mahjong rules must not be mixed into sanma state.
- This time, mainly support the current local three-player East configuration. Keep the core structure configurable for player count, slots, and rules; it is not required to simultaneously implement autonomous four-player mahjong play.
- Preserve dry-run and an explicit execution switch. dry-run does not send any mouse input, including menus, switches, and settlement confirmations; when encountering states that require these operations, only report the pending steps. Live-machine operations follow the authorization obtained during subsequent implementation; already-authorized routine steps are not repeatedly confirmed.
- First establish a baseline, then do meaningful targeted tests and specified regressions. Do not delete assertions, skip failing tests, falsify logs, or change unknown into a constant to claim completion.
- Necessary interface migrations are allowed; it is forbidden to build another parallel architecture to bypass the original types or verification problems; replacing old paths must explain the migration and compatibility scope.

## 3. Current Facts, Blockers, and Existing Entry Points

Verify the site and files when starting implementation; the following are known starting points, not already-completed goals.

| Item | Current Situation | This Requirement |
|---|---|---|
| Window filtering | Title "雀魂", class name `UnityWndClass`; title-only matching once selected the wrong window | Keep title + class name filtering, and disambiguate when there are multiple candidates |
| Capture and clicking | PrintWindow can capture occluded windows; Unity does not accept the attempted background message clicks | Reuse background capture and brief foreground input, and do not claim to implement pure background clicking |
| Tile model | The real-time configuration still points to `artifacts/model-select-20260929/convnext_tiny_best.pt` | Migrate to gold-label enhanced D, and record weights and version evidence |
| ROI | Old reference 1920×1156, current diagnosis 2582×1550, captured by whole window including the title bar | Recalibrate with the current screen; you cannot simply scale the old boxes and claim the redraw is complete |
| Metadata | The saved frames' bottom/left/right wind positions and dealer are readable, but the left and right scores are incomplete, blocking layout locking | Remove the score dependency, and fix the necessary glyph and validation coupling |
| Wind position templates | The South/West directories contain glyphs belonging to the other side | Review labels and sources tile by tile; do not blindly swap entire directories |
| Concealed kan | bridge treats all melds as `opened=True` | The open/concealed type must run through perception, tracking, bridge, rules, and features |
| Special context | Information such as the supplementary draw is not fully connected; some fields use False/None placeholders | Distinguish known from unknown, and complete the context required for actions |
| Interaction | Closed loops such as multiple kan candidates and riichi tile selection are incomplete; some button templates are missing | Multi-step interaction state machine and a real evidence library |
| Endgame | The model path once ended by silently guessing after 90 seconds | Share the settlement/confirmation/lobby lifecycle; silence is only for watchdog |
| Time | The configuration is still 5+20, and the real countdown is not connected | Synchronize 7+20, share the deadline, and cancel expired results |
| Live-machine collection | Verified one match returning to the lobby, 106 keyframes, with no complete mahjong semantic observation | Preserve the UI results; they cannot be treated as acceptance of a full Agent match or a complete hand record |
| Regression | Last time 405 passed, 8 failed, and collection-related targeted tests passed | Re-measure the baseline, handle failures by category, and report the real results |

Prioritize reuse:

- `majsoul/live/`: window, classifier, buttons, menus, bridge, loop, collector, notifications, CLI.
- `majsoul/calibration/`, `majsoul/perception/`, `majsoul/state/`: calibration, vision, tracking, and verification.
- `majsoul/rules.py`, `majsoul/jev.py`, and the existing unified decision interface.
- `configs/live.steam.yaml`、`collect.steam.yaml`、`roi.yaml`、`vision.yaml`、`sanma.json`、`menu.steam.yaml`。
- `artifacts/live/observe-collect-debug/frame.png`: the starting point for local recalibration; areas that do not appear still require real frames to be added.
- `artifacts/tile-augment-20261002/REPORT.md`、`split_manifest.json`、`runs/D_s*/checkpoint.pt`。
- `docs/auto-collection.md`, `artifacts/live/collect-loop-validation.json`, and related test reports.

These paths are used to locate assets; you must not turn the content of a certain screenshot into a business condition.

## 4. Target Architecture and Two Modes

```text
Steam window → background capture → Frame with timestamp, window geometry, and version
                       ├─ Lightweight UI observation → shared SessionLifecycle
                       │                 ├─ Lobby/Matchmaking/Table
                       │                 ├─ Between-hand settlement/confirmation
                       │                 └─ Whole-match result/return to lobby
                       └─ ROI → segmentation/rotation correction → gold-label CNN + necessary metadata
                                           → Observation → Tracker
                                           → validate State by action dependency
run:     DecisionWindow → LegalActions → Jev／DeepSeek → Executor → Verification
collect: new hand ready → check client auto options → client plays on behalf
Shared:    keyframes/events/diagnostics, endgame screenshots, async Feishu, stop and exception recovery
```

- UI observation must not wait for the complete hand, scores, or remote model responses; settlement and stop signals must always be processable.
- The basic `collect` loop is independent of semantic recognition. On failure it still saves images, semantics are unavailable, and events must not be fabricated.
- `run` discards come from the model or an explicitly recorded legal fallback; it must not rely on the client's automatic draw-and-discard to impersonate model success.
- When switching into `run`, check and turn off auto win, auto draw-and-discard, auto North extraction, and "no chi/pon/kan" that would conflict with the Agent, and confirm the state; auto tile sorting may be kept, but click positions must be re-observed.
- When switching into `collect`, ensure auto win, no chi/pon/kan, auto draw-and-discard, and auto North extraction are ON; auto tile sorting is handled per existing convention, and unknown must not be treated as enabled.
- Mode switching cancels old decisions and clears pending actions, and is executed only in a recognizable, safe state.

## 5. Gold-label model integration and tile-face diagnostics

### 5.1 Weights and inference contract

- By default use `artifacts/tile-augment-20261002/runs/D_s42/checkpoint.pt`, which belongs to the user-specified manually gold-labeled full-augmentation D series, with a fixed seed for reproducibility; do not repeatedly select seeds based on test scores.
- If an official release manifest exists, verify it and use the D weights specified by the manifest and record the basis. If the file is corrupted or incompatible, report a blocker; do not silently fall back to old weights.
- Strictly verify the checkpoint's `model_name`, the order of the 37-class `labels`, `state_dict`, experiment, and seed; mismatched layers must not be ignored.
- Align with the training evaluation transforms: BGR→RGB, 224×224, interpolation, ImageNet normalization, and orientation handling. Verify the actual implementation; do not assume consistency based only on comments.
- Record the weight path, SHA-256, class order, preprocessing version, and device; logs must not merely say "CNN enabled".
- The real-time path explicitly requires the gold-label CNN. Missing torch/torchvision, weight loading failure, or insufficient resources must produce explicit errors; the template tile classifier is for explicit diagnostics only and must not silently substitute.
- Prefer GPU when available; CPU must be budget-tested. Limit batch, cache, and queues to avoid keeping multiple large models resident simultaneously, and do not retry memory errors indefinitely or automatically reconfigure the training environment.

### 5.2 Perception and evidence

- Perform ROI, segmentation, slot rotation correction, and sideways-tile candidate handling first, then call the CNN; the classifier is not responsible for fixing incorrect crops.
- Preserve the original orientation, tile bounding box, owning slot, top-k, and confidence; the semantics of riichi sideways tiles must not be lost after rotation correction.
- Cover the concealed hand, draw-tile gaps, concealed hand shortening after melds, discard row wrapping, sideways tiles, red fives, dora, and North extraction.
- Tiles unavailable in sanma are constrained by the rules; misrecognized illegal tiles remain errors/unknown and are not automatically replaced with the "closest legal tile".
- Probability thresholds and top-1/top-2 margins are configurable. Low quality remains unknown; do not temporarily lower thresholds for a particular frame.
- Diagnostics are divided into ROI geometry, segmentation, orientation, classification rejection, and temporal conflict; do not lump them all together as "the model doesn't recognize it".
- Establish an independent Steam local regression set, distinguishing development samples from frozen acceptance samples; most of the original gold-label data comes from same-source recordings, and the original test accuracy does not substitute for real-time validation.
- For model switching, complete loading, preprocessing, and frozen regression first, then do live testing; a single prediction that looks correct must not be used in place of acceptance.

## 6. Local ROI recalibration and basic scaling

### 6.1 Recalibration deliverables

- Start from the current 2582×1550 diagnostic frame; during implementation, verify the actual size, DPI, borders, and content position.
- Prefer continuing the whole-window-including-title-bar convention to reduce unrelated migration. If switching to the client area/content area, migrate ROI, menus, drawers, buttons, clicks, and export tools uniformly; do not mix them.
- Save normalized coordinates, reference size, capture convention, calibration date, version, and source image; keep the old version.
- Required regions: own concealed hand/draw tile, actual slot discards/melds/North extraction, dora, round wind and round number, honba/riichi sticks/remaining tile count, seat wind/dealer, buttons, countdown, riichi/kan candidates, auto options, between-hand and whole-match results, confirmation, and lobby anchors.
- The score ROI may be kept for diagnostics; recognition is not required, and it is not a mandatory item for the main flow.
- Check with multiple frames covering early/mid/late rounds, concealed hand shortening, melds, animations, and settlement; a single empty discard-pile frame must not be used to prove all scenarios work.
- Top slots, buttons, result pages, etc. not present in the diagnostic frames must be explicitly marked as evidence pending; supplement them from real live scenes or historical frames, and do not fabricate generic boxes or templates.
- Output full images with boxes, named crops, cross-frame comparisons, a problem list, and coordinate descriptions. Submit ROIs to the user for review per existing convention; continue independent offline work while waiting, and do not claim candidate ROIs are accepted.

### 6.2 Geometry gate

Define `Δ = abs((W/H)/(W0/H0) - 1)`; the configuration states whether W/H comes from the whole window or the content area.

- Main acceptance is fixed to the verified local window/DPI configuration. Window changes invalidate old frames, click targets, and pending decisions.
- When the ratio is consistent or content scaling is proven to be uniform, use a single scaling factor and necessary offset.
- When the ratio differs slightly, per existing project convention configurable separate x/y scaling may be used, but it must be made clear that this is not strictly uniform; width-preserving/height-preserving requires handling offsets and verifying anchors.
- Even after Δ passes, still check content anchors and crop completeness; do not treat "below threshold" as proof of successful recognition.
- Do not immediately expand the existing threshold. New thresholds are based on multi-size crop validation; unverified values may only be marked as candidates and must not be claimed to be safe.
- When outside the verified range, anchors drift, or the UI reflows, stop Mahjong clicks, save diagnostics, and require restoring the calibrated size/ratio or recalibrating.
- When crops are complete but glyph classification fails, report a recognition problem; do not repeatedly ask the user to resize the window.
- Multi-device, multi-DPI, automatic content-area localization, and complex projection selection are deferred to Extra.

## 7. Removing real-time score dependency and metadata tiering

It is not enough to just turn off the score reader; it must run through the entire pipeline.

### 7.1 Handling unknown scores

- The current score is `None`/explicitly unavailable, with the reason attached; it must not be filled with 0, 35000, a stale score, or a model guess. The initial 35000 may still be retained as a rule parameter.
- Check and adapt the required score paths in the observation schema, metadata.complete, layout resolver, state validator, tracker diff, bridge, features, logs, and model input.
- Unknown scores do not participate in total score conservation, score-difference arithmetic, or riichi stick deduction event inference; independently valid tile count, tile type, seat wind, dealer, and timing checks are retained.
- If the old offline score recognition is retained, maintain it as an independent compatibility capability; do not use the removal of the main pipeline dependency to delete failing tests or conceal old defects.
- This version does not provide score difference, placement, or target points strategies, and the model prompt clearly states that scores are unavailable.
- For restrictions that depend on scores, such as the riichi points threshold, use the client's explicitly enabled button as evidence of availability, combined with locally implemented validation; do not default to the condition being satisfied just because there is no score.
- Distinguish theoretical hand value under tile-type conditions from each player's current score; follow sanma rules and do not disguise it as complete round payoff or actual final points.

### 7.2 Layout and required fields

- Use the physical slots bottom/top/left/right, with the self seat at bottom; legal occupied combinations come from configuration, and do not fix player numbers to left/right.
- Lock the layout using evidence such as stable occupancy, seat wind/dealer, and UI anchors. When multiple candidates still hold, keep it unknown; do not choose the first one.
- Confirmation counts and stabilization time are configurable; score unavailable does not participate in vetoing.
- The physical combination is locked for the whole match; a new hand only updates seat winds, dealer, round number, and counts. Only a new match, an explicit resync, or a layout change resets the combination.
- Recheck the South/West templates image by image and establish regressions; do not blindly swap directories.
- Determine decision_ready based on field quality and action dependencies: the self concealed hand, target position, phase, etc. must be reliable; unknown scores do not block it, and other fields are handled according to the action's actual dependencies.
- Opponents' concealed kan hidden tile faces are a legal blind spot; do not uniformly freeze all discards; tile-count or risk features that depend on it remain uncertain.
- Takeover mid-match records have incomplete history; do not assert "no furiten/no riichi/no special action" based on missing history.

## 8. Timing state, concealed kan, and special context

### 8.1 State and events

- Use stable frames and diffs; animations/occlusion do not directly overwrite confirmed state; retain evidence, time, and confidence.
- Distinguish GameEvent, AgentAction, and UIEvent. PASS is an action log; RON/TSUMO/specific exhaustive draw types require outcome evidence.
- Collect the table recovery sequence number separately from East round number and honba; consecutive dealer repeats with the same round_id can also identify a new deal.
- Hand tile count depends on phase, melds, and replacement draws; do not always require 13/14.
- Unify the entity/count conventions for called discards, tiles entering melds, and dora display to avoid double counting. Check tile-type counts and red-five counts only against provable facts.
- When skipped frames prevent recovery of intermediate actions, record the incomplete interval and RESYNC; do not fabricate draws or discards.

### 8.2 Concealed kan and meld structure

It is recommended to retain at least the following for each meld group:

```text
kind: pon | chi | daiminkan | ankan | kakan | unknown
tiles: known tile faces and unknown, preserving uncertainty about red-five identity
opened: true | false | unknown
source_slot / called_tile / added_tile: fill in only when there is evidence
box / orientation / evidence_refs / confidence
```

- Run through perception → observation → tracker → bridge → rules/features; do not uniformly reset to opened=True at the bridge.
- Sanma does not generate chi; a general enumeration does not mean four-player mahjong must be implemented this time.
- Concealed kan remains closed; open kan and added kan remain open melds. unknown must not default to open or concealed.
- For the self seat's concealed kan, jointly verify the hand before the operation, legal candidates, the actual choice, the hand/melds after the operation, and the replacement draw; seeing four identical tiles does not mean a concealed kan has been made.
- Treat face-down tiles geometrically/by appearance as backs or occlusion; do not force 37-class tile-face CNN classification.
- For opponents' concealed kan, record only what is visible and historically provable; do not guess hidden tiles or red-five identity.
- Added kan is associated with the original pon, preserving the source; do not treat it as a newly added group or a concealed kan.
- When the self seat's open/concealed status is unknown and affects action legality, suppress the relevant actions and resync; do not approve riichi on the basis of "I did not see an open meld."

### 8.3 Special rule context

- Complete the context for replacement draws after kan/North extraction, rinshan source, riichi lock, legal discard constraints, furiten, and chankan/robbing North responses, or clearly define the capability boundary.
- Do not use a fixed False to disguise "proving there is no temporary furiten/no replacement draw"; insufficient history is unknown, constrained together with the client's explicitly enabled button.
- Count North extraction separately and as its own event; do not treat it as an open meld; closed hand, ippatsu, and other effects follow the confirmed `configs/sanma.json`.
- Restrictions such as kan/North extraction after riichi are combined with the client's available buttons and local validation; rules not independently implemented are clearly recorded, and do not claim the local judge is complete.
- North, red fives, and sanma points follow the project rules; do not apply four-player mahjong outcomes.

## 9. Buttons, Decision Windows, and Multi-Step Interaction

### 9.1 Real Button Evidence

- Complete and verify ron, tsumo, pon, kan, riichi, nuki, pass, abort, and unify the client's Traditional Chinese glyphs with logical labels.
- Cover available/disabled, normal/hover, and button combinations; do not treat decorative text, animations, or disabled buttons as clickable targets.
- Templates include source frame, ROI, mode, and label; for genuinely missing ones, generate a to-be-added case; do not use synthesized text as real-machine evidence.
- Unknown buttons follow existing conventions: during the response phase, when PASS is clearly clickable, the reason may be recorded and then skipped; during the discard phase, only record, do not blindly skip or select an unknown action. When a target is occluded or the phase changes, pause the related action.
- After unknown deduplication, notify; do not flood the screen frame by frame. Always PASS on unknown cannot serve as a substitute for full support.

### 9.2 Decision Windows

- Trigger only at discrete windows such as own discard, response to opponent discard, riichi tile selection, and kan candidates; do not call the model repeatedly frame by frame.
- Identify windows using match/hand epoch, window type, state version, and action set; at most one valid pending decision per window.
- Action parameters must be locatable: multiple copies of the same tile, red five/normal five, and different positions cannot be arbitrarily mapped using only `discard_5p`.
- Cancel old decisions when the window closes, key state changes, settlement occurs, or stopping occurs; late results must not reopen expired windows.

### 9.3 Multi-Step Operations

- Handle explicitly: button submit → wait for candidates → re-read candidates → select legal target → wait for result → verify/resynchronize.
- Riichi: click riichi → recognize selectable discards → generate only allowed candidates → select tile → verify discard and riichi evidence. Clicking the riichi button does not count as completing riichi.
- Kan: distinguish self kan/response kan → when there are multiple candidates, bind to the real tile face → select → verify meld, concealed hand, and replacement draw. Be compatible with the single-step case where no candidates appear.
- Handle the discard after pon according to the client and locally confirmed restrictions; do not treat it as a normal draw phase.
- Nuki/North extraction verifies the concealed hand, the nuki/North extraction display, and the replacement draw; do not rely only on the button disappearing.
- Ron, tsumo, and nine terminals and nine honors require subsequent acceptance/result evidence; the button disappearing is insufficient to deem success.
- Re-observe the target and remaining time at each step; when uncertain whether the previous action was executed, do not resubmit.

## 10. Decision Backend and the 7+20 Deadline

### 10.1 Unified Decision

- Reuse Jev's official Choice `/v1/systemone` adapter and the DeepSeek client; Jev is primary, deepseek-flash is fallback, with a unified interface.
- Send only structured visible state, unknown, and limited legal_actions; do not let the model recognize tiles from screenshots, locate the mouse, or generate legality on its own.
- Validate the returned structure, action_id, probabilities, and their relationship to the action set; do not execute illegal output.
- Client buttons provide timing and availability evidence for some rules; local rules supplement implemented constraints; clarify the boundary between the two.
- Do not provide fabricated score difference/placement features; model confidence is not equal to win rate.
- Record backend errors, refusals, timeouts, and fallback; fallback actions must not be labeled as successful Jev selections.

### 10.2 Time Budget

- During implementation, synchronize rules, configuration, comments, and documentation to a base of 7 seconds and a reserve of 20 seconds, and verify the client's actual mode.
- Use a monotonic clock, starting from the earliest reliable actionable evidence, and deduct time already spent on detection, capture, recognition, model, clicking, and review.
- Prefer reading the real countdown, distinguishing base and reserve consumption; the reserve reset timing is based on client evidence, and it is not restored to 20 seconds every time.
- When the reserve state cannot be read, use a provably conservative base budget, mark it as degraded, and do not claim that the 20 seconds have been utilized.
- Jev, DeepSeek, retries, and multi-step operations share an absolute deadline; fallback does not get a new 7/27 seconds.
- Configure the click and review margins, and calibrate them against local P50/P95/max latency; do not test only the API.
- Separate state freshness from the action deadline: a new screenshot can confirm that the state is still valid, but it does not restore consumed time.
- While waiting for the model, continue lightweight UI observation and stop handling; use bounded concurrency, discard expired results, and do not accumulate requests.
- If local fallback is allowed, select only from the latest legal actions according to general interpretable rules and record it. When the state is unreliable, do not blindly discard; allow the client to time out and report failure; client-timeout actions do not count as Agent success.

## 11. Background Capture, Transient Input, and Visual Verification

### 11.1 Window and Input

- Reuse PrintWindow, 64-bit handles, and GDI release; check for black frames, ghosting, freezing, and abnormal dimensions.
- Handle minimization according to the existing inactive-restore strategy; on failure, pause; do not recognize minimized ghost images.
- Before clicking, verify HWND/process, geometry, ROI version, latest frame freshness, phase, and that the target still exists.
- The game does not need to remain in the foreground for a long time. Observation losing focus does not block background work; before actual input, confirm that the game foreground has been obtained; on failure, do not click.
- After clicking, restore the original foreground and applicable mouse position as soon as possible; if the user actively switches/moves during this time, respect the user's new operation and do not force it back to the old position.
- Record actual focus acquisition and restoration failures; evaluate multi-step actions step by step; do not keep the foreground occupied by pinning on top for long periods.
- Esc, Ctrl+C, and the STOP file are effective; stopping does not wait for remote models or Feishu timeouts.

### 11.2 Action Verification

| Action | Evidence to verify |
|---|---|
| Discard | Target hand removal, river tile or called-tile record corresponds to the selected tile, phase change; handle animation and immediate calls |
| Riichi | Target discard and riichi indicator / sideways tile / client evidence; do not rely on score-deduction OCR |
| Pon / Kan | Changes in concealed hand and melds, source association, subsequent phase; record the replacement draw for kan |
| Nuki/North extraction | Changes in concealed hand and nuki/North display, replacement draw context |
| Ron / Tsumo / Nine Terminals and Nine Honors | Result overlay or explicit acceptance evidence; if the type is unclear, record only the settlement as unknown |
| PASS | The response window closes and there is no call/win by the player; distinguish active skip from timeout; ambiguity is unknown |
| Settlement confirmation | Enter the next settlement, stable table, or lobby |

- Return verified/failed/unknown/cancelled; unknown does not count as success.
- Before retrying, confirm it has not yet been executed, the window is still valid, and it will not cause a toggle flip or duplicate submission; otherwise resynchronize.
- Keep pre- and post-execution images, target, time, and state, so it can answer "what was clicked, what happened on the client."

## 12. Shared Inter-hand / End-of-match Lifecycle and Result Screenshots

Suggested states: UNKNOWN, LOBBY, MATCHING, TABLE_LOADING, TABLE_READY, PLAYING, HAND_RESULT, MATCH_RESULT_CANDIDATE, RETURNING_LOBBY, PAUSED, STOPPED. Existing naming may be adapted, but the semantics must be clear.

- When a marked settlement is confirmed to appear, mark the settlement as pending and pause new mahjong decisions; do not directly increment the hand (small round).
- The in-hand automatic countdown waits according to the client. Only after the confirmation disappears and the table is confirmed to have stably recovered do you enter a new hand (small round); if it is confirmed again midway, continue the settlement.
- Do not hardcode "one win, two exhaustive draws" as ground truth; support multi-stage settlements, dealer repeats, transitions, and re-confirmation.
- Gold-colored or long-persisting confirmation is only used as a candidate/processing basis; do not declare the entire match over based only on color, 8 seconds, or 90 seconds of silence.
- Combine the whole-match results/ranking UI with evidence of returning to the lobby; only after the lobby is recognized does the whole match count as complete and allow matching again.
- Do not rely on real-time scores or a fixed end at East 3; entering South, dealer repeats, and early endings follow the client flow.
- The lifecycle shares confirmation/navigation; mode hooks are responsible for per-hand automatic options or Agent state initialization, avoiding two sets of end-of-match logic.

Result screenshots and Feishu:

1. Before the final results page closes, save a stable, readable screenshot of the complete game; do not screenshot only the button, and do not pass off the lobby as the result.
2. OCR of scores/rankings is not required. When intermediate and final settlements cannot be distinguished, keep the sequence and supplement real templates; do not fabricate successful end-of-match recognition.
3. The image must first be successfully written to disk and queued for sending before confirmation continues; do not block the UI waiting for the network. Send the actual image message/attachment; do not send only local path text.
4. Deduplicate by session_id + match_id + image hash; save time, path, send status, and platform return identifier.
5. Failed sends have bounded retries, and pending-send records are retained; being queued does not equal being delivered. Exit with bounded waiting, and clearly report if not delivered.
6. If the lobby has been returned to but the result image was missed, the lifecycle may record completion, but screenshot acceptance fails and it does not count as fully complete.
7. Each match notification includes the actual result image, mode, match number, elapsed time, lobby-return status, and exception summary; do not guess scores/rankings.

## 13. Collection Loop and Archive Contract

- Keep the verified collect; after each initial hand/new hand (small round) is stably ready, check the four switches.
- Click only when clearly OFF, skip when ON, re-observe when unknown; recheck after clicking. Blind retries or treating repeated clicks as recovery are prohibited.
- If a settlement appears during panel operation, immediately hand it back to the lifecycle and recheck on the next hand (small round).
- After bounded retries fail, archive, notify, and pause/stop; do not pretend collection is normal when automatic options are not enabled.
- Keyframes are triggered by local ROI changes, semantic changes, provable events, before/after operations, UI transitions, and low-frequency heartbeats, avoiding whole-image averaging that misses a single discard.
- Store session metadata, keyframes, observations, events, decisions/execution, diagnostics, and result images; remain compatible with existing formats, and version new fields.
- Each record includes capture time, window/DPI/ROI/model version, mode/player_count, match number, hand (small round) recovery sequence number, image reference, and quality/source.
- Scores are unavailable; unknown tiles/events/history are clearly marked; original images are not gold-label data and are not automatically used for training.
- Sparse keyframes do not guarantee complete draw-and-discard coverage; record missed frames and incomplete intervals, and do not claim to have recovered a complete tile record.
- 2026-10-07 user supplement: archives must make it easy to find reference frames. For historical and subsequent data, provide a thumbnail index filterable by button visibility, provable events, visible melds, batch/match/hand (small round), and recognition quality, supporting viewing the original image and neighboring saved frames; automatic candidates must not masquerade as manual gold-label data. Subsequent event-triggered collection should save before/after context and indicate missing frames, and must not replace searchable event evidence with dense timed screenshots.
- The session directory is unique; overwriting is prohibited. Images must be decodable and references consistent; write failures/insufficient space must notify and stop safely.
- At the end, write session_end and the actual number of matches, release handles, tasks, and workers, and do not leave behind undisclosed resident processes.

## 14. Feishu, unknown Learning Loop, and Exception Recovery

### 14.1 Notification Conventions

- Continue using the session `oc_54dfa592aa8d0d5e5adefc017a969262`, read from configuration, not scattered as hardcoding.
- When implementing notifications, read `C:/Users/胡/.agents/skills/Feishu-Communicate-With-Codex/SKILL.md` and follow it; explain to the user first before first use. Do not rebuild/replace the session.
- Only read credentials from existing configuration/environment, do not output secrets; network access requiring a proxy continues to use the configured local 7890, not hardcoded in business code.
- Stage notifications: baseline completion, model/ROI acceptance, state/interaction fixes, Agent first match results, continuous collection acceptance, key blockers, and final report. Aggregate and deduplicate, do not notify per frame.
- The notification worker has a bounded queue, network/subprocess timeout, and retry limit; it does not block UI/countdown/stop, and results and error evidence take priority.

### 14.2 unknown Review

- For the first unknown, resync first; when 3~5 consecutive usable stable frames are still unknown, generate a deduplicated HUMAN_REVIEW case according to configuration.
- Attach the original image, ROI, single tile/glyph, top-k, context, and version, distinguishing classification, segmentation, and geometric issues.
- Backs, occlusion animations, and frozen afterimages are not used as ordinary recognition error training.
- Manual labels enter the versioned verified dataset and training queue through the existing GUI/review process; Feishu only reminds, and labels are not directly changed based on vague chat answers.
- Do not immediately train on a single sample or automatically overwrite the online model; upgrades must freeze regression and the release checklist.

### 14.3 Failure Strategy

- Distinguish window disappearance/minimization, capture, geometry, semantics, unknown UI, model, execution, disconnection/reconnection, notification, and storage failures.
- Recognized popups/reconnections are recovered via restricted configured paths, then check the current hand and history; do not blindly click unknown popups.
- The watchdog triggers screenshots, limited retries, RESYNC, notification, or pause, and does not treat long silence as a successful final state.
- Recovery cancels old decisions and rebuilds necessary state, without using pre-popup coordinates or old answers.
- Stop after reaching the configured limit, report evidence and specific gaps, and do not idle indefinitely. Reproducible local errors still belong to the main task and cannot be pushed to Extra.

## 15. Phased Implementation and Delivery

| Phase | Work | Deliverables and Exit Conditions |
|---|---|---|
| P0 Baseline | Read docs, verify entry points/weights/live site, test baseline | Reproducible report, attribution of existing failures, file-level plan |
| P1 Local Perception | Gold-label D, ROI recalibration, necessary glyphs, cancel score required | Model manifest, ROI check package, Steam regression; lock the real layout when score is unknown, hand ROI to user for inspection |
| P2 State Rules | Open/concealed melds, draw/unknown, quality gates, cross-hand and RESYNC | Multi-frame and rule tests, concealed kan does not mistakenly open the door, no fabricated scores/events |
| P3 Interaction Execution | Buttons, multi-step tile selection, decision window, 7+20, verification | Offline timing closed loop, timeout/cancel tests, key actions executable |
| P4 Lifecycle | Shared settlement/lobby, result image, asynchronous Feishu, mode isolation | Multiple confirmations/dealer-repeat replay, save image before closing, notification state truthful |
| P5 dry-run | Local background observation, model invocation, target mapping, no mouse input sent | Auditable logs, no score blocking, targets/budget/legality valid |
| P6 Agent Live | Execute a full sanma East match based on implementation authorization | Model truly participates and is verified, cross-hand, result image delivered, return to lobby |
| P7 Continuous Collection | Bounded verification of cross-match loop and per-hand options | Initial target of 3 consecutive matches, configurable; each match has complete image/lobby/save/notification, exit at end |
| P8 Delivery | Regression, instructions, evidence, limitations, and memory | Test and live reports, start/stop/recalibration methods, MEMORY update |

- Independent work can proceed in an interleaved manner, but blind clicking cannot be started by skipping perceptual evidence.
- Rare concealed kan/added kan/nine terminals and nine honors are first covered with real historical frames and meaningful offline tests; report "offline verified, not encountered live", do not wait indefinitely for live encounters, and do not claim all were live-tested.
- Missing necessary real templates and inability to execute safely counts as incomplete and cannot be masked by a blanket PASS.
- After the first `run` match passes, support configurable match count/continue matching, defaulting to one match for easy inspection; infinite running is explicitly selected by a subsequent run instruction, and acceptance leaves no resident process.
- `collect` retains continuous running and a match count limit; the P7 bounded target covers at least two cross-match rematches.
- When user ROI inspection/additional evidence is truly necessary, list all gaps at once; continue work that does not depend on that input.

## 16. Verification Commands and Acceptance Details

```powershell
.venv\Scripts\python -m pytest
.venv\Scripts\python -m majsoul --help
.venv\Scripts\python -m majsoul.live --help
.venv\Scripts\python -m majsoul.live doctor --config configs/live.steam.yaml
.venv\Scripts\python -m majsoul.live observe --config configs/live.steam.yaml --output artifacts/live/observe-workflow
.venv\Scripts\python -m majsoul.state images <ordered screenshot paths> --live-config configs/live.steam.yaml --export-roi --output artifacts/live/replay.json
.venv\Scripts\python -m majsoul.live run --config configs/live.steam.yaml
# Execute only after obtaining implementation/live authorization:
.venv\Scripts\python -m majsoul.live run --config configs/live.steam.yaml --execute
.venv\Scripts\python -m majsoul.live collect --matches 3
```

The above is the existing entry point form; verify with the actual `--help` before execution; update help for new options, and do not write unimplemented parameters as already available. When temporary directory permissions are abnormal, use a separate `--basetemp` within the project, and do not arbitrarily delete occupied directories.

Must cover:

- gold D loading, labels/transform consistency, no silent fallback on failure.
- Local ROI across frames, size changes invalidating old targets, over-ratio gating.
- All score unknown can still lock layout and establish necessary state; real seat wind/tile count conflicts are still intercepted.
- South/West glyph regression, legal slots, and full-match locking.
- Concealed kan closed hand, open kan/added kan opens the door, opponent backs unknown are not forced to classify, and restrictions on related actions when own hand is unknown.
- Riichi, multiple kan candidates, red five/normal five, discard after pon, nuki/North extraction draw replacement.
- Decision deduplication, window-close cancellation, state change discards old answers, fallback shares deadline, notification blocking does not slow operations.
- Button disappearance with no result evidence does not report a win success; PASS timeout ambiguity does not count as active success.
- Multiple confirmations, multi-stage draw, dealer repeat, final match return to lobby, next match matching.
- Mode auto-option isolation, ON does not click repeatedly, panel settles mid-way.
- Score screenshot saved to disk before closing, actual image sent, failure retry/deduplication, non-path text.
- Occlusion, minimize-restore failure, focus failure, user moves mouse, STOP/Esc, disk and network exceptions.
- Image decoding, reference, version/timestamp, no overwrite, unknown not fabricated.

Completion criteria:

1. At least one full match actually participated in decision execution by the Agent, including entering the table, cross-hand, final score and return-to-lobby evidence; full-session client takeover does not count as passing.
2. Continuous collection completes cross-match goals within bounds, and each match's final screenshot and send result are verifiable.
3. Statistics by decision window for model selection, fallback, cancellation, client timeout, verification failure/unknown, not just showing successful clicks; unverified actions do not count as success.
4. Key targeted tests on this machine pass. For historical full-suite failures, explain item by item the baseline, impact and whether they block delivery; if not passed, cannot say full suite passed.
5. Key blockers unresolved cannot be marked as workflow complete; clearly list the offline/real-machine coverage status of rare actions.
6. Original image, state version, action, verification and notification correspond to each other, and main judgments are reproducible.
7. Illegal model output being executed, stale result clicks, unresolved target misclicks or duplicate submissions are all acceptance blockers. Core discard continuously relying on client timeout cannot count as autonomous play complete; occasional timeouts must be quantified and explained, not hidden.
8. This acceptance goal is process correctness and traceability, not winning or ranking as completion conditions, nor inferring strategy strength from one match result.

## 17. Delivery and Final Report

Use the existing directory; do not force creation of a new src tree. At minimum deliver:

- Implementation/configuration changes and necessary migration notes.
- Gold-label weight list, hashes, preprocessing and runtime device.
- Local ROI boxed diagram, crop comparison, user inspection status, recalibration method.
- score optional contract, bright/dark melds, shared lifecycle and multi-step interaction description.
- Targeted/full results, replay and real-machine session index.
- Agent first match and continuous collection score images, Feishu delivered/undelivered records.
- dry-run, execution, continue running, stop, fault recovery description.
- Verified, uncovered, incomplete and Extra lists.
- MEMORY long-term decisions, corrections and measured facts; do not store secrets, do not record plans as completed.

Finally, first state whether the standard is met, then report modification reasons, tests, real-machine evidence and limitations. State whether there are still background processes, data locations, whether notifications were actually delivered, and whether MEMORY was updated.

## 18. Extra: Advance only after local workflow passes

- Verification on other devices, resolutions, DPI, aspect ratios, themes/skins.
- Automatic content-area localization, title bar/whitespace separation, anchor registration, multiple candidate projections.
- Broader Steam independent-source data, confidence calibration, cross-device regression and training.
- Restore real-time score OCR, ranking/score-difference strategy, automatic final-score parsing.
- Four-player Agent, complete referee for different rooms/rules.
- Longer stability, further performance optimization and denser hand-history reconstruction.

Extra must not mask local errors. Necessary tiles, buttons, settlement, timing and click issues in the current window must still be resolved.

## 19. First Instruction to the Implementer

First complete P0, output baseline and file-level plan; then start from P1's "gold-label D integration + cancel score required + local ROI recalibration". Prioritize removing actual blockers; do not rewrite from scratch the already-working capture, menu and collection loop.

Continuously advance authorized work, giving reviewable results and Feishu progress at each stage. When input is needed, state the specific gap and continue independent tasks; hardcoding specific samples, fabricating unknown, skipping action verification or writing unimplemented solutions as complete is prohibited.
