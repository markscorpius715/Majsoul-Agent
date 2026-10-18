# majsoul: Steam Mahjong Soul visual agent

Part of the client's built-in auto-play collection flow has been implemented and verified live; three consecutive matches of acceptance have not yet passed:

```powershell
.venv\Scripts\python -m majsoul.live collect
```

Background screenshots, briefly stealing focus to check the per-round auto option, continuing to match after settlement returns to the lobby; key frames and events are saved by default to `E:\majsoul_dataset`. For full instructions and how to stop, see [Auto-collection instructions](docs/auto-collection.md). This path is independent of the full tile recognition gate of the model agent described below.

For the current fixed ROI model agent entry point, asynchronous decision cancellation and acceptance boundaries, see [Live workflow instructions](docs/live-workflow.md). For historical screenshot retrieval and offline replay reusing the live configuration, see [Reference gallery instructions](docs/dataset-reference-browser.md).

In both modes the final score image is saved before confirmation and submitted for asynchronous sending; for new failed tasks you can preview with `python -m majsoul.live notify-retry --directory '<session directory>/notifications'`, add `--execute` for bounded resend, without starting the game or collection. Continuous collection is currently disabled per user request.

The current goal is governed by `GOAL_PROMPT.md`: fixed ROI → template/OCR/tile classifier → temporal state → legal actions → decision backend → execution and visual verification. Historical recording ROI was once reviewed by the user; the current Steam ROI still requires multi-frame checking. Single-frame vision, temporal state, buttons and decisions have partial implementation and regression, but the Agent's full whole-match and continuous collection acceptance have not yet passed. Phase 5 only restores events provable by table difference; interaction events and action verification are recorded separately, and detecting a button does not replace the action having been executed.

The existing YOLO, rule, model and click flows are historical implementations and do not mean the new vision scheme has passed acceptance. The model agent's full unattended match is still not accepted; the above collection entry uses the client's built-in auto option to play on its behalf. The old `run` / `replay` commands below still correspond to YOLO/detection JSON, not the full video replay required by the goal.

## Fixed ROI: frame extraction, recalibration and cross-frame verification

Install `.venv\Scripts\python -m pip install -e ".[vision,test]"`. The historical YOLO flow additionally needs `.[yolo]`; public Parquet data auditing needs `.[data]`. The new tools do not call models or click the game.

It is recommended to use the resident annotation window:

```powershell
.venv\Scripts\python roi_labeler.py
```

It opens `artifacts/roi/reference.png` by default, loads and saves `configs/roi.yaml`. Drag boxes on the left, select all required ROIs on the right; green is an existing region, orange is the current region. Supports marking as unknown, undo, horizontal mirroring of left/right regions, switching reference frames of the same layout, save and exit. Switching reference frames does not lose already-drawn regions, so melds, nuki and operation buttons can be labeled separately in different situations. You can also specify an image or edit only some regions:

```powershell
.venv\Scripts\python roi_labeler.py artifacts/roi/action-survey/frame_6750.png --names action_buttons melds_bottom kita_bottom
```

The equivalent module command is `.venv\Scripts\python -m majsoul.calibration gui <image> --config configs/roi.yaml`. The window only edits offline images, it will not capture the game window, call models or perform clicks. You must click save before closing the window; unconfirmed regions remain "unknown".

```powershell
.venv\Scripts\python -m majsoul.calibration extract artifacts/input/QQ20260920-202822.mp4 --frame 300 --output artifacts/roi/reference.png
.venv\Scripts\python -m majsoul.calibration edit artifacts/roi/reference.png --players 3 --kita --config configs/roi.yaml
.venv\Scripts\python -m majsoul.calibration extract artifacts/input/QQ20260920-202822.mp4 --frame 1590 --output artifacts/roi/validation.png
.venv\Scripts\python -m majsoul.calibration debug artifacts/roi/validation.png --config configs/roi.yaml --output artifacts/roi/validation --active-slots bottom top left
```

Frame numbers start from 0; the JSON next to the extracted frames records the original path, frame number, time and resolution. First check the reference frame and its adjacent frames, avoiding distortion and animation occlusion; successful OpenCV decoding does not prove the image is undamaged. The user has confirmed that this recording has some recording distortion.

The editor boxes regions in order by name, Enter/Space accepts, Esc marks that region as unknown (null), Ctrl+C exits without saving. At the end you can enter a custom English ROI name to continue boxing. Existing YAML will reuse screen slots and layout variants; partial modifications can use `--names self_hand melds_bottom`. To change player count or layout, save to a new config path.

The edit preview is limited to within 1280×800 by default, preserving aspect ratio; on small screens you can specify `--preview-size 960 600`. Normalized coordinates are mapped by the actual preview size, and the config still records the original image resolution. Close that window after each boxing to avoid stacking multiple windows.

Required regions include round number, remaining tiles, honba, riichi sticks, dora, own concealed hand, action buttons, turn prompt, and the score, seat wind/dealer indicator, discards and melds for the four screen slots `bottom/top/left/right`; when nuki/North extraction is enabled, add a nuki area for each slot. In sanma, three slot combinations may occur: `bottom+top+left`, `bottom+top+right`, `bottom+left+right`. At the start of the match, lock `seat_layout` once based on repeated consistent seat wind/score anchor observations, and reuse it for the whole match; when the hand changes, only update East/South/West seat winds, dealer and round number, without reassigning screen positions. Even if left/right side melds are placed diagonally, first cover the full range of variation with axis-aligned rectangles, and perform direction and perspective normalization later during recognition.

`self_hand` is the parent range for all visible tiles of the own player, including the concealed hand, drawn tile and melds, so it is allowed to overlap heavily with `melds_bottom`; nuki/North extraction display still uses the separate `kita_bottom`. The drawn tile position changes with the number of melds, so a separate fixed `drawn_tile` ROI is not required; later, within the parent range, identify the drawn tile by tile spacing and consecutive frames, and prioritize parsing `melds_bottom` then excluding melds from the hand candidates. A red seat wind is the dealer state, so `seat_<slot>` carries both seat wind and dealer recognition, and `dealer_<slot>` is no longer separately labeled.

`melds_<slot>` marks the complete spatial envelope that melds may occupy from 0 groups to the maximum number of groups, including the stacked positions of sideways-called tiles, added kan and concealed kan, and does not include discards, tile wall or nuki/North extraction display. The ROI still uses axis-aligned rectangles; side melds, even if rotated or with perspective, are fully framed and a small amount of background at the rectangle corners is allowed. In the recognition stage, use the three slots locked for the whole match, then rotate/perspective-normalize according to the slot direction, and group by centerline, group spacing and temporal order; the same tile in an overlapping region belongs to only one group, and when disambiguation is impossible, keep it unknown and stop clicking.

The current `configs/roi.yaml` uses schema v2, storing four spatial slots and three sanma layouts. The old numeric regions have been losslessly migrated to `0→bottom, 1→top, 2→left, 3→right`; missing right-side regions remain `null` and are not automatically guessed from a single frame. After manually clicking "horizontally mirror to the corresponding left/right region", the coverage range still needs to be checked in the corresponding layout screen.

The configuration stores normalized `x/y` coordinates and the reference resolution. `debug` loads the configuration in another screen and outputs `roi_debug.png`, `crops/<ROI name>.png` and `roi_report.json`; the report lists uncalibrated regions and target size evaluations. Aspect ratio drift smaller than `max_auto_aspect_drift` (default 3%) is scaled separately according to the x/y proportions; exceeding the threshold rejects automatic selection and prompts to use `--projection all` to export candidate crops for stretch, width-preserving three alignments and height-preserving three alignments. After passing in the whole-match locked `--active-slots`, only that layout's slots are exported. Acceptance still requires recognition tests or manual inspection; the same ratio does not guarantee the same UI.

```powershell
.venv\Scripts\python -m majsoul.calibration debug <new-frame.png> --config configs/roi.yaml --output artifacts/roi/projection-review --projection all
```

This recording has a resolution of 1920×1156 and includes the title bar. When the client area screenshot, window ratio or UI layout changes, recalibration is required, and the recording coordinates cannot be reused directly. Empty meld areas, buttons that have not appeared and occluded areas require more frames for confirmation; keeping null is better than guessing.

Jev has been integrated according to TypeSafe's official System One Choice API; the policy is still invoked through the unified `choose(state, actions) -> Action` interface, and vision, rules and control do not depend on a specific provider. For configuration and validation boundaries, see [Jev integration notes](docs/jev-integration.md).

## Fixed ROI tile recognition

`majsoul.perception` first segments within the ROI by white tile face and relative geometric size, then recognizes tile faces using HOG, color descriptors and multi-template cosine distance. The classifier supports the project's 37 tile classes; when templates are insufficient, the distance is too large, or the gap between the top two classes is too small, it returns `unknown`. The current template library is located at `artifacts/templates/tiles/`, containing 26 classes and 64 real crops manually verified from this recording; missing classes are not guessed by code.

Check segmentation only:

```powershell
.venv\Scripts\python -m majsoul.perception segment artifacts/roi/review/frame_1590.png --output artifacts/perception/check-segmentation
```

Segment and recognize:

```powershell
.venv\Scripts\python -m majsoul.perception recognize artifacts/roi/review/frame_1590.png --output artifacts/perception/check-recognition
```

The result includes `recognition_debug.png` with boxes and labels, as well as machine-readable `recognition_report.json`. `complete=false` or `issues` in the report means the frame must not be sent to the subsequent state layer. Animations, occlusions and off-grid UI elements within the discard ROI are reported; the drawn tile does not use a fixed ROI, but is determined by the spacing between the last tile and the rest of the hand and returns `drawn=true`.

For real validation details, see [Phase 2 tile recognition report](docs/phase2-tile-recognition.md).

After the whole-match layout is locked, all visible tiles of that layout can be read:

```powershell
.venv\Scripts\python -m majsoul.perception visible artifacts/roi/review/frame_1590.png --active-slots bottom top left --output artifacts/perception/check-visible
.venv\Scripts\python -m majsoul.perception visible <yonma-frame.png> --players 4 --no-kita --active-slots bottom top left right --output artifacts/perception/yonma-visible
```

`--active-slots` must come from the whole-match-level `seat_layout` and cannot be re-guessed per hand. This command covers the own hand, discards/melds/nuki and dora indicators for all valid slots, preserving sideways tile orientation and meld grouping. For Phase 3 real-machine results and boundaries, see [all visible tiles report](docs/phase3-visible-tiles.md).

Read the round number, remaining tiles, honba, riichi sticks, scores, seat winds and dealer for the same locked layout:

```powershell
.venv\Scripts\python -m majsoul.perception metadata artifacts/roi/review/frame_1590.png --active-slots bottom top left --output artifacts/perception/check-metadata
```

Fixed fonts are segmented by HSV and then recognized using real glyph templates in `artifacts/templates/glyphs/`; side scores and seat winds are first rotation-normalized, and red seat wind tiles also provide the dealer state. Outputs `metadata_debug.png` and `metadata_report.json`; any unknown glyph, animation occlusion or layout contradiction will make `complete=false`. For real-machine results and uncovered classes, see [Phase 4 metadata report](docs/phase4-metadata.md).

## Fixed ROI continuous state tracking

`majsoul.state` combines all visible tiles and metadata into a `FrameObservation` that allows unknown values and field confidence, and only commits `MahjongState` after consecutive consistency. Discards, melds, nuki/North extraction, remaining tiles, scores, riichi sticks, dora and hand changes only allow locally explainable changes; when cross-frame gaps occur, record `RESYNC` and set `history_complete` to false, without guessing missing events.

When the meld group structure is reliable but individual tile faces are unknown, a generic `MELD` may be committed, while retaining in state
`uncertain_fields` and setting `decision_ready=false`; later recognition of the tile face is only perceptual refinement and will not
generate duplicate events. When sparse snapshots cannot distinguish the order of draw and discard, `DRAW.player` remains `null` and the order ambiguity is recorded,
without guessing the drawer; but even if the drawn tile interval marker is missed, as long as "the multiset of the hand before and after + own discards + wall tiles
decrease" uniquely proves one own draw, it is still marked `certainty=inferred` and the conservation evidence is recorded.
A baseline `ROUND_START` produced by first joining midway through a hand is likewise explicitly marked
`certainty=inferred, joined_mid_round=true`, and does not pretend that the real start time was observed.

The true decision gate exists only in a validated `MahjongState`: it must be in `TRACKING`, with continuous history and no
uncertain fields. State transitions also verify the tiles consumed by the hand multiset, wall tile consumption, own discards, nuki/North extraction and melds;
hand replacements with no source or melds that do not match discards/calls only trigger resynchronization.

Perform offline checks on screenshot sequences or recording intervals:

```powershell
.venv\Scripts\python -m majsoul.state images artifacts/roi/review/frame_1590.png artifacts/roi/review/frame_1605.png --output artifacts/state/check-images.json
.venv\Scripts\python -m majsoul.state video artifacts/input/QQ20260920-202822.mp4 --start 405 --end 466 --stride 15 --output artifacts/state/check-video.json
```

When `--active-slots` is omitted, the valid sanma layouts in the configuration are tried separately; only when complete metadata, seat winds/dealer,
round number and score conservation jointly support a unique candidate, and two consecutive frames agree, is it locked as the whole-match layout. Ambiguity or no candidate will
break the confirmation streak; after locking, hand changes do not reassign slots. Cached frames used for confirmation are sent back to the state
tracker in their original temporal order, so events are not lost due to layout prechecking. `--active-slots` can still be used as an explicit audit parameter.

The second command automatically locks `bottom/top/left`, commits a 420-frame midway baseline in the real recording, and at frame 465 restores `DISCARD(bottom, 1s) → DRAW(top)`. Of the 629 samples audited every 15 frames throughout the segment, only 17 frames simultaneously pass completeness and certainty validation, after which it enters resynchronization due to a long gap; it cannot be regarded as a complete game record. For implementation, evidence and unmet acceptance items, see [Phase 5 state tracking report](docs/phase5-state-tracker.md).

## Continuous perception learning and manual review

Unknown will not be forcibly mapped to an existing class. Classification uncertainty, state rule conflicts or visual novelty can all trigger
unknown; the first anomaly first requests re-observation/resynchronization, and only if a similar anomaly persists for 4 consecutive stable frames is one deduplicated review
package generated, entering `SAFE_PAUSE/HUMAN_REVIEW`. Manual labels are only added to the verified dataset and
training queue, and do not immediately train or replace the model.

The main annotation entry point is the local interactive interface: the left side filters pending/verified/rejected cases, and the right side simultaneously displays
crop, full frame, previous and next frames, model candidates and AI reference suggestions. You must click a 37-class tile button or enter a label and
confirm a second time before the sample is stored; you can also fill in a reason to reject an invalid crop. The interface reuses the CLI's source and label validation,
and will not change data due to viewing or skipping.

```powershell
.venv\Scripts\python review_labeler.py --kind tile --reviewer human
# Equivalent entry point:
.venv\Scripts\python -m majsoul.continual --cases artifacts/unknown_cases `
  --datasets datasets gui --kind tile --reviewer human
```

Codex's `reviewer_suggestions` and model candidates are for reference only in the interface and will not be automatically filled in or submitted; only
manual clicking of "Confirm and Store" writes to the ground truth and training queue.

```powershell
.venv\Scripts\python -m majsoul.continual --help
.venv\Scripts\python -m majsoul.continual --cases artifacts/unknown_cases --datasets datasets collect-video artifacts/input/QQ20260920-202822.mp4 --start 180 --end 286 --stride 15 --active-slots bottom top left --report artifacts/unknown_cases/collection.json
.venv\Scripts\python -m majsoul.state video artifacts/input/QQ20260920-202822.mp4 --start 180 --end 286 --stride 15 --active-slots bottom top left --review-unknowns --output artifacts/state/phase5/review-smoke.json
.venv\Scripts\python -m majsoul.continual register-video '屏幕录制 2026-09-23 005221.mp4' --source-id captured_yonma_20260923_005221 --manifest datasets/manifests/sources/captured_yonma_20260923_005221.json --players 4 --mode yonma
.venv\Scripts\python -m majsoul.continual crop-video '屏幕录制 2026-09-23 005221.mp4' --source-manifest datasets/manifests/sources/captured_yonma_20260923_005221.json --output-root datasets/raw/captured/captured_yonma_20260923_005221/tile_crops --stride 15 --active-slots bottom top left right --players 4 --no-kita --resume
.venv\Scripts\python -m majsoul.continual --cases artifacts/unknown_cases collect-video '屏幕录制 2026-09-23 005221.mp4' --players 4 --no-kita --source-id captured_yonma_20260923_005221 --active-slots bottom top left right --include-metadata --report artifacts/datasets/captured_yonma_20260923/review.json
```

`crop-video` covers the specified complete timeline and stabilizes the tile faces with a default sampling rate of 2 Hz; it does not merely capture a short interval,
nor does it repeatedly save identical tiles from every compressed frame. Progress is atomically updated every 100 sampled frames, and `--resume` continues from the last
checkpoint. All output is raw crops with `label=null, training_allowed=false`, deduplicated jointly by region, direction,
visual hash, and normalized pixel difference; model candidates never masquerade as manual labels.

The Feishu skill path and target are configured only through parameters or environment configuration, and are not written into business code. When automatic delivery is enabled, the first
persistent unknown will send a to-do reminder, crop, complete frame, and review-package.zip; after success it writes the delivery audit and
enters `SAFE_PAUSE`; the actual label is completed in the local GUI:

```powershell
$env:MAJSOUL_FEISHU_HELPER='<skill-directory>\scripts\feishu_send.mjs'
$env:MAJSOUL_FEISHU_CHAT_ID='<chat-id>'
.venv\Scripts\python -m majsoul.continual --cases artifacts/unknown_cases collect-video <video> --active-slots bottom top left --notify-feishu --report artifacts/unknown_cases/review.json
.venv\Scripts\python -m majsoul.continual --cases artifacts/unknown_cases notify-case case_000001
```

A real smoke test has converged the persistently unknown `river_left[0]` into one case; the user confirmed it is South Wind and it has entered
the training queue; rescanning the same interval added 0 new cases. State replay will also return
`safe_pause=true` after generating post-frame evidence, and will not continue processing the anomalous sequence. For the data directory, manual labeling commands, Feishu message format, and model upgrade regression gate, see
[Continual Perception Learning Guide](docs/continual-perception.md). Public data will not be automatically downloaded or mixed into the training set before license, category mapping, and data leakage
review.

When a case that has been verified but has not yet entered a new model is hit again, it still remains safely paused, but `/label` will not be requested again;
the message will display the existing label and wait for batch training and the regression gate.
For invalid crops use `reject <case_id> <reason> --reviewer <name>`; the rejected state participates in deduplication but will not be written into
the verified dataset or training queue. Four-player recordings are used only for cross-mode shared visual content; mode-specific rules and
state labels must not enter sanma data. Every case, verified sample, and training queue entry must carry
`player_count/mode`; historical records can only be migrated with an explicit `backfill-context`, and the game mode must not be guessed from the tile faces.

## Environment and Local Verification

A `.venv` has currently been created, and the rules library, YOLO, and test dependencies have been installed. Offline tests cover ROI layout migration, full-match layout locking and reset, all visible tiles, metadata awareness, continuous state events and resynchronization, persistent unknown debouncing/deduplication, manual label queue, scaling mapping, the Jev interface, the simulated click closed loop, and SSE streaming; real game clicking has not yet been performed. For the latest passing count, see the Phase report. This environment inherits the base packages already present on the machine; on another machine, an independent environment can be created:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[vision,test]"
.venv\Scripts\python -m pytest -q
.venv\Scripts\python -m majsoul --help
.venv\Scripts\python -m majsoul doctor
.venv\Scripts\python -m majsoul doctor --check-api
```

`doctor` does not output secrets; only `--check-api` sends one minimal model request. In the actual test on 2026-09-20, `deepseek-flash` successfully returned the agreed action ID, with one request taking about 1.15 seconds; this does not mean that all match decisions can be completed within that time. The installed Torch is currently a CPU build, and GPU inference has not been verified.

Configuration priority: process environment variables > project `.env` > `../PuwuQuiz/.env` > default values. Supports `KEY=VALUE`, the reference file's `KEY:VALUE`, and `#model-name:`. Only model-related configuration is extracted; credentials from the reference project are not copied.

| Variable | Default Value |
| --- | --- |
| `DEEPSEEK_API_KEY` / `API_KEY` | Read from the above configuration sources |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com` |
| `DEEPSEEK_MODEL` | `deepseek-flash` |
| `DEEPSEEK_TIMEOUT_SECONDS` | `8` |
| `DEEPSEEK_THINKING` | `disabled`, thinking mode is turned off for real-time decisions |

Model requests default to `stream=true`, receiving SSE incrementally; a valid action is returned only after the complete JSON, normal termination, and the stream termination marker are received. The total deadline covers the entire request, including connection, reading, and continuous heartbeats; the log distinguishes the first valid content segment, the complete decision, and the total elapsed time.

The user's latest agreement is **7+20**. The real-time path reads the actual countdown (white base + gold reserve, or a single reserve field), and uses the displayed remaining time only after continuous confirmation, deducting display quantization and action margin; when it cannot be read clearly, it falls back to the base budget already elapsed. The absolute deadline for each window only shortens; fallback and multi-step operations share the deadline, and if the re-read before clicking is zero, input is rejected. The 8 seconds in the environment is an independent call/timing upper limit; on a real machine it is additionally limited by the current remaining budget and observation age. Real materials and offline tests at two sizes have passed; continuous real-machine timing still awaits acceptance, see [Real-time workflow](docs/live-workflow.md).

Two batches of real API requests, 96 in total, have been made for 8 types of simulated situations. The final version succeeded in streaming 22/24 times, with a median of 0.72 seconds and P95 of 0.98 seconds for successful requests; the two batches combined had 14 connection errors, which cannot be regarded as a stability guarantee. See [Complete Timing Report](docs/latency-report-2026-09-20.md). You can retest yourself:

```powershell
.venv\Scripts\python -m majsoul benchmark --rounds 3 --modes stream buffered
```

When the model fails, returns an ID outside the candidates, or the response format is incorrect, this execution stops; it will not generate click coordinates on its own based on model output, nor will it silently switch models. `efficiency` is a shanten/ukeire baseline for offline debugging, not a calibrated offense/defense strategy.

## Steam Client Integration

1. Open Steam Mahjong Soul and enter the friend room to be tested.
2. List window handles:

```powershell
.venv\Scripts\python -m majsoul windows
```

In the command below, replace `<HWND>` with the integer handle of the Mahjong Soul window in the output. The handle may change after the window is reopened.

```powershell
.venv\Scripts\python -m majsoul capture --hwnd <HWND> --count 30 --interval 1 --start-delay 5
```

Switch back to Mahjong Soul within 5 seconds after startup. The screenshot captures only the window's **client area**, excluding the title bar; mouse coordinates and the screenshot use the same DPI coordinate system. Window loss of focus, minimization, movement, or scaling will block the current click. Press Esc or Ctrl+C in the terminal to stop.

3. Use the captured client-area screenshot for layout calibration:

```powershell
.venv\Scripts\python -m majsoul calibrate --image artifacts/captures/<screenshot name>.png --output configs/layout.local.json
```

In order, box the concealed hand, drawn tile, dora indicators, the three players' discards, melds, and the nuki/North extraction area. Regions should not overlap. Numbering is uniformly **0=below (self), 1=above, 2=left**. The concealed hand area does not include the separate drawn tile. `row_axis`, `row_direction`, and `tile_direction` control the discard order for different orientations, and need to be modified according to the actual screenshot; they cannot all default to left-to-right on screen.

`configs/layout.steam.draft.json` is the initial reference for the user-provided screenshot: it records only the concealed hand and dora regions that can be confirmed, and deducts the title bar; the remaining regions are `null`, and loading will clearly indicate that they have not yet been calibrated. It is not an executable layout. After changing the window aspect ratio, UI scaling, or tile back/interface theme, the regions should be rechecked.

4. Configure the dedicated YOLO weights and category mapping, then first observe one decision:

```powershell
.venv\Scripts\python -m majsoul run --hwnd <HWND> --weights models/mahjong.pt --layout configs/layout.local.json --once --start-delay 5
```

By default, only a decision is generated. To execute and verify clicks, use the calibrated configuration:

```powershell
.venv\Scripts\python -m majsoul run --hwnd <HWND> --weights models/mahjong.pt --layout configs/layout.local.json --execute --start-delay 5
```

Before each click, retake a frame and check the situation, window position, target, and action legality; expired decisions are rejected. After clicking, the corresponding change must be observed (for example, the discarded tile appears in the self player's discards); otherwise, save a diagnostic screenshot and stop, without repeating the click.

## Recognizer Output Convention

Detection boxes use client-area pixel `xyxy`. The weight's category names can directly use the format in the table below, or be mapped through the `labels` dictionary in the layout. This is not a requirement to hard-code the coordinates of these UI elements in the match code.

| Semantic category | Meaning |
| --- | --- |
| `tile:1m`, `tile:0p`, `tile:4z`, etc. | Tile face; `0p/0s` red five, `1z..7z` East/South/West/North/White/Green/Red |
| `button:ron/tsumo/nuki/pon/kan/riichi/pass/abort` | **Enabled and clickable** action buttons, each as a complete category name |
| `phase:discard/response/riichi_select/waiting/result` | Current phase, each as a complete category name |
| `meta:seat:3z` | Own seat wind; East/South/West are `1z/2z/3z` respectively |
| `meta:round:1z`, `meta:number:1`, `meta:honba:0` | Round wind, hand number, honba count, combined as the match identifier |
| `riichi:0`, `riichi:1`, `riichi:2` | Corresponding player has declared riichi |
| `closed:0:5p` | Player 0's five-pin kan is a concealed kan; complete meld identification info is still required |
| `ignore` | Explicitly ignore, e.g. other players' face-down tile backs |

The above labels can also be produced by subsequent CV/OCR adapters, then merged with YOLO boxes. The fixed-ROI main workflow has implemented reading the numbers, seat winds and dealer in this recording; **the phase/button recognition of the new main workflow is not yet implemented, and there are no corresponding Steam UI classification weights**. Tile-face detection weights alone are insufficient to complete automatic play. Subsequent visual adaptation still needs to cover button enabled/disabled, riichi markers and key phases; the screenshot capture entry point is already in place.

This project does not treat "not detected" in YOLO as provably nonexistent. It can already check low confidence, duplicate boxes, tile count, tile type, number of tiles and missing metadata, but these checks cannot detect all missed detections (e.g. missing one opponent discard or riichi marker). Recall rate needs to be evaluated on independent complete match data before considering long-duration operation.

## Rules and current boundaries

`configs/sanma.json` records the user's rule screenshots. Implemented:

- Filtering 2-8 man, red-five count constraints, sanma man dora `1m↔9m`.
- Shanten for standard form, seven pairs, kokushi; effective tile count after subtracting visible tiles.
- Furiten from recorded discards, temporary/riichi furiten from passed wins; preserving the history of called-away discards across frames.
- Tsumogiri after riichi, tenpai restrictions on riichi tile selection, same-tile kuikae restrictions after observing a pon.
- Win points with sanma tsumo loss, nuki North added dora (not as a yaku), ron on North does not automatically add the chankan yaku.
- Genbutsu/suji information for candidate discards, number of kept dora, and conditional points for winning on the next draw after reaching tenpai.

The following still have clear boundaries:

- **Legal actions are the intersection of local constraints and client-enabled buttons, not a fully independent referee.** Winning yaku, special kan restrictions, nine types nine tiles and other timing eligibility are determined by client buttons; the local scoring tool is not the sole basis for automatic clicking of a win.
- Concealed kan backs visually and identity recovery from observed own-hand diffs already have offline coverage; opponent hidden identities remain unknown. Multi-candidate kan tile selection adaptation is not yet complete; the riichi tile selection phase is connected but the full live closed loop has not yet been accepted.
- When joining mid-game or crossing unobserved events, no claim is made to have complete history; a fully observed new hand can re-establish that hand's history. The current fixed-ROI entry point is connected to the menu and the shared settlement/lobby lifecycle; disconnect recovery and a complete Agent full match have not yet been accepted.
- Remaining tiles, scores, riichi sticks and seat winds can already be read from a normal single frame; the new state layer has recovered discards and the next player's draw in one segment of real adjacent frames, but the complete-observation recall rate for an entire recording is insufficient, and a complete event sequence cannot yet be reconstructed. Ranking, exact turn count, complete ippatsu/dual-riichi timing and opponent hidden hands remain incomplete. `turn_lower_bound` is only the number of own discards observed by the historical implementation.
- `expected_value` and deal-in probability are kept as `null`; suji information is not equal to safety probability. Tenpai points are conditional values under a limited context, not the expected value of the hand's payoff.
- Honba payments, double yakuman, kiriage mangan, cumulative yakuman, etc. not shown in the screenshots use config/underlying-library default assumptions, and the point output lists these assumptions. Game rules such as starting point/South entry are recorded, but a full-match ranking strategy or independent hand simulator has not yet been implemented.

## Offline use and recording

```powershell
.venv\Scripts\python -m majsoul analyze --hand 19m123456p123s111z
.venv\Scripts\python -m majsoul score --hand 123456789p123s11z --win-index 13 --seat 2z --tsumo --riichi
.venv\Scripts\python -m majsoul replay --layout configs/layout.local.json --input artifacts/run/observations.jsonl --policy efficiency
```

`analyze` accepts a 14-tile concealed hand; states with melds are input via the Python API or observation replay. `score`'s `win-index` starts from 0. Replay does not operate the window; only specifying `--policy deepseek` sends model requests.

Run logs are in `artifacts/run/`: `observations.jsonl` saves raw detections and time/window ranges, `events.jsonl` saves states, candidate decisions and verification results, and PNGs save before/after-action or verification-failure frames. Screenshots and logs are not version-controlled. YOLO settings are also written in the project's `artifacts/`.

Dependency references: [Ultralytics detection output](https://docs.ultralytics.com/modes/predict/)、[MahjongRepository rule library](https://github.com/MahjongRepository/mahjong)。the project adapts to the installed `mahjong 2.x` interface adaptation, with sanma-related differences handled separately in `majsoul/rules.py`.
# Jev interface update (2026-09-21)

Added the Jev Choice interface, DeepSeek automatic fallback and shared deadline. For configuration and diagnostics see [Jev integration notes](docs/jev-integration.md). Real Jev calls have succeeded before; the current Steam ROI multi-frame check, fixed-ROI state layer and full Agent match have not all been accepted, and interface success cannot substitute for workflow completion. Real-time decision-making is connected to single-request asynchronous observation and cancellation; see [real-time workflow notes](docs/live-workflow.md) for details.
