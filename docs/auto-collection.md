# Steam Mahjong Soul automatic collection loop

The collection path uses the client's built-in auto play, independent of Jev/DeepSeek and full tile-face recognition.

```powershell
# Continuous collection; by default saved to E:\majsoul_dataset\collect-date-time-unique-suffix\
.venv\Scripts\python -m majsoul.live collect

# Exit after completing one match and confirming return to the lobby
.venv\Scripts\python -m majsoul.live collect --matches 1
```

The configuration entry is `configs/collect.steam.yaml`, sharing the window, perception and Feishu configuration from `configs/live.steam.yaml`.
No decision API needs to be started.

## Actual loop

1. Filter games by title and configured window class, excluding browser and dialog windows whose titles also contain "雀魂".
2. From the recognized lobby, execute Ranked match → Bronze room → Sanma East. If already at the table or settlement page, take over directly; wait on unrecognizable pages, do not blindly click menus.
3. After continuously seeing the table UI anchors, open the triangle panel and check each item: auto-win, no call pon chi kan, auto draw-discard, auto nuki North. Only click switches that are clearly OFF, re-screenshot to confirm ON, then close the panel. Auto-sort tiles keeps its original state.
4. In-hand blue confirm waits for the client countdown. After confirm appears, mark it as pending check; only after it disappears and the table is stably seen again does it proceed to the next hand's option check. If confirm appears again midway, the wait resets, so two confirms from an exhaustive draw will not be treated as two hands.
5. Gold confirm or a confirm that persists without disappearing is clicked at intervals, re-observing before each click. Only detecting the lobby counts as one completed match; if the table is restored after the confirm click, continue the current match.
   When the final results page first appears, the shared end-of-match component first saves the full results image, binds the hash and persists the notification task, then handles confirm; there is no need to wait for return to the lobby to send the image. After returning to the lobby, a separate completion text is sent. New-version end-of-match notifications that were not successfully delivered can be resent independently; see [live workflow](live-workflow.md) for the entry point.
6. Automatically start the next match when the `--matches` limit is not reached. The default 0 means continuous looping.

## Background and Stopping

`PrintWindow` captures the entire window, so screenshots can still be taken when the game is occluded by other windows. When minimized, first restore the window without activating it, then take the screenshot; this does not mean reading the table from a minimized residual image is supported.

The Unity client still requires real foreground mouse input. Each click briefly takes the foreground and attempts to restore the original window and mouse position; if the user actively switches windows or moves the mouse during this period, the user's actions are respected. Expired screenshots, coordinates after moving/scaling, and focus acquisition failures all result in the click being rejected.

You can stop with Ctrl+C or Esc. You can also create an empty file `STOP` in the current capture directory to stop before the next capture, without needing to switch the foreground or press a global hotkey:

```powershell
New-Item 'E:\majsoul_dataset\<current session>\STOP' -ItemType File
```

The output directory name is printed when the session starts. Reusing a non-empty session directory is prohibited, to avoid overwriting existing images and sequence numbers.

## Data Contract

- `session.json`: session information such as source, sanma mode, automatic options, and calibration dimensions.
- `frames/*.jpg`: keyframes for screen changes, UI transitions, option reviews, semantic changes, and confirmed events; for static pages, heartbeat frames are retained according to configuration.
- `observations.jsonl`: image path, capture time, round/match index, window, mode, and semantic observations and validation issues when available. Images and records correspond one-to-one.
- `events.jsonl`: capture control events and provable tracker `GameEvent`s, with time, round/match index, and the corresponding image path.

`round_index` is the table restoration sequence number observed by the collector; it starts from 1 when taking over mid-match and must not be treated as the East round number or match count in the game. These fields are read from semantic observations only when recognition succeeds.

When tile face or seat glyph recognition fails, the raw keyframe is still saved; `observation=null` and semantics are marked unavailable, without fabricating opponent tiles, seats, or mahjong events. UI control events and mahjong events remain distinguished. Captured images are not manual gold labels and do not automatically enter training.

Change detection performs local grid pooling within the calibrated ROI, to avoid a single new discard being diluted by the whole-image average. `visual_change_grid` controls local sensitivity, `visual_change_threshold` controls the difference threshold, and `keyframe_min_seconds` limits the save frequency.

Templates are taken from the fixed triangular labels in existing real UI screenshots; the template path, color threshold, coordinates, and wait time are all configurable. Window aspect ratio changes still obey the project's ROI projection gate.

Repeated option review failures, unknown pages, or settlement timeouts that persist will be archived and notified via the established Feishu target, then stop; Feishu failures are recorded in the log. Completion of each match is also notified via Feishu.

## Validation Scope

Targeted tests cover initial-round enabling, skipping when already ON, distinguishing OFF/unknown, settlement appearing mid-panel, multiple confirmations for an exhaustive draw, instantaneous table animation, canceling the click before the countdown disappears, multiple confirmations at the end of the match, cross-match looping, the STOP file, raw frame fallback, window class filtering, expired screenshots, and window movement.

Legacy template perception/state replay failures in the full test suite still need to be fixed separately; passing the UI loop must not be treated as passing complete hand reconstruction.
