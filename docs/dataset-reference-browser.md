# Reference Image Library for Collected Data

Run from the project root directory:

```powershell
.venv\Scripts\python -m majsoul.dataset_browser --source E:\majsoul_dataset --output artifacts\dataset-browser --scan-ui
```

Open `artifacts/dataset-browser/index.html`. The page opens directly on the local machine, requires no server to start, and does not modify the original data. It supports Chinese keywords, button/UI/event/meld/quality tags, collection batches, and match and hand filtering; click a thumbnail to view the original image and nearby saved frames, and it supports exporting the current filter as JSON.

- `button:ron` means a ron button was detected, not that a ron has occurred. `event:RON` comes only from provable event records.
- `meld:ankan` comes from the meld type in saved observations; melds not recognized by older versions will not automatically become ankan without re-recognition.
- `context:button:ron` means it belongs to the segment before or after the ron button trigger, not that the frame itself has a button.
- All automatic filtering results are reference candidates and cannot directly enter the manually gold-labeled dataset.
- Warnings are retained for issues such as missing images, corrupted records and geometric mismatches. Original images, records and old directories are not deleted.
- The generated page is a snapshot; after adding data or modifying templates, rerun the command to update it. The initial full UI scan takes several minutes.

The new collector keeps a capacity-limited JPEG ring buffer in memory. When an action button first appears, a provable event, a confirmed transition, a final result, or a key failure is encountered, it writes the before/after frames and `clip.json` into `references/<number>/`. By default, it requests 3 seconds before and after, at 0.5-second intervals, with at most 32 frames per segment, all configured via `configs/collect.steam.yaml`.

The clip records the actual capture time, trigger tags, maximum frame interval, actual before/after coverage and early-stop status. Before/after screenshots are not associated across matches or hands. `complete` means the end of the collection window has been observed, and does not guarantee that every draw and discard is complete; the actual coverage and gaps are based on the clip records. Low-frequency heartbeats and local-change keyframes are still used as a retrieval fallback.

Frames missing from historical data cannot be recovered through the index. Follow-up work to improve availability includes continuing to improve semantic observation and manually reviewing stable unknown frames, rather than adding unlimited timed screenshots.

After finding reference images, you can directly reuse the live configuration for offline diagnosis, without restarting a match or starting collection:

```powershell
.venv\Scripts\python -m majsoul.state images <screenshot paths in order> --live-config configs/live.steam.yaml --export-roi --output artifacts/live/replay.json
```

This mode reuses the live ROI, gold-label model and its identity verification, button templates, rules, score switches and stable-frame settings, overriding the corresponding standalone offline options; it does not start the menu, mouse, model decision API or Feishu notifications. The report includes the classifier list, complete observations, verification issues and status updates, and the crop inspection package is located in the `roi/` directory next to the output file with its extension removed. When `--live-config` is omitted, the original offline template mode is still used.

Seats are by default stably recognized from metadata. When a single image has seat occlusion, first look at the original image; only after manually confirming the actual occupancy should you use a legal combination such as `--active-slots bottom left right` to diagnose the hand. Manual occupancy will not fabricate missing seat winds, nor will it bypass status verification. When the ratio exceeds the live gate, stop the replay and first check the projection or recalibrate.

The timing of `images` is inferred from the file sequence number and `--fps`, and is only suitable for checking frames with a clear order and sampling interval; you cannot use a few arbitrary filtered screenshots to prove continuous events or real operation latency. The original collection time is still based on `observations.jsonl` and the segment list.
