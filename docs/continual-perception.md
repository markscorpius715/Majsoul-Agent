# Continual Perception Learning Loop

## Goals and boundaries

The system no longer requires collecting a "complete dataset" covering all rare tile faces and buttons before starting work. The initial version is bootstrapped by verified
Mahjong Soul data; during operation, continuously, stably, and deduplicated unknowns are saved as manual review cases.

This is not single-sample online training:

1. On first anomaly, first request re-observation/resynchronization, but do not fabricate `GameEvent.RESYNC` when there is no valid new snapshot;
2. Only when a similar anomaly persists across 3–5 consecutive stable frames does it enter `SAFE_PAUSE/HUMAN_REVIEW`;
3. Manual labels only enter `datasets/verified/` and the training queue;
4. Only after accumulating to a configurable batch can a candidate be trained;
5. A candidate is only allowed to be promoted after passing golden/historical regression.

Offline replay pauses processing; future real-time operation only prohibits automatic Agent actions, and does not control or pause the Mahjong Soul client.
Offline reports use `resync_requested` to represent the control signal, and only a complete, stable new snapshot that is non-locally conflicting with the previous state
will submit a real `GameEvent.RESYNC`.

## Unknown determination

`majsoul.continual` simultaneously supports three types of signals:

- model uncertainty: insufficient confidence or candidate margin;
- state inconsistency: for example, a fifth copy of the same tile, or losing multiple concealed hand tiles without any meld;
- novel visual pattern: the nearest distance between the crop and the verified visual index is too large.

Therefore high-confidence errors can still be intercepted by rule conflicts or visual novelty. The default thresholds are in
`configs/continual.json`, the current consecutive-frame threshold is 4, and the manual-label training batch threshold is 50.
These are bootstrap configurations and must be calibrated with a subsequent golden set; the current values must not be described as final thresholds.
Persistent counting closes by consecutive frames: once a region recovers to known in the current frame, its streak is immediately reset to zero.

## Review package and commands

The main manual review entry point is the offline desktop annotator:

```powershell
.venv\Scripts\python review_labeler.py --kind tile --reviewer human
# or
.venv\Scripts\python -m majsoul.continual --cases artifacts/unknown_cases `
  --datasets datasets gui --kind tile --reviewer human
```

The left queue can be filtered by pending/verified/rejected and tile/action/meld/ui_state/metadata; the right side
displays the crop, full frame, previous and next frames, source, rule reason, model candidates and AI suggestions. Tiles provide the full 37-class buttons,
other types can use presets or manual input. Both confirm and reject have a second confirmation; skipping or viewing does not write data. The GUI
calls `ReviewCaseStore`, sharing the same labels, gameplay source, verified manifest and training
queue validation with the CLI, and does not maintain a second set of ingestion logic.

For a full recording, first build an independent raw crop corpus:

```powershell
.venv\Scripts\python -m majsoul.continual crop-video `
  '屏幕录制 2026-09-23 005221.mp4' `
  --source-manifest datasets/manifests/sources/captured_yonma_20260923_005221.json `
  --output-root datasets/raw/captured/captured_yonma_20260923_005221/tile_crops `
  --stride 15 --active-slots bottom top left right --players 4 --no-kita --resume
```

The scan covers the complete timeline, by default sampling one frame every 15 frames to capture persistently visible tile faces, and merges duplicate crops
produced by identical compressed frames into an occurrence count. Progress is atomically saved every 100 sampled frames; after an interruption you must use the same scan
signature and `--resume` to continue. The raw manifest's labels are always null and `training_allowed=false`;
model output can only become review candidates. Pseudo-candidates produced by geometric segmentation on animations, character occlusion or blank UI are also kept
in the raw audit layer with a segmentation issue recorded; they must subsequently be filtered by persistent multi-frame presence, visual novelty and manual
`/reject`, and difficult examples must not be silently deleted just to make the sample count "look correct".

Offline scan example:

```powershell
.venv\Scripts\python -m majsoul.continual `
  --cases artifacts/unknown_cases --datasets datasets `
  collect-video artifacts/input/QQ20260920-202822.mp4 `
  --start 180 --end 286 --stride 15 --active-slots bottom top left `
  --report artifacts/unknown_cases/collection-180-285.json
```

A case contains `full_frame.png`, `crop.png`, the previous and next frames, `state.json` and
`metadata.json`. The metadata stores the reason, model version, candidates, source, frame number, content hash and visual hash.
When model candidates are clearly unreliable, Codex can first inspect the crop and context and write ordered
`reviewer_suggestions`. This field only represents AI visual review suggestions; the case remains pending and will not enter
the verified dataset or training queue; only the `label` confirmed by the user is ground truth. The reproducible command is:

```powershell
.venv\Scripts\python -m majsoul.continual --cases artifacts/unknown_cases `
  suggest case_000004 4z --reviewer codex_visual_review `
  --note "the tile face glyph is North; preferred 4z"
```

If the manual label matches the suggestion, the audit status is recorded as `confirmed_by_human`; otherwise it is recorded as
`corrected_by_human` and the correct label is retained. This allows measuring the hit rate of AI-assisted review, but AI
suggestions will never replace manual confirmation.
The notification text uniformly gives:

```text
/label <case_id> <label>
/reject <case_id> <reason>
```

The current real smoke test generates `case_000001`: the South Wind tile at `river_left[0]`. The template classifier's first candidate is
`2z` (0.730), but the second candidate is too close, so a strong guess is rejected per the rules. A second scan of the same interval adds 0 new cases,
proving that deduplication works across processes. The user then confirms it is the South Wind in the discard river, the case is labeled `2z` and
enters the verified dataset and training queue; no model is trained or published immediately.

When the same visual state appears again before a model update, it will still `SAFE_PAUSE`, but persistent deduplication will return the existing case.
If the case is already verified, the message only shows the verified label and "waiting for batch training/regression gate", and will not again
give the `/label` instruction; only newly created pending cases request a manual label.

After confirming the label, execute:

```powershell
.venv\Scripts\python -m majsoul.continual `
  --cases artifacts/unknown_cases --datasets datasets `
  label case_000001 2z --reviewer human
```

This command only writes the verified sample, manifest and training queue; it does not train a model.

Feishu is only used for to-do reminders; the chat bridge's strict single-line commands are retained as a fallback path when the GUI is unavailable:

```powershell
.venv\Scripts\python -m majsoul.continual `
  --cases artifacts/unknown_cases --datasets datasets `
  ingest-command "/label 1 2z" --reviewer human
.venv\Scripts\python -m majsoul.continual `
  --cases artifacts/unknown_cases --datasets datasets `
  ingest-command "/reject 5 unusable crop" --reviewer human
```

`/reject` only changes the case audit status and continues to participate in deduplication; it does not write to the verified dataset or training queue.
Text with extra natural language, path characters, case zero or extra actions will be rejected, to avoid ordinary chat being mistaken
for data label/reject commands. A blank tile face may be the white board `5z`, and must be confirmed together with the full frame; it cannot be rejected based on the crop alone.

Automatic Feishu delivery reuses the send helper of the `Feishu-Communicate-With-Codex` skill, and does not rewrite
the Feishu API in the project. `notify-case` first performs a configuration precheck with the same target, then sends the to-do text, crop, full frame and a ZIP containing the previous and next frames/
state/metadata; only if all sends succeed is the de-identified delivery audit written into the case metadata.
The same message is idempotent for the same target. `collect-video --notify-feishu` performs the
same delivery after the first persistent unknown completes and returns `safe_pause=true`, and will not continue batching past the anomaly. The first delivery includes all images and the ZIP;
when text updates such as AI suggestions are added for the same case, only the new text is sent, avoiding repeated sending of large attachments.

```powershell
$env:MAJSOUL_FEISHU_HELPER='<skill-directory>\scripts\feishu_send.mjs'
$env:MAJSOUL_FEISHU_CHAT_ID='<chat-id>'
$env:HTTP_PROXY='http://127.0.0.1:7890'
$env:HTTPS_PROXY='http://127.0.0.1:7890'
.venv\Scripts\python -m majsoul.continual --cases artifacts/unknown_cases `
  notify-case case_000127
```

For four-player recordings, `--players 4 --no-kita --active-slots bottom top left right` can be used to explicitly override the game mode;
`--include-metadata` also collects score, seat wind and center count unknown. The case records the source id and
gameplay source; only cross-mode shared visual samples that have been manually confirmed can enter training.

Phase 5 replay can directly enable the same review gate:

```powershell
.venv\Scripts\python -m majsoul.state video <video> `
  --active-slots bottom top left --review-unknowns --output <report.json>
```

Once sustained unknown reaches the threshold, the command first saves the next frame context, then returns `safe_pause=true` and stops;
It will not automatically train, send clicks, or continue skipping anomalies.

## Data Versions and Public Data

The directory contract is in `datasets/README.md`, and the initial manifest is
`datasets/manifests/dataset_v0.json`. Any public data must be individually verified before import:

- Whether the license permits this project's download, training and redistribution methods;
- Whether the class mapping is consistent with the internal 37 classes and red-five convention;
- Whether the images are truly the target Mahjong Soul UI domain;
- Whether they duplicate golden/test frames and cause data leakage;
- Whether the source, hash and usage weight are written into the manifest.

Therefore, there is currently no automatic downloading or mixing in of datasets mentioned by users based solely on the number of web images. Mahjong Soul real-machine and manual verification
Samples are high-weight target-domain data; generic mahjong tiles can at most serve as low-weight warm start/pretrain.

The preliminary source audit on 2026-09-22 has been written into `dataset_v0.json`:

- Roboflow `core/mahjong-soul-8bdp2`: CC BY 4.0, version report 178 images, 38 classes, including red fives;
  but the overview only shows 70 images and there is an extra `test` class, so it can only be included after reviewing the actual export package.
- Hugging Face `pjura/mahjong_souls_tiles`: Apache-2.0, 1,977 Mahjong Soul single-tile samples,
  34 classes; suitable for target-domain warm start, but red fives must be supplemented from other verified sources.
- GitHub `Camerash/mahjong-dataset`: the repository is MIT, but the README states that the original images were mostly scraped from
  Google/eBay/Alibaba, and the rights chain for individual images is unclear, so it is currently only for reference, not training.

The old port 7899 was unreachable at the time. After the user uniformly changed the proxy to 7890, the Dataset Viewer API returned
train=1,080, test=897, totaling 1,977 rows; both parquet files have been downloaded to `datasets/raw/public/` and
pinned with revision and SHA-256. The reproducible audit command is:

```powershell
.venv\Scripts\python -m majsoul.continual audit-parquet `
  --split train=datasets/raw/public/huggingface_pjura_mahjong_souls_tiles/train-0000.parquet `
  --split test=datasets/raw/public/huggingface_pjura_mahjong_souls_tiles/test-0000.parquet `
  --report artifacts/datasets/huggingface_pjura_mahjong_souls_tiles/audit.json `
  --contact-sheet artifacts/datasets/huggingface_pjura_mahjong_souls_tiles/contact-sheet.png
```

All images can be decoded, and the contact sheet confirms the `b→s`, `n→m`, `p→p` mappings as well as the wind/triple tile mappings;
This set has no red fives. The audit also found 39 groups of completely duplicate original bytes/decoded pixels, of which 34 groups cross
train/test, covering all 34 classes, so the original test split is prohibited from being used as golden or a trusted validation set.
1-bit dHash gives 62 cross-split near-neighbor pairs, of which 47 are cross-label; this shows that low-resolution tile faces will collide
based on perceptual hashing alone, and near neighbors can only enter manual review, not automatic deletion. For detailed sources, mappings and admission thresholds, see
`datasets/manifests/sources/huggingface_pjura_mahjong_souls_tiles.json`. The current status is still
`downloaded_audited_not_ingested`, and training admission is false.
