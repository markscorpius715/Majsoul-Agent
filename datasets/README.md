# Continual perception datasets

This directory is governed by `GOAL_PROMPT.md` section 23.1.

- `raw/`: immutable public, captured, and unknown source material;
- `verified/`: human-verified samples grouped by task and label;
- `rejected/`: unusable or incorrectly proposed samples;
- `golden_set/`: frozen regression inputs with reviewed expected labels;
- `manifests/`: versioned provenance, split, training queue, and release records.

Large image data is local by default. Manifests must record source, license when
applicable, content hash, human verification, split, and model provenance. A
human label adds a sample to the training queue only; it never trains or
promotes a model immediately.

The current captured-corpus workflow is packaged as four scripts in
`tools/deepseek_annotation_pipeline/`: video extraction, ROI segmentation,
DeepSeek Flash classification, and binary human review. For an existing crop
corpus, run only steps 3 and 4:

```powershell
.venv\Scripts\python tools\deepseek_annotation_pipeline\03_classify_with_deepseek.py `
  datasets\raw\captured\captured_yonma_20260923_005221\tile_crops\manifest.json
.venv\Scripts\python tools\deepseek_annotation_pipeline\04_human_review.py `
  datasets\raw\captured\captured_yonma_20260923_005221\tile_crops\manifest.json `
  --reviewer human
```

Every crop is sent as an actual image to `deepseek-flash`. Strict parsing
accepts only all requested ids, the 37 tile labels or `reject`, and finite
confidence values. Invalid batches are split and retried. Successful model
results remain `awaiting_human_confirmation`; they are not pseudo-labels.

The review UI displays only completed DeepSeek classifications. Press `1` to
Pass the current final label. Press `2` to reject that prediction, then choose
the corrected tile label or explicitly mark the crop invalid; closing the
correction dialog writes nothing. Source-video frames remain available on
demand. Manual class selection appears only after rejecting a prediction.
Pending samples are sorted by `occurrence_count` descending. Samples with the
same frequency are interleaved by final-label prediction and physical
`region/position/orientation`, prioritizing the least represented verified
position buckets. The **Verified statistics** window lists the verified count for all
37 tile labels, including zero-count labels.

When browsing `verified`, press `2` to correct an accidental confirmation:
choose another tile label to move the derived sample to the corrected class,
or choose invalid to remove it from verified data. Every such change is kept
in the append-only `review_corrections` audit table; the immutable raw crop is
never changed.

Raw corpus manifests stay immutable. Predictions and human decisions are stored in
`manifests/corpus_reviews/<corpus-id>.sqlite3`; a verified row includes
`training_status=pending` and its copied `verified/tiles/<label>/...` path.
Rejected rows retain reviewer, time, reason, source mode, and content hash but
never enter verified data. Browsing, filtering, context loading, or skipping
does not write a decision; only an explicit confirmed label or rejection does.

Captured videos are registered with `register-video` before mining. A source
manifest records its hash, player count, mode, and allowed/excluded training
scope. Review replies are exact `/label` or `/reject` commands; rejection is
audited and deduplicated but never enters verified data or the training queue.

`crop-video` creates a resumable corpus below
`raw/captured/<source-id>/tile_crops/`. Its manifest covers the requested full
timeline, stores source/player-count/mode/region/orientation provenance, and
keeps every crop unverified and ineligible for training. Repeated compressed
frames are suppressed only when region, orientation, dHash and normalized
pixel distance all agree. This raw corpus is input to review and clustering;
it is not a pseudo-labeled dataset. Animation/overlay false candidates may
remain in this audit layer with segmentation issues; persistence, novelty,
and explicit human rejection decide admission rather than silent heuristics.

Review cases and every later verified/training-queue entry must contain
`player_count` and `mode`. Legacy cases require an explicit source-context
backfill before labeling; visual content is never used to guess the game mode.

Ordered `reviewer_suggestions` may be attached to a pending case after an AI
review of the crop and context. Suggestions never grant training admission.
Only a separate human `label` action may copy the crop into `verified/`; the
suggestion is then retained as `confirmed_by_human` or `corrected_by_human`
for audit and reviewer-assistance evaluation.

Public Parquet sources can be audited without ingestion after installing the
optional data dependency:

```powershell
.venv\Scripts\python -m pip install -e ".[data]"
.venv\Scripts\python -m majsoul.continual audit-parquet `
  --split train=path/to/train.parquet --split test=path/to/test.parquet `
  --report artifacts/datasets/source/audit.json `
  --contact-sheet artifacts/datasets/source/contact-sheet.png
```

The report checks decoding, class counts, file and pixel hashes, exact
cross-split leakage, and review-only perceptual near matches. Its admission
flag is always false; a separate reviewed transformation must create a new
versioned dataset.
