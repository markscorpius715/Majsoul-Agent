# Yonma Video Visual Data Source Checkpoint (2026-09-23)

## Source and Admission

The user authorizes `屏幕录制 2026-09-23 005221.mp4` for related content recognition training. The original video is
3070×1822, 30 FPS, 162587 frames, about 90 minutes; the file content hash, size, gameplay, and admission scope are recorded in
`datasets/manifests/sources/captured_yonma_20260923_005221.json`。

The original video status is `registered_unlabeled_review_only` and cannot be used for direct training. The allowed scope includes only cross-sanma/yonma
shared tile faces, glyphs, discards, melds, screen orientation, right-side slots, and general UI. Yonma score conservation, nuki/North extraction,
sanma event rules, and unmanually confirmed model candidates are all prohibited from entering sanma state/rule training.

## Adaptive ROI Empirical Evidence

The existing reference resolution is 1920×1156. The new video's aspect ratio drift is 1.4487%, below the configured 3% automatic threshold,
so it is scaled separately by normalized x/y proportions; none of the 30 calibrated regions go out of bounds. The inspection evidence is located in
`artifacts/roi/four-player-20260923-auto/`。

Frames exceeding the threshold no longer directly report "different ratio" and terminate, nor are they silently stretched. The command
`majsoul.calibration debug ... --projection all` exports different candidate crops for independent stretching, width-preserving top/middle/bottom alignment,
and height-preserving left/middle/right alignment, awaiting recognition results or manual review selection.

## First Round of Continuous Unknown Capture

Frames 1800–1920 are sampled every 15 frames, processing 9 frames in total and generating 8 globally deduplicated review cases; the source field
is fixed as `captured_yonma_20260923_005221`, `training_started=false`. One of the crops is a blank
tile face; combined with the full-frame position, it may be the white dragon `5z` in the right player's discard pond, and it cannot be judged as a tile back or
negative sample based on the local image alone. This also proves that neither the classifier's first candidate nor the developer's visual guess can replace manual contextual confirmation.

A new `reject` command handles manually confirmed invalid crops: the rejection record is kept in the case for subsequent deduplication, but is not copied to
the verified dataset, nor does it enter the training queue. A full scan should proceed in segments and be manually reviewed; this checkpoint
does not claim that a new model has already been trained or released.

## Full-Segment Mahjong Tile Crop Corpus

`crop-video` has been used to cover the complete timeline 0–162586 at a 15-frame step, ultimately sampling 10,840 frames;
the last sampled checkpoint is 162585. Geometric segmentation produced 425,737 candidate crops, which after joint deduplication by "same-region family +
same orientation + dHash + normalized pixel difference" retained 32,784 representative crops, with cumulative repeated occurrences of
392,953. Both the manifest sample count and the on-disk PNG file count are 32,784, and the crops total 463,976,589 bytes.

The corpus is located at `datasets/raw/captured/captured_yonma_20260923_005221/tile_crops/`, manifest
with per-sample source, player_count=4, mode=yonma, region, orientation, frame number, time, content hash and
occurrence count. All entries are `label=null`, `training_allowed=false`.

The complete corpus is now connected to the local visual manual labeler: running
`.venv\Scripts\python review_labeler.py --reviewer <name>` automatically discovers this corpus, displays in pages
all 32,784 images awaiting labeling, and can filter by region, segmentation issue, status or crop id. The raw
video frames of first/last occurrence are only read on demand after a manual click. The raw manifest remains read-only; manual confirmations and rejections are written to
`datasets/manifests/corpus_reviews/captured_yonma_20260923_005221_tile_crops.sqlite3`。
Only confirmed samples are copied to `datasets/verified/tiles/<label>/` and enter pending in the same transaction's review record
training queue; rejected samples only retain the reason and source audit.

10,373 sampled frames reached the minimum tile candidate count, 467 frames were skipped; 7,302 frames contain segmentation issues
such as grouping ambiguity, animation or occlusion. The cross-time audit image is located at
`artifacts/datasets/captured_yonma_20260923_005221/crop-audit-final.png`. The audit confirms the corpus
retains both valid tile faces and animation/character/UI pseudo-candidates; the latter must be excluded through persistent multi-frame unknown, visual novelty
and manual `/reject`, and must not be silently treated as negative samples or training labels.

## Safe improvements under unlabeled conditions

The score digit `5` in new recordings uses a rounded orange font, while the existing two `5` templates both come from a centered cyan pixel font;
under the existing descriptor, the nearest class for the new glyph is `8` rather than `5`. Therefore one must not force a guess by loosening distance/margin,
and must wait for manual labels of the score metadata before adding the corresponding style template.

The seat wind `North` consists of multiple disconnected strokes, and the old segmentation mistakenly treated it as two characters. The seat wind ROI semantically contains only one
wind-position character, and now all valid components are merged after noise filtering before classification. Since the current template library does not yet have a verified
`wind_north`, the merged complete character still returns unknown. At the same time, even if one seat wind is unknown, it immediately checks
whether the remaining known seat winds are duplicated; a 60-second frame detected that both bottom/right were read as South wind and explicitly reported a consistency error,
not allowing the high-confidence misrecognition on the right to enter the state layer.
