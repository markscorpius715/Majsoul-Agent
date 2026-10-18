# Phase 2: this player's hand and one group of river recognition

## Status

The Phase 1 ROI passed user review on 2026-09-22. This phase implements and verifies offline:

- upright concealed-hand segmentation within `self_hand`, left-to-right ordering and draw-gap determination;
- grid row splitting within `river_bottom`, left-to-right ordering, orientation preservation and rejection of off-grid candidates;
- a shared 37-class tile-name template classifier;
- explicit reporting of `unknown`, occlusion, animation and off-grid UI contamination;
- JSON recognition reports and original-image overlay debug images.

The implementation is in `majsoul/perception/`, with thresholds centralized in `configs/vision.yaml`. The template library uses real UI crops and is in `artifacts/templates/tiles/`; the Phase 2 checkpoint covered 22 classes and 33 templates, and it has been incrementally expanded after advancing to Phase 3. Classes lacking templates must return `unknown`, and similar tiles must not be substituted.

## Real-screen verification

On a 1920×1156 recorded screen:

| Frame | Hand | Lower river | Result |
| --- | ---: | ---: | --- |
| 300 | 14 | 1 | Segmentation correct; the last tile is recognized as a draw by spacing |
| 1575 | 10 | 10 | All 20 tiles recognized, order correct |
| 1590 | 10 | 10 | All 20 tiles recognized, order correct |
| 1605 | 10 | 10 | All 20 tiles recognized, order consistent with the two adjacent frames |
| 8100 | 13 | 6 | All 19 tiles recognized, order correct |

The stable sequence for the three frames 1575, 1590, 1605 is:

- Hand: `5p 5p 4s 0s 5s 6s 7s 7s 8s 6z`
- Discards: `1m 1p 1s 7p 4p 3z 2z 1m 9p 7z`

Recognition evidence is located in `artifacts/perception/phase2/recognize-*/`. Among these, `recognize-1590/recognition_debug.png` and `recognition_report.json` are the primary acceptance samples.

Frame 6300 has animation/occlusion, and the segmenter reports `animation or occlusion`. In frame 6750, the operation selection UI falls into the wide discard ROI; the parser excludes candidates that do not match the discard row spacing and reports `Off-grid river components ignored`; frames with issues are not treated as complete observations.

## Verification Commands

```powershell
.venv\Scripts\python -m pytest tests/test_perception.py -q
.venv\Scripts\python -m majsoul.perception recognize artifacts/roi/review/frame_1590.png --output artifacts/perception/check-recognition
```

The tests include real adjacent-frame stability, normal tiles, red five sou, cross-scale recognition of small discard tiles by the hand template, rotation normalization, rejection of random unknown images, animation occlusion, and operation UI contamination.

## Boundaries and Next Phase

The current template library does not cover all 37 categories; new themes, different rendering scales, and tiles that have not yet appeared must continue to have reviewed templates collected. No confirmable sideways riichi tile for the self player was found in the real footage, so although direction normalization already has real tile-face rotation tests, end-to-end verification still needs to be supplemented on footage where a real sideways tile appears.

Phase 3 will read the three screen slots locked for the whole match, and extend to the remaining discards, melds, dora, and nuki/North extraction. The left and right regions need slot direction normalization first; overlap between `self_hand` and `melds_bottom` must be deduplicated by tile instance, and kept unknown when disambiguation is impossible.
