# Phase 4: Numbers and Match Metadata

## Implementation

Phase 4 uses fixed ROIs and real-machine glyph templates to read:

- The `round_id` composed of the round wind, hand number, and honba count;
- The remaining tile count and the number of riichi sticks in the pot;
- The scores of all valid screen slots;
- The seat wind of each slot and the red dealer status.

Digits, prefixes, and wind positions are first segmented by configured HSV ranges, then normalized into binary glyphs and compared with
the real templates in `artifacts/templates/glyphs/`. Classification simultaneously checks the maximum distance
and the gap between the top two classes; when a template is missing or ambiguous, it returns `unknown`. The top, left, and right slots are recognized before
Rotate to a unified orientation according to `configs/vision.yaml`, then map the output boxes back to screen coordinates.

Adjacent digits of the left-side score may merge into a single connected component due to perspective and anti-aliasing. The parser splits them based on the typical
character width and local projection valleys within the same score, without hardcoding specific scores. For the dealer's seat wind, it first detects
the red tile base, then extracts the dark-colored wind-position character from inside the tile base; when the red coverage ratio falls in an uncertain interval, it will not
forcibly determine the dealer.

## Real-Machine Evidence

Normal frames in the same recording yield the following complete results:

| Frame | round_id | Remaining tiles | Honba | bottom / top / left scores | Dealer |
| --- | --- | ---: | ---: | --- | --- |
| 300 | `1z:2:0` | 48 | 0 | 42000 / 30000 / 33000 | left (East) |
| 1590 | `1z:2:0` | 21 | 1 | 42000 / 30000 / 32000 | left (East) |
| 2700 | `1z:2:1` | 49 | 0 | 40000 / 28000 / 37000 | left (East) |
| 6300 | `1z:3:0` | 24 | 2 | 23800 / 43200 / 36000 | bottom (East) |
| 8100 | `1z:3:1` | 37 | 0 | 37800 / 37200 / 30000 | bottom (East) |

The machine-readable reports and boxed original images for 1590 and 6300 are located at:

- `artifacts/perception/phase4/frame1590-metadata/`
- `artifacts/perception/phase4/frame6300-metadata/`

Settlement, transition, character animation, or recording distortion frames such as 4230, 6855, 6870, 6900, and 6930 will
return `complete=false`, and will not construct complete hand metadata from partial residual digits.

## Usage

```powershell
.venv\Scripts\python -m majsoul.perception metadata artifacts/roi/review/frame_1590.png --active-slots bottom top left --output artifacts/perception/check-metadata
```

`--active-slots` must come from match-level layout locking. The output is `metadata_debug.png` and
`metadata_report.json`; only when the overall report and every sub-reading are complete may the subsequent state layer
accept that frame.

## Still to be covered

The current recording only actually tests `bottom+top+left`, and only East round and East/South/West seat winds appear. The template library has no
South round text, North seat wind, negative score sign, or real right-side player screen. New themes, different rendering scales, and
these categories that have not appeared must all be supplemented with templates from real screens; until then, keep `unknown`. What Phase 4
completes is single-frame metadata perception; continuous event recovery belongs to Phase 5 and cannot be replaced by single-frame results.
