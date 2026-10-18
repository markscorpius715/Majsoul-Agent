# Phase 3: All Visible Tiles

## Implementation

Phase 3 reads on the `seat_layout` locked for the whole match:

- The self hand and drawn tile positions;
- The discards of the three valid screen slots;
- The melds and group numbers of the three valid slots;
- The dora indicators;
- The nuki/North tiles of each slot.

`configs/vision.yaml` stores the rotation angles for `bottom/top/left/right` and the relative geometric ranges of each type of region. The image is first rotated to a unified viewpoint, and the recognition boxes are then mapped back to the original screen coordinates. Adjacent white tile faces in the left discard pile become connected into a single connected component due to perspective; the parser splits them based on the median size of normal tile faces in the same frame and projection valleys, without using specific tile names or screenshot coordinates.

Meld recognition preserves the sideways call direction and groups by spatial intervals in standardized coordinates. Each group must have 3 or 4 tiles; when the number of groups, tile faces, or occlusion cannot be determined, an issue is reported and no false meld is constructed. The large tile faces in `self_hand` and the smaller flat-laid tile faces in `melds_bottom` use different geometric ranges, so the same instance in an overlapping ROI will not be duplicated into both the hand and the melds.

## Real-Machine Evidence

On the fixed layout `bottom+top+left` of `frame_1590.png`, the complete recognition result is:

- Hand: 10 tiles;
- Discards: bottom 10 tiles, top 11 tiles, left 9 tiles;
- Melds: bottom is 4 nine sou, top is 3 white dragons, left is empty;
- Dora indicators: `7s 6z`;
- Nuki/North: bottom 1 tile, top 1 tile, left 0 tiles.

The overall report is `complete=true`, with no issues. The original image overlay and JSON are located at:

- `artifacts/perception/phase3/frame1590-visible/recognition_debug.png`
- `artifacts/perception/phase3/frame1590-visible/recognition_report.json`

The action/character animation in frame 6300 causes the meld and nuki/North regions to return occlusion issues; the upper discard animation in frame 8100 also contaminates the upper meld ROI. Neither is treated as a complete observation. At the time of the Phase 3 checkpoint, 110 tests passed; after adding Phase 4, the current full suite is 118 items.

## Usage

```powershell
.venv\Scripts\python -m majsoul.perception visible artifacts/roi/review/frame_1590.png --active-slots bottom top left --output artifacts/perception/check-visible
```

`--active-slots` must be locked by continuous consistent observation from the start of the match and reused for the whole match. The command will reject slot combinations that are not part of a legal layout in `configs/roi.yaml`.

## Still Awaiting Real-Machine Coverage

The current footage only actually covers `bottom+top+left`. The right side uses a configured rotation opposite to the left side and reviewed ROIs, but real footage verification with a player on the right is still needed. The footage also has no confirmable sideways riichi tile; more arrangements of multiple melds, closed kans, and added kans still need to be supplemented with new screenshots. In the absence of such evidence, the relevant situations remain `unknown` or incomplete.
