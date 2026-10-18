# Tile template bank

These are labeled crops from the recorded Mahjong Soul UI at 1920×1156. The
classifier rescales each crop before comparison, so both hand-size and
river-size examples can contribute to one bank.

Sources:

- `frame300_*`: `artifacts/roi/review/frame_300.png`
- `frame1590_*`: `artifacts/roi/review/frame_1590.png`
- `frame8100_*`: `artifacts/roi/survey/frame_8100.png`

Folder names use the project's 37-class tile notation. Missing classes are
deliberately left absent: the classifier must return `unknown` instead of
guessing a label that the bank does not support. New UI themes or resolutions
need separately reviewed real crops.
