# Metadata glyph template bank

These binary templates were segmented from reviewed 1920×1156 Mahjong Soul
recording frames. Labels cover digits `0` through `9`, the remaining-tile and
counter prefixes, the round suffix, and the East/South/West wind glyphs visible
in the recording.

Sources include frames 0, 300, 1590, 2700, 5400, 6300, 7150, and 8100. The
score templates deliberately include bottom, top, and left renderings after
slot rotation because perspective and anti-aliasing differ by screen slot.

Missing visual classes are not synthesized. In particular, the recording has
no South-round label, North seat, negative score, or real occupied-right-slot
sample. Those cases must remain `unknown` until separately reviewed real
templates are added. A new UI theme, resolution, or rendering scale also needs
a separately validated template bank.
