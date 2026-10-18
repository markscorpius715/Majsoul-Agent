import cv2

from .roi_config import ROIConfig, default_layout, required_regions


def select_region(name, image, selector, preview_size):
    height, width = image.shape[:2]
    if (len(preview_size) != 2 or
            any(type(value) is not int or value <= 0 for value in preview_size)):
        raise ValueError("preview_size must contain two positive pixel dimensions")
    scale = min(1.0, preview_size[0] / width, preview_size[1] / height)
    preview_width = max(1, round(width * scale))
    preview_height = max(1, round(height * scale))
    preview = (cv2.resize(image, (preview_width, preview_height), interpolation=cv2.INTER_AREA)
               if scale < 1 else image)
    x, y, w, h = selector(f"ROI: {name}", preview, showCrosshair=True, fromCenter=False)
    # Normalization in preview space is equivalent to mapping to original pixels
    # and then normalizing. Use actual rounded dimensions on each axis.
    return ([x / preview_width, y / preview_height,
             (x+w) / preview_width, (y+h) / preview_height] if w and h else None)


def edit_rois(image, *, player_count, screen_slots=None, layout_variants=None,
              self_slot=None, kita, existing=None,
              names=None, selector=None, ask=input, preview_size=(1280, 800)):
    """Cancel a selection to record unknown, never invent coordinates."""
    selector = selector or cv2.selectROI
    height, width = image.shape[:2]
    if existing:
        existing.pixel_boxes((width, height))
        regions = dict(existing.regions)
        screen_slots = list(existing.screen_slots)
        layout_variants = [list(variant) for variant in existing.layout_variants]
        self_slot = existing.self_slot
        player_count = existing.player_count
        kita = existing.kita
    else:
        default_slots, default_variants, default_self = default_layout(player_count)
        screen_slots = screen_slots or default_slots
        layout_variants = layout_variants or default_variants
        self_slot = self_slot or default_self
        regions = dict.fromkeys(required_regions(screen_slots, kita))
    config = ROIConfig(reference_size=[width, height], player_count=player_count,
                       screen_slots=screen_slots, layout_variants=layout_variants,
                       self_slot=self_slot, kita=kita, regions=regions)
    config.validate()
    try:
        for name in names or list(regions):
            print(f"Select {name}: Enter/Space accepts; Esc marks unknown. Ctrl+C aborts without saving.", flush=True)
            regions[name] = select_region(name, image, selector, preview_size)
            config.validate()
            cv2.destroyAllWindows()
        while True:
            name = ask("Additional ROI name (letters/numbers/underscores; Enter finishes): ").strip()
            if not name:
                break
            if name in regions:
                print("Name already exists; use --names to edit it.")
                continue
            regions[name] = None
            try:
                config.validate()
            except ValueError:
                del regions[name]
                print("Invalid ROI name.")
                continue
            regions[name] = select_region(name, image, selector, preview_size)
            config.validate()
            cv2.destroyAllWindows()
    finally:
        cv2.destroyAllWindows()
    return config
