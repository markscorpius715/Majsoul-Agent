"""Interactive, persistent ROI editor for offline screenshots."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from PIL import Image, ImageTk

from .roi_config import ROIConfig, default_layout, required_regions


ROI_DESCRIPTIONS = {
    "round_info": "Round wind and hand number (e.g. East 2)",
    "remaining_tiles": "Remaining tiles in the wall",
    "honba": "Honba count",
    "riichi_sticks": "Riichi sticks on the table",
    "dora_indicators": "All dora indicator slots",
    "self_hand": "Parent range of all visible tiles of the self player: concealed hand, drawn tile and melds; does not include the separate nuki/North extraction display area",
    "drawn_tile": "Legacy config compatibility item; the drawn tile is now recognized by spacing and timing within self_hand, no separate label needed",
    "action_buttons": "Range where action buttons such as pon, kan, win, skip appear",
    "turn_indicator": "Center board and current action prompt",
    "countdown": "Action countdown (7+20 second countdown)",
    "confirm_button": "Confirm buttons for between-hands and whole-match settlement",
    "auto_toggle_panel": "Auto option panel (auto sort hand/auto win/no call pon kan/auto discard drawn tile/auto nuki North)",
    "result_overlay": "Overall range of the between-hands or whole-match result overlay",
    "lobby_anchor": "Lobby anchor (e.g. ranked match banner), used to confirm return to lobby",
}

MELD_ANNOTATION_GUIDE = """Meld ROI annotation specification

1. self_hand is the parent range of all visible tiles of the self player, and may heavily overlap with melds_bottom.
2. melds_<slot> only describes the full range in which that slot's meld may appear; it includes positions that may be occupied by sideways-called tiles, added kan and concealed kan, and does not include the discard pond, wall or nuki/North extraction display area.
3. Do not shrink boxes to avoid overlap. Take the spatial envelope across multiple frames from 0 melds to the maximum number of melds, and leave a small margin.
4. Even if tiles are rotated or have perspective, first fully enclose top/left/right with an axis-aligned rectangle. Later rotate by slot and perform perspective normalization, then segment each meld group.
5. It is acceptable if rectangle corners contain a small amount of discard pond or background; later use direction, centerline, group spacing and timing attribution, and when disambiguation is impossible keep it unknown and stop clicking."""


def describe_region(name: str) -> str:
    if name in ROI_DESCRIPTIONS:
        return ROI_DESCRIPTIONS[name]
    for prefix, label in (
        ("score_", "Player score"), ("seat_", "Player seat wind and dealer color (red indicates dealer)"),
        ("dealer_", "Legacy config compatibility item; dealer status shares the area with seat"), ("river_", "Full candidate range of the discard pond"),
        ("melds_", "Full spatial envelope of melds; overlap with the coarse boxes of hand/discard pond is allowed, and sideways tiles are fully enclosed"),
        ("kita_", "Full candidate range of the nuki/North extraction display"),
    ):
        if name.startswith(prefix):
            return f"{label}; screen slot {name[len(prefix):]}"
    return "Custom region"


def fit_image(image_size: tuple[int, int], viewport_size: tuple[int, int]):
    """Return scale and centered offset for an image inside a viewport."""
    width, height = image_size
    view_width, view_height = viewport_size
    if min(width, height, view_width, view_height) <= 0:
        raise ValueError("Image and viewport dimensions must be positive")
    scale = min(view_width / width, view_height / height)
    shown = (max(1, round(width * scale)), max(1, round(height * scale)))
    offset = ((view_width - shown[0]) / 2, (view_height - shown[1]) / 2)
    return scale, offset, shown


def normalize_canvas_rect(start, end, image_size, scale, offset, minimum_pixels=3):
    """Map a dragged canvas rectangle into normalized image coordinates."""
    if scale <= 0:
        raise ValueError("scale must be positive")
    width, height = image_size
    ox, oy = offset
    points = []
    for cx, cy in (start, end):
        x = min(width, max(0.0, (cx - ox) / scale))
        y = min(height, max(0.0, (cy - oy) / scale))
        points.append((x, y))
    (x1, y1), (x2, y2) = points
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    if x2 - x1 < minimum_pixels or y2 - y1 < minimum_pixels:
        return None
    return [x1 / width, y1 / height, x2 / width, y2 / height]


def mirror_horizontal(rect):
    """Mirror one normalized rectangle across the vertical center line."""
    if rect is None:
        return None
    x1, y1, x2, y2 = rect
    return [1 - x2, y1, 1 - x1, y2]


def opposite_side_region(name, screen_slots):
    """Find the same region kind for the opposite left/right screen slot."""
    if "_" not in name:
        return None
    kind, suffix = name.rsplit("_", 1)
    if suffix not in ("left", "right") or suffix not in screen_slots:
        return None
    opposite = "right" if suffix == "left" else "left"
    return f"{kind}_{opposite}" if opposite in screen_slots else None


class ROIEditorApp:
    def __init__(self, root, image_path, config_path, *, player_count=3,
                 screen_slots=None, layout_variants=None, self_slot=None,
                 kita=True, names=None):
        import tkinter as tk
        from tkinter import ttk

        self.tk = tk
        self.ttk = ttk
        self.root = root
        self.config_path = Path(config_path)
        self.image_path = Path(image_path)
        self.image = self._read_image(self.image_path)
        self.image_size = self.image.size
        if self.config_path.exists():
            self.config = ROIConfig.load(self.config_path)
            self.config.pixel_boxes(self.image_size)
        else:
            default_slots, default_variants, default_self = default_layout(player_count)
            slots = screen_slots or default_slots
            variants = layout_variants or default_variants
            own_slot = self_slot or default_self
            regions = dict.fromkeys(required_regions(slots, kita))
            self.config = ROIConfig(reference_size=list(self.image_size), player_count=player_count,
                                    screen_slots=slots, layout_variants=variants,
                                    self_slot=own_slot, kita=kita, regions=regions)
            self.config.validate()
        requested = names or list(self.config.regions)
        unknown = [name for name in requested if name not in self.config.regions]
        if unknown:
            raise ValueError("Unknown ROI names: " + ", ".join(unknown))
        self.names = list(requested)
        self.current = next((name for name in self.names if self.config.regions[name] is None), self.names[0])
        self.history = []
        self.dirty = False
        self.drag_start = None
        self.drag_item = None
        self.photo = None
        self.scale = 1.0
        self.offset = (0.0, 0.0)
        self.shown_size = self.image_size
        self._render_pending = False

        root.title("Mahjong Soul ROI annotator")
        root.minsize(1000, 650)
        root.geometry(f"{min(1600, root.winfo_screenwidth()-80)}x{min(950, root.winfo_screenheight()-100)}")
        root.protocol("WM_DELETE_WINDOW", self.close)

        main = ttk.Panedwindow(root, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True)
        image_frame = ttk.Frame(main)
        side = ttk.Frame(main, padding=10, width=420)
        main.add(image_frame, weight=4)
        main.add(side, weight=1)

        self.canvas = tk.Canvas(image_frame, background="#181818", cursor="crosshair", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", self.schedule_render)
        self.canvas.bind("<ButtonPress-1>", self.begin_drag)
        self.canvas.bind("<B1-Motion>", self.update_drag)
        self.canvas.bind("<ButtonRelease-1>", self.end_drag)

        ttk.Label(side, text="ROI region", font=("Microsoft YaHei UI", 13, "bold")).pack(anchor=tk.W)
        self.tree = ttk.Treeview(side, columns=("status", "description"), show="headings", height=22)
        self.tree.heading("status", text="Status")
        self.tree.heading("description", text="Name and meaning")
        self.tree.column("status", width=58, stretch=False, anchor=tk.CENTER)
        self.tree.column("description", width=330, stretch=True)
        scroll = ttk.Scrollbar(side, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        self.tree.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        scroll.place(in_=self.tree, relx=1, rely=0, relheight=1, anchor="ne")
        self.tree.bind("<<TreeviewSelect>>", self.select_tree_item)

        self.current_label = ttk.Label(side, wraplength=390, justify=tk.LEFT)
        self.current_label.pack(fill=tk.X, pady=(10, 6))
        self.progress_label = ttk.Label(side)
        self.progress_label.pack(fill=tk.X, pady=(0, 8))

        nav = ttk.Frame(side)
        nav.pack(fill=tk.X, pady=2)
        ttk.Button(nav, text="Previous", command=lambda: self.move(-1)).pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Button(nav, text="Next", command=lambda: self.move(1)).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(6, 0))
        edit = ttk.Frame(side)
        edit.pack(fill=tk.X, pady=2)
        ttk.Button(edit, text="Mark as unknown", command=self.mark_unknown).pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Button(edit, text="Undo", command=self.undo).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(6, 0))
        ttk.Button(side, text="Mirror horizontally to the corresponding left/right region", command=self.mirror_to_opposite).pack(fill=tk.X, pady=2)
        ttk.Button(side, text="View meld annotation specification", command=self.show_meld_guide).pack(fill=tk.X, pady=2)
        files = ttk.Frame(side)
        files.pack(fill=tk.X, pady=2)
        ttk.Button(files, text="Switch reference frame", command=self.open_image).pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Button(files, text="Save", command=self.save).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(6, 0))
        ttk.Button(side, text="Save and exit", command=self.save_and_close).pack(fill=tk.X, pady=(8, 2))

        help_text = ("Operation: hold the left button on the image and drag out a rectangle; click the list on the right to switch regions.\n"
                     "Shortcuts: Ctrl+S to save, Ctrl+O to change frame, Ctrl+Z to undo, Delete to mark unknown, ↑/↓ to switch.\n"
                     "bottom/top/left/right are screen slots; at the start of the match lock three of them and reuse them for the whole match.\n"
                     "Changing the frame only changes the background; drawn boxes are retained. For regions that cannot be confirmed, keep them unknown.")
        ttk.Label(side, text=help_text, wraplength=390, justify=tk.LEFT, foreground="#555555").pack(fill=tk.X, pady=(8, 0))

        root.bind("<Control-s>", lambda _: self.save())
        root.bind("<Control-o>", lambda _: self.open_image())
        root.bind("<Control-z>", lambda _: self.undo())
        root.bind("<Delete>", lambda _: self.mark_unknown())
        root.bind("<Up>", lambda _: self.move(-1))
        root.bind("<Down>", lambda _: self.move(1))
        root.bind("<Escape>", lambda _: self.cancel_drag())
        self.refresh_tree()
        root.after_idle(self.render)

    @staticmethod
    def _read_image(path):
        with Image.open(path) as source:
            return source.convert("RGB")

    def schedule_render(self, _event=None):
        if not self._render_pending:
            self._render_pending = True
            self.root.after_idle(self.render)

    def render(self):
        self._render_pending = False
        width = max(1, self.canvas.winfo_width())
        height = max(1, self.canvas.winfo_height())
        self.scale, self.offset, self.shown_size = fit_image(self.image_size, (width, height))
        resized = self.image.resize(self.shown_size, Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(resized)
        self.canvas.delete("all")
        self.canvas.create_image(*self.offset, anchor="nw", image=self.photo)
        for name, rect in self.config.regions.items():
            if rect is None:
                continue
            x1, y1, x2, y2 = self.rect_to_canvas(rect)
            active = name == self.current
            color = "#ffb000" if active else "#39d353"
            self.canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=3 if active else 1)
            label = self.canvas.create_text(x1 + 3, y1 + 3, anchor="nw", text=name, fill="#111111",
                                            font=("Arial", 9, "bold"))
            box = self.canvas.bbox(label)
            if box:
                self.canvas.create_rectangle(*box, fill=color, outline=color)
                self.canvas.tag_raise(label)

    def rect_to_canvas(self, rect):
        width, height = self.image_size
        ox, oy = self.offset
        return (ox + rect[0] * width * self.scale, oy + rect[1] * height * self.scale,
                ox + rect[2] * width * self.scale, oy + rect[3] * height * self.scale)

    def begin_drag(self, event):
        ox, oy = self.offset
        sw, sh = self.shown_size
        if not (ox <= event.x <= ox + sw and oy <= event.y <= oy + sh):
            return
        self.drag_start = (event.x, event.y)
        self.drag_item = self.canvas.create_rectangle(event.x, event.y, event.x, event.y,
                                                       outline="#00d9ff", width=2, dash=(5, 3))

    def update_drag(self, event):
        if self.drag_start and self.drag_item:
            self.canvas.coords(self.drag_item, *self.drag_start, event.x, event.y)

    def end_drag(self, event):
        if not self.drag_start:
            return
        rect = normalize_canvas_rect(self.drag_start, (event.x, event.y), self.image_size,
                                     self.scale, self.offset)
        self.cancel_drag()
        if rect is None:
            return
        self._change(self.current, rect)
        self.move(1, prefer_missing=True)

    def cancel_drag(self):
        if self.drag_item:
            self.canvas.delete(self.drag_item)
        self.drag_start = None
        self.drag_item = None

    def _change(self, name, value):
        old = deepcopy(self.config.regions[name])
        if old == value:
            return
        self.history.append((name, old))
        self.config.regions[name] = value
        self.config.validate()
        self.dirty = True
        self.refresh_tree()
        self.render()

    def mark_unknown(self):
        self._change(self.current, None)
        self.move(1, prefer_missing=True)

    def undo(self):
        if not self.history:
            return
        name, value = self.history.pop()
        self.config.regions[name] = value
        self.current = name
        self.dirty = True
        self.refresh_tree()
        self.render()

    def mirror_to_opposite(self):
        from tkinter import messagebox
        target = opposite_side_region(self.current, self.config.screen_slots)
        rect = self.config.regions[self.current]
        if target is None or target not in self.config.regions:
            messagebox.showinfo("Cannot mirror", "The current region is not a configured left/right player region.", parent=self.root)
            return
        if rect is None:
            messagebox.showinfo("Cannot mirror", "Please annotate the current region first, then perform mirroring.", parent=self.root)
            return
        if self.config.regions[target] is not None and not messagebox.askyesno(
                "Overwrite existing region?", f"{target} already annotated, Overwrite with the mirrored result?", parent=self.root):
            return
        self._change(target, mirror_horizontal(rect))
        self.current = target
        self.refresh_tree()
        self.render()

    def show_meld_guide(self):
        from tkinter import messagebox
        messagebox.showinfo("Meld ROI annotation specification", MELD_ANNOTATION_GUIDE, parent=self.root)

    def move(self, delta, prefer_missing=False):
        index = self.names.index(self.current)
        candidates = [self.names[(index + delta * step) % len(self.names)]
                      for step in range(1, len(self.names) + 1)]
        if prefer_missing:
            self.current = next((name for name in candidates if self.config.regions[name] is None), candidates[0])
        else:
            self.current = candidates[0]
        self.refresh_tree()
        self.render()

    def refresh_tree(self):
        selected = self.current
        existing = set(self.tree.get_children())
        for name in self.names:
            status = "Annotated" if self.config.regions[name] is not None else "Unknown"
            values = (status, f"{name} — {describe_region(name)}")
            if name in existing:
                self.tree.item(name, values=values)
            else:
                self.tree.insert("", "end", iid=name, values=values)
        self.tree.selection_set(selected)
        self.tree.see(selected)
        self.current_label.configure(text=f"current: {selected}\n{describe_region(selected)}")
        marked = sum(self.config.regions[name] is not None for name in self.names)
        self.progress_label.configure(text=f"progress: {marked}/{len(self.names)} annotated; {len(self.names)-marked} unknown")

    def select_tree_item(self, _event=None):
        selection = self.tree.selection()
        if selection and selection[0] != self.current:
            self.current = selection[0]
            self.current_label.configure(text=f"current: {self.current}\n{describe_region(self.current)}")
            self.render()

    def open_image(self):
        from tkinter import filedialog, messagebox
        path = filedialog.askopenfilename(title="Select a reference frame with the same layout", initialdir=str(self.image_path.parent),
                                          filetypes=[("Image", "*.png *.jpg *.jpeg *.bmp *.webp"), ("All files", "*.*")])
        if not path:
            return
        try:
            image = self._read_image(path)
            self.config.pixel_boxes(image.size)
        except (OSError, ValueError) as exc:
            messagebox.showerror("Unable to switch reference frame", str(exc), parent=self.root)
            return
        self.image_path = Path(path)
        self.image = image
        self.image_size = image.size
        self.render()

    def save(self):
        from tkinter import messagebox
        try:
            self.config.save(self.config_path)
        except (OSError, ValueError) as exc:
            messagebox.showerror("Save failed", str(exc), parent=self.root)
            return False
        self.dirty = False
        missing = self.config.validate()
        messagebox.showinfo("Saved", f"saved to: \n{self.config_path}\n\n remaining {len(missing)} unknown regions. ",
                            parent=self.root)
        return True

    def save_and_close(self):
        if self.save():
            self.root.destroy()

    def close(self):
        from tkinter import messagebox
        if self.dirty and not messagebox.askyesno("Discard unsaved changes?", "Current changes have not been saved. Exit anyway?",
                                                  parent=self.root):
            return
        self.root.destroy()


def run_roi_gui(image_path, config_path="configs/roi.yaml", *, player_count=3,
                screen_slots=None, layout_variants=None, self_slot=None,
                kita=True, names=None):
    import tkinter as tk

    root = tk.Tk()
    try:
        ROIEditorApp(root, image_path, config_path, player_count=player_count,
                     screen_slots=screen_slots, layout_variants=layout_variants,
                     self_slot=self_slot, kita=kita, names=names)
    except Exception:
        root.destroy()
        raise
    root.mainloop()
