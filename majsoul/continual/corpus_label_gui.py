"""Desktop labeler for large captured tile-crop corpora."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from .corpus_review import CorpusReviewStore, CropReviewRecord
from .deepseek_classifier import CLASSIFICATION_LABELS
from .label_gui import ReviewLabelerApp, preset_labels
from .schemas import ReviewStatus


def discover_crop_manifests(dataset_root: str | Path) -> tuple[Path, ...]:
    """Find complete captured crop corpora below a dataset root."""
    root = Path(dataset_root) / "raw" / "captured"
    manifests = []
    for path in sorted(root.glob("*/tile_crops/manifest.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            continue
        if (value.get("label_status") == "unverified" and
                value.get("training_allowed") is False and
                isinstance(value.get("samples"), list)):
            manifests.append(path)
    return tuple(manifests)


def _preview(path: Path, size: tuple[int, int], *, upscale: bool = True) -> Image.Image:
    with Image.open(path) as source:
        image = source.convert("RGB")
    if upscale:
        image = ImageOps.contain(image, size, Image.Resampling.NEAREST)
    else:
        image.thumbnail(size, Image.Resampling.LANCZOS)
    background = Image.new("RGB", size, "#20242a")
    background.paste(
        image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
    return background


class SourceFrameReader:
    """Resolve and lazily seek the registered source video for visual context."""

    def __init__(self, corpus_manifest: Path):
        self.corpus_manifest = corpus_manifest
        self.corpus = json.loads(
            corpus_manifest.read_text(encoding="utf-8-sig"))
        self.source_manifest_path = self._resolve_source_manifest()
        self.source_manifest = (
            json.loads(self.source_manifest_path.read_text(encoding="utf-8-sig"))
            if self.source_manifest_path else {})
        self.video_path = self._resolve_video()
        self._capture = None

    def _resolve_source_manifest(self) -> Path | None:
        raw = self.corpus.get("source_manifest")
        if not raw:
            return None
        candidate = Path(raw)
        checks = [candidate, Path.cwd() / candidate]
        for parent in self.corpus_manifest.parents:
            checks.append(parent / candidate)
        for path in checks:
            if path.is_file():
                return path.resolve()
        return None

    def _resolve_video(self) -> Path | None:
        raw = self.source_manifest.get("path")
        if raw:
            candidate = Path(str(raw))
            bases = [Path.cwd()]
            if self.source_manifest_path:
                bases.append(self.source_manifest_path.parent)
            for path in [candidate, *(base / candidate for base in bases)]:
                if path.is_file():
                    return path.resolve()
        expected_size = self.source_manifest.get("bytes")
        if isinstance(expected_size, int):
            matches = [path.resolve() for path in Path.cwd().glob("*.mp4")
                       if path.is_file() and path.stat().st_size == expected_size]
            if len(matches) == 1:
                return matches[0]
        return None

    def read(self, frame_index: int) -> Image.Image:
        if self.video_path is None:
            raise FileNotFoundError(
                "Source video not found; crop can still be annotated, but the full frame context cannot be displayed.")
        import cv2
        if self._capture is None:
            self._capture = cv2.VideoCapture(str(self.video_path))
        if not self._capture.isOpened():
            raise OSError(f"Cannot open source video: {self.video_path}")
        self._capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = self._capture.read()
        if not ok:
            raise OSError(f"Cannot read source video frame {frame_index} frame")
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None


class CorpusLabelerApp:
    """Paged visual review UI for every crop in one immutable corpus."""

    PAGE_SIZE = 300

    def __init__(self, root, manifest: str | Path, dataset_root: str | Path, *,
                 cases_root: str | Path = "artifacts/unknown_cases",
                 reviewer: str = "human_gui", ai_reviewed_only: bool = False):
        import tkinter as tk
        from tkinter import ttk

        self.tk = tk
        self.ttk = ttk
        self.root = root
        self.dataset_root = Path(dataset_root)
        self.cases_root = Path(cases_root)
        self.store = CorpusReviewStore(manifest, dataset_root)
        self.ai_reviewed_only = ai_reviewed_only
        self.frame_reader = SourceFrameReader(Path(manifest))
        self.records: list[CropReviewRecord] = []
        self.page_records: list[CropReviewRecord] = []
        self.current: CropReviewRecord | None = None
        self.page = 0
        self.photos: dict[str, Any] = {}

        self.status_var = tk.StringVar(value="pending")
        self.region_var = tk.StringVar(value="all")
        self.issue_var = tk.StringVar(value="all")
        self.predicted_label_var = tk.StringVar(value="all")
        self.query_var = tk.StringVar()
        self.reviewer_var = tk.StringVar(value=reviewer)
        self.label_var = tk.StringVar()
        self.reject_var = tk.StringVar(
            value="deepseek_classification_rejected_by_human")
        self.prediction_var = tk.StringVar(value="Waiting for reviewable candidates")
        self.progress_var = tk.StringVar()
        self.sample_info_var = tk.StringVar(value="No crops match the filter criteria")
        self.context_info_var = tk.StringVar(value="Load source frame context on demand")

        root.title("Mahjong Soul Manual confirmation of AI review results" if ai_reviewed_only else "Mahjong Soul Visualization of complete images to be annotated")
        root.minsize(1180, 760)
        root.geometry(
            f"{min(1720, root.winfo_screenwidth()-50)}x"
            f"{min(1000, root.winfo_screenheight()-70)}")
        root.protocol("WM_DELETE_WINDOW", self.close)

        toolbar = ttk.Frame(root, padding=(10, 8))
        toolbar.pack(fill=tk.X)
        self._combo(toolbar, "Status", self.status_var,
                    ("pending", "all", "verified", "rejected"), 10)
        self._combo(toolbar, "Model candidates", self.predicted_label_var,
                    ("all", *CLASSIFICATION_LABELS), 10)
        self._combo(toolbar, "Region", self.region_var,
                    ("all", *self.store.regions()), 18)
        self._combo(toolbar, "Quality", self.issue_var,
                    ("all", "clean", "issues"), 9)
        ttk.Label(toolbar, text="ID search").pack(side=tk.LEFT)
        query = ttk.Entry(toolbar, textvariable=self.query_var, width=20)
        query.pack(side=tk.LEFT, padx=(5, 12))
        query.bind("<Return>", lambda _event: self.refresh(reset_page=True))
        ttk.Label(toolbar, text="Annotator").pack(side=tk.LEFT)
        ttk.Entry(toolbar, textvariable=self.reviewer_var, width=14).pack(
            side=tk.LEFT, padx=(5, 10))
        ttk.Button(toolbar, text="Refresh", command=lambda: self.refresh(
            reset_page=True)).pack(side=tk.LEFT)
        ttk.Button(toolbar, text="Verified statistics", command=self.show_verified_stats).pack(
            side=tk.LEFT, padx=(8, 0))
        ttk.Button(toolbar, text="Anomaly case window", command=self.open_cases).pack(
            side=tk.LEFT, padx=(8, 0))

        pagebar = ttk.Frame(root, padding=(10, 0, 10, 7))
        pagebar.pack(fill=tk.X)
        ttk.Button(pagebar, text="← Previous page", command=lambda: self.change_page(-1)).pack(
            side=tk.LEFT)
        ttk.Button(pagebar, text="Next page →", command=lambda: self.change_page(1)).pack(
            side=tk.LEFT, padx=(6, 0))
        ttk.Label(pagebar, textvariable=self.progress_var).pack(side=tk.RIGHT)

        main = ttk.Panedwindow(root, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 8))
        queue_frame = ttk.Frame(main, width=490)
        detail = ttk.Frame(main)
        main.add(queue_frame, weight=2)
        main.add(detail, weight=5)

        columns = (
            "id", "prediction", "confidence", "location", "uses",
            "reviewed_at", "status")
        self.tree = ttk.Treeview(
            queue_frame, columns=columns, show="headings", selectmode="browse")
        for name, title, width in (
                ("id", "Crop ID", 190), ("prediction", "Model candidates", 75),
                ("confidence", "Confidence", 65), ("location", "Physical location", 210),
                ("uses", "Appearance", 50),
                ("reviewed_at", "Review/modification time", 185),
                ("status", "Status", 70)):
            self.tree.heading(name, text=title)
            self.tree.column(name, width=width, stretch=name == "id")
        scroll = ttk.Scrollbar(queue_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.bind("<<TreeviewSelect>>", self._select_tree)

        ttk.Label(
            detail, textvariable=self.sample_info_var, justify=tk.LEFT,
            font=("Microsoft YaHei UI", 10, "bold"), wraplength=1120).pack(
                fill=tk.X, pady=(0, 6))

        images = ttk.Panedwindow(detail, orient=tk.HORIZONTAL)
        images.pack(fill=tk.BOTH, expand=True)
        crop_frame = ttk.LabelFrame(images, text="Crops to be annotated", padding=6)
        context_frame = ttk.LabelFrame(images, text="Source frame context (on demand)", padding=6)
        images.add(crop_frame, weight=2)
        images.add(context_frame, weight=3)
        self.crop_label = ttk.Label(crop_frame, anchor=tk.CENTER)
        self.crop_label.pack(fill=tk.BOTH, expand=True)
        self.context_label = ttk.Label(
            context_frame, textvariable=self.context_info_var,
            anchor=tk.CENTER, justify=tk.CENTER)
        self.context_label.pack(fill=tk.BOTH, expand=True)
        context_nav = ttk.Frame(context_frame)
        context_nav.pack(fill=tk.X, pady=(6, 0))
        ttk.Button(context_nav, text="Load first appearance frame",
                   command=lambda: self.load_context("first_frame")).pack(
                       side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(context_nav, text="Load last appearance frame",
                   command=lambda: self.load_context("last_frame")).pack(
                       side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 0))

        label_frame = ttk.LabelFrame(detail, text="Manual review of model candidates", padding=8)
        label_frame.pack(fill=tk.X, pady=(8, 0))
        ttk.Label(
            label_frame, textvariable=self.prediction_var,
            font=("Microsoft YaHei UI", 15, "bold"), anchor=tk.CENTER).pack(
                fill=tk.X, pady=(2, 8))
        actions = ttk.Frame(label_frame)
        actions.pack(fill=tk.X)
        self.verify_button = ttk.Button(
            actions, text="1 · Pass", command=self.verify_label)
        self.verify_button.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.reject_button = ttk.Button(
            actions, text="2 · Reject / Correct", command=self.reject_sample)
        self.reject_button.pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 0))
        navigation = ttk.Frame(label_frame)
        navigation.pack(fill=tk.X, pady=(5, 0))
        ttk.Button(navigation, text="← Previous", command=lambda: self.move(-1)).pack(
            side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(navigation, text="Skip / Next →", command=lambda: self.move(1)).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 0))

        ttk.Label(
            root,
            text=(("The pending-confirmation list only includes images already reviewed by AI (including uncertain items); to add a new batch, click refresh."
                   if ai_reviewed_only else "The pending-confirmation list shows classified images.") +
                  "Press 1 Pass: manually confirm the current candidate; tile-type images are saved to datasets/verified/tiles/<label>/;"
                  "Press 2 Reject: reject the current label and select the correct label or an invalid sample."
                  "When viewing verified, press 2 to re-correct or void."
                  "When viewing rejected, press 2 to reclassify a mistakenly rejected sample as the correct tile type."
                  "pending is sorted by occurrence count from high to low,"
                  "and when counts are equal, tile types and physical positions are balanced."
                  "verified is sorted by most recent review or modification time."
                  "Canceling the correction window will not write data. Alt+←/→ only browses."),
            foreground="#555555", padding=(10, 0, 10, 8)).pack(fill=tk.X)

        for variable in (self.status_var, self.region_var, self.issue_var,
                         self.predicted_label_var):
            variable.trace_add("write", lambda *_args: self.refresh(reset_page=True))
        root.bind("<F5>", lambda _event: self.refresh(reset_page=True))
        root.bind("<Alt-Left>", lambda _event: self.move(-1))
        root.bind("<Alt-Right>", lambda _event: self.move(1))
        root.bind("1", lambda event: self._review_hotkey(event, pass_result=True))
        root.bind("2", lambda event: self._review_hotkey(event, pass_result=False))
        self.refresh(reset_page=True)

    def _combo(self, parent, title, variable, values, width):
        self.ttk.Label(parent, text=title).pack(side=self.tk.LEFT)
        box = self.ttk.Combobox(
            parent, textvariable=variable, state="readonly", width=width,
            values=values)
        box.pack(side=self.tk.LEFT, padx=(5, 12))

    def refresh(self, *, reset_page: bool = False,
                preferred_index: int | None = None) -> None:
        if reset_page:
            self.page = 0
        selected_status = self.status_var.get()
        if selected_status == ReviewStatus.PENDING.value:
            order = "frequency_balanced_positions"
        elif selected_status == ReviewStatus.VERIFIED.value:
            order = "reviewed_at_desc"
        else:
            order = "manifest"
        self.records = self.store.scan(
            status=selected_status, region=self.region_var.get(),
            issue_filter=self.issue_var.get(), query=self.query_var.get(),
            classification=("classified"
                            if selected_status == ReviewStatus.PENDING.value
                            else "all"),
            predicted_label=self.predicted_label_var.get(), order=order,
            ai_reviewed_only=(self.ai_reviewed_only and
                              selected_status not in {ReviewStatus.VERIFIED.value,
                                                      ReviewStatus.REJECTED.value}))
        total_pages = max(1, (len(self.records) + self.PAGE_SIZE - 1) // self.PAGE_SIZE)
        self.page = min(self.page, total_pages - 1)
        start = self.page * self.PAGE_SIZE
        self.page_records = self.records[start:start + self.PAGE_SIZE]
        for iid in self.tree.get_children():
            self.tree.delete(iid)
        for record in self.page_records:
            meta = record.metadata
            self.tree.insert("", "end", iid=record.sample_id, values=(
                record.sample_id, record.prediction_label or "",
                (f"{record.prediction_confidence:.3f}"
                 if record.prediction_confidence is not None else ""),
                " / ".join(str(meta.get(field) or "unknown") for field in (
                    "region", "position", "orientation")),
                meta.get("occurrence_count", 1),
                record.reviewed_at or "",
                record.status))
        counts = self.store.counts()
        prediction_counts = self.store.prediction_counts()
        self.progress_var.set(
            f"filter {len(self.records)} | number {self.page + 1}/{total_pages} page | "
            f"{'AI reviewed, pending manual confirmation' if self.ai_reviewed_only else 'All candidates'} | "
            f"classified {prediction_counts['classified']}/{len(self.store.samples)} | "
            f"pending {counts['pending']} | verified {counts['verified']} | "
            f"rejected {counts['rejected']}")
        if self.page_records:
            index = min(preferred_index or 0, len(self.page_records) - 1)
            selected = self.page_records[index]
            self.tree.selection_set(selected.sample_id)
            self.tree.see(selected.sample_id)
            self.show_record(selected)
        else:
            self.clear_record()

    def change_page(self, delta: int) -> None:
        if not self.records:
            return
        total_pages = (len(self.records) + self.PAGE_SIZE - 1) // self.PAGE_SIZE
        target = min(max(self.page + delta, 0), total_pages - 1)
        if target == self.page:
            return
        self.page = target
        self.refresh()

    def _select_tree(self, _event=None) -> None:
        selected = self.tree.selection()
        if not selected:
            return
        record = next((item for item in self.page_records
                       if item.sample_id == selected[0]), None)
        if record:
            self.show_record(record)

    def show_record(self, record: CropReviewRecord) -> None:
        self.current = record
        meta = record.metadata
        issues = ", ".join(meta.get("segmentation_issues") or []) or "None"
        audit = ""
        if record.status == ReviewStatus.VERIFIED.value:
            audit = (
                f"\n verified label={record.label}  annotator={record.reviewed_by}  "
                f"review/modified at={record.reviewed_at}")
        elif record.status == ReviewStatus.REJECTED.value:
            audit = f"\n rejection reason={record.rejection_reason}  annotator={record.reviewed_by}"
        ai_review = self.store.latest_ai_review(record.sample_id)
        if ai_review:
            decision = {"keep": "Candidates consistent", "correct": "Corrected candidate",
                        "uncertain": "uncertain"}[ai_review["decision"]]
            audit += f"\nAI verification: {decision}; {ai_review['reason']}"
            if record.status == ReviewStatus.PENDING.value:
                audit += "(still pending manual confirmation)"
        self.sample_info_var.set(
            f"{record.sample_id}  |  {meta.get('region')} / {meta.get('position')} / "
            f"{meta.get('orientation')}  |  status={record.status}\n"
            f"frame={meta.get('first_frame')}–{meta.get('last_frame')}  "
            f"time={meta.get('first_timestamp')}–{meta.get('last_timestamp')}s  "
            f"occurrences={meta.get('occurrence_count', 1)} times  |  "
            f"players={meta.get('player_count')} mode={meta.get('mode')}\n"
            f"segmentation issues: {issues}{audit}")
        self.label_var.set(record.prediction_label or "")
        confidence = (f"{record.prediction_confidence:.3f}"
                      if record.prediction_confidence is not None else "n/a")
        self.prediction_var.set(
            f"{record.prediction_label or 'Unclassified'}    confidence={confidence}\n"
            f"source: {record.prediction_model or 'None'}")
        pending = (record.status == ReviewStatus.PENDING.value and
                   record.prediction_label is not None)
        verified = record.status == ReviewStatus.VERIFIED.value
        rejected = record.status == ReviewStatus.REJECTED.value
        self.verify_button.configure(text="1 · Pass")
        self.reject_button.configure(
            text=("2 · Correct verified" if verified else
                  "2 · Correct rejected" if rejected else "2 · Reject / Correct"))
        self.verify_button.configure(state="normal" if pending else "disabled")
        self.reject_button.configure(
            state="normal" if pending or verified or rejected else "disabled")
        self._show_crop(record.path)
        self.photos.pop("context", None)
        self.context_label.configure(image="")
        self.context_info_var.set("Load source frame context on demand")

    def _show_crop(self, path: Path) -> None:
        from PIL import ImageTk
        if not path.is_file():
            self.crop_label.configure(image="", text=f"Missing image: {path}")
            self.photos.pop("crop", None)
            return
        photo = ImageTk.PhotoImage(_preview(path, (500, 500)))
        self.photos["crop"] = photo
        self.crop_label.configure(image=photo, text="")

    def load_context(self, frame_field: str) -> None:
        from tkinter import messagebox
        from PIL import ImageTk
        if not self.current:
            return
        frame_index = int(self.current.metadata[frame_field])
        self.root.configure(cursor="watch")
        self.root.update_idletasks()
        try:
            image = self.frame_reader.read(frame_index)
            image.thumbnail((880, 510), Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", (880, 510), "#20242a")
            canvas.paste(
                image, ((880 - image.width) // 2, (510 - image.height) // 2))
            photo = ImageTk.PhotoImage(canvas)
            self.photos["context"] = photo
            self.context_info_var.set("")
            self.context_label.configure(image=photo, text="")
        except (OSError, ValueError, KeyError) as exc:
            messagebox.showerror("Failed to load context", str(exc), parent=self.root)
        finally:
            self.root.configure(cursor="")

    def clear_record(self) -> None:
        self.current = None
        self.sample_info_var.set("No crops match the filter criteria")
        self.prediction_var.set("Waiting for reviewable candidates")
        self.crop_label.configure(image="", text="")
        self.context_label.configure(image="", text="Load source frame context on demand")
        self.photos.clear()
        self.verify_button.configure(state="disabled")
        self.reject_button.configure(state="disabled")

    def move(self, delta: int) -> None:
        if not self.current or not self.page_records:
            return
        index = self.page_records.index(self.current)
        target_index = index + delta
        if 0 <= target_index < len(self.page_records):
            target = self.page_records[target_index]
            self.tree.selection_set(target.sample_id)
            self.tree.see(target.sample_id)
            self.show_record(target)
        elif delta > 0:
            self.change_page(1)
        elif delta < 0:
            old_page = self.page
            self.change_page(-1)
            if self.page != old_page and self.page_records:
                target = self.page_records[-1]
                self.tree.selection_set(target.sample_id)
                self.tree.see(target.sample_id)
                self.show_record(target)

    def _review_hotkey(self, event, *, pass_result: bool):
        focused = self.root.focus_get()
        if focused is not None and focused.winfo_class() in {
                "Entry", "TEntry", "Text", "TCombobox"}:
            return None
        if pass_result:
            self.verify_label()
        else:
            self.reject_sample()
        return "break"

    def verify_label(self) -> None:
        from tkinter import messagebox
        if not self.current or self.current.status != ReviewStatus.PENDING.value:
            return
        label = self.current.prediction_label
        reviewer = self.reviewer_var.get().strip()
        if not label or not reviewer:
            messagebox.showerror(
                "Incomplete information", "The current sample has no candidate labels or the annotator is empty.",
                parent=self.root)
            return
        index = self.page_records.index(self.current)
        try:
            if label == "reject":
                self.store.reject(
                    self.current.sample_id, rejected_by=reviewer,
                    reason="deepseek_reject_confirmed_by_human")
            else:
                self.store.label(
                    self.current.sample_id, label, verified_by=reviewer)
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            messagebox.showerror("Failed to save", str(exc), parent=self.root)
            return
        self.refresh(preferred_index=index)

    def reject_sample(self) -> None:
        from tkinter import messagebox
        if (not self.current or self.current.status not in {
                ReviewStatus.PENDING.value, ReviewStatus.VERIFIED.value,
                ReviewStatus.REJECTED.value}):
            return
        reviewer = self.reviewer_var.get().strip()
        if not reviewer:
            messagebox.showerror(
                "Incomplete information", "The annotator must be filled in.", parent=self.root)
            return
        is_verified = self.current.status == ReviewStatus.VERIFIED.value
        is_rejected = self.current.status == ReviewStatus.REJECTED.value
        current_label = (
            self.current.label if is_verified else
            "reject" if is_rejected else self.current.prediction_label)
        corrected = self._ask_correction(
            current_label=current_label,
            source_name=("verified" if is_verified else
                         "rejected" if is_rejected else "Model candidates"))
        if corrected is None:
            return
        index = self.page_records.index(self.current)
        try:
            if is_verified:
                self.store.correct_verified(
                    self.current.sample_id, corrected, corrected_by=reviewer)
            elif is_rejected:
                self.store.correct_rejected(
                    self.current.sample_id, corrected, corrected_by=reviewer)
            elif corrected == "reject":
                self.store.reject(
                    self.current.sample_id, rejected_by=reviewer,
                    reason="deepseek_prediction_corrected_to_invalid_by_human")
            else:
                self.store.label(
                    self.current.sample_id, corrected, verified_by=reviewer)
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            messagebox.showerror("Correction failed", str(exc), parent=self.root)
            return
        if is_verified:
            # A correction updates reviewed_at, so it becomes the newest item.
            self.refresh(reset_page=True, preferred_index=0)
        else:
            self.refresh(preferred_index=index)

    def _ask_correction(self, *, current_label: str | None,
                        source_name: str) -> str | None:
        """Require an explicit corrected class after rejecting a prediction."""
        if self.current is None:
            return None
        dialog = self.tk.Toplevel(self.root)
        dialog.title(f"corrected {source_name} classification")
        dialog.transient(self.root)
        dialog.resizable(False, False)
        selected: dict[str, str | None] = {"value": None}

        self.ttk.Label(
            dialog,
            text=(f"{source_name} current label: {current_label}\n"
                  "Please select the correct label; it will only be written after the selection is confirmed."),
            justify=self.tk.CENTER,
            font=("Microsoft YaHei UI", 10, "bold"),
            padding=(12, 12, 12, 8)).pack(fill=self.tk.X)
        grid = self.ttk.Frame(dialog, padding=(10, 0, 10, 8))
        grid.pack(fill=self.tk.BOTH, expand=True)

        def choose(value: str) -> None:
            selected["value"] = value
            dialog.destroy()

        for index, (label, display) in enumerate(preset_labels("tile")):
            button = self.ttk.Button(
                grid, text=display, command=lambda value=label: choose(value))
            if label == current_label:
                button.configure(state="disabled")
            button.grid(
                row=index // 10, column=index % 10,
                sticky="ew", padx=2, pady=2)
        for column in range(10):
            grid.columnconfigure(column, weight=1)

        footer = self.ttk.Frame(dialog, padding=(10, 0, 10, 12))
        footer.pack(fill=self.tk.X)
        invalid = self.ttk.Button(
            footer, text="Invalid sample (not a usable tile face)",
            command=lambda: choose("reject"))
        if current_label == "reject":
            invalid.configure(state="disabled")
        invalid.pack(side=self.tk.LEFT, fill=self.tk.X, expand=True)
        self.ttk.Button(
            footer, text="Cancel", command=dialog.destroy).pack(
                side=self.tk.LEFT, padx=(8, 0))
        dialog.bind("<Escape>", lambda _event: dialog.destroy())
        dialog.protocol("WM_DELETE_WINDOW", dialog.destroy)
        dialog.grab_set()
        dialog.wait_window()
        return selected["value"]

    def show_verified_stats(self) -> None:
        """Show all tile classes, including classes with zero verified samples."""
        window = self.tk.Toplevel(self.root)
        window.title("Verified tile-type statistics")
        window.transient(self.root)
        window.geometry("430x680")
        summary = self.tk.StringVar()
        self.ttk.Label(
            window, textvariable=summary,
            font=("Microsoft YaHei UI", 11, "bold"),
            padding=(10, 10, 10, 6)).pack(fill=self.tk.X)

        frame = self.ttk.Frame(window, padding=(10, 0, 10, 10))
        frame.pack(fill=self.tk.BOTH, expand=True)
        tree = self.ttk.Treeview(
            frame, columns=("label", "name", "count"), show="headings")
        for column, title, width in (
                ("label", "Label", 60), ("name", "Tile type", 210),
                ("count", "Verified count", 110)):
            tree.heading(column, text=title)
            tree.column(column, width=width, anchor="center")
        scroll = self.ttk.Scrollbar(
            frame, orient=self.tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=scroll.set)
        tree.pack(side=self.tk.LEFT, fill=self.tk.BOTH, expand=True)
        scroll.pack(side=self.tk.RIGHT, fill=self.tk.Y)

        counts = self.store.verified_label_counts()
        for label, display in preset_labels("tile"):
            tree.insert("", "end", values=(
                label, display.removeprefix(f"{label} "), counts[label]))
        summary.set(
            f"total {sum(counts.values())} items verified samples · 37 tile classes(including 0)")

    def open_cases(self) -> None:
        window = self.tk.Toplevel(self.root)
        ReviewLabelerApp(
            window, self.cases_root, self.dataset_root,
            reviewer=self.reviewer_var.get().strip() or "human_gui",
            initial_kind="all")

    def close(self) -> None:
        self.frame_reader.close()
        self.root.destroy()


def run_corpus_label_gui(
        manifest: str | Path, dataset_root: str | Path, *,
        cases_root: str | Path = "artifacts/unknown_cases",
        reviewer: str = "human_gui", ai_reviewed_only: bool = False) -> None:
    import tkinter as tk
    root = tk.Tk()
    CorpusLabelerApp(
        root, manifest, dataset_root,
        cases_root=cases_root, reviewer=reviewer, ai_reviewed_only=ai_reviewed_only)
    root.mainloop()
