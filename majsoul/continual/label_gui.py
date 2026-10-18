"""Desktop human-labeling UI for continual-perception review cases."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from ..actions import AgentActionType
from .schemas import ReviewKind, ReviewStatus
from .store import ReviewCaseStore


_HONOR_NAMES = ("East", "South", "West", "North", "White", "Green", "Red")
_SUIT_NAMES = {"m": "man", "p": "pin", "s": "sou"}


def tile_label_text(label: str) -> str:
    """Return a compact human-readable tile name for one internal label."""
    rank, suit = label[0], label[1]
    if suit == "z":
        return f"{label} {_HONOR_NAMES[int(rank) - 1]}"
    prefix = "red five" if rank == "0" else rank
    return f"{label} {prefix} {_SUIT_NAMES[suit]}"


def preset_labels(kind: str | ReviewKind) -> tuple[tuple[str, str], ...]:
    """Return domain labels for buttons; arbitrary tasks still use text input."""
    kind = ReviewKind(kind)
    if kind == ReviewKind.TILE:
        labels = []
        for suit in "mps":
            for rank in "123456789":
                label = rank + suit
                labels.append((label, tile_label_text(label)))
            red = "0" + suit
            labels.append((red, tile_label_text(red)))
        for rank in "1234567":
            label = rank + "z"
            labels.append((label, tile_label_text(label)))
        return tuple(labels)
    if kind == ReviewKind.ACTION:
        return tuple((action.value, action.value) for action in AgentActionType)
    if kind == ReviewKind.MELD:
        return tuple((label, label) for label in ("MELD", "CHI", "PON", "KAN"))
    return ()


@dataclass(frozen=True)
class ReviewRecord:
    case_id: str
    path: Path
    metadata: dict[str, Any]

    @property
    def status(self) -> str:
        return self.metadata["status"]

    @property
    def kind(self) -> str:
        return self.metadata["kind"]


class ReviewCaseIndex:
    """Read-only queue index used by both the GUI and headless tests."""

    def __init__(self, cases_root: str | Path):
        self.cases_root = Path(cases_root)
        self.errors: list[str] = []

    def scan(self, *, status: str = "pending", kind: str = "all") -> list[ReviewRecord]:
        if status not in {"all", *(item.value for item in ReviewStatus)}:
            raise ValueError(f"Unsupported status filter: {status}")
        if kind not in {"all", *(item.value for item in ReviewKind)}:
            raise ValueError(f"Unsupported kind filter: {kind}")
        self.errors = []
        records = []
        for metadata_path in sorted(self.cases_root.glob("case_*/metadata.json")):
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
                ReviewStatus(metadata["status"])
                ReviewKind(metadata["kind"])
                if metadata.get("case_id") != metadata_path.parent.name:
                    raise ValueError("case_id does not match directory name")
            except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
                self.errors.append(f"{metadata_path}: {exc}")
                continue
            if status != "all" and metadata["status"] != status:
                continue
            if kind != "all" and metadata["kind"] != kind:
                continue
            records.append(ReviewRecord(
                metadata["case_id"], metadata_path.parent, metadata))
        return records

    def counts(self) -> Counter:
        return Counter(record.status for record in self.scan(status="all"))


def _fit_preview(path: Path, size: tuple[int, int], *, upscale: bool) -> Image.Image:
    with Image.open(path) as source:
        image = source.convert("RGB")
    if not upscale:
        image.thumbnail(size, Image.Resampling.LANCZOS)
    else:
        image = ImageOps.contain(image, size, Image.Resampling.LANCZOS)
    background = Image.new("RGB", size, "#20242a")
    background.paste(
        image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
    return background


class ReviewLabelerApp:
    def __init__(self, root, cases_root: str | Path, dataset_root: str | Path, *,
                 reviewer: str = "human_gui", initial_kind: str = "tile"):
        import tkinter as tk
        from tkinter import ttk

        self.tk = tk
        self.ttk = ttk
        self.root = root
        self.cases_root = Path(cases_root)
        self.dataset_root = Path(dataset_root)
        self.store = ReviewCaseStore(self.cases_root, self.dataset_root)
        self.index = ReviewCaseIndex(self.cases_root)
        self.records: list[ReviewRecord] = []
        self.current: ReviewRecord | None = None
        self.photos: dict[str, Any] = {}
        self.context_name = "full_frame.png"

        self.status_var = tk.StringVar(value="pending")
        self.kind_var = tk.StringVar(value=initial_kind)
        self.reviewer_var = tk.StringVar(value=reviewer)
        self.label_var = tk.StringVar()
        self.reject_var = tk.StringVar()
        self.progress_var = tk.StringVar()
        self.case_info_var = tk.StringVar(value="No cases match the filter criteria")
        self.candidate_var = tk.StringVar()
        self.context_title_var = tk.StringVar(value="Full frame")

        root.title("Mahjong Soul Human-in-the-loop data labeler")
        root.minsize(1150, 720)
        root.geometry(
            f"{min(1680, root.winfo_screenwidth()-60)}x"
            f"{min(980, root.winfo_screenheight()-80)}")

        toolbar = ttk.Frame(root, padding=(10, 8))
        toolbar.pack(fill=tk.X)
        ttk.Label(toolbar, text="Status").pack(side=tk.LEFT)
        status_box = ttk.Combobox(
            toolbar, textvariable=self.status_var, state="readonly", width=10,
            values=("pending", "all", "verified", "rejected"))
        status_box.pack(side=tk.LEFT, padx=(5, 12))
        ttk.Label(toolbar, text="Type").pack(side=tk.LEFT)
        kind_box = ttk.Combobox(
            toolbar, textvariable=self.kind_var, state="readonly", width=12,
            values=("all", *(item.value for item in ReviewKind)))
        kind_box.pack(side=tk.LEFT, padx=(5, 12))
        ttk.Label(toolbar, text="Annotator").pack(side=tk.LEFT)
        ttk.Entry(toolbar, textvariable=self.reviewer_var, width=18).pack(
            side=tk.LEFT, padx=(5, 12))
        ttk.Button(toolbar, text="Refresh", command=self.refresh).pack(side=tk.LEFT)
        ttk.Label(toolbar, textvariable=self.progress_var).pack(side=tk.RIGHT)
        status_box.bind("<<ComboboxSelected>>", lambda _event: self.refresh())
        kind_box.bind("<<ComboboxSelected>>", lambda _event: self.refresh())

        main = ttk.Panedwindow(root, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        queue_frame = ttk.Frame(main, width=400)
        detail = ttk.Frame(main)
        main.add(queue_frame, weight=1)
        main.add(detail, weight=4)

        columns = ("case", "kind", "region", "status")
        self.tree = ttk.Treeview(
            queue_frame, columns=columns, show="headings", selectmode="browse")
        for name, title, width in (
                ("case", "Case", 110), ("kind", "Type", 70),
                ("region", "Region", 150), ("status", "Status", 75)):
            self.tree.heading(name, text=title)
            self.tree.column(name, width=width, stretch=name == "region")
        scroll = ttk.Scrollbar(queue_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.bind("<<TreeviewSelect>>", self._select_tree)

        ttk.Label(
            detail, textvariable=self.case_info_var, justify=tk.LEFT,
            font=("Microsoft YaHei UI", 10, "bold"), wraplength=1180).pack(
                fill=tk.X, pady=(0, 6))

        images = ttk.Frame(detail)
        images.pack(fill=tk.BOTH, expand=True)
        crop_frame = ttk.LabelFrame(images, text="Crops to be annotated", padding=6)
        context_frame = ttk.LabelFrame(images, textvariable=self.context_title_var, padding=6)
        crop_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False)
        context_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(8, 0))
        self.crop_label = ttk.Label(crop_frame, anchor=tk.CENTER)
        self.crop_label.pack(fill=tk.BOTH, expand=True)
        self.context_label = ttk.Label(context_frame, anchor=tk.CENTER)
        self.context_label.pack(fill=tk.BOTH, expand=True)
        context_nav = ttk.Frame(context_frame)
        context_nav.pack(fill=tk.X, pady=(6, 0))
        for text, filename in (("Previous frame", "previous_frame.png"),
                               ("Full frame", "full_frame.png"),
                               ("Next frame", "next_frame.png")):
            ttk.Button(
                context_nav, text=text,
                command=lambda name=filename: self.show_context(name)).pack(
                    side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        lower = ttk.Panedwindow(detail, orient=tk.HORIZONTAL)
        lower.pack(fill=tk.X, pady=(8, 0))
        candidate_frame = ttk.LabelFrame(lower, text="Reference info (cannot be auto-ingested)", padding=8)
        label_frame = ttk.LabelFrame(lower, text="Manual label", padding=8)
        lower.add(candidate_frame, weight=1)
        lower.add(label_frame, weight=2)
        ttk.Label(
            candidate_frame, textvariable=self.candidate_var, justify=tk.LEFT,
            wraplength=420).pack(fill=tk.BOTH, expand=True)

        self.preset_frame = ttk.Frame(label_frame)
        self.preset_frame.pack(fill=tk.X)
        manual = ttk.Frame(label_frame)
        manual.pack(fill=tk.X, pady=(8, 3))
        ttk.Label(manual, text="Label").pack(side=tk.LEFT)
        self.label_entry = ttk.Entry(manual, textvariable=self.label_var, width=22)
        self.label_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)
        self.verify_button = ttk.Button(
            manual, text="Confirm and ingest", command=self.verify_label)
        self.verify_button.pack(side=tk.LEFT)
        rejection = ttk.Frame(label_frame)
        rejection.pack(fill=tk.X, pady=3)
        ttk.Label(rejection, text="Rejection reason").pack(side=tk.LEFT)
        ttk.Entry(rejection, textvariable=self.reject_var).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=6)
        self.reject_button = ttk.Button(
            rejection, text="Reject sample", command=self.reject_case)
        self.reject_button.pack(side=tk.LEFT)
        navigation = ttk.Frame(label_frame)
        navigation.pack(fill=tk.X, pady=(5, 0))
        ttk.Button(navigation, text="← Previous", command=lambda: self.move(-1)).pack(
            side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(navigation, text="Skip / Next →", command=lambda: self.move(1)).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 0))

        help_text = (
            "Click a preset or enter an internal label, then click \"Confirm and ingest\"; model/AI candidates are not adopted automatically."
            "After confirmation, the sample enters the verified dataset and the training queue, but training does not start immediately."
            "Shortcuts: Ctrl+Enter to confirm, ←/→ to switch, F5 to refresh.")
        ttk.Label(root, text=help_text, foreground="#555555", padding=(10, 0, 10, 8)).pack(
            fill=tk.X)

        root.bind("<F5>", lambda _event: self.refresh())
        root.bind("<Left>", lambda _event: self.move(-1))
        root.bind("<Right>", lambda _event: self.move(1))
        root.bind("<Control-Return>", lambda _event: self.verify_label())
        self.refresh()

    def refresh(self, *, preferred_case: str | None = None):
        current_id = preferred_case or (self.current.case_id if self.current else None)
        self.records = self.index.scan(
            status=self.status_var.get(), kind=self.kind_var.get())
        existing = set(self.tree.get_children())
        wanted = {record.case_id for record in self.records}
        for iid in existing - wanted:
            self.tree.delete(iid)
        for record in self.records:
            values = (
                record.case_id, record.kind, record.metadata.get("region", ""),
                record.status)
            if record.case_id in existing:
                self.tree.item(record.case_id, values=values)
            else:
                self.tree.insert("", "end", iid=record.case_id, values=values)
        counts = self.index.counts()
        queue_count = self._training_queue_count()
        self.progress_var.set(
            f"filter {len(self.records)} | pending {counts['pending']} | "
            f"verified {counts['verified']} | rejected {counts['rejected']} | "
            f"training queue {queue_count}" +
            (f" | Read error {len(self.index.errors)}" if self.index.errors else ""))
        selected = next(
            (record for record in self.records if record.case_id == current_id),
            self.records[0] if self.records else None)
        if selected:
            self.tree.selection_set(selected.case_id)
            self.tree.see(selected.case_id)
            self.show_record(selected)
        else:
            self.clear_record()

    def _training_queue_count(self) -> int:
        path = self.dataset_root / "manifests" / "training_queue.json"
        if not path.exists():
            return 0
        try:
            value = json.loads(path.read_text(encoding="utf-8-sig"))
            return len(value) if isinstance(value, list) else 0
        except (OSError, json.JSONDecodeError):
            return 0

    def _select_tree(self, _event=None):
        selection = self.tree.selection()
        if not selection:
            return
        record = next(
            (item for item in self.records if item.case_id == selection[0]), None)
        if record:
            self.show_record(record)

    def show_record(self, record: ReviewRecord):
        self.current = record
        metadata = record.metadata
        self.case_info_var.set(
            f"{record.case_id}  |  {record.kind} / {metadata.get('region')}  |  "
            f"status={record.status}  |  frame={metadata.get('frame_index')}  "
            f"time={metadata.get('timestamp')}s\n"
            f"source={metadata.get('source')}  |  players={metadata.get('player_count')}  "
            f"mode={metadata.get('mode')}  |  reasons={', '.join(metadata.get('reasons') or [])}")
        self.candidate_var.set(self._format_candidates(metadata))
        self.label_var.set("")
        self.reject_var.set("")
        self._render_presets(record.kind)
        pending = record.status == ReviewStatus.PENDING.value
        self.verify_button.configure(state="normal" if pending else "disabled")
        self.reject_button.configure(state="normal" if pending else "disabled")
        self._show_image(record.path / "crop.png", crop=True)
        self.show_context(self.context_name)

    @staticmethod
    def _format_candidates(metadata: dict) -> str:
        lines = []
        suggestions = metadata.get("reviewer_suggestions") or []
        if suggestions:
            latest = suggestions[-1]
            lines.append("AI suggestion (reference only):" + ", ".join(latest.get("labels") or []))
            if latest.get("note"):
                lines.append("Basis:" + latest["note"])
        lines.append("Model candidate:")
        candidates = metadata.get("candidates") or []
        if candidates:
            lines.extend(
                f"  {item['label']}  {float(item['score']):.3f}" for item in candidates)
        else:
            lines.append("  None")
        if metadata.get("label"):
            lines.append("Verified label:" + metadata["label"])
        if metadata.get("rejection_reason"):
            lines.append("Rejection reason:" + metadata["rejection_reason"])
        return "\n".join(lines)

    def _render_presets(self, kind: str):
        for child in self.preset_frame.winfo_children():
            child.destroy()
        labels = preset_labels(kind)
        if not labels:
            self.ttk.Label(
                self.preset_frame,
                text="This type has no fixed categories; enter the confirmed label below.",
                foreground="#555555").grid(row=0, column=0, sticky="w")
            return
        columns = 10 if kind == ReviewKind.TILE.value else 5
        for index, (label, display) in enumerate(labels):
            button = self.ttk.Button(
                self.preset_frame, text=display,
                command=lambda value=label: self.label_var.set(value))
            button.grid(
                row=index // columns, column=index % columns,
                sticky="ew", padx=2, pady=2)
        for column in range(columns):
            self.preset_frame.columnconfigure(column, weight=1)

    def _show_image(self, path: Path, *, crop: bool):
        target = self.crop_label if crop else self.context_label
        key = "crop" if crop else "context"
        if not path.is_file():
            target.configure(image="", text=f"Missing image: {path.name}")
            self.photos.pop(key, None)
            return
        from PIL import ImageTk
        preview = _fit_preview(
            path, (360, 360) if crop else (850, 480), upscale=crop)
        photo = ImageTk.PhotoImage(preview)
        self.photos[key] = photo
        target.configure(image=photo, text="")

    def show_context(self, filename: str):
        self.context_name = filename
        titles = {
            "previous_frame.png": "Previous frame",
            "full_frame.png": "Full frame",
            "next_frame.png": "Next frame",
        }
        self.context_title_var.set(titles[filename])
        if self.current:
            self._show_image(self.current.path / filename, crop=False)

    def clear_record(self):
        self.current = None
        self.case_info_var.set("No cases match the filter criteria")
        self.candidate_var.set("")
        self.crop_label.configure(image="", text="")
        self.context_label.configure(image="", text="")
        self.photos.clear()
        self.verify_button.configure(state="disabled")
        self.reject_button.configure(state="disabled")
        self._render_presets(ReviewKind.METADATA.value)

    def move(self, delta: int):
        if not self.records:
            return
        if self.current not in self.records:
            target = self.records[0]
        else:
            index = self.records.index(self.current)
            target = self.records[(index + delta) % len(self.records)]
        self.tree.selection_set(target.case_id)
        self.tree.see(target.case_id)
        self.show_record(target)

    def verify_label(self):
        from tkinter import messagebox
        if not self.current or self.current.status != ReviewStatus.PENDING.value:
            return
        label = self.label_var.get().strip()
        reviewer = self.reviewer_var.get().strip()
        if not label or not reviewer:
            messagebox.showerror(
                "Incomplete information", "Label and annotator are required.", parent=self.root)
            return
        if not messagebox.askyesno(
                "Confirm manual label",
                f"{self.current.case_id}\n label: {label}\n\n"
                "After confirmation it will enter the verified dataset and the training queue. Continue?",
                parent=self.root):
            return
        current_id = self.current.case_id
        try:
            self.store.label(current_id, label, verified_by=reviewer)
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            messagebox.showerror("Failed to save", str(exc), parent=self.root)
            return
        self.refresh()

    def reject_case(self):
        from tkinter import messagebox
        if not self.current or self.current.status != ReviewStatus.PENDING.value:
            return
        reason = self.reject_var.get().strip()
        reviewer = self.reviewer_var.get().strip()
        if not reason or not reviewer:
            messagebox.showerror(
                "Incomplete information", "Rejection reason and annotator are required.", parent=self.root)
            return
        if not messagebox.askyesno(
                "Confirm rejected sample",
                f"{self.current.case_id}\n reason: {reason}\n\n"
                "Rejected items will not enter the training set. Continue?",
                parent=self.root):
            return
        try:
            self.store.reject(
                self.current.case_id, rejected_by=reviewer, reason=reason)
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            messagebox.showerror("Rejection failed", str(exc), parent=self.root)
            return
        self.refresh()


def run_label_gui(cases_root: str | Path = "artifacts/unknown_cases",
                  dataset_root: str | Path = "datasets", *,
                  reviewer: str = "human_gui", initial_kind: str = "tile"):
    import tkinter as tk

    root = tk.Tk()
    try:
        ReviewLabelerApp(
            root, cases_root, dataset_root,
            reviewer=reviewer, initial_kind=initial_kind)
    except Exception:
        root.destroy()
        raise
    root.mainloop()
