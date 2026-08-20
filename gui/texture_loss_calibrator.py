"""Development GUI for calibrating composite texture-loss weights."""
from __future__ import annotations

import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageTk

from procedural_texture_kernel import TextureLossWeights, calculate_texture_loss, load_image


IMAGE_TYPES = [("Images", "*.png *.jpg *.jpeg *.tif *.tiff *.bmp"), ("All files", "*.*")]
WEIGHTS = (
    ("Spectrum", "spectrum", "spectrum_loss", 1.0),
    ("Absolute spectrum", "absolute_spectrum", "absolute_spectrum_loss", 0.25),
    ("Oriented spectrum", "oriented_spectrum", "oriented_spectrum_loss", 0.25),
    ("Histogram", "histogram", "histogram_loss", 0.5),
    ("Autocorrelation", "autocorrelation", "autocorrelation_loss", 0.75),
    ("Gradient", "gradient", "gradient_loss", 0.5),
    ("MSE", "mse", "mse_loss", 1.0),
    ("Local structure", "local_structure", "local_structure_loss", 0.0),
    ("Local contrast", "local_contrast", "local_contrast_loss", 0.0),
)


def evaluate_images(reference: np.ndarray, candidate: np.ndarray,
                    weights: TextureLossWeights) -> dict[str, float]:
    """Validate a pair and evaluate their composite texture loss."""
    if reference.shape != candidate.shape:
        raise ValueError(
            "images must have the same dimensions "
            f"(got {reference.shape[1]} x {reference.shape[0]} and "
            f"{candidate.shape[1]} x {candidate.shape[0]})"
        )
    return calculate_texture_loss(reference, candidate, weights)


class TextureLossCalibrator(tk.Tk):
    """Compare two rasters while interactively tuning texture-loss weights."""

    def __init__(self) -> None:
        super().__init__()
        self.title("Texture Loss Weight Calibrator")
        self.geometry("1050x720")
        self.minsize(760, 560)
        self.arrays: list[np.ndarray | None] = [None, None]
        self.photos: list[ImageTk.PhotoImage | None] = [None, None]
        self.events: queue.Queue = queue.Queue()
        self.pending: set[int] = set()
        self.revision = 0
        self.recalculate_job: str | None = None
        self.weight_vars = {key: tk.StringVar(value=str(default))
                            for _, key, _, default in WEIGHTS}
        self.value_labels: dict[str, ttk.Label] = {}
        self.contribution_labels: dict[str, ttk.Label] = {}
        self._build_ui()

    def _build_ui(self) -> None:
        ttk.Label(
            self,
            text=("Load a reference and candidate raster with identical dimensions. "
                  "Images are compared as normalized grayscale."),
        ).pack(fill="x", padx=12, pady=(12, 6))
        views = ttk.Frame(self)
        views.pack(fill="both", expand=True, padx=8)
        self.preview_labels: list[ttk.Label] = []
        self.path_labels: list[ttk.Label] = []
        for index, title in enumerate(("Reference image", "Candidate image")):
            frame = ttk.LabelFrame(views, text=title)
            frame.pack(side="left", fill="both", expand=True, padx=4)
            preview = ttk.Label(frame, text="No image loaded", anchor="center")
            preview.pack(fill="both", expand=True, padx=6, pady=6)
            self.preview_labels.append(preview)
            path_label = ttk.Label(frame, anchor="center")
            path_label.pack(fill="x", padx=6)
            self.path_labels.append(path_label)
            ttk.Button(frame, text=f"Load {title}",
                       command=lambda slot=index: self.load(slot)).pack(pady=(4, 8))

        controls = ttk.LabelFrame(self, text="Texture loss weights")
        controls.pack(fill="x", padx=12, pady=8)
        for column, heading in enumerate(("Component", "Weight", "Raw loss", "Weighted term")):
            ttk.Label(controls, text=heading).grid(row=0, column=column, padx=8, pady=4)
        for row, (label, key, loss_key, _) in enumerate(WEIGHTS, start=1):
            ttk.Label(controls, text=label).grid(row=row, column=0, sticky="w", padx=8, pady=3)
            entry = ttk.Entry(controls, textvariable=self.weight_vars[key], width=12)
            entry.grid(row=row, column=1, padx=8, pady=3)
            entry.bind("<KeyRelease>", self._schedule_calculation)
            raw = ttk.Label(controls, text="--", width=18, anchor="e")
            raw.grid(row=row, column=2, padx=8, pady=3)
            self.value_labels[loss_key] = raw
            contribution = ttk.Label(controls, text="--", width=18, anchor="e")
            contribution.grid(row=row, column=3, padx=8, pady=3)
            self.contribution_labels[key] = contribution
        controls.columnconfigure(4, weight=1)

        result = ttk.Frame(self)
        result.pack(fill="x", padx=12, pady=(2, 6))
        ttk.Label(result, text="Final texture loss:").pack(side="left")
        self.total_label = ttk.Label(result, text="--", font=("TkDefaultFont", 16, "bold"))
        self.total_label.pack(side="left", padx=8)
        ttk.Button(result, text="Recalculate", command=self.recalculate).pack(side="right")
        self.status = ttk.Label(self, text="Load two images to begin", anchor="w")
        self.status.pack(fill="x", padx=12, pady=(0, 10))

    def load(self, slot: int) -> None:
        path = filedialog.askopenfilename(filetypes=IMAGE_TYPES)
        if not path:
            return
        try:
            array = load_image(path)
        except ValueError as exc:
            messagebox.showerror("Load failed", str(exc))
            return
        other = self.arrays[1 - slot]
        if other is not None and array.shape != other.shape:
            messagebox.showerror(
                "Image dimensions differ",
                "Both images must have the same dimensions.\n\n"
                f"Selected: {array.shape[1]} x {array.shape[0]}\n"
                f"Loaded: {other.shape[1]} x {other.shape[0]}",
            )
            return
        self.arrays[slot] = array
        self._show(array, slot)
        self.path_labels[slot].configure(
            text=f"{Path(path).name}  ({array.shape[1]} x {array.shape[0]})")
        self.recalculate()

    def _show(self, array: np.ndarray, slot: int) -> None:
        image = Image.fromarray(np.uint8(np.clip(array, 0, 1) * 255), "L")
        image.thumbnail((480, 350))
        photo = ImageTk.PhotoImage(image)
        self.photos[slot] = photo
        self.preview_labels[slot].configure(image=photo, text="")

    def _schedule_calculation(self, _event=None) -> None:
        if self.recalculate_job is not None:
            self.after_cancel(self.recalculate_job)
        self.recalculate_job = self.after(250, self.recalculate)

    def _read_weights(self) -> TextureLossWeights:
        try:
            values = {key: float(variable.get()) for key, variable in self.weight_vars.items()}
        except ValueError as exc:
            raise ValueError("weights must be numbers") from exc
        return TextureLossWeights(**values)

    def recalculate(self) -> None:
        self.recalculate_job = None
        # Invalidate any in-flight result, including when the new input is incomplete.
        self.revision += 1
        if any(array is None for array in self.arrays):
            self.status.configure(text="Load two images to begin")
            return
        try:
            weights = self._read_weights()
        except ValueError as exc:
            self.status.configure(text=f"Invalid weights: {exc}")
            self.total_label.configure(text="--")
            return
        revision = self.revision
        reference, candidate = self.arrays
        self.pending.add(revision)
        self.status.configure(text="Calculating texture loss...")

        def worker() -> None:
            try:
                metrics = evaluate_images(reference, candidate, weights)  # type: ignore[arg-type]
                self.events.put((revision, "result", metrics, weights))
            except Exception as exc:
                self.events.put((revision, "error", exc, weights))

        threading.Thread(target=worker, daemon=True).start()
        self.after(25, self._poll)

    def _poll(self) -> None:
        try:
            while True:
                revision, kind, payload, weights = self.events.get_nowait()
                self.pending.discard(revision)
                if revision != self.revision:
                    continue
                if kind == "error":
                    self.status.configure(text=f"Calculation failed: {payload}")
                else:
                    self._display_result(payload, weights)
        except queue.Empty:
            pass
        if self.pending:
            self.after(25, self._poll)

    def _display_result(self, metrics: dict[str, float], weights: TextureLossWeights) -> None:
        self.total_label.configure(text=f"{metrics['texture_loss']:.8g}")
        for _, key, loss_key, _ in WEIGHTS:
            raw = metrics[loss_key]
            self.value_labels[loss_key].configure(text=f"{raw:.8g}")
            self.contribution_labels[key].configure(text=f"{getattr(weights, key) * raw:.8g}")
        self.status.configure(text="Texture loss is the sum of weighted terms divided by total weight")


def main() -> None:
    TextureLossCalibrator().mainloop()


if __name__ == "__main__":
    main()
