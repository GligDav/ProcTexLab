"""Development GUI for inspecting adaptive loss weights per pyramid band."""
from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageTk

from procedural_texture_kernel import LaplacianPyramid, WeightEstimator, load_image
from gui.decomposition_viewer import IMAGE_TYPES, preview_values


class WeightEstimatorViewer(tk.Tk):
    """Small diagnostic client; it is intentionally separate from the main GUI."""

    def __init__(self) -> None:
        super().__init__()
        self.title("Weight Estimator Developer Viewer")
        self.geometry("850x650")
        self.source: np.ndarray | None = None
        self.source_path: Path | None = None
        self.band_arrays: tuple[np.ndarray, ...] = ()
        self.pyramid: LaplacianPyramid | None = None
        self.photo: ImageTk.PhotoImage | None = None
        self.estimator = WeightEstimator()
        self.band_count = tk.IntVar(value=5)
        self.base_sigma = tk.DoubleVar(value=1.0)
        self.band_index = tk.IntVar(value=0)
        self._build_ui()

    def _build_ui(self) -> None:
        controls = ttk.Frame(self); controls.pack(fill="x", padx=10, pady=10)
        ttk.Button(controls, text="Load grayscale raster", command=self.load).pack(side="left")
        ttk.Label(controls, text="Bands").pack(side="left", padx=(15, 4))
        ttk.Spinbox(controls, from_=1, to=12, width=4,
                    textvariable=self.band_count).pack(side="left")
        ttk.Label(controls, text="Base sigma").pack(side="left", padx=(15, 4))
        ttk.Entry(controls, width=7, textvariable=self.base_sigma).pack(side="left")
        ttk.Button(controls, text="Generate", command=self.generate).pack(side="left", padx=10)

        selector = ttk.Frame(self); selector.pack(fill="x", padx=10)
        ttk.Label(selector, text="Selected band").pack(side="left")
        self.band_box = ttk.Combobox(selector, state="readonly", width=50)
        self.band_box.pack(side="left", padx=8, fill="x", expand=True)
        self.band_box.bind("<<ComboboxSelected>>", self.select_band)

        body = ttk.Frame(self); body.pack(fill="both", expand=True, padx=10, pady=10)
        self.preview = ttk.Label(body, anchor="center")
        self.preview.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        details = ttk.Frame(body); details.grid(row=0, column=1, sticky="nsew")
        body.columnconfigure(0, weight=3); body.columnconfigure(1, weight=2); body.rowconfigure(0, weight=1)

        feature_frame = ttk.LabelFrame(details, text="Band features")
        feature_frame.pack(fill="x", pady=(0, 10))
        self.feature_labels = self._value_rows(feature_frame,
            ("spectral_entropy", "spectral_anisotropy", "autocorrelation_strength",
             "gradient_coherence", "kurtosis", "raw_excess_kurtosis"))
        weight_frame = ttk.LabelFrame(details, text="Normalized loss weights")
        weight_frame.pack(fill="x")
        self.weight_labels = {}
        self.weight_bars = {}
        for row, name in enumerate(("spectrum", "histogram", "autocorrelation", "gradient")):
            ttk.Label(weight_frame, text=name.replace("_", " ").title()).grid(row=row, column=0, sticky="w", padx=5, pady=3)
            bar = ttk.Progressbar(weight_frame, maximum=1.0, length=150)
            bar.grid(row=row, column=1, padx=5)
            label = ttk.Label(weight_frame, width=8, anchor="e")
            label.grid(row=row, column=2, padx=5)
            self.weight_bars[name], self.weight_labels[name] = bar, label
        self.status = ttk.Label(self, text="Load an image to begin", anchor="w")
        self.status.pack(fill="x", padx=10, pady=(0, 10))

    @staticmethod
    def _value_rows(parent, names):
        labels = {}
        for row, name in enumerate(names):
            ttk.Label(parent, text=name.replace("_", " ").title()).grid(row=row, column=0, sticky="w", padx=5, pady=3)
            labels[name] = ttk.Label(parent, width=14, anchor="e")
            labels[name].grid(row=row, column=1, padx=5)
        return labels

    def load(self) -> None:
        path = filedialog.askopenfilename(filetypes=IMAGE_TYPES)
        if not path: return
        try:
            self.source = load_image(path)
        except ValueError as exc:
            messagebox.showerror("Load failed", str(exc)); return
        self.source_path = Path(path); self.generate()

    def generate(self) -> None:
        if self.source is None: return
        try:
            self.pyramid = LaplacianPyramid(self.band_count.get(), self.base_sigma.get())
            self.band_arrays = self.pyramid.decompose(self.source)
        except (ValueError, tk.TclError) as exc:
            self.status.configure(text=f"Invalid settings: {exc}"); return
        self.band_box["values"] = [self._band_name(i) for i in range(len(self.band_arrays))]
        self.band_box.current(0); self.show_band(0)

    def select_band(self, _event=None) -> None:
        self.show_band(self.band_box.current())

    def _band_name(self, index: int) -> str:
        assert self.pyramid is not None
        if self.pyramid.bands == 1: return "Band 1 — identity"
        if index == self.pyramid.bands - 1:
            return f"Band {index + 1} — low-pass sigma {self.pyramid.sigmas[-1]:g}px"
        low = "source" if index == 0 else f"sigma {self.pyramid.sigmas[index - 1]:g}"
        return f"Band {index + 1} — {low} to sigma {self.pyramid.sigmas[index]:g}px"

    def show_band(self, index: int) -> None:
        if not 0 <= index < len(self.band_arrays): return
        band = self.band_arrays[index]
        result = self.estimator.analyze(band)
        image = Image.fromarray(np.uint8(preview_values(
            band, signed=index < len(self.band_arrays) - 1) * 255), "L")
        image.thumbnail((480, 450)); self.photo = ImageTk.PhotoImage(image)
        self.preview.configure(image=self.photo)
        for name, value in result.features.to_dict().items():
            self.feature_labels[name].configure(text=f"{value:.6g}")
        for name, value in result.weights.to_dict().items():
            self.weight_bars[name]["value"] = value
            self.weight_labels[name].configure(text=f"{value:.4f}")
        self.status.configure(text=(f"{self.source_path.name if self.source_path else 'Raster'} | "
            f"band {index + 1}/{len(self.band_arrays)} | {band.shape[1]} x {band.shape[0]} | "
            f"weight sum {sum(result.weights.to_dict().values()):.12g}"))


def main() -> None:
    WeightEstimatorViewer().mainloop()


if __name__ == "__main__":
    main()
