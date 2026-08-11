"""Tkinter development application for the public kernel API."""
from __future__ import annotations
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
from PIL import Image, ImageTk
from procedural_texture_kernel import (FitConfig, SUPPORTED_COMPONENT_FAMILIES,
                                       TextureFitter, load_image)

ATOM_LABELS = {
    "sinusoid": "Sinusoid", "gabor": "Gabor", "gaussian_rbf": "Gaussian RBF",
    "perlin_noise": "Perlin noise", "wavelet": "Wavelet",
    "voronoi_noise": "Voronoi noise", "fbm": "Fractal Brownian motion (fBm)",
    "ridged_multifractal": "Ridged multifractal", "turbulence_noise": "Turbulence noise",
    "domain_warped_noise": "Domain-warped noise",
    "anisotropic_gaussian": "Anisotropic Gaussian", "line": "Line / ridge / bar",
    "step_edge": "Step / sigmoid edge", "dog_log": "DoG / LoG",
    "polynomial_trend": "Polynomial trend", "radial_wave": "Radial wave",
    "spiral_wave": "Spiral wave", "sparse_impulse": "Sparse impulse / spot",
    "binary_primitive": "Binary primitives",
}

class TestApplication(tk.Tk):
    """Small visual harness; fitting runs on a worker thread."""
    def __init__(self):
        super().__init__(); self.title("Procedural Texture Kernel"); self.geometry("1100x720")
        self.source = None; self.result = None; self.images = [None, None, None]
        self.events = queue.Queue(); self.running = False; self.extent_job = None
        controls = ttk.Frame(self); controls.pack(fill="x", padx=8, pady=8)
        self.components = tk.IntVar(value=8); self.iterations = tk.IntVar(value=40)
        self.resolution = tk.IntVar(value=96); self.seed = tk.IntVar(value=0)
        self.min_improvement = tk.DoubleVar(value=1e-6)
        for label, variable in (("Components", self.components), ("Iterations", self.iterations),
                                ("Fit resolution", self.resolution), ("Seed", self.seed),
                                ("Min improvement", self.min_improvement)):
            ttk.Label(controls, text=label).pack(side="left", padx=(8,2))
            ttk.Entry(controls, textvariable=variable, width=6).pack(side="left")
        ttk.Button(controls, text="Load Image", command=self.load).pack(side="left", padx=8)
        self.fit_button = ttk.Button(controls, text="Fit", command=self.fit); self.fit_button.pack(side="left")
        weight_controls = ttk.LabelFrame(self, text="Texture loss weights")
        weight_controls.pack(fill="x", padx=16, pady=(0, 4))
        self.spectrum_weight = tk.DoubleVar(value=1.0)
        self.histogram_weight = tk.DoubleVar(value=0.5)
        self.autocorrelation_weight = tk.DoubleVar(value=0.75)
        self.gradient_weight = tk.DoubleVar(value=0.5)
        for label, variable in (("Spectrum", self.spectrum_weight),
                                ("Histogram", self.histogram_weight),
                                ("Autocorrelation", self.autocorrelation_weight),
                                ("Gradient", self.gradient_weight)):
            ttk.Label(weight_controls, text=label).pack(side="left", padx=(12, 2))
            ttk.Entry(weight_controls, textvariable=variable, width=8).pack(side="left")
        atom_controls = ttk.LabelFrame(self, text="Allowed procedural atoms")
        atom_controls.pack(fill="x", padx=16, pady=(0, 4))
        self.atom_enabled = {
            family: tk.BooleanVar(value=True) for family in SUPPORTED_COMPONENT_FAMILIES
        }
        columns = 5
        for index, family in enumerate(SUPPORTED_COMPONENT_FAMILIES):
            label = ATOM_LABELS.get(family, family.replace("_", " ").title())
            ttk.Checkbutton(atom_controls, text=label,
                            variable=self.atom_enabled[family]).grid(
                                row=index // columns, column=index % columns,
                                sticky="w", padx=10, pady=2)
        extent_controls = ttk.Frame(self); extent_controls.pack(fill="x", padx=16)
        ttk.Label(extent_controls, text="Result UV extent").pack(side="left")
        self.extent = tk.DoubleVar(value=1.0)
        self.extent_value = ttk.Label(extent_controls, text="1.0×  ([0, 1)²)", width=18)
        self.extent_value.pack(side="right")
        ttk.Scale(extent_controls, from_=1.0, to=4.0, variable=self.extent,
                  command=self._schedule_extent_preview).pack(side="left", fill="x", expand=True, padx=8)
        views = ttk.Frame(self); views.pack(fill="both", expand=True)
        self.labels = []
        for title in ("Source", "Procedural continuation", "Residual (original domain, contrast scaled)"):
            frame = ttk.LabelFrame(views, text=title); frame.pack(side="left", fill="both", expand=True, padx=4)
            label = ttk.Label(frame, anchor="center"); label.pack(fill="both", expand=True); self.labels.append(label)
        self.progress = ttk.Progressbar(self, maximum=100); self.progress.pack(fill="x", padx=8)
        self.status = ttk.Label(self, text="Load an image to begin"); self.status.pack(fill="x", padx=8, pady=8)

    def _show(self, array, slot, residual=False):
        values = np.asarray(array, float)
        if residual:
            limit = max(float(np.max(np.abs(values))), 1e-12); values = .5 + .5 * values / limit
        values = np.clip(values, 0, 1)
        image = Image.fromarray(np.uint8(values * 255), "L"); image.thumbnail((350, 470))
        photo = ImageTk.PhotoImage(image); self.images[slot] = photo; self.labels[slot].configure(image=photo)

    def _schedule_extent_preview(self, _value=None):
        extent = self.extent.get()
        self.extent_value.configure(text=f"{extent:.1f}×  ([0, {extent:.1f})²)")
        if self.extent_job is not None:
            self.after_cancel(self.extent_job)
        # Avoid evaluating a large atom list for every intermediate slider event.
        self.extent_job = self.after(150, self._update_extent_preview)

    def _update_extent_preview(self):
        self.extent_job = None
        if self.result is None or self.source is None:
            return
        extent = self.extent.get()
        height, width = self.source.shape
        scale = min(1.0, 350.0 / max(width, height))
        preview_width = max(1, round(width * scale))
        preview_height = max(1, round(height * scale))
        continuation = self.result.evaluate_region(
            preview_width, preview_height, (0.0, extent), (0.0, extent)
        )
        self._show(continuation, 1)

    def load(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg *.tif *.tiff *.bmp"), ("All", "*.*")])
        if not path: return
        try: self.source = load_image(path)
        except ValueError as exc: messagebox.showerror("Load failed", str(exc)); return
        self.result = None; self.extent.set(1.0); self._show(self.source, 0)
        self.extent_value.configure(text="1.0×  ([0, 1)²)")
        self.status.configure(text=f"Loaded {self.source.shape[1]} × {self.source.shape[0]}")

    def fit(self):
        if self.source is None or self.running: return
        try:
            config = self._build_config()
        except (ValueError, tk.TclError) as exc: messagebox.showerror("Invalid settings", str(exc)); return
        self.running = True; self.fit_button.state(["disabled"])
        def worker():
            try:
                result = TextureFitter(config).fit(self.source,
                    lambda stage, value, msg: self.events.put(("progress", value, msg)))
                self.events.put(("result", result))
            except Exception as exc: self.events.put(("error", exc))
        threading.Thread(target=worker, daemon=True).start(); self.after(50, self._poll)

    def _build_config(self) -> FitConfig:
        """Read and validate all editable fitting controls."""
        families = tuple(family for family in SUPPORTED_COMPONENT_FAMILIES
                         if self.atom_enabled[family].get())
        if not families:
            raise ValueError("select at least one procedural atom family")
        return FitConfig(seed=self.seed.get(), max_components=self.components.get(),
                         max_iterations=self.iterations.get(), fitting_resolution=self.resolution.get(),
                         component_families=families,
                         min_improvement=self.min_improvement.get(),
                         spectrum_weight=self.spectrum_weight.get(),
                         histogram_weight=self.histogram_weight.get(),
                         autocorrelation_weight=self.autocorrelation_weight.get(),
                         gradient_weight=self.gradient_weight.get())

    def _poll(self):
        try:
            while True:
                event = self.events.get_nowait()
                if event[0] == "progress": self.progress["value"] = event[1] * 100; self.status.configure(text=event[2])
                elif event[0] == "result":
                    result = event[1]; self.result = result
                    self._update_extent_preview(); self._show(result.residual, 2, True)
                    m = result.metrics
                    self.status.configure(text=f"Texture loss {m['texture_loss']:.6f}   RMSE {m['rmse']:.5f} (diagnostic)   PSNR {m['psnr']:.2f} dB")
                    self.running = False; self.fit_button.state(["!disabled"])
                else:
                    messagebox.showerror("Fit failed", str(event[1])); self.running = False; self.fit_button.state(["!disabled"])
        except queue.Empty: pass
        if self.running: self.after(50, self._poll)

def main(): TestApplication().mainloop()
if __name__ == "__main__": main()
