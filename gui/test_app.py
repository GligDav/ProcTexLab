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
from .spectral_diagnostics_viewer import SpectralDiagnosticsDialog
from .measurement_diagnostics_viewer import MeasurementDiagnosticsDialog

ATOM_LABELS = {
    "sinusoid": "Sinusoid", "gabor": "Gabor", "gaussian_rbf": "Gaussian RBF",
    "spectral_noise": "Spectral noise bundle",
    "perlin_noise": "Perlin noise", "thresholded_noise": "Thresholded noise",
    "masked_noise": "Masked region detail", "wavelet": "Wavelet",
    "shader_graph": "Shader graph region mix",
    "voronoi_noise": "Voronoi noise", "fbm": "Fractal Brownian motion (fBm)",
    "ridged_multifractal": "Ridged multifractal", "turbulence_noise": "Turbulence noise",
    "domain_warped_noise": "Domain-warped noise",
    "warped_ridged_multifractal": "Warped ridged multifractal",
    "warped_ridge_detail": "Warped ridge-conditioned detail",
    "anisotropic_gaussian": "Anisotropic Gaussian", "line": "Line / ridge / bar",
    "step_edge": "Step / sigmoid edge", "dog_log": "DoG / LoG",
    "polynomial_trend": "Polynomial trend", "radial_wave": "Radial wave",
    "spiral_wave": "Spiral wave", "sparse_impulse": "Sparse impulse / spot",
    "binary_primitive": "Binary primitives", "simple_constant": "Simple constant",
}
DEFAULT_DECOMPOSITION_BANDS = 5

class TestApplication(tk.Tk):
    """Small visual harness; fitting runs on a worker thread."""
    def __init__(self):
        super().__init__(); self.title("Procedural Texture Kernel"); self.geometry("1100x720")
        self.attributes("-fullscreen", True)
        self.bind("<Escape>", lambda _event: self.attributes("-fullscreen", False))
        self.bind("<F11>", self._toggle_fullscreen)
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.source = None; self.result = None; self.images = [None, None, None]
        self.events = queue.Queue(); self.running = False; self.extent_job = None
        self.cancel_event = threading.Event(); self.worker_thread = None
        self.closing = False
        controls = ttk.Frame(self); controls.pack(fill="x", padx=8, pady=8)
        self.components = tk.IntVar(value=12); self.iterations = tk.IntVar(value=60)
        self.resolution = tk.IntVar(value=192); self.seed = tk.IntVar(value=0)
        self.noise_seed_candidates = tk.IntVar(value=4)
        self.spectral_noise_modes = tk.IntVar(value=32)
        self.max_frequency = tk.DoubleVar(value=0.0)
        self.decomposition_bands = tk.IntVar(value=DEFAULT_DECOMPOSITION_BANDS)
        self.band_workers = tk.IntVar(value=3)
        self.candidate_workers = tk.IntVar(value=1)
        self.compute_backend = tk.StringVar(value="numpy")
        self.gpu_batch_size = tk.IntVar(value=16)
        self.min_improvement = tk.DoubleVar(value=1e-6)
        for label, variable in (("Components", self.components), ("Iterations", self.iterations),
                                ("Fit resolution", self.resolution),
                                ("Bands", self.decomposition_bands), ("Seed", self.seed),
                                ("Noise seeds", self.noise_seed_candidates),
                                ("Max freq (0=auto)", self.max_frequency),
                                ("Min improvement", self.min_improvement)):
            ttk.Label(controls, text=label).pack(side="left", padx=(8,2))
            ttk.Entry(controls, textvariable=variable, width=6).pack(side="left")
        ttk.Button(controls, text="Load Image", command=self.load).pack(side="left", padx=8)
        self.fit_button = ttk.Button(controls, text="Fit", command=self.fit); self.fit_button.pack(side="left")
        self.cancel_button = ttk.Button(controls, text="Cancel Fit", command=self.cancel_fit)
        self.cancel_button.pack(side="left", padx=(4, 0)); self.cancel_button.state(["disabled"])
        self.export_button = ttk.Button(controls, text="Export JSON", command=self.export_json)
        self.export_button.pack(side="left", padx=(4, 0)); self.export_button.state(["disabled"])
        self.spectrum_button = ttk.Button(controls, text="Spectrum", command=self.show_spectrum)
        self.spectrum_button.pack(side="left", padx=(8, 0)); self.spectrum_button.state(["disabled"])
        self.measurements_button = ttk.Button(
            controls, text="Measurements", command=self.show_measurements)
        self.measurements_button.pack(side="left", padx=(4, 0))
        self.measurements_button.state(["disabled"])
        weight_controls = ttk.LabelFrame(self, text="Texture loss weights")
        weight_controls.pack(fill="x", padx=16, pady=(0, 4))
        self.adaptive_weights = tk.BooleanVar(value=True)
        ttk.Checkbutton(weight_controls, text="Estimate statistical weights per band",
                        variable=self.adaptive_weights,
                        command=self._update_weight_controls).pack(
                            anchor="w", padx=8, pady=(2, 0))
        self.spectrum_weight = tk.DoubleVar(value=1.0)
        self.histogram_weight = tk.DoubleVar(value=0.5)
        self.autocorrelation_weight = tk.DoubleVar(value=0.75)
        self.gradient_weight = tk.DoubleVar(value=0.5)
        self.mse_weight = tk.DoubleVar(value=1.0)
        self.local_structure_weight = tk.DoubleVar(value=0.5)
        self.local_contrast_weight = tk.DoubleVar(value=0.5)
        self.absolute_spectrum_weight = tk.DoubleVar(value=0.25)
        self.oriented_spectrum_weight = tk.DoubleVar(value=0.25)
        self.statistical_weight_entries = []
        weight_grid = ttk.Frame(weight_controls)
        weight_grid.pack(fill="x", padx=4, pady=(0, 4))
        for index, (label, variable) in enumerate((("Spectrum", self.spectrum_weight),
                                ("Absolute spectrum", self.absolute_spectrum_weight),
                                ("Oriented spectrum", self.oriented_spectrum_weight),
                                ("Histogram", self.histogram_weight),
                                ("Autocorrelation", self.autocorrelation_weight),
                                ("Gradient", self.gradient_weight),
                                ("MSE", self.mse_weight),
                                ("Local structure", self.local_structure_weight),
                                ("Local contrast", self.local_contrast_weight))):
            cell = ttk.Frame(weight_grid)
            cell.grid(row=index // 5, column=index % 5, sticky="w", padx=6, pady=2)
            ttk.Label(cell, text=label).pack(side="left", padx=(0, 2))
            entry = ttk.Entry(cell, textvariable=variable, width=8)
            entry.pack(side="left")
            if variable not in (self.mse_weight, self.absolute_spectrum_weight,
                                self.oriented_spectrum_weight,
                                self.local_structure_weight,
                                self.local_contrast_weight):
                self.statistical_weight_entries.append(entry)
        self._update_weight_controls()
        detail_controls = ttk.LabelFrame(self, text="High-frequency residual refinement")
        detail_controls.pack(fill="x", padx=16, pady=(0, 4))
        self.detail_refinement = tk.BooleanVar(value=True)
        self.detail_components = tk.IntVar(value=12)
        self.detail_min_frequency = tk.DoubleVar(value=12.0)
        self.detail_hf_threshold = tk.DoubleVar(value=.85)
        ttk.Checkbutton(detail_controls, text="Enable when HF ratio is below threshold",
                        variable=self.detail_refinement).pack(side="left", padx=8)
        for label, variable in (("Detail atoms", self.detail_components),
                                ("Min frequency", self.detail_min_frequency),
                                ("HF threshold", self.detail_hf_threshold)):
            ttk.Label(detail_controls, text=label).pack(side="left", padx=(12, 2))
            ttk.Entry(detail_controls, textvariable=variable, width=7).pack(side="left")
        ttk.Label(detail_controls, text="Band workers").pack(
            side="left", padx=(12, 2))
        ttk.Entry(detail_controls, textvariable=self.band_workers,
                  width=4).pack(side="left")
        ttk.Label(detail_controls, text="Candidate workers").pack(
            side="left", padx=(12, 2))
        ttk.Entry(detail_controls, textvariable=self.candidate_workers,
                  width=4).pack(side="left")
        ttk.Label(detail_controls, text="Backend").pack(side="left", padx=(12, 2))
        ttk.Combobox(detail_controls, textvariable=self.compute_backend,
                     values=("numpy", "cupy"), state="readonly", width=7).pack(side="left")
        ttk.Label(detail_controls, text="GPU batch").pack(side="left", padx=(12, 2))
        ttk.Entry(detail_controls, textvariable=self.gpu_batch_size,
                  width=4).pack(side="left")
        ttk.Label(detail_controls, text="Spectral modes").pack(
            side="left", padx=(12, 2))
        ttk.Entry(detail_controls, textvariable=self.spectral_noise_modes,
                  width=5).pack(side="left")
        refit_controls = ttk.LabelFrame(self, text="Joint atom amplitude refinement")
        refit_controls.pack(fill="x", padx=16, pady=(0, 4))
        self.joint_amplitude_refit = tk.BooleanVar(value=True)
        self.amplitude_refit_interval = tk.IntVar(value=2)
        self.joint_parameter_refinement = tk.BooleanVar(value=True)
        self.parameter_refinement_passes = tk.IntVar(value=1)
        self.parameter_refinement_atom_limit = tk.IntVar(value=8)
        self.band_aware_candidates = tk.BooleanVar(value=True)
        ttk.Checkbutton(refit_controls, text="Jointly refit amplitudes",
                        variable=self.joint_amplitude_refit).pack(side="left", padx=8)
        ttk.Label(refit_controls, text="Every N accepted atoms").pack(
            side="left", padx=(12, 2))
        ttk.Entry(refit_controls, textvariable=self.amplitude_refit_interval,
                  width=7).pack(side="left")
        ttk.Checkbutton(refit_controls, text="Refine recent atom parameters at end",
                        variable=self.joint_parameter_refinement).pack(
                            side="left", padx=(12, 2))
        ttk.Label(refit_controls, text="Passes").pack(side="left", padx=(8, 2))
        ttk.Entry(refit_controls, textvariable=self.parameter_refinement_passes,
                  width=4).pack(side="left")
        ttk.Label(refit_controls, text="Recent atoms").pack(
            side="left", padx=(8, 2))
        ttk.Entry(refit_controls, textvariable=self.parameter_refinement_atom_limit,
                  width=4).pack(side="left")
        ttk.Checkbutton(refit_controls, text="Band-aware atom roles",
                        variable=self.band_aware_candidates).pack(side="left", padx=12)
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

    def _toggle_fullscreen(self, _event=None):
        self.attributes("-fullscreen", not bool(self.attributes("-fullscreen")))

    def _update_weight_controls(self):
        state = ["disabled"] if self.adaptive_weights.get() else ["!disabled"]
        for entry in self.statistical_weight_entries:
            entry.state(state)

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
        self.export_button.state(["disabled"])
        self.spectrum_button.state(["disabled"])
        self.measurements_button.state(["disabled"])
        self.extent_value.configure(text="1.0×  ([0, 1)²)")
        self.status.configure(text=f"Loaded {self.source.shape[1]} × {self.source.shape[0]}")

    def fit(self):
        if self.source is None or self.running: return
        try:
            config = self._build_config()
        except (ValueError, tk.TclError) as exc: messagebox.showerror("Invalid settings", str(exc)); return
        self.cancel_event.clear()
        self.running = True; self.fit_button.state(["disabled"])
        self.cancel_button.state(["!disabled"])
        def worker():
            try:
                result = TextureFitter(config).fit(self.source,
                    lambda stage, value, msg: self.events.put(("progress", value, msg)),
                    self.cancel_event.is_set)
                self.events.put(("result", result))
            except Exception as exc:
                event = "cancelled" if self.cancel_event.is_set() else "error"
                self.events.put((event, exc))
        self.worker_thread = threading.Thread(target=worker, name="texture-fit-controller")
        self.worker_thread.start(); self.after(50, self._poll)

    def cancel_fit(self):
        """Request cooperative shutdown of the controller and kernel workers."""
        if not self.running:
            return
        self.cancel_event.set()
        self.cancel_button.state(["disabled"])
        self.status.configure(text="Cancelling fit and stopping worker threads...")

    def export_json(self):
        if self.result is None:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".json", filetypes=[("JSON", "*.json"), ("All", "*.*")],
            initialfile="fit.json")
        if not path:
            return
        try:
            self.result.save_json(path)
        except OSError as exc:
            messagebox.showerror("Export failed", str(exc))
            return
        self.status.configure(text=f"Exported fit result to {path}")

    def _on_close(self):
        """Keep Tk alive until cooperative cancellation has joined all workers."""
        self.closing = True
        if self.running:
            self.cancel_fit()
        self._finish_close()

    def _finish_close(self):
        if self.worker_thread is not None and self.worker_thread.is_alive():
            self.after(50, self._finish_close)
            return
        self.destroy()

    def show_spectrum(self):
        if self.result is None:
            return
        SpectralDiagnosticsDialog(
            self, self.result.metadata["spectral_diagnostics"])

    def show_measurements(self):
        if self.result is None or self.source is None:
            return
        MeasurementDiagnosticsDialog(self, self.source, self.result.reconstruction)

    def _build_config(self) -> FitConfig:
        """Read and validate all editable fitting controls."""
        families = tuple(family for family in SUPPORTED_COMPONENT_FAMILIES
                         if self.atom_enabled[family].get())
        if not families:
            raise ValueError("select at least one procedural atom family")
        return FitConfig(seed=self.seed.get(), max_components=self.components.get(),
                         noise_seed_candidates=self.noise_seed_candidates.get(),
                         spectral_noise_modes=self.spectral_noise_modes.get(),
                         max_iterations=self.iterations.get(), fitting_resolution=self.resolution.get(),
                         decomposition_bands=self.decomposition_bands.get(),
                         band_workers=self.band_workers.get(),
                         candidate_workers=self.candidate_workers.get(),
                         compute_backend=self.compute_backend.get(),
                         gpu_batch_size=self.gpu_batch_size.get(),
                         component_families=families,
                         max_frequency=(None if self.max_frequency.get() <= 0
                                        else self.max_frequency.get()),
                         min_improvement=self.min_improvement.get(),
                         adaptive_texture_weights=self.adaptive_weights.get(),
                         spectrum_weight=self.spectrum_weight.get(),
                         histogram_weight=self.histogram_weight.get(),
                         autocorrelation_weight=self.autocorrelation_weight.get(),
                         gradient_weight=self.gradient_weight.get(),
                         mse_weight=self.mse_weight.get(),
                         local_structure_weight=self.local_structure_weight.get(),
                         local_contrast_weight=self.local_contrast_weight.get(),
                         absolute_spectrum_weight=self.absolute_spectrum_weight.get(),
                         oriented_spectrum_weight=self.oriented_spectrum_weight.get(),
                         detail_refinement=self.detail_refinement.get(),
                         detail_max_components=self.detail_components.get(),
                         detail_min_frequency=self.detail_min_frequency.get(),
                         detail_hf_ratio_threshold=self.detail_hf_threshold.get(),
                         joint_amplitude_refit=self.joint_amplitude_refit.get(),
                         amplitude_refit_interval=self.amplitude_refit_interval.get(),
                         joint_parameter_refinement=self.joint_parameter_refinement.get(),
                         parameter_refinement_passes=self.parameter_refinement_passes.get(),
                         parameter_refinement_atom_limit=
                             self.parameter_refinement_atom_limit.get(),
                         band_aware_candidates=self.band_aware_candidates.get())

    def _poll(self):
        try:
            while True:
                event = self.events.get_nowait()
                if event[0] == "progress": self.progress["value"] = event[1] * 100; self.status.configure(text=event[2])
                elif event[0] == "result":
                    result = event[1]; self.result = result
                    self.spectrum_button.state(["!disabled"])
                    self.measurements_button.state(["!disabled"])
                    self.export_button.state(["!disabled"])
                    self._update_extent_preview(); self._show(result.residual, 2, True)
                    m = result.metrics
                    objective = result.metadata["objective"]
                    mode = "adaptive per-band" if objective["weight_mode"] == "adaptive_per_band" else "manual"
                    detail = result.metadata["detail_refinement"]
                    detail_text = (f"detail +{detail['components']}" if detail["accepted"]
                                   else f"detail {detail.get('reason', 'not accepted')}")
                    self.status.configure(text=f"Band objective {objective['final']:.6f} ({mode})   {detail_text}   Full-image texture loss {m['texture_loss']:.6f}   RMSE {m['rmse']:.5f}   PSNR {m['psnr']:.2f} dB")
                    self.running = False; self.fit_button.state(["!disabled"])
                    self.cancel_button.state(["disabled"])
                elif event[0] == "cancelled":
                    self.running = False; self.fit_button.state(["!disabled"])
                    self.cancel_button.state(["disabled"])
                    self.progress["value"] = 0
                    self.status.configure(text="Fit cancelled; worker threads stopped")
                else:
                    if not self.closing:
                        messagebox.showerror("Fit failed", str(event[1]))
                    self.running = False; self.fit_button.state(["!disabled"])
                    self.cancel_button.state(["disabled"])
        except queue.Empty: pass
        if self.running: self.after(50, self._poll)

def main(): TestApplication().mainloop()
if __name__ == "__main__": main()
