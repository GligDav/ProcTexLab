"""Tkinter development application for the public kernel API."""
from __future__ import annotations
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
from PIL import Image, ImageTk
from procedural_texture_kernel import FitConfig, TextureFitter, load_image

class TestApplication(tk.Tk):
    """Small visual harness; fitting runs on a worker thread."""
    def __init__(self):
        super().__init__(); self.title("Procedural Texture Kernel"); self.geometry("1100x650")
        self.source = None; self.images = []; self.events = queue.Queue(); self.running = False
        controls = ttk.Frame(self); controls.pack(fill="x", padx=8, pady=8)
        self.components = tk.IntVar(value=8); self.iterations = tk.IntVar(value=40)
        self.resolution = tk.IntVar(value=96); self.seed = tk.IntVar(value=0)
        for label, variable in (("Components", self.components), ("Iterations", self.iterations),
                                ("Fit resolution", self.resolution), ("Seed", self.seed)):
            ttk.Label(controls, text=label).pack(side="left", padx=(8,2))
            ttk.Entry(controls, textvariable=variable, width=6).pack(side="left")
        ttk.Button(controls, text="Load Image", command=self.load).pack(side="left", padx=8)
        self.fit_button = ttk.Button(controls, text="Fit", command=self.fit); self.fit_button.pack(side="left")
        views = ttk.Frame(self); views.pack(fill="both", expand=True)
        self.labels = []
        for title in ("Source", "Reconstruction", "Residual (contrast scaled)"):
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
        photo = ImageTk.PhotoImage(image); self.images.append(photo); self.labels[slot].configure(image=photo)

    def load(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg *.tif *.tiff *.bmp"), ("All", "*.*")])
        if not path: return
        try: self.source = load_image(path)
        except ValueError as exc: messagebox.showerror("Load failed", str(exc)); return
        self._show(self.source, 0); self.status.configure(text=f"Loaded {self.source.shape[1]}×{self.source.shape[0]}")

    def fit(self):
        if self.source is None or self.running: return
        try:
            config = FitConfig(seed=self.seed.get(), max_components=self.components.get(),
                               max_iterations=self.iterations.get(), fitting_resolution=self.resolution.get())
        except (ValueError, tk.TclError) as exc: messagebox.showerror("Invalid settings", str(exc)); return
        self.running = True; self.fit_button.state(["disabled"])
        def worker():
            try:
                result = TextureFitter(config).fit(self.source,
                    lambda stage, value, msg: self.events.put(("progress", value, msg)))
                self.events.put(("result", result))
            except Exception as exc: self.events.put(("error", exc))
        threading.Thread(target=worker, daemon=True).start(); self.after(50, self._poll)

    def _poll(self):
        try:
            while True:
                event = self.events.get_nowait()
                if event[0] == "progress": self.progress["value"] = event[1] * 100; self.status.configure(text=event[2])
                elif event[0] == "result":
                    result = event[1]; self._show(result.reconstruction, 1); self._show(result.residual, 2, True)
                    m = result.metrics; self.status.configure(text=f"RMSE {m['rmse']:.5f}   MAE {m['mae']:.5f}   PSNR {m['psnr']:.2f} dB")
                    self.running = False; self.fit_button.state(["!disabled"])
                else:
                    messagebox.showerror("Fit failed", str(event[1])); self.running = False; self.fit_button.state(["!disabled"])
        except queue.Empty: pass
        if self.running: self.after(50, self._poll)

def main(): TestApplication().mainloop()
if __name__ == "__main__": main()
