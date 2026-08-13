"""Debug GUI for inspecting reconstructable raster frequency bands."""
from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageTk

from procedural_texture_kernel import LaplacianPyramid, load_image


IMAGE_TYPES = [("Images", "*.png *.jpg *.jpeg *.tif *.tiff *.bmp"),
               ("All files", "*.*")]


def decompose_image(image: np.ndarray, bands: int = 5,
                    base_sigma: float = 1.0) -> tuple[LaplacianPyramid, tuple[np.ndarray, ...]]:
    """Decompose an image using the same public implementation as the fitter."""
    pyramid = LaplacianPyramid(bands=bands, base_sigma=base_sigma)
    return pyramid, pyramid.decompose(image)


def preview_values(array: np.ndarray, signed: bool = False) -> np.ndarray:
    """Map an image or signed frequency band into displayable grayscale."""
    values = np.asarray(array, dtype=np.float64)
    if signed:
        limit = max(float(np.max(np.abs(values))), 1e-12)
        values = 0.5 + 0.5 * values / limit
    return np.clip(values, 0.0, 1.0)


class DecompositionViewer(tk.Tk):
    """Load a raster and display its Laplacian bands and reconstruction."""

    def __init__(self) -> None:
        super().__init__()
        self.title("Texture Decomposition Viewer")
        self.geometry("1100x760")
        self.minsize(720, 520)
        self.source: np.ndarray | None = None
        self.source_path: Path | None = None
        self.photos: list[ImageTk.PhotoImage] = []
        self.bands = tk.IntVar(value=5)
        self.base_sigma = tk.DoubleVar(value=1.0)
        self._build_ui()

    def _build_ui(self) -> None:
        controls = ttk.Frame(self)
        controls.pack(fill="x", padx=10, pady=10)
        ttk.Button(controls, text="Load raster", command=self.load).pack(side="left")
        ttk.Label(controls, text="Bands").pack(side="left", padx=(16, 4))
        ttk.Spinbox(controls, from_=1, to=12, textvariable=self.bands,
                    width=5, command=self.refresh).pack(side="left")
        ttk.Label(controls, text="Base sigma (px)").pack(side="left", padx=(16, 4))
        sigma_entry = ttk.Entry(controls, textvariable=self.base_sigma, width=8)
        sigma_entry.pack(side="left")
        sigma_entry.bind("<Return>", lambda _event: self.refresh())
        sigma_entry.bind("<FocusOut>", lambda _event: self.refresh())
        ttk.Button(controls, text="Refresh", command=self.refresh).pack(side="left", padx=10)

        self.canvas = tk.Canvas(self, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.canvas.pack(fill="both", expand=True, padx=(10, 0))
        self.gallery = ttk.Frame(self.canvas)
        self.gallery_window = self.canvas.create_window((0, 0), window=self.gallery, anchor="nw")
        self.gallery.bind("<Configure>", self._update_scroll_region)
        self.canvas.bind("<Configure>", self._resize_gallery)
        self.canvas.bind_all("<MouseWheel>", self._scroll)

        self.status = ttk.Label(self, text="Load a raster image to begin", anchor="w")
        self.status.pack(fill="x", padx=10, pady=(6, 10))

    def _update_scroll_region(self, _event=None) -> None:
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _resize_gallery(self, event) -> None:
        self.canvas.itemconfigure(self.gallery_window, width=event.width)

    def _scroll(self, event) -> None:
        self.canvas.yview_scroll(int(-event.delta / 120), "units")

    def load(self) -> None:
        path = filedialog.askopenfilename(filetypes=IMAGE_TYPES)
        if not path:
            return
        try:
            self.source = load_image(path)
        except ValueError as exc:
            messagebox.showerror("Load failed", str(exc))
            return
        self.source_path = Path(path)
        self.refresh()

    def refresh(self) -> None:
        if self.source is None:
            return
        try:
            pyramid, bands = decompose_image(
                self.source, self.bands.get(), self.base_sigma.get())
        except (ValueError, tk.TclError) as exc:
            self.status.configure(text=f"Invalid decomposition settings: {exc}")
            return
        reconstruction = pyramid.reconstruct(bands)
        error = reconstruction - self.source
        self._clear_gallery()
        self._add_preview("Source", self.source, 0, signed=False)
        for index, band in enumerate(bands):
            # Laplacian differences are signed; the final Gaussian residual is
            # an ordinary low-pass image and should retain its native contrast.
            self._add_preview(self._band_title(index, pyramid), band, index + 1,
                              signed=index < len(bands) - 1)
        self._add_preview("Reconstruction", reconstruction, len(bands) + 1, signed=False)
        self._add_preview("Reconstruction error (contrast scaled)", error,
                          len(bands) + 2, signed=True)
        maximum_error = float(np.max(np.abs(error)))
        name = self.source_path.name if self.source_path else "Raster"
        self.status.configure(
            text=(f"{name}  |  {self.source.shape[1]} x {self.source.shape[0]}  |  "
                  f"{len(bands)} bands  |  max reconstruction error {maximum_error:.3g}"))

    @staticmethod
    def _band_title(index: int, pyramid: LaplacianPyramid) -> str:
        if pyramid.bands == 1:
            return "Band 1: identity"
        if index == 0:
            return f"Band 1: source - blur sigma {pyramid.sigmas[0]:g}px"
        if index == pyramid.bands - 1:
            return f"Band {index + 1}: low-pass sigma {pyramid.sigmas[-1]:g}px"
        return (f"Band {index + 1}: sigma {pyramid.sigmas[index - 1]:g}"
                f"-{pyramid.sigmas[index]:g}px")

    def _clear_gallery(self) -> None:
        for widget in self.gallery.winfo_children():
            widget.destroy()
        self.photos.clear()

    def _add_preview(self, title: str, array: np.ndarray, index: int,
                     signed: bool) -> None:
        row, column = divmod(index, 3)
        frame = ttk.LabelFrame(self.gallery, text=title)
        frame.grid(row=row, column=column, sticky="nsew", padx=5, pady=5)
        self.gallery.columnconfigure(column, weight=1)
        image = Image.fromarray(np.uint8(preview_values(array, signed) * 255), "L")
        image.thumbnail((330, 245))
        photo = ImageTk.PhotoImage(image)
        self.photos.append(photo)
        ttk.Label(frame, image=photo, anchor="center").pack(fill="both", expand=True, padx=4, pady=4)
        ttk.Label(frame, text=(f"min {np.min(array):.4g}   max {np.max(array):.4g}   "
                               f"RMS {np.sqrt(np.mean(array * array)):.4g}"),
                  anchor="center").pack(fill="x", padx=4, pady=(0, 4))


def main() -> None:
    DecompositionViewer().mainloop()


if __name__ == "__main__":
    main()
