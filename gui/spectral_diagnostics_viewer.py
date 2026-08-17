"""Tk viewer for target/reconstruction spectral diagnostics."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk
import numpy as np


BAND_LABELS = {
    "dc": "DC",
    "very_low": "Very low",
    "low": "Low",
    "mid": "Mid",
    "high": "High",
    "very_high": "Very high",
}


def _finite_log_range(*series: np.ndarray) -> tuple[float, float]:
    """Return a stable base-10 plot range for non-negative power values."""
    values = np.concatenate([np.asarray(item, dtype=float).ravel() for item in series])
    positive = values[np.isfinite(values) & (values > 0)]
    if positive.size == 0:
        return -12.0, 0.0
    upper = float(np.log10(np.max(positive)))
    lower = max(float(np.log10(np.min(positive))), upper - 12.0)
    if upper - lower < 1.0:
        lower = upper - 1.0
    return lower, upper


def _format_number(value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    if value == 0:
        return "0"
    return f"{value:.4g}"


class SpectralDiagnosticsDialog(tk.Toplevel):
    """Display absolute radial PSD curves and band-energy comparisons."""

    def __init__(self, parent: tk.Misc, diagnostics: dict):
        super().__init__(parent)
        self.title("Spectral diagnostics")
        self.geometry("860x650")
        self.minsize(620, 480)
        self.diagnostics = diagnostics

        summary = ttk.Frame(self)
        summary.pack(fill="x", padx=10, pady=(10, 4))
        ratio = float(diagnostics["high_frequency_ratio"])
        ttk.Label(summary, text="Combined high-frequency energy ratio:").pack(side="left")
        ttk.Label(summary, text=f"{ratio:.3f}  (result / target)",
                  font=("TkDefaultFont", 10, "bold")).pack(side="left", padx=6)

        plot_frame = ttk.LabelFrame(self, text="Absolute radial power spectrum (log scale)")
        plot_frame.pack(fill="both", expand=True, padx=10, pady=6)
        self.canvas = tk.Canvas(plot_frame, background="white", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=4, pady=4)
        self.canvas.bind("<Configure>", self._draw_plot)

        table_frame = ttk.LabelFrame(self, text="Frequency-band energy")
        table_frame.pack(fill="x", padx=10, pady=(0, 10))
        columns = ("band", "range", "target", "result", "target_fraction",
                   "result_fraction", "ratio")
        self.table = ttk.Treeview(table_frame, columns=columns, show="headings", height=6)
        headings = ("Band", "Nyquist range", "Target energy", "Result energy",
                    "Target %", "Result %", "Ratio")
        widths = (90, 110, 105, 105, 85, 85, 75)
        for column, heading, width in zip(columns, headings, widths):
            self.table.heading(column, text=heading)
            self.table.column(column, width=width, anchor="center", stretch=True)
        self.table.pack(fill="x", padx=4, pady=4)
        self._populate_table()

    def _populate_table(self) -> None:
        for band in self.diagnostics["bands"]:
            if band["name"] == "dc":
                frequency_range = "0 (mean)"
            else:
                frequency_range = (f"{band['frequency_min']:.3g}–"
                                   f"{band['frequency_max']:.3g}")
            self.table.insert("", "end", values=(
                BAND_LABELS.get(band["name"], band["name"]), frequency_range,
                _format_number(float(band["target_energy"])),
                _format_number(float(band["result_energy"])),
                f"{100 * float(band['target_fraction']):.2f}",
                f"{100 * float(band['result_fraction']):.2f}",
                _format_number(float(band["ratio"])),
            ))

    def _draw_plot(self, _event=None) -> None:
        self.canvas.delete("all")
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()
        if width < 100 or height < 100:
            return
        left, right, top, bottom = 62, 18, 20, 42
        plot_width = width - left - right
        plot_height = height - top - bottom
        target = self.diagnostics["target"]
        result = self.diagnostics["result"]
        frequency = np.asarray(target["frequency"], dtype=float)
        target_power = np.asarray(target["power"], dtype=float)
        result_power = np.asarray(result["power"], dtype=float)
        if frequency.size == 0:
            return
        y_min, y_max = _finite_log_range(target_power, result_power)
        x_max = max(float(np.max(frequency)), 1.0)

        def x_coordinate(value: float) -> float:
            return left + plot_width * value / x_max

        def y_coordinate(value: float) -> float:
            log_value = np.log10(max(value, 10 ** y_min))
            return top + plot_height * (y_max - log_value) / (y_max - y_min)

        self.canvas.create_rectangle(left, top, left + plot_width, top + plot_height,
                                     outline="#777777")
        for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
            x = left + plot_width * fraction
            value = x_max * fraction
            self.canvas.create_line(x, top, x, top + plot_height, fill="#e6e6e6")
            self.canvas.create_text(x, top + plot_height + 16, text=f"{value:.2g}")
        for index in range(5):
            value = y_min + (y_max - y_min) * index / 4
            y = top + plot_height * (4 - index) / 4
            self.canvas.create_line(left, y, left + plot_width, y, fill="#e6e6e6")
            self.canvas.create_text(left - 7, y, text=f"10^{value:.1f}", anchor="e")

        for power, color, label in ((target_power, "#1f77b4", "Target"),
                                    (result_power, "#d62728", "Result")):
            points = [(x_coordinate(float(x)), y_coordinate(float(y)))
                      for x, y in zip(frequency, power)]
            if len(points) > 1:
                self.canvas.create_line(*[coordinate for point in points for coordinate in point],
                                        fill=color, width=2)
            elif points:
                x, y = points[0]
                self.canvas.create_oval(x - 2, y - 2, x + 2, y + 2, fill=color, outline=color)
            legend_x = left + plot_width - 95
            legend_y = top + 12 + (0 if label == "Target" else 20)
            self.canvas.create_line(legend_x, legend_y, legend_x + 24, legend_y,
                                    fill=color, width=2)
            self.canvas.create_text(legend_x + 30, legend_y, text=label, anchor="w")
        self.canvas.create_text(left + plot_width / 2, height - 8,
                                text="Frequency / axial Nyquist")
        self.canvas.create_text(12, top + plot_height / 2, text="PSD", angle=90)
