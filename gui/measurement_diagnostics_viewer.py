"""Tk viewer for spatial and directional texture measurements."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

import numpy as np

from procedural_texture_kernel.measurement_diagnostics import compare_measurements


def _format(value: float, unit: str = "") -> str:
    if not np.isfinite(value):
        return "n/a"
    if unit == "fraction":
        return f"{100 * value:.2f}%"
    return f"{value:.5g}"


def _ratio_status(ratio: float, tolerance: float = .1) -> str:
    if not np.isfinite(ratio):
        return "n/a"
    if abs(ratio - 1.0) <= tolerance:
        return "matched"
    return "excess" if ratio > 1 else "deficit"


class MeasurementDiagnosticsDialog(tk.Toplevel):
    """Display measurements that complement the radial spectrum dialog."""

    def __init__(self, parent: tk.Misc, target: np.ndarray, result: np.ndarray):
        super().__init__(parent)
        self.title("Texture measurements")
        self.geometry("900x680")
        self.minsize(680, 520)
        self.diagnostics = compare_measurements(target, result)

        ttk.Label(self, text=("Ratios are result / target. Values near 1 are matched; "
                              "below 1 indicate a deficit."),
                  wraplength=850).pack(fill="x", padx=10, pady=(10, 4))
        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True, padx=10, pady=(4, 10))
        overview = ttk.Frame(notebook)
        contrast = ttk.Frame(notebook)
        direction = ttk.Frame(notebook)
        notebook.add(overview, text="Overview")
        notebook.add(contrast, text="Local contrast")
        notebook.add(direction, text="Directionality")
        self._build_overview(overview)
        self._build_contrast(contrast)
        self._build_direction(direction)

    @staticmethod
    def _table(parent, columns, headings, widths, height):
        table = ttk.Treeview(parent, columns=columns, show="headings", height=height)
        for column, heading, width in zip(columns, headings, widths):
            table.heading(column, text=heading)
            table.column(column, width=width, anchor="center", stretch=True)
        table.pack(fill="both", expand=True, padx=8, pady=8)
        return table

    def _build_overview(self, parent):
        table = self._table(parent, ("metric", "target", "result", "ratio", "status"),
                            ("Metric", "Target", "Result", "Ratio", "Assessment"),
                            (190, 120, 120, 100, 110), 8)
        for row in self.diagnostics["summary"]:
            table.insert("", "end", values=(row["name"], _format(row["target"], row["unit"]),
                         _format(row["result"], row["unit"]), _format(row["ratio"]),
                         _ratio_status(row["ratio"])))
        threshold = self.diagnostics["edge_threshold"]
        ttk.Label(parent, text=f"Strong edges use the target's 90th-percentile gradient threshold ({threshold:.5g}).",
                  wraplength=800).pack(fill="x", padx=10, pady=(0, 10))

    def _build_contrast(self, parent):
        table = self._table(parent,
            ("scale", "tm", "rm", "mr", "tp", "rp", "pr"),
            ("Gaussian sigma", "Target mean", "Result mean", "Mean ratio",
             "Target p95", "Result p95", "p95 ratio"),
            (100, 110, 110, 100, 110, 110, 100), 8)
        for row in self.diagnostics["local_contrast"]:
            table.insert("", "end", values=(f"{row['sigma']:g} px",
                _format(row["target_mean"]), _format(row["result_mean"]),
                _format(row["mean_ratio"]), _format(row["target_p95"]),
                _format(row["result_p95"]), _format(row["p95_ratio"])))

    def _build_direction(self, parent):
        columns = ("wedge", "target", "result", "target_pct", "result_pct", "ratio")
        table = self._table(parent, columns,
            ("Frequency angle", "Target energy", "Result energy", "Target %", "Result %", "Ratio"),
            (130, 120, 120, 100, 100, 90), 8)
        for row in self.diagnostics["oriented_spectrum"]:
            table.insert("", "end", values=(
                f"{row['angle_min']:.1f}-{row['angle_max']:.1f} deg",
                _format(row["target_energy"]), _format(row["result_energy"]),
                f"{100 * row['target_fraction']:.2f}",
                f"{100 * row['result_fraction']:.2f}", _format(row["ratio"])))
        ttk.Label(parent, text=("Angles describe FFT frequency direction; visible ridges run "
                                "approximately perpendicular to that direction."),
                  wraplength=800).pack(fill="x", padx=10, pady=(0, 10))
