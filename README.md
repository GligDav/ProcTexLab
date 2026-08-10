# Procedural Texture Kernel

This repository implements the standalone Python kernel for approximating a 2D raster scalar field with a compact sum of procedural atoms. It follows the accompanying HSPD research design: boundary-aware spectral initialization, sparse residual selection, separable amplitude estimation, bounded nonlinear refinement, and coarse-to-fine/local atom candidates. It deliberately contains no Blender dependency.

## Installation

Python 3.10 or newer is required.

```bash
python -m pip install -e ".[dev]"
```

The runtime dependencies are NumPy, SciPy, and Pillow. Tkinter is normally included with desktop Python distributions.

## Public API

```python
from procedural_texture_kernel import FitConfig, TextureFitter, load_image

image = load_image("roughness.png")
result = TextureFitter(FitConfig(
    max_components=8,
    max_iterations=40,
    fitting_resolution=96,
    seed=0,
)).fit(image)

reconstruction = result.evaluate(width=image.shape[1], height=image.shape[0])
# Inspect procedural continuation over two source-sized UV intervals.
extended = result.evaluate_region(512, 512, (0, 2), (0, 2))
print(result.metrics)
print(result.model.components)
result.save_json("fit.json")
```

`FitResult` retains the full-resolution reconstruction, signed residual, metrics, and fitting history. `TextureFitter.fit` also accepts optional `progress_callback(stage, progress, message)` and `cancel_callback()` callables. Cancellation is checked between atom stages.

## Architecture

- `coordinates.py` owns the one coordinate convention and cached grid construction.
- `components.py` defines typed, serializable sinusoid, Gabor, and Gaussian RBF atoms.
- `model.py` evaluates and serializes the bias/plane plus sparse atom sum.
- `fitting.py` contains analysis, candidate pursuit, variable projection, and OMP amplitude refitting.
- `api.py` exposes configuration, fitter, and result objects.
- `io.py` handles raster loading and scalar normalization; `metrics.py` is reusable numerical evaluation.
- `gui/test_app.py` and `examples/basic_usage.py` are replaceable clients of the public API.

The kernel never imports GUI modules. Image I/O contains no optimizer logic.

## Fitting behavior and configuration

The fitter converts inputs to float64 grayscale in `[0, 1]`, optionally downsamples only the fitting raster, estimates DC and a plane, then iterates against the residual. A Hann-windowed FFT proposes global Fourier atoms. Residual extrema propose localized RBF and oriented Gabor atoms at several scales. The best candidate is refined with bounded SciPy nonlinear least squares while its amplitude is eliminated analytically (variable projection). After every selection, all active amplitudes and the global terms are refit together using least squares with a small ridge augmentation—an OMP-like update.

Important `FitConfig` fields are `max_components`, nonlinear `max_iterations`, `fitting_resolution`, enabled `component_families`, FFT candidate count and frequency bounds, `min_improvement`, `ridge`, and `fit_plane`. Defaults are bounded and deterministic. `seed` is serialized into metadata and reserved for stochastic dictionary extensions; the current dictionary is deterministic.

This baseline implements the paper's most identifiable first benchmark (Fourier/Gabor/RBF). Perlin/simplex seed-bank inversion, wavelet residual atoms, atom merging, explicit tiling constraints, LASSO, and GPU acceleration are future extensions rather than incomplete hidden code paths.

## Coordinates and procedural model

Pixels use normalized half-open coordinates: `u = column / width`, `v = row / height`. The origin is top-left; U increases right and V increases down. Frequencies are cycles per normalized coordinate interval, phase and orientation are radians. With the image-coordinate V axis, positive angles appear clockwise on a conventional screen. Model evaluation is resolution independent.

The component types are:

- `SinusoidComponent`: amplitude, U/V frequency vector, phase.
- `GaborComponent`: amplitude, center, two Gaussian widths, carrier frequency, orientation, phase.
- `GaussianRBFComponent`: amplitude, center, Gaussian width.

`ProceduralTextureModel.to_dict()` emits JSON primitives with schema version 1 and the coordinate-system identifier. `save_json`, `load_json`, and `from_dict` support round trips and reject unsupported schemas or component types.

## Input and metrics

2D grayscale and 3/4-channel RGB(A) arrays are accepted. Integer formats (including uint16) are divided by their dtype range; floating inputs already in `[0, 1]` are preserved, while out-of-range finite values are min/max normalized. RGB uses linear coefficients 0.2126/0.7152/0.0722; alpha is ignored. Empty, malformed, NaN, and infinite inputs raise descriptive errors.

Reported metrics are MSE, RMSE, MAE, PSNR (assuming normalized peak 1), normalized RMSE, and correlation.

## GUI and example

Run the threaded development GUI:

```bash
python -m gui.test_app
```

It loads common raster formats, edits the principal settings, shows progress, and displays source, reconstruction, contrast-scaled residual, and metrics. The **Min improvement** field sets the minimum candidate score and MSE decrease required to retain another atom; lowering it permits smaller residual improvements and potentially larger models. The **Result UV extent** slider evaluates `[0, extent)²` at a bounded preview resolution, making procedural continuation and repetition visible without changing the fitted model. Tk widgets are updated only on the main thread and duplicate fits are disabled.

Run the self-contained synthetic example:

```bash
python examples/basic_usage.py
```

It creates a known two-sinusoid target through the public model, fits it, prints metrics, and writes previews plus JSON under `example_output/`.

## Testing

```bash
python -m compileall procedural_texture_kernel gui examples
pytest
```

Tests cover atom evaluation, coordinates, model composition and JSON round trips, metrics, image normalization, deterministic synthetic fitting, constant fields, and non-square rasters.

## Current limitations

The model is scalar/grayscale and the GUI is a development tool. Candidate scoring is intentionally compact and materializes only a small active design matrix, but local position search is not FFT-accelerated. Very stochastic, photographic, sharp-edged, or high-entropy fields may require many atoms and are often better represented by conventional textures. Output is not clipped during model synthesis, preserving the true least-squares residual. Periodic seam constraints, noise seed banks, wavelets, robust/perceptual losses, color-channel fitting, and batch/GPU paths are not yet implemented.

## Future Blender integration

The intended boundary is:

```text
Raster -> TextureFitter -> ProceduralTextureModel -> JSON/typed parameters
       -> future Blender adapter -> shader node graph or compact GPU evaluator
```

A Blender adapter should depend only on the public model and map bias to a Value node, sinusoids to Wave/Math chains, and localized Gabor/RBF atoms to node groups or shader code. Fitting internals, Tkinter, and SciPy optimizer state do not cross that boundary.
