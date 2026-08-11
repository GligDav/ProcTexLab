# Procedural Texture Kernel

This repository implements the standalone Python kernel for approximating a 2D raster scalar field with a compact sum of procedural atoms. It follows the accompanying HSPD research design: boundary-aware spectral initialization, sparse residual proposals, statistical texture matching, bounded nonlinear refinement, and coarse-to-fine/local atom candidates. It deliberately contains no Blender dependency.

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
- `components.py` defines typed, serializable sinusoid, Gabor, Gaussian RBF, seeded Perlin-noise, and wavelet atoms.
- `model.py` evaluates and serializes the bias/plane plus sparse atom sum.
- `fitting.py` contains spectral analysis, residual-based candidate proposals, statistical selection, and bounded atom refinement.
- `texture_loss.py` implements the weighted multi-scale spectrum, histogram, autocorrelation, and gradient-statistics objective.
- `api.py` exposes configuration, fitter, and result objects.
- `io.py` handles raster loading and scalar normalization; `metrics.py` is reusable numerical evaluation.
- `gui/test_app.py` and `examples/basic_usage.py` are replaceable clients of the public API.

The kernel never imports GUI modules. Image I/O contains no optimizer logic.

## Fitting behavior and configuration

The fitter converts inputs to float64 grayscale in `[0, 1]`, optionally downsamples only the fitting raster, estimates DC and a plane, then iterates against the residual. A Hann-windowed FFT proposes global Fourier atoms. Residual extrema propose localized RBF and oriented Gabor atoms at several scales. Pixel residual correlation is used only to initialize candidates; it is not the optimization objective. Candidate selection, stopping, amplitude adjustment, and bounded nonlinear refinement minimize a weighted statistical texture loss composed of multi-scale log-power spectra, intensity-distribution CDF distance, normalized spatial autocorrelation, and periodic gradient magnitude/orientation statistics. Consequently, a translated but statistically equivalent texture can score well even with poor pixel-wise MSE.

The loss weights are exposed by `FitConfig` as `spectrum_weight`, `histogram_weight`, `autocorrelation_weight`, and `gradient_weight`. Defaults are `1.0`, `0.5`, `0.75`, and `0.5`. The reported `texture_loss` is the weighted mean, keeping its scale stable when all weights are multiplied by the same factor.

Important `FitConfig` fields are `max_components`, nonlinear `max_iterations`, `fitting_resolution`, enabled `component_families`, FFT candidate count and frequency bounds, `min_improvement`, the four texture-loss weights, and `fit_plane`. `ridge` stabilizes the initial DC/plane estimate. Defaults are bounded and deterministic. `seed` is serialized into metadata and reserved for stochastic dictionary extensions; the current dictionary is deterministic.

The fitter includes Fourier, Gabor, RBF, seeded Perlin-noise, and localized wavelet candidates. Perlin candidates use a deterministic seed bank derived from `FitConfig.seed`; atom merging, explicit tiling constraints, LASSO, simplex noise, and GPU acceleration remain future extensions.

## Coordinates and procedural model

Pixels use normalized half-open coordinates: `u = column / width`, `v = row / height`. The origin is top-left; U increases right and V increases down. Frequencies are cycles per normalized coordinate interval, phase and orientation are radians. With the image-coordinate V axis, positive angles appear clockwise on a conventional screen. Model evaluation is resolution independent.

The component types are:

- `SinusoidComponent`: amplitude, U/V frequency vector, phase.
- `GaborComponent`: amplitude, center, two Gaussian widths, carrier frequency, orientation, phase.
- `GaussianRBFComponent`: amplitude, center, Gaussian width.
- `PerlinNoiseComponent`: amplitude, base frequency, octave count, persistence, lacunarity, UV offset, and deterministic seed. Its normalized fractal gradient-noise basis continues procedurally outside the source UV range.
- `WaveletComponent`: amplitude, center, anisotropic U/V scales, and orientation. It uses a localized 2D Mexican-hat (Ricker) basis for residual blobs, spots, and band-pass detail.

All five names can be selected through `FitConfig.component_families`: `sinusoid`, `gabor`, `gaussian_rbf`, `perlin_noise`, and `wavelet`. They are enabled by default. Components can also be constructed and added directly:

```python
from procedural_texture_kernel import PerlinNoiseComponent, WaveletComponent

model.add(PerlinNoiseComponent(amplitude=0.15, frequency=6, octaves=3, seed=12))
model.add(WaveletComponent(amplitude=-0.1, center_u=0.4, center_v=0.6, scale_u=0.08))
```

`ProceduralTextureModel.to_dict()` emits JSON primitives with schema version 1 and the coordinate-system identifier. `save_json`, `load_json`, and `from_dict` support round trips and reject unsupported schemas or component types.

## Input and metrics

2D grayscale and 3/4-channel RGB(A) arrays are accepted. Integer formats (including uint16) are divided by their dtype range; floating inputs already in `[0, 1]` are preserved, while out-of-range finite values are min/max normalized. RGB uses linear coefficients 0.2126/0.7152/0.0722; alpha is ignored. Empty, malformed, NaN, and infinite inputs raise descriptive errors.

The primary reported metric is `texture_loss`, accompanied by `spectrum_loss`, `histogram_loss`, `autocorrelation_loss`, and `gradient_loss`. MSE, RMSE, MAE, PSNR, normalized RMSE, and correlation remain available strictly as pixel-aligned diagnostics.

## GUI and example

Run the threaded development GUI:

```bash
python -m gui.test_app
```

It loads common raster formats, edits the principal settings, shows progress, and displays source, reconstruction, contrast-scaled residual, and metrics. The **Allowed procedural atoms** checkboxes enable or disable sinusoid, Gabor, Gaussian RBF, Perlin-noise, and wavelet candidates; at least one must remain selected. The four fields under **Texture loss weights** directly control the spectrum, histogram, autocorrelation, and gradient contributions. Weights must be finite and non-negative, and at least one must be positive. The **Min improvement** field sets the minimum decrease in composite texture loss required to retain another atom; lowering it permits smaller statistical improvements and potentially larger models. The **Result UV extent** slider evaluates `[0, extent)²` at a bounded preview resolution, making procedural continuation and repetition visible without changing the fitted model. Tk widgets are updated only on the main thread and duplicate fits are disabled.

To compare two same-sized rasters and interactively calibrate the four objective
weights, run:

```bash
python -m gui.texture_loss_calibrator
```

The calibrator displays both normalized grayscale inputs, each raw loss component,
its weighted term, and the final normalized `texture_loss`. Editing a weight
automatically recalculates the result.

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

Tests cover all atom evaluations, seeded Perlin determinism, wavelet localization, model composition and JSON round trips, coordinates, metrics, image normalization, deterministic synthetic fitting, constant fields, and non-square rasters.

## Current limitations

The model is scalar/grayscale and the GUI is a development tool. Candidate scoring evaluates compact procedural candidates directly, but local position search is not FFT-accelerated. Very stochastic, photographic, sharp-edged, or high-entropy fields may require many atoms and are often better represented by conventional textures. Output is not clipped during model synthesis, preserving both genuine statistics and the signed diagnostic residual. Periodic seam constraints, broader/adaptive noise seed searches, discrete wavelet decompositions, SSIM/perceptual losses, color-channel fitting, and batch/GPU paths are not yet implemented.

## Future Blender integration

The intended boundary is:

```text
Raster -> TextureFitter -> ProceduralTextureModel -> JSON/typed parameters
       -> future Blender adapter -> shader node graph or compact GPU evaluator
```

A Blender adapter should depend only on the public model and map bias to a Value node, sinusoids to Wave/Math chains, and localized Gabor/RBF atoms to node groups or shader code. Fitting internals, Tkinter, and SciPy optimizer state do not cross that boundary.
