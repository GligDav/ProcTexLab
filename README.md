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
    max_components=12,
    max_iterations=60,
    fitting_resolution=192,
    decomposition_bands=5,
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
- `texture_loss.py` implements the weighted multi-scale spectrum, histogram, autocorrelation, gradient-statistics, and MSE objective.
- `spectral_diagnostics.py` reports absolute radial power spectra and target/result band-energy ratios without normalizing away high-frequency deficits.
- `weight_estimator.py` independently describes a pyramid band and heuristically proposes normalized weights for the four statistical loss terms.
- `api.py` exposes configuration, fitter, and result objects.
- `io.py` handles raster loading and scalar normalization; `metrics.py` is reusable numerical evaluation.
- `gui/test_app.py` and `examples/basic_usage.py` are replaceable clients of the public API.

The kernel never imports GUI modules. Image I/O contains no optimizer logic.

## Fitting behavior and configuration

The fitter converts inputs to float64 grayscale in `[0, 1]`, optionally downsamples only the fitting raster, and decomposes only the target into a reconstructable Laplacian pyramid. Each target band gets its own independently optimized procedural submodel and its own `max_components` atom budget. These submodels are added into the final `ProceduralTextureModel`; procedural candidates themselves are never decomposed during fitting. Thus 8 bands with `max_components=4` permit up to 32 atoms (fewer when a band reaches the configured stopping criterion early). A Hann-windowed FFT proposes global Fourier atoms. Residual extrema propose localized atoms at several scales. Pixel residual correlation initializes candidates, while selection and refinement minimize the configured statistical texture loss within the current band.

By default, `adaptive_texture_weights=True` analyzes every decomposed target band
and independently normalizes its spectrum, histogram, autocorrelation, and
gradient weights. The configured `mse_weight` is added unchanged to each band
objective. Extracted features and effective weights are recorded in each entry of
`FitResult.metadata["bands"]`. Set `adaptive_texture_weights=False` to apply the
manual `spectrum_weight`, `histogram_weight`, `autocorrelation_weight`, and
`gradient_weight` values to every band. Their defaults remain `1.0`, `0.5`,
`0.75`, and `0.5`; `mse_weight` defaults to `1.0`. The additional absolute
band-energy spectrum term defaults to `0.25` and prevents normalized spectral
shape from hiding an overall contrast or high-frequency deficit. Losses are
weighted means.

An orientation-aware absolute spectrum term also defaults to `0.25`. It splits
each radial band into eight unoriented Fourier wedges, so matching radial energy
with the wrong dominant direction is penalized.

Important `FitConfig` fields are `max_components`, nonlinear `max_iterations`, `fitting_resolution`, enabled `component_families`, FFT candidate count and frequency bounds, `min_improvement`, adaptive/manual texture-loss controls, and `fit_plane`. `decomposition_method` defaults to `"laplacian"`; `decomposition_bands` defaults to 5, and `decomposition_base_sigma` defaults to 1.0 pixels. Successive Gaussian cutoffs double in sigma (approximately octave-spaced), with Laplacian differences plus the final low-pass residual summing to the input within floating-point tolerance. Set the band count to 1 for identity decomposition. `ridge` stabilizes the initial DC/plane estimate. Defaults are bounded and deterministic. `seed` is serialized into metadata.

`max_frequency=None` selects 45% of the smaller fitting dimension, retaining an
anti-aliasing margin. An explicit cycle-per-UV value still overrides it. Input
downsampling applies a Gaussian anti-aliasing filter before resampling.

With `band_aware_candidates=True` (the default), high-frequency Laplacian bands search detail families, the low-pass residual searches coherent structure families, and middle bands allow both roles. A user selection containing only families outside a preferred role is preserved as a fallback; set the option to false to use every selected family in every band.

Candidate initialization is residual-adaptive: dominant spectral peaks propose noise frequencies, local structure tensors and support estimates initialize oriented atom directions and sizes, and the positive residual coverage initializes thresholded-noise masks. `noise_seed_candidates` controls the bounded deterministic seed bank (default 4); at most three noise frequencies are retained per iteration.

`masked_noise` is a deliberately constrained compositional atom: it applies independent detail noise to either side of a coherent mask without introducing a general shader-graph search space. Both mask sides are proposed, while mask shape and detail placement are refined continuously.

The experimental `shader_graph` family adds a serializable, acyclic scalar DAG with component, constant, add, multiply, smoothstep, one-minus, and mix nodes. Graphs are topologically ordered and capped at 64 nodes. The fitter intentionally searches only a five-node `noise mask -> smoothstep -> mix(two detail fields)` topology; arbitrary graph-topology evolution is not enabled.

Nonlinear refinement covers the principal smooth and structural families, including thresholded/ridged/domain-warped noise, fBm/turbulence, Voronoi, anisotropic Gaussians, lines, step edges, and DoG/LoG atoms. The fitter retains the original projected candidate whenever the bounded optimizer fails to improve its objective.

Set `detail_refinement=True` to run an adaptive high-frequency residual pass after
the normal multiband fit. It activates only when the fitting-resolution
`high_frequency_ratio` is below `detail_hf_ratio_threshold`, high-pass filters
the signed reconstruction residual, and fits a separate budget of detail atoms.
The candidate is retained only when both absolute HF-energy error and full-image
MSE improve. Configuration includes `detail_max_components`,
`detail_min_frequency`, `detail_min_improvement`, `detail_base_sigma`, and
`detail_component_families`. Detailed before/after diagnostics and the acceptance
decision are stored in `result.metadata["detail_refinement"]`.

With `joint_amplitude_refit=True` (the default), the fitter periodically solves
all linear atom amplitudes, bias, and plane coefficients together. The refit runs
after every `amplitude_refit_interval` accepted atoms and once at the end when
needed. A refitted state is retained only if the complete band texture objective
does not increase. Each decision is recorded in the band iteration metadata.

With `joint_parameter_refinement=True`, the final atoms in each band are revisited
by coordinate-descent nonlinear optimization after greedy construction. The
default performs one pass over the eight most recently accepted atoms; configure
`parameter_refinement_passes` and `parameter_refinement_atom_limit` to trade fit
quality for runtime. Every pass and accepted replacement is recorded per band.

The fitter includes Fourier, Gabor, RBF, seeded Perlin-noise, and localized wavelet candidates. Perlin candidates use a deterministic seed bank derived from `FitConfig.seed`; atom merging, explicit tiling constraints, LASSO, simplex noise, and GPU acceleration remain future extensions.

## Coordinates and procedural model

Pixels use normalized half-open coordinates: `u = column / width`, `v = row / height`. The origin is top-left; U increases right and V increases down. Frequencies are cycles per normalized coordinate interval, phase and orientation are radians. With the image-coordinate V axis, positive angles appear clockwise on a conventional screen. Model evaluation is resolution independent.

The component types are:

- `SinusoidComponent`: amplitude, U/V frequency vector, phase.
- `GaborComponent`: amplitude, center, two Gaussian widths, carrier frequency, orientation, phase.
- `GaussianRBFComponent`: amplitude, center, Gaussian width.
- `PerlinNoiseComponent`: amplitude, base frequency, octave count, persistence, lacunarity, UV offset, and deterministic seed. Its normalized fractal gradient-noise basis continues procedurally outside the source UV range.
- `ThresholdedNoiseComponent`: a rotated fBm field remapped through a smooth threshold, with controllable threshold and edge width for coherent high-contrast regions.
- `MaskedNoiseComponent`: independent detail noise restricted to either side of a smooth thresholded-noise region mask.
- `ShaderGraphComponent`: an embedded validated scalar DAG; currently fitted as a coherent mask mixing two independently seeded detail fields.
- `WaveletComponent`: amplitude, center, anisotropic U/V scales, and orientation. It uses a localized 2D Mexican-hat (Ricker) basis for residual blobs, spots, and band-pass detail.
- Noise families: `VoronoiNoiseComponent`, `FractalBrownianMotionComponent`, `RidgedMultifractalComponent`, `TurbulenceNoiseComponent`, and `DomainWarpedNoiseComponent`. Ridged multifractals fold every octave independently and expose ridge offset/power, rotation, and anisotropy for vein-like structures.
- Geometric/local atoms: `AnisotropicGaussianComponent`, `LineComponent`, `StepEdgeComponent`, `DifferenceOfGaussiansComponent` (`dog` or `log` mode), and `BinaryPrimitiveComponent` (disk, box, ring, or checker).
- Structured/global atoms: `PolynomialTrendComponent`, `RadialWaveComponent`, `SpiralWaveComponent`, and `SparseImpulseComponent`.

All registered names can be selected through `FitConfig.component_families`; they are enabled by default. Components can also be constructed and added directly. The fitter initializes `thresholded_noise` candidates from source-noise quantiles at several frequencies and edge widths, then refines amplitude, frequency, offset, rotation, threshold, and edge width against the configured objective.

```python
from procedural_texture_kernel import PerlinNoiseComponent, WaveletComponent

model.add(PerlinNoiseComponent(amplitude=0.15, frequency=6, octaves=3, seed=12))
model.add(WaveletComponent(amplitude=-0.1, center_u=0.4, center_v=0.6, scale_u=0.08))
```

`ProceduralTextureModel.to_dict()` emits JSON primitives with schema version 1 and the coordinate-system identifier. `save_json`, `load_json`, and `from_dict` support round trips and reject unsupported schemas or component types.

## Input and metrics

2D grayscale and 3/4-channel RGB(A) arrays are accepted. Integer formats (including uint16) are divided by their dtype range; floating inputs already in `[0, 1]` are preserved, while out-of-range finite values are min/max normalized. RGB uses linear coefficients 0.2126/0.7152/0.0722; alpha is ignored. Empty, malformed, NaN, and infinite inputs raise descriptive errors.

The primary reported metric is `texture_loss`, accompanied by `spectrum_loss`, `histogram_loss`, `autocorrelation_loss`, multiscale `gradient_loss`, `local_contrast_loss`, `local_structure_loss`, and `mse_loss`. The optional local-contrast term compares local standard-deviation distributions at several Gaussian scales and is controlled by `local_contrast_weight`. MSE is also part of the configurable fitting objective; RMSE, MAE, PSNR, normalized RMSE, and correlation remain available as diagnostics.

The optional local-structure objective uses a cached multiscale, oriented complex
filter bank. It compares response magnitude distributions, correlations between
neighboring orientations, cross-scale magnitude correlations, and cross-scale
phase coherence. Enable it with `local_structure_weight`; its scale count,
orientation count, and spatial pooling size are controlled by
`local_structure_scales`, `local_structure_orientations`, and
`local_structure_block_size`. When enabled, residual correlation preselects at
most `local_structure_candidate_limit` atoms for this expensive loss. The public
API defaults its weight to zero because
candidate evaluation becomes substantially more expensive; the development GUI
defaults it to `0.5` for visual experiments.

Every fit also records full-resolution absolute spectral diagnostics under
`result.metadata["spectral_diagnostics"]`. The public `radial_power_spectrum`
and `compare_spectra` functions can be used independently. Frequencies are
reported as fractions of the axial Nyquist frequency; the comparison separates
DC, very-low, low, mid, high, and very-high energy and includes a combined
`high_frequency_ratio` (result divided by target).

## GUI and example

Run the threaded development GUI:

```bash
python -m gui.test_app
```

It loads common raster formats, edits the principal settings, shows progress, and displays source, reconstruction, contrast-scaled residual, and metrics. The **Allowed procedural atoms** checkboxes enable or disable the registered component families; at least one must remain selected. Under **Texture loss weights**, per-band estimation is enabled by default and disables the four overridden statistical fields. Clear the checkbox to use them manually. MSE remains editable in both modes. Weights must be finite and non-negative, and at least one must be positive. The **Min improvement** field sets the minimum decrease in composite texture loss required to retain another atom; lowering it permits smaller statistical improvements and potentially larger models. The **Result UV extent** slider evaluates `[0, extent)²` at a bounded preview resolution, making procedural continuation and repetition visible without changing the fitted model. Tk widgets are updated only on the main thread and duplicate fits are disabled.

After a fit, the **Spectrum** button opens the full-resolution target/result
diagnostics: absolute radial PSD curves on a logarithmic scale, per-band absolute
and normalized energies, and the combined high-frequency energy ratio.

The adjacent **Measurements** button opens complementary full-resolution
diagnostics. Its tabs compare global contrast and gradient tails, local contrast
at four Gaussian scales, strong-edge density, and absolute directional Fourier
energy in eight orientation wedges. All ratios are reported as result divided by
target so that deficits and excesses are directly visible.

To compare two same-sized rasters and interactively calibrate the four objective
weights, run:

```bash
python -m gui.texture_loss_calibrator
```

The calibrator displays both normalized grayscale inputs, each raw loss component,
its weighted term, and the final normalized `texture_loss`. Editing a weight
automatically recalculates the result.

To inspect the Laplacian pyramid used by the fitting objective, run:

```bash
python -m gui.decomposition_viewer
```

To inspect experimental per-band descriptors and weight proposals, run
`python -m gui.weight_estimator_viewer`. The standalone API is:

```python
from procedural_texture_kernel import WeightEstimator

result = WeightEstimator().analyze(pyramid_band)
print(result.features.to_dict(), result.weights.to_dict())
```

Spectral entropy measures broadband energy, spectral anisotropy measures the
directionality of frequency second moments, off-center autocorrelation measures
repetition, gradient coherence measures aligned edge orientations, and normalized
absolute excess kurtosis measures non-Gaussian/heavy-tailed amplitudes. Normalized
descriptors are approximately `[0, 1]`; raw excess kurtosis is retained as a
diagnostic. Configurable non-negative linear scores add broadband/anisotropic
importance to spectrum, heavy-tail importance to histogram, repetition importance
to autocorrelation, and directional importance to gradient, then normalize to one.
These coefficients are heuristic starting values. Circular autocorrelation can
reflect boundary discontinuities, tiny/constant bands carry little evidence, and
the descriptors do not determine universally optimal loss weights. The fitter now
uses the estimator immediately before constructing each per-band `TextureLoss`.
The extractor and mapper remain public and replaceable for future experiments.

The viewer accepts any supported raster, exposes the band count and base Gaussian
sigma, and displays the source, every signed frequency band, the reconstructed
image, and the contrast-scaled reconstruction error. Band previews map zero to
middle gray and report their numeric range and RMS energy.

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
