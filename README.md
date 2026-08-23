# Procedural Texture Kernel

This repository implements the standalone Python kernel for approximating a 2D raster scalar field with a compact sum of procedural atoms. It follows the accompanying HSPD research design: boundary-aware spectral initialization, sparse residual proposals, statistical texture matching, bounded nonlinear refinement, and coarse-to-fine/local atom candidates. It deliberately contains no Blender dependency.

## Installation

Python 3.10 or newer is required.

```bash
python -m pip install -e ".[dev]"
```

The runtime dependencies are NumPy, SciPy, and Pillow. Tkinter is normally included with desktop Python distributions.

For optional NVIDIA CUDA 12 candidate batching, install the GPU extra into the
same virtual environment (the base install does not pull in CUDA packages):

```bash
python -m pip install -e ".[dev,gpu]"
```

For a different CUDA major version, install the matching official CuPy wheel
instead of `cupy-cuda12x`.

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

`FitResult` retains the full-resolution reconstruction, signed residual, metrics, and fitting history. `TextureFitter.fit` also accepts optional `progress_callback(stage, progress, message)` and `cancel_callback()` callables. Cancellation is cooperative and is checked between fitting stages, candidate tasks, and nonlinear optimizer evaluations; parallel worker pools are shut down before cancellation returns.

## Architecture

- `coordinates.py` owns the one coordinate convention and cached grid construction.
- `components.py` defines typed, serializable Fourier bundles, sinusoid, Gabor, Gaussian RBF, seeded Perlin-noise, and wavelet atoms.
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

Independent decomposition bands can be fitted concurrently with `band_workers`.
The public API defaults to one worker; the development GUI defaults to three.
Threaded results retain band order and deterministic model composition, while
progress is aggregated across bands. Values above the band count are harmlessly
capped. Because numerical libraries may also use internal threads, increasing
the setting beyond the available physical cores can reduce performance.

Candidate projection and scoring within each band can also run concurrently with
`candidate_workers`. It defaults to one so existing callers keep conservative
CPU and memory use; use a value such as 2–4 when fitting one or a few bands with
large candidate sets. Candidate results are consumed in their original order,
so worker scheduling does not change tie-breaking or the fitted model. Running
both levels multiplies concurrency (`band_workers * candidate_workers`), and each
worker holds temporary fitting-resolution arrays. Keep that product near the
available physical core count and reduce either setting if NumPy/SciPy is already
configured to use several native threads.

The optimizer avoids calculating zero-weight texture features during inner
probes. Full loss components are still calculated at recorded iterations and by
`calculate_texture_loss`, so public diagnostics and fit decisions are unchanged.

Set `compute_backend="cupy"` (the **Backend** selector in the development GUI)
to initialize CuPy and replace compatible NumPy/SciPy work with its CUDA
equivalents. Initialization verifies a CUDA device with a real allocation and
kernel reduction before fitting starts; a missing/incompatible CuPy or CUDA
installation raises a clear error instead of silently changing the requested
backend. Image resizing and Gaussian-pyramid construction use CuPy and
`cupyx.scipy.ndimage`; supported candidate families then remain batched on the
device for array operations, reductions, FFTs, and loss evaluation.
Component and model grid evaluation also select their array namespace from the
provided coordinate arrays. Passing CuPy `u`/`v` grids therefore returns CuPy
arrays without changing component parameters or serialization. Seeded noise
keeps deterministic permutation generation on the CPU and transfers only that
small lookup table; its per-pixel work runs on the GPU.

`gpu_batch_size` defaults to 16 and bounds the main device allocation,
approximately `batch_size * height * width * 8` bytes per resident float64
array. Several arrays coexist, so start conservatively at high resolutions and
increase while monitoring VRAM. The target and coordinate grids stay resident
for a complete band. Fourier, spectral-noise, Gabor, Gaussian/RBF, wavelet,
line/edge, DoG/LoG, polynomial, radial/spiral, and constant candidates use CUDA;
other families fall back to CPU and their counts are recorded in band metadata.
Local-structure loss currently makes candidate scoring fall back to CPU because
its complex filter-bank feature has not yet been ported.

CuPy does not provide a drop-in equivalent of `scipy.optimize.minimize`, so
bounded nonlinear atom refinement and the serializable public model/result
boundary intentionally remain on the CPU. Backend scope and whether acceleration
was active are recorded in `FitResult.metadata["backend_scope"]` and
`FitResult.metadata["backend_accelerated"]`.

GPU reductions and FFTs are numerically equivalent but not bit-for-bit identical
to NumPy. Consequently, extremely close candidate ties can produce a different
model. Validate GPU fits with tolerances appropriate to float64 numerical work.

With `band_aware_candidates=True` (the default), high-frequency Laplacian bands search detail families, the low-pass residual searches coherent structure families, and middle bands allow both roles. A user selection containing only families outside a preferred role is preserved as a fallback; set the option to false to use every selected family in every band.

Candidate initialization is residual-adaptive: dominant spectral peaks propose noise frequencies, local structure tensors and support estimates initialize oriented atom directions and sizes, and the positive residual coverage initializes thresholded-noise masks. `noise_seed_candidates` controls the bounded deterministic seed bank (default 4); at most three noise frequencies are retained per iteration. The `spectral_noise` candidate packs the strongest unique residual FFT modes into one atom; `spectral_noise_modes` controls its bounded mode count (default 32).

`masked_noise` is a deliberately constrained compositional atom: it applies independent detail noise to either side of a coherent mask without introducing a general shader-graph search space. Both mask sides are proposed, while mask shape and detail placement are refined continuously.

The experimental `shader_graph` family adds a serializable, acyclic scalar DAG with component, constant, add, multiply, smoothstep, one-minus, and mix nodes. Graphs are topologically ordered and capped at 64 nodes. The fitter intentionally searches only a five-node `noise mask -> smoothstep -> mix(two detail fields)` topology; arbitrary graph-topology evolution is not enabled.

Nonlinear refinement covers the principal smooth and structural families, including thresholded/ridged/domain-warped noise, fBm/turbulence, Voronoi, anisotropic Gaussians, lines, step edges, and DoG/LoG atoms. The fitter retains the original projected candidate whenever the bounded optimizer fails to improve its objective.

Set `detail_refinement=True` to run an adaptive high-frequency residual pass after
the normal multiband fit. It activates only when the fitting-resolution
`high_frequency_ratio` is below `detail_hf_ratio_threshold`, high-pass filters
the signed reconstruction residual, and fits a separate budget of detail atoms.
The candidate must improve absolute HF-energy error. When `mse_weight` is
positive it must also avoid worsening full-image MSE; an MSE weight of zero
removes MSE from both detail fitting and this final acceptance gate.
Configuration includes `detail_max_components`,
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

The fitter includes individual Fourier modes, compact spectral-noise bundles, Gabor, RBF, seeded Perlin-noise, and localized wavelet candidates. Spectral bundles target the high-frequency energy that a small atom budget would otherwise miss, while using integer FFT modes so the result tiles over the source UV interval. Perlin candidates use a deterministic seed bank derived from `FitConfig.seed`; atom merging, LASSO, and simplex noise remain future extensions.

## Coordinates and procedural model

Pixels use normalized half-open coordinates: `u = column / width`, `v = row / height`. The origin is top-left; U increases right and V increases down. Frequencies are cycles per normalized coordinate interval, phase and orientation are radians. With the image-coordinate V axis, positive angles appear clockwise on a conventional screen. Model evaluation is resolution independent.

The component types are:

- `SinusoidComponent`: amplitude, U/V frequency vector, phase.
- `SpectralNoiseComponent`: amplitude plus a deterministic weighted bundle of U/V Fourier frequencies and phases. Its basis is RMS-normalized so candidate projection controls the overall contrast.
- `GaborComponent`: amplitude, center, two Gaussian widths, carrier frequency, orientation, phase.
- `GaussianRBFComponent`: amplitude, center, Gaussian width.
- `PerlinNoiseComponent`: amplitude, base frequency, octave count, persistence, lacunarity, UV offset, and deterministic seed. Its normalized fractal gradient-noise basis continues procedurally outside the source UV range.
- `ThresholdedNoiseComponent`: a rotated fBm field remapped through a smooth threshold, with controllable threshold and edge width for coherent high-contrast regions.
- `MaskedNoiseComponent`: independent detail noise restricted to either side of a smooth thresholded-noise region mask.
- `WarpedRidgeDetailComponent`: independent fine noise restricted to either side of a smooth mask derived from anisotropic, domain-warped multifractal ridges.
- `ShaderGraphComponent`: an embedded validated scalar DAG; currently fitted as a coherent mask mixing two independently seeded detail fields.
- `WaveletComponent`: amplitude, center, anisotropic U/V scales, and orientation. It uses a localized 2D Mexican-hat (Ricker) basis for residual blobs, spots, and band-pass detail.
- Noise families: `VoronoiNoiseComponent`, `FractalBrownianMotionComponent`, `RidgedMultifractalComponent`, `TurbulenceNoiseComponent`, `DomainWarpedNoiseComponent`, and `WarpedRidgedMultifractalComponent`. Ridged multifractals fold every octave independently and expose ridge offset/power, rotation, and anisotropy for vein-like structures. The warped-ridged variant bends those anisotropic ridges with an independently seeded smooth vector field.
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

It opens fullscreen by default; press **Escape** to leave fullscreen or **F11** to toggle it. It loads common raster formats, edits the principal settings, shows progress, and displays source, reconstruction, contrast-scaled residual, and metrics. The **Allowed procedural atoms** checkboxes enable or disable the registered component families; at least one must remain selected. Under **Texture loss weights**, per-band estimation is enabled by default and disables the four overridden statistical fields. Clear the checkbox to use them manually. MSE remains editable in both modes. Weights must be finite and non-negative, and at least one must be positive. The **Min improvement** field sets the minimum decrease in composite texture loss required to retain another atom; lowering it permits smaller statistical improvements and potentially larger models. The **Result UV extent** slider evaluates `[0, extent)²` at a bounded preview resolution, making procedural continuation and repetition visible without changing the fitted model. Tk widgets are updated only on the main thread and duplicate fits are disabled. **Cancel Fit** cooperatively stops the controller, band, and candidate workers; closing the window during a fit performs the same shutdown before destroying Tk.

### Development GUI input reference

`gui/test_app.py` exposes the following fitting inputs. Increasing a budget
usually broadens the search at the cost of time and memory, but does not
guarantee a better result because atoms are retained only when they improve the
configured objective.

| Input | Purpose and influence on the result |
| --- | --- |
| **Components** | Maximum atoms accepted per decomposition band. More atoms permit a more complex, slower model; zero leaves only the fitted bias/plane. |
| **Iterations** | Maximum nonlinear optimizer iterations for each proposed atom. Higher values tune parameters more thoroughly and take longer. |
| **Fit resolution** | Maximum image dimension used during optimization. Larger sources are anti-aliased and downsampled, although final diagnostics use source resolution. Higher values retain finer evidence and cost more memory/time. |
| **Bands** | Number of independently fitted, additive Laplacian bands. More bands isolate scales more finely but multiply work; one disables decomposition. |
| **Seed** | Controls deterministic stochastic candidate generation. Changing it explores different noise and impulse candidates. |
| **Noise seeds** | Number of consecutive seeds tried for each eligible noise family and frequency. More seeds broaden and slow the search. |
| **Max freq (0=auto)** | Highest cycles per normalized UV interval. Zero chooses an image-dependent anti-aliasing limit. Higher values admit finer atoms; lower values suppress fine detail. |
| **Min improvement** | Smallest composite-loss decrease needed to accept an atom. Raising it stops weak additions sooner; lowering it can retain subtler components. |

Texture-loss weights are normalized by their sum, so their ratios matter more
than their absolute scale.

| Input | Purpose and influence on the result |
| --- | --- |
| **Estimate statistical weights per band** | Derives Spectrum, Histogram, Autocorrelation, and Gradient weights from each band's entropy, directionality, repetition, and edge coherence. When selected, those four entries are disabled; the other weights remain manual. |
| **Spectrum** | Matches normalized multiscale log-power spectra, emphasizing frequency distribution rather than absolute energy. |
| **Absolute spectrum** | Matches absolute Fourier energy in radial bands, discouraging too little or too much texture contrast. |
| **Oriented spectrum** | Matches absolute spectral energy by frequency band and direction, helping aligned grain and stripes. |
| **Histogram** | Matches cumulative tonal distributions without requiring corresponding pixels to align. |
| **Autocorrelation** | Matches cyclic spatial correlation and repetition around the origin. |
| **Gradient** | Matches multiscale edge magnitudes and orientation distributions. |
| **MSE** | Matches pixels directly. More weight favors spatial alignment and restrains statistically plausible but displaced results. |
| **Local structure** | Matches oriented-filter magnitudes and cross-scale relationships. It captures richer structure but is one of the most expensive terms. |
| **Local contrast** | Matches distributions of neighborhood standard deviation at several scales. |

The high-frequency section controls an optional detail-only pass after combining
the fitted bands.

| Input | Purpose and influence on the result |
| --- | --- |
| **Enable when HF ratio is below threshold** | Fits the high-pass residual when reconstructed high-frequency energy is deficient. The detail model is retained only when it improves that deficit and does not worsen enabled MSE. |
| **Detail atoms** | Maximum atoms in the extra detail fit. More allow finer recovery and take longer; zero prevents useful refinement. |
| **Min frequency** | Lowest detail-atom frequency. Raising it focuses the pass on finer structure. |
| **HF threshold** | Triggers the pass when result-to-target high-frequency energy is below this ratio. Values nearer one trigger on smaller deficits. |
| **Band workers** | Laplacian bands fitted concurrently. More workers may reduce elapsed time while increasing simultaneous CPU and memory use. |
| **Candidate workers** | CPU threads used to initialize and score candidates inside a band. More can accelerate expensive pools but add overhead. |
| **Backend** | `numpy` uses CPU arrays. `cupy` requests CUDA and requires compatible optional CuPy plus a usable device. |
| **GPU batch** | Homogeneous candidates evaluated together on CUDA. Larger batches reduce launch overhead but use more device memory; this does not affect NumPy fitting. |
| **Spectral modes** | Strongest Fourier modes stored in a spectral-noise bundle. More modes reproduce broader detail in one atom at greater evaluation/serialization cost. |

The joint-refinement section revisits atoms already accepted by the greedy fit.

| Input | Purpose and influence on the result |
| --- | --- |
| **Jointly refit amplitudes** | Periodically solves the bias, plane, and every atom amplitude together. A refit is reverted if it worsens the objective. |
| **Every N accepted atoms** | Interval between amplitude refits. Smaller values refit more often and cost more. |
| **Refine recent atom parameters at end** | Runs end-of-band nonlinear refinement on recent atoms and keeps only objective improvements. |
| **Passes** | Maximum parameter-refinement passes. More passes cost more; processing stops early when none improve. |
| **Recent atoms** | Maximum most-recent atoms revisited per pass. Increasing it exposes more of the model to refinement. |
| **Band-aware atom roles** | Favors detail families in high-frequency bands and structural families in low-frequency bands. Disabling it offers all enabled families to every band. |

Every **Allowed procedural atoms** checkbox determines whether that family can be
proposed. Oscillatory/detail families include sinusoid, spectral noise, Gabor,
wavelet, DoG/LoG, sparse impulse, and line. Geometric families include Gaussian,
edge, polynomial, radial/spiral, constant, and binary primitives. The Perlin,
thresholded/masked, Voronoi, fBm, turbulence, ridged, warped, ridge-detail, and
shader-graph families describe stochastic or coherent regions. Disabling a
family narrows and speeds the search but prevents its representation from being
used; at least one family must remain enabled.

The remaining controls affect workflow or display rather than model fitting.

| Input | Purpose and influence on the result |
| --- | --- |
| **Load Image** | Selects a Pillow-supported raster, converts color to luminance, normalizes it to `[0, 1]`, and clears the previous result. |
| **Fit** | Validates all settings and starts a worker-thread fit. It is disabled while fitting. |
| **Cancel Fit** | Requests cancellation of the current fit and waits for its controller and kernel worker pools to stop. Window close uses the same path. |
| **Export JSON** | Writes the completed `FitResult`—model, metrics, and fitting metadata—to a chosen JSON file. |
| **Spectrum** | Opens source/reconstruction radial power-spectrum diagnostics after a fit. |
| **Measurements** | Opens contrast, gradient, edge-density, local-contrast, and directional-energy diagnostics after a fit. |
| **Result UV extent** | Changes only the continuation preview from `[0, 1)²` up to `[0, 4)²`. It neither refits nor mutates the model; larger extents reveal procedural continuation and repetition. |

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

## Improvement ideas

The current reconstruction is close enough that further work should focus on
procedural generalization, resolution, and runtime rather than simply adding
more atom families. Suggested remaining work, roughly in priority order, is:

1. **Define and measure non-memorizing texture similarity.** A zero MSE weight
   only removes the explicit pixel-error term; it does not make an exact copy a
   bad solution. An exact copy also minimizes spectrum, histogram,
   autocorrelation, gradient, local-contrast, and local-structure discrepancies.
   Add continuation-aware measurements such as statistics over several UV
   windows, random translations, phase-scrambled controls, patch-distribution
   distances, and held-out crops. Model complexity and repetition should be
   reported alongside source-domain loss.
2. **Finish the zero-MSE audit.** High-frequency refinement now inherits
   `mse_weight` and disables its MSE acceptance gate at zero, but several
   proposal mechanisms remain deliberately pixel aligned: the initial plane,
   residual amplitude projection, joint least-squares amplitude proposals, FFT
   phases, and per-band residual fitting. Their proposals are accepted using the
   configured texture objective, yet they strongly bias the search toward the
   source realization. Experiment with phase-free or randomized-phase spectral
   proposals, translation-ensemble objectives, statistic-based amplitude
   initialization, and an explicit `spatial_alignment_weight` separate from
   MSE. Add metadata and tests that distinguish proposal scoring from final
   objective scoring.
3. **Improve high-resolution fitting performance.** Profile feature extraction,
   candidate evaluation, basis synthesis, and nonlinear refinement separately.
   Cache reusable bases and target features, evaluate candidates in batches,
   FFT-accelerate localized correlation, avoid recomputing the complete model
   for single-atom trials, and use coarse-to-fine parameter refinement. After
   the CPU path is measured, consider process-level parallelism or array/GPU
   backends. This should allow a larger `fitting_resolution` without changing
   fitting semantics.
4. **Use a progressive-resolution schedule.** Fit structure at low resolution,
   transfer the model to successively larger rasters, and unlock finer bands and
   higher frequencies at each stage. Re-optimize only parameters affected by a
   new stage. This should be cheaper and more stable than running every family
   against the full-resolution raster from the start.
5. **Add sparse model selection and consolidation.** Penalize component count
   and spectral-bundle mode count, prune atoms whose removal does not harm the
   statistical objective, merge redundant atoms, and compare LASSO/elastic-net
   amplitude refits with the current ridge solve. This is especially important
   for preventing `SpectralNoiseComponent` from becoming a compact raster
   encoding rather than a procedural description.
6. **Strengthen continuation and boundary behavior.** Add optional periodic seam
   loss, boundary-conditioned candidate generation, and diagnostics over UV
   extents larger than one. Separate explicitly tileable components from
   components intended to vary indefinitely, and test both behaviors at several
   output resolutions.
7. **Expand the multiscale representation where diagnostics justify it.** Useful
   candidates include steerable or wavelet-packet decompositions, locally
   modulated spectral noise, improved conditional detail masks, broader adaptive
   seed search, and cross-band dependencies. Add a family only when a measured
   residual statistic cannot be represented efficiently by the existing warped
   ridge, masked-detail, and spectral-bundle atoms.

The MSE issue should therefore not be treated only as a weight-propagation bug.
With all discrepancy losses minimized by the source image, the optimizer has no
reason to prefer a statistically equivalent but different realization. A future
"texture synthesis" mode should explicitly reward generalization or phase
freedom, while a separate "reconstruction" mode can retain the current
pixel-aligned proposals.

## Current limitations

The model is scalar/grayscale and the GUI is a development tool. Candidate scoring evaluates compact procedural candidates directly, but local position search is not FFT-accelerated. Very stochastic, photographic, sharp-edged, or high-entropy fields may require many atoms and are often better represented by conventional textures. Output is not clipped during model synthesis, preserving both genuine statistics and the signed diagnostic residual. Periodic seam constraints, broader/adaptive noise seed searches, discrete wavelet decompositions, SSIM/perceptual losses, color-channel fitting, and batch/GPU paths are not yet implemented.

## Blender add-on

The `blender_addon` package imports a fitter result JSON into Blender 4.5 LTS as
an editable scalar shader node group and material. It accepts both a complete
`FitResult` envelope and a bare schema-v1 model. The generated material converts
Blender UV coordinates to the fitter's top-left, V-down coordinate convention,
keeps the signed model value available on the group, and routes a clamped copy to
Principled BSDF Roughness by default. Base Color, Metallic, Alpha, Emission
Strength, and Bump routing are also available in the import dialog.

To install it:

1. Create a ZIP whose top-level folder is `blender_addon` (the folder containing
   `__init__.py`). From the repository root, PowerShell users can run
   `Compress-Archive -Path blender_addon -DestinationPath ptk_blender_addon.zip`.
2. In Blender, open **Edit > Preferences > Add-ons**, choose **Install from
   Disk**, select the ZIP, and enable **Procedural Texture Kernel Importer**.
3. Select a mesh, open **Material Properties > Procedural Texture Kernel**, and
   choose **Import PTK Fitter Result**. Select a fitter JSON such as
   `blender_addon/example_result.json`.

Analytic component families and spectral bundles are translated literally.
Seeded Perlin/composite noise, Voronoi, and sparse-impulse families do not match
Blender's native texture algorithms; importing a model containing those families
therefore stops unless **Compact (Approximate)** is explicitly enabled. Every
approximation is reported and saved in material custom properties. Large spectral
models show an estimated node count and are protected by a configurable hard
limit. Reimport builds and validates a replacement group before swapping it into
the material, preserving user routing if an import fails.

The add-on uses only Blender's bundled Python standard library and `bpy`, so it
does not add project or Blender Python dependencies.

## Future Blender integration

The intended boundary is:

```text
Raster -> TextureFitter -> ProceduralTextureModel -> JSON/typed parameters
       -> future Blender adapter -> shader node graph or compact GPU evaluator
```

A Blender adapter should depend only on the public model and map bias to a Value node, sinusoids to Wave/Math chains, and localized Gabor/RBF atoms to node groups or shader code. Fitting internals, Tkinter, and SciPy optimizer state do not cross that boundary.
