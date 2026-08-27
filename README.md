# Procedural Texture Kernel

This repository implements the standalone Python kernel for approximating a 2D raster scalar field with a compact sum of procedural atoms. It follows the accompanying HSPD research design: boundary-aware spectral initialization, sparse residual proposals, statistical texture matching, bounded nonlinear refinement, and coarse-to-fine/local atom candidates. It deliberately contains no Blender dependency.

## Installation

Python 3.10 or newer is required.

The recommended installation and launch procedure is to run `./run_app.ps1`
from PowerShell on Windows, or `sh ./run_app.sh` on Linux and macOS. These
helpers check for Python, create a local `.venv` when needed, install the
project dependencies from `pyproject.toml`, optionally install the GPU extra,
and launch the development GUI. When the GUI closes, the helper deactivates the
virtual environment and exits. If Python is unavailable, it recommends
installing Python 3.14 before trying again.

For a manual development installation, run:

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

## GUI and example

To open the development GUI, run:

```bash
python -m gui.test_app
```

The window opens in fullscreen mode. Press **Escape** to leave fullscreen, or
press **F11** to switch fullscreen mode on or off.

A typical workflow is:

1. Select **Load Image** and choose an image.
2. Adjust the main settings if needed. The defaults are a good starting point.
3. Select **Fit** and wait while the model is built. The progress area shows
   what the program is doing.
4. Compare the original image with the reconstructed result and the difference
   between them. You can also inspect the reported measurements.
5. Select **Export JSON** to save the finished procedural model.

The **Allowed procedural atoms** checkboxes control which kinds of patterns the
program may use to rebuild the image. At least one must be selected. Under
**Texture loss weights**, **Estimate statistical weights per band** is selected
by default. In this mode, the program chooses the Spectrum, Histogram,
Autocorrelation, and Gradient weights automatically for each scale of detail.
Clear the checkbox if you want to enter those weights yourself. The MSE weight
can always be edited. All weights must be zero or greater, and at least one
weight must be greater than zero.

**Min improvement** controls when the program stops adding patterns. A lower
value may produce a more detailed but larger model and take longer. The
**Result UV extent** slider previews the model beyond the original image area,
which helps reveal repetition and procedural continuation. It changes only the
preview, not the fitted model.

Only one fit can run at a time. Select **Cancel Fit** to stop the current fit
safely. Closing the window while a fit is running performs the same safe
shutdown.

### Development GUI input reference

The controls below determine how the procedural model is built. Settings that
allow more work or more components usually increase processing time and memory
use. They may improve the result, but improvement is not guaranteed because the
program keeps a new pattern only when it improves the selected measurements.

| Input | Purpose and influence on the result |
| --- | --- |
| **Components** | Maximum number of patterns that may be kept for each detail band. A higher value can represent a more complex image, but takes longer and creates a larger model. A value of zero keeps only the basic brightness or gradient. |
| **Iterations** | Maximum number of attempts used to fine-tune each proposed pattern. Higher values may tune it more accurately, but take longer. |
| **Fit resolution** | Maximum width or height used while building the model. Higher values preserve more fine detail, but require more time and memory. Large images are reduced to this size during fitting; final measurements still use the original size. |
| **Bands** | Number of layers used to separate broad shapes from fine details. More bands give the program finer control over different scales, but increase the work. A value of one turns this separation off. |
| **Seed** | Selects a repeatable set of random pattern candidates. Using the same settings and seed gives the same candidates; changing the seed explores different possibilities. |
| **Noise seeds** | Number of variations tried for each suitable noise pattern and frequency. More variations broaden the search, but take longer. |
| **Max freq (0=auto)** | Limits how fine the generated patterns may be. Zero lets the program choose a suitable limit for the image. Higher values allow finer detail; lower values favor broader patterns. |
| **Min improvement** | Minimum improvement required before a new pattern is kept. A higher value stops sooner; a lower value may retain subtle details and create a larger model. |

The texture-loss weights control what “a good match” means. Their proportions
matter more than their exact values. For example, `2, 1, 1` has the same balance
as `1, 0.5, 0.5`.

| Input | Purpose and influence on the result |
| --- | --- |
| **Estimate statistical weights per band** | Automatically chooses the Spectrum, Histogram, Autocorrelation, and Gradient weights for each scale of detail. When selected, those four fields cannot be edited. Other weights remain manual. |
| **Spectrum** | Matches the overall balance of broad, medium, and fine patterns, regardless of their exact position. |
| **Absolute spectrum** | Matches the amount of contrast at different pattern sizes, helping prevent a result that looks too flat or too strong. |
| **Oriented spectrum** | Matches both pattern size and direction, which is useful for grain, fibers, and stripes. |
| **Histogram** | Matches the overall distribution of light and dark values without requiring individual pixels to appear in the same place. |
| **Autocorrelation** | Matches repeated spacing and recurring structure in the image. |
| **Gradient** | Matches the strength and direction of edges at several scales. |
| **MSE** | Compares corresponding pixels directly. A higher value favors a result that lines up closely with the source image. |
| **Local structure** | Matches more complex relationships between nearby shapes and details at different scales. It can capture richer structure, but is one of the slowest measurements. |
| **Local contrast** | Matches how much light and dark variation appears within small neighborhoods at several sizes. |

The high-frequency section controls an optional final pass for fine details.
This pass runs after the main model has been assembled.

| Input | Purpose and influence on the result |
| --- | --- |
| **Enable when HF ratio is below threshold** | Runs the extra pass when the result is missing enough fine detail. The added detail is kept only if it improves the result without worsening pixel matching when MSE is enabled. |
| **Detail atoms** | Maximum number of patterns allowed in the extra detail pass. More patterns may recover finer detail, but take longer. A value of zero effectively disables the pass. |
| **Min frequency** | Sets the coarsest pattern allowed in the detail pass. Raising it makes the pass focus on finer detail. |
| **HF threshold** | Decides how much fine detail may be missing before the extra pass starts. Values closer to one make it start for smaller differences. |
| **Band workers** | Number of detail bands processed at the same time. More workers can finish sooner, but use more processor capacity and memory. |
| **Candidate workers** | Number of CPU tasks used at the same time while testing patterns within a band. More may improve speed, but can also add overhead. |
| **Backend** | Select `numpy` to use the CPU. Select `cupy` to use a compatible NVIDIA CUDA graphics card; this requires the optional CuPy package and a working CUDA setup. |
| **GPU batch** | Number of similar pattern candidates tested together on the GPU. Larger values may be faster, but use more graphics memory. This setting has no effect with the NumPy backend. |
| **Spectral modes** | Number of strong frequency patterns stored together in one spectral-noise component. More modes can capture broader fine detail, but make the model slower and larger. |

The joint-refinement section lets the program revisit patterns it has already
accepted and improve how they work together.

| Input | Purpose and influence on the result |
| --- | --- |
| **Jointly refit amplitudes** | Occasionally readjusts the strength of all accepted patterns together. A change is undone if it makes the result worse. |
| **Every N accepted atoms** | Sets how often those strengths are readjusted. Smaller values do this more often and take more time. |
| **Refine recent atom parameters at end** | Fine-tunes recently added patterns after each band is fitted. Only improvements are kept. |
| **Passes** | Maximum number of fine-tuning rounds. More rounds may take longer; the program stops early when a round makes no improvement. |
| **Recent atoms** | Maximum number of recently added patterns fine-tuned in each round. A higher value revisits more of the model. |
| **Band-aware atom roles** | Uses fine-detail pattern types mainly for fine-detail bands and broad structural types mainly for broad bands. Turn it off to allow every selected type in every band. |

Each **Allowed procedural atoms** checkbox represents a family of patterns the
program can use. Some create waves and fine details, some create geometric
shapes or edges, and others create noise-like or natural-looking regions.
Disabling families can make the search faster, but also prevents the program
from using those kinds of patterns. At least one family must remain enabled.

The buttons above the checkboxes provide quick selections. **Select all**
enables every family. **Deselect all** clears the list so you can build your own
selection; remember that a fit still requires at least one family. **Select all
supported by non compact flow** enables only the pattern families that the
Blender add-on can reproduce exactly through its non-compact analytic import.
Use this preset when you plan to export the result to Blender without enabling
compact approximation. The preset excludes `shader_graph` because the fitter's
current shader graph contains Perlin patterns, which this Blender import mode
cannot reproduce exactly.

The remaining controls manage files, fitting, and previews. They do not change
how the model is built unless stated otherwise.

| Input | Purpose and influence on the result |
| --- | --- |
| **Load Image** | Opens a supported image, converts it to grayscale, and clears the previous result. |
| **Fit** | Checks the settings and starts building the model. It is unavailable while a fit is already running. |
| **Cancel Fit** | Safely stops the current fit. Closing the window during a fit uses the same process. |
| **Export JSON** | Saves the finished model, its measurements, and fitting information to a JSON file. |
| **Spectrum** | After a fit, opens a detailed comparison of the source and result at different pattern sizes. |
| **Measurements** | After a fit, opens comparisons for contrast, edges, local variation, gradients, and directional patterns. |
| **Result UV extent** | Expands the preview from one image-sized area up to four areas in each direction. This reveals continuation and repetition without rebuilding or changing the model. |

After a fit, select **Spectrum** for a full-resolution comparison of how much
broad, medium, and fine detail appears in the source and the result. The window
includes detailed graphs and shows whether the result has too little or too much
high-frequency detail.

Select **Measurements** for other full-resolution comparisons, including overall
contrast, gradients, local contrast at several scales, strong edges, and the
amount of detail in different directions. Ratios compare the result with the
source: values below `1` indicate less of a feature, values above `1` indicate
more, and `1` indicates a match.

The repository also includes several specialized viewing tools. To compare two
images of the same size and experiment with four matching weights, run:

```bash
python -m gui.texture_loss_calibrator
```

The calibrator shows both images in grayscale, the difference measured by each
method, the effect of each weight, and the final combined texture-loss value.
Changing a weight updates the result immediately.

To see how an image is separated into broad shapes and progressively finer
detail bands, run:

```bash
python -m gui.decomposition_viewer
```

To inspect the image properties used for automatic per-band weight selection,
run `python -m gui.weight_estimator_viewer`. Developers can access the same
analysis through the API:

```python
from procedural_texture_kernel import WeightEstimator

result = WeightEstimator().analyze(pyramid_band)
print(result.features.to_dict(), result.weights.to_dict())
```

The estimator looks for varied detail, dominant directions, repetition, aligned
edges, and unusual light or dark values. It converts these properties into
suggested Spectrum, Histogram, Autocorrelation, and Gradient weights, normalized
so they add up to one. These suggestions are useful starting points rather than
universally best settings: image boundaries and bands with very little detail
can affect the analysis. The fitter runs this estimator separately for each
detail band when automatic weight estimation is enabled.

The viewer accepts any supported image. You can choose the number of bands and
the base blur amount. It displays the source, each detail band, the reconstructed
image, and an enhanced view of the reconstruction error. Each band also reports
its numeric range and average energy.

To run a self-contained example without choosing an image, use:

```bash
python examples/basic_usage.py
```

The example creates a simple image from two wave patterns, fits a procedural
model to it, prints the measurements, and saves preview images and JSON in
`example_output/`.

## Blender add-on

The `blender_addon` package imports a fitter result JSON into Blender 4.5 LTS as
an editable scalar shader node group and material. It accepts both a complete
`FitResult` envelope and a bare schema-v1 model. The generated material converts
Blender UV coordinates to the fitter's top-left, V-down coordinate convention,
keeps the signed model value available on the group, and routes a clamped copy to
Principled BSDF Roughness by default. Base Color, Metallic, Alpha, Emission
Strength, and Bump routing are also available in the import dialog.

To install it:

1. From the repository root, run `python blender_addon/build_addon.py`. This
   creates a clean `ptk_blender_addon-1.0.2.zip` without Python caches or test
   data; its top-level folder is `blender_addon`.
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

## Glossary

- **Amplitude** — The strength of a component: increasing its magnitude makes
  that component's effect more visible. A negative amplitude reverses its light
  and dark contribution.
- **Anti-aliasing** — Smoothing applied when reducing an image or limiting fine
  patterns so they do not create false jagged edges or moiré patterns.
- **Atom / component** — One simple mathematical pattern used as a building
  block of the reconstructed texture. Examples include a wave, line, spot,
  edge, or noise pattern. In this project, the two words are often used
  interchangeably.
- **Autocorrelation** — A measurement of how strongly an image resembles a
  shifted copy of itself. It helps identify repeated spacing and patterns.
- **Backend** — The system used to perform numerical calculations. The NumPy
  backend uses the CPU, while the optional CuPy backend uses a compatible
  NVIDIA GPU.
- **Band / decomposition band** — One layer of an image containing a particular
  range of detail sizes. Fine bands contain small details; coarse bands contain
  broad shapes and gradual changes.
- **Bias** — A single value added across the whole model to set its basic
  brightness level.
- **Candidate** — A possible atom that the fitter tests before deciding whether
  to add it to the model.
- **Contrast** — The amount of difference between light and dark values. A
  high-contrast texture has stronger light and dark variations.
- **CPU** — The computer's general-purpose processor. It runs the default NumPy
  fitting backend.
- **CUDA** — NVIDIA's platform for running calculations on compatible graphics
  cards. It is required by this project's optional CuPy backend.
- **CuPy** — A Python array library similar to NumPy that performs supported
  calculations on an NVIDIA GPU.
- **Deterministic** — Producing the same result whenever the same input,
  settings, and seed are used.
- **FFT (Fast Fourier Transform)** — A fast method for separating an image into
  its component frequencies. The fitter uses it to find strong repeating or
  wave-like patterns.
- **Fit / fitting** — The process of searching for and adjusting procedural
  components so their combined result resembles the source image.
- **Fourier analysis** — A way to describe an image as a combination of waves
  with different frequencies, strengths, and directions. FFT is the fast
  calculation used to perform this analysis.
- **Frequency** — How often a pattern repeats across an area. Low frequencies
  describe broad, slow changes; high frequencies describe fine details.
- **GPU** — A graphics processor that can also run many numerical calculations
  in parallel. A compatible NVIDIA GPU can optionally accelerate parts of the
  fitting process.
- **Gradient** — The amount and direction of brightness change between nearby
  pixels. Strong gradients usually represent visible edges.
- **Grayscale** — An image containing brightness values but no color. Black,
  gray, and white represent different brightness levels.
- **Histogram** — A count of how often different brightness values occur in an
  image, without considering where those values appear.
- **High-frequency detail** — Small, rapidly changing features such as fine
  grain, tiny scratches, or sharp edges.
- **JSON** — A text-based file format used here to save the procedural model,
  its settings, measurements, and metadata.
- **Laplacian pyramid** — A method of separating an image into several bands,
  from fine detail to broad structure, which can later be added together to
  recreate the image.
- **Loss / objective** — A numerical score describing the difference between
  the source and reconstructed images. Lower values indicate a closer match.
- **Luminance** — The perceived brightness of a color. Color images are
  converted to luminance because the current model works with grayscale values.
- **Metadata** — Additional information saved about a fit, such as settings,
  progress records, backend details, and per-band measurements.
- **MSE (Mean Squared Error)** — The average squared difference between matching
  pixels in two images. It strongly favors placing features in the same
  locations as the source.
- **Normalization** — Converting values to a consistent scale so they can be
  compared or processed reliably, commonly the range from `0` to `1`.
- **NumPy** — The Python numerical array library used by the default CPU
  backend.
- **Optimizer / optimization** — The part of the fitter that repeatedly adjusts
  component settings to reduce the loss score.
- **Perlin noise** — A smoothly varying, repeatable noise pattern commonly used
  to imitate natural variation such as clouds, stone, or terrain.
- **Plane** — A simple gradual brightness slope across the image. Together with
  the bias, it represents the model's most basic large-scale brightness.
- **Procedural texture** — A texture described by mathematical rules and
  parameters instead of only by a fixed grid of stored pixels. It can be
  evaluated at different sizes or beyond the original image area.
- **Raster image / raster texture** — An image stored as a rectangular grid of
  pixels, such as a PNG, JPEG, or TIFF file. Enlarging a raster does not create
  new underlying detail.
- **Reconstruction / result** — The image produced by evaluating all fitted
  procedural components together.
- **Residual** — The difference left after subtracting the reconstruction from
  the source image. It shows what the current model has not yet reproduced.
- **Seed** — A number used to create a repeatable sequence of random-looking
  choices. Changing it lets the fitter explore different noise candidates.
- **Serialization** — Converting the model and its settings into data that can
  be saved and loaded, such as the project's JSON output.
- **Scalar field** — A set of positions where each position has one numerical
  value. A grayscale texture is a two-dimensional scalar field whose values
  represent brightness.
- **Spectrum / power spectrum** — A description of how much of an image's
  variation belongs to broad, medium, and fine frequencies. It does not require
  features to appear at the same positions.
- **Texture-loss weight** — A value controlling how much one comparison method,
  such as Spectrum or MSE, contributes to the combined loss score.
- **UV coordinates / UV extent** — Two coordinates, U and V, used to locate
  points on a texture. The original texture occupies the range from `0` up to
  `1` in each direction; a larger extent previews the procedural model beyond
  that original area.
