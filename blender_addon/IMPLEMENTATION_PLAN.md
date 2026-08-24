# Blender Add-on Implementation Plan

## Scope and objective

Build a Blender add-on that imports the versioned procedural model contained in a fitter JSON file and turns it into a reusable shader node group and material. The add-on is a consumer of the completed fitter output: it must not import, modify, or depend on the fitting code at run time.

The first target should be Blender 4.5 LTS. Keep Blender-version-specific socket names and node settings behind a small compatibility module so later Blender releases can be supported without changing the importer or component builders.

The imported scalar is suitable for non-color material channels such as Roughness, Height, or a mask. The default generated material should expose the scalar through a group output and connect a clamped copy to Principled BSDF Roughness. It should leave the unclamped value available because the fitted model is a signed additive expression and premature clamping changes the result.

## Input contract

`example_result.json` is a fit-result envelope:

```text
{
  "schema_version": 1,             # result envelope
  "model": {
    "schema_version": 1,           # model schema to validate
    "coordinate_system": "uv_normalized_top_left_v_down_half_open",
    "bias": number,
    "trend_u": number,
    "trend_v": number,
    "components": [{"type": string, "amplitude": number, ...}]
  },
  "metrics": {...},
  "metadata": {...}
}
```

For convenience, also accept a bare model object whose top-level object contains `coordinate_system`, `bias`, and `components`. Ignore `metrics` and `metadata` when building the shader, but retain the source path, schema, metric summary, and import warnings as custom properties on the material.

Validate before creating any Blender data:

- the document is a JSON object;
- an envelope has a `model` object;
- model `schema_version` is exactly `1`;
- `coordinate_system` is exactly `uv_normalized_top_left_v_down_half_open`;
- `components` is a list and every component has a registered `type`;
- required values are finite numbers, integer/count fields are in safe ranges, enum values are known, and parallel spectral arrays have equal lengths;
- component and spectral-mode counts are below configurable safety limits.

Never silently skip an unsupported component. The operator should abort by default with a message listing its type and index. An explicit **Allow Approximate Components** option may build a material with warnings.

## Theory of operation

### Coordinate conversion

The kernel evaluates pixels at half-open, top-left image coordinates: `u = column / width`, `v = row / height`, with positive V downward. Blender's UV map uses positive V upward. Start from the active UV output of a Texture Coordinate node, separate X/Y, and define:

```text
u = UV.x
v = 1 - UV.y
p = (u, v)
```

This makes fitted centers, angles, offsets, and frequency vectors agree with the kernel. Positive fitted angles then retain their documented clockwise appearance. The half-open rule describes raster sampling and requires no resolution-dependent correction in a continuous shader.

Expose optional group inputs `UV`, `U Scale`, `V Scale`, `U Offset`, and `V Offset`. The default path uses the object's active UV map. Apply user scale/offset before the V flip only if the UI describes them as Blender-UV transforms; otherwise expose a single fitted-space vector input and document that its Y axis points down. The latter is less ambiguous and is preferred for version 1.

### Model expression

Construct one scalar expression:

```text
value = bias + trend_u * (u - 0.5) + trend_v * (v - 0.5)
for component in components, preserving JSON order:
    value = value + component_expression(component, u, v)
```

Each component builder returns its complete contribution, including `amplitude`. Accumulate contributions with a balanced binary tree of Add nodes rather than a long chain. This reduces graph depth and makes large spectral bundles more manageable. Do not normalize the final sum.

Create the implementation as nested node groups:

1. one top-level model group;
2. one reusable group definition per component family/formula;
3. one component instance per JSON component;
4. for `spectral_noise`, an internal balanced sum of mode expressions.

Name nodes deterministically (`PTK_0007_spectral_noise`, `PTK_0007_mode_0012`) and place component frames in rows. Store the original component index and JSON type as node custom properties. The generated graph should be readable, but correctness and bounded generation time take priority over decorative layout.

### Component translation

The formulas in `procedural_texture_kernel/components.py` are the specification. Port them literally with Separate/Combine XYZ, Vector Math, and Math nodes. Important base translations are:

| JSON type | Shader construction | Fidelity |
| --- | --- | --- |
| `sinusoid` | `amplitude * cos(2*pi*(frequency_u*u + frequency_v*v) + phase)` | Exact |
| `spectral_noise` | Sum the same sinusoid for every zipped frequency/weight/phase tuple, apply the kernel's RMS normalization, then amplitude | Exact, but potentially very large |
| `gabor` | Rotate centered coordinates, anisotropic Gaussian envelope, cosine carrier, amplitude | Exact |
| `gaussian_rbf` | Centered squared radius, `exp(-r2/(2*sigma^2))`, amplitude | Exact |
| `wavelet` | Rotated/scaled radius and the kernel's normalized 2-D Mexican-hat expression | Exact |
| `anisotropic_gaussian` | Rotate centered coordinates and evaluate the two-axis Gaussian | Exact |
| `line` | Rotated local coordinates, smooth width and length gates, multiply gates and amplitude | Exact if the same smoothstep convention is used |
| `step_edge` | Oriented signed distance followed by the kernel smooth transition | Exact |
| `dog_log` | Difference of Gaussian expressions, or the kernel LoG expression according to `mode` | Exact |
| `polynomial_trend` | Polynomial in centered `u` and `v`, then amplitude | Exact |
| `radial_wave` | Radius from center, sinusoidal phase, optional exponential decay, amplitude | Exact |
| `spiral_wave` | Radial phase plus `arms * atan2(y,x)`, optional decay, amplitude | Exact where Blender's Arctan2 operation is available |
| `binary_primitive` | Rotated disk/box/ring/checker signed tests and thresholds | Exact if boundary conventions match |
| `simple_constant` | Value node equal to amplitude | Exact |
| `perlin_noise`, `fbm`, `turbulence_noise` | Port the kernel permutation/hash, fade, gradient, octave loop as generated Math-node subgraphs | Exact but expensive; Blender Noise Texture is only an opt-in approximation |
| `thresholded_noise` | Exact seeded fBm group, rotation, kernel normalization and smooth threshold | Exact once exact noise exists |
| `masked_noise` | Exact mask fBm and detail fBm groups, optional one-minus, multiply | Exact once exact noise exists |
| `ridged_multifractal` | Exact octave expansion with per-octave fold, ridge offset/power, rotation, anisotropy | Exact once exact noise exists |
| `domain_warped_noise` | Generate the two kernel warp fields, displace coordinates, then exact fBm | Exact once exact noise exists |
| `warped_ridged_multifractal`, `warped_ridge_detail` | Compose exact warp, ridge, threshold-mask, and detail groups | Exact once exact noise exists |
| `voronoi_noise` | Port the kernel seeded cell hash and nearest-point search | Exact but expensive; Blender Voronoi is only an opt-in approximation |
| `sparse_impulse` | Port the kernel deterministic cell/hash and radial impulse expression | Exact but expensive |

Do not assume Blender's Noise, Voronoi, or Gabor textures use the same algorithms, seeding, normalization, octave definitions, or output ranges as the kernel. They can produce a useful procedural resemblance, but cannot be labeled an exact import. Likewise, the kernel `seed` cannot generally be represented by merely translating the input vector of a native Noise Texture.

Before coding each family, copy its evaluate formula and constants into an add-on-side formula note/test fixture. This is not a dependency on the fitter; it is a frozen implementation of schema version 1.

### Exactness strategy and graph size

Implement in three milestones:

1. **Exact analytic core:** bias/plane, sinusoid, spectral bundle, Gabor, Gaussian/RBF, wavelet, geometric, polynomial, radial/spiral, binary, and constant families.
2. **Exact deterministic primitives:** the kernel's seeded gradient noise, hash functions, Voronoi, and sparse impulses, followed by all composite noise families.
3. **Optional compact approximation mode:** use native Blender Noise/Voronoi/Gabor nodes when the user explicitly accepts approximation.

`example_result.json` currently contains many `spectral_noise` components, each with many Fourier modes. A literal graph can therefore contain thousands of Math nodes. The operator must estimate node count before creation and show the estimate in its confirmation text. Provide:

- a configurable hard node limit;
- cancellation/progress reporting between component builds;
- balanced sums;
- reuse of family node-group definitions;
- a `Compact (Approximate)` mode only for supported noise families, never as an implicit fallback.

Do not use a generated raster, baked image, or image texture as a fallback because that defeats the repository's objective. OSL may be investigated later as an optional Cycles-only compact backend, but it must not be the primary implementation because it does not give equivalent Eevee support or a native editable node graph.

### Material construction and assignment

The import operator should offer these destinations:

- create a new material and assign it to the active object's active material slot (default);
- create a material without assignment;
- replace a previously generated PTK node group in the active material.

Use `bpy.data.materials.new`, set `use_nodes = True`, create a `ShaderNodeTree` group with a float `Value` output, and instance it in the material. Add a Principled BSDF and Material Output only when creating a new material. Connect `clamp(Value, 0, 1)` to Roughness by default. Also expose optional one-click routing to Base Color (grayscale), Metallic, Alpha, Emission Strength, or Bump Height; routing is presentation logic and must not alter the generated model group.

Replacement should be transactional: parse and validate first, build a new temporary node group, then swap the material's group-node reference only after success. Remove the old generated group only when it has no remaining users. A failed import must leave the prior material intact.

## Suggested add-on layout

```text
blender_addon/
  __init__.py             # bl_info, registration
  operators.py            # import/reimport operators and file selector
  panels.py               # Shader Editor / Material properties UI
  properties.py           # Scene or Material property group
  schema.py               # JSON loading and strict schema-v1 validation
  builder.py              # transaction, material and top-level group creation
  node_utils.py           # typed socket lookup, linking, balanced sums, layout
  compatibility.py        # Blender-version and socket-name differences
  components/
    analytic.py
    geometric.py
    noise.py
    composites.py
  tests/
    fixtures/             # small hand-authored schema-v1 models
```

The Blender add-on should use only Blender's bundled Python standard library and `bpy`; do not package NumPy, SciPy, Pillow, CuPy, or the fitter package.

## Pseudocode

### Registration and UI

```python
class PTK_OT_import_material(Operator, ImportHelper):
    bl_idname = "ptk.import_material"
    filename_ext = ".json"
    filter_glob: StringProperty(default="*.json", options={"HIDDEN"})
    allow_approximate: BoolProperty(default=False)
    assign_to_active: BoolProperty(default=True)

    def execute(self, context):
        try:
            document = load_json_utf8(self.filepath)
            model = extract_and_validate_model(document)
            estimate = estimate_node_count(model, self.allow_approximate)
            enforce_limits(estimate)
            material, warnings = import_transactionally(
                context, model, self.filepath,
                allow_approximate=self.allow_approximate,
                assign=self.assign_to_active,
            )
        except ImportError as error:
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}
        self.report({"WARNING"} if warnings else {"INFO"}, summarize(warnings))
        return {"FINISHED"}
```

### Parsing and validation

```python
def extract_and_validate_model(document):
    require_mapping(document)
    model = document["model"] if "model" in document else document
    require(model["schema_version"] == 1)
    require(model["coordinate_system"] == EXPECTED_COORDINATES)
    require_finite(model.get("bias", 0.0), "bias")
    require_finite(model.get("trend_u", 0.0), "trend_u")
    require_finite(model.get("trend_v", 0.0), "trend_v")
    for index, component in enumerate(require_list(model, "components")):
        validator = COMPONENT_VALIDATORS.get(component.get("type"))
        require(validator is not None, f"component {index}: unknown type")
        validator(component, index)
    return model
```

### Transactional graph build

```python
def import_transactionally(context, model, source_path, allow_approximate, assign):
    temporary_group = bpy.data.node_groups.new(unique_temp_name(), "ShaderNodeTree")
    try:
        add_group_output_socket(temporary_group, "Value", "NodeSocketFloat")
        u, v = build_fitted_coordinates(temporary_group)
        terms = [constant(model.get("bias", 0.0))]
        terms += build_plane_terms(u, v, model)

        warnings = []
        for index, component in enumerate(model["components"]):
            builder = COMPONENT_BUILDERS[component["type"]]
            term, component_warnings = builder(
                temporary_group, u, v, component,
                exact=not allow_approximate,
                index=index,
            )
            terms.append(term)
            warnings.extend(component_warnings)
            update_progress(index + 1, len(model["components"]))

        value = balanced_add(temporary_group, terms)
        link(value, group_output_input("Value"))
        validate_generated_tree(temporary_group)

        material = create_or_prepare_material_without_destroying_previous()
        commit_group_and_material(material, temporary_group, source_path, model, warnings)
        if assign:
            assign_material_to_active_object(context, material)
        return material, warnings
    except Exception:
        bpy.data.node_groups.remove(temporary_group)
        raise
```

### Example analytic builders

```python
def build_sinusoid(tree, u, v, c):
    phase = add(
        multiply(TAU, add(multiply(c["frequency_u"], u),
                          multiply(c["frequency_v"], v))),
        c.get("phase", 0.0),
    )
    return multiply(c["amplitude"], math_node("COSINE", phase))

def build_spectral_noise(tree, u, v, c):
    validate_equal_lengths(c, "frequencies_u", "frequencies_v", "weights", "phases")
    modes = []
    for fu, fv, weight, phase in zip_fields(c):
        angle = add(multiply(TAU, add(multiply(fu, u), multiply(fv, v))), phase)
        modes.append(multiply(weight, math_node("COSINE", angle)))
    basis = multiply(kernel_rms_normalizer(c["weights"]), balanced_add(tree, modes))
    return multiply(c["amplitude"], basis)
```

### Reimport

```python
def reimport_active_material(context):
    material = context.object.active_material
    source = material.get("ptk_source_path")
    require(source, "active material was not imported by PTK")
    # Build and validate a fresh group first; preserve user material routing.
    new_group = build_from_file(source)
    old_group_node = find_tagged_model_group_node(material.node_tree)
    old_group = old_group_node.node_tree
    old_group_node.node_tree = new_group
    remove_if_orphan(old_group)
```

## Verification plan

Automate validation with Blender background mode and small fixtures, not with the large sample alone.

1. Create one fixture per component type plus mixed, negative-amplitude, empty-component, and invalid-schema fixtures.
2. Independently sample the generated shader at known UV coordinates and compare it with golden values exported from the kernel when the schema-v1 formula fixture is created.
3. Use strict tolerances for analytic components and separately documented tolerances for explicitly approximate native-node mappings.
4. Test the V flip with asymmetric values and off-center components; a symmetric texture will not detect the error.
5. Test JSON envelope and bare-model inputs, NaN/Infinity rejection, unknown types, malformed spectral arrays, safety limits, cancellation, rollback, assignment, and reimport.
6. Test both Cycles and Eevee rendering for the native node backend.
7. Load `example_result.json` as an integration/performance case and record node count, import time, `.blend` size, shader compile time, and viewport responsiveness.

Definition of done for exact mode: every registered schema-v1 component imports without raster data, invalid data fails before material mutation, and sampled values agree with the kernel within the chosen floating-point tolerance. Definition of done for approximate mode: every approximation is opt-in, visibly reported, and stored in material metadata.

## Blender API and SDK references

- [Blender Python API home](https://docs.blender.org/api/current/)
- [Add-on tutorial](https://docs.blender.org/manual/en/latest/advanced/scripting/addon_tutorial.html)
- [`bpy.types.Operator`](https://docs.blender.org/api/current/bpy.types.Operator.html) and [`bpy_extras.io_utils.ImportHelper`](https://docs.blender.org/api/current/bpy_extras.io_utils.html#bpy_extras.io_utils.ImportHelper) for the JSON import action and file browser
- [`bpy.types.Panel`](https://docs.blender.org/api/current/bpy.types.Panel.html) for Material Properties or Shader Editor UI
- [`bpy.props`](https://docs.blender.org/api/current/bpy.props.html) for operator and add-on settings
- [`bpy.utils.register_class`](https://docs.blender.org/api/current/bpy.utils.html#bpy.utils.register_class) for add-on registration
- [`bpy.data`](https://docs.blender.org/api/current/bpy.data.html), [`BlendDataMaterials`](https://docs.blender.org/api/current/bpy.types.BlendDataMaterials.html), and [`BlendDataNodeTrees`](https://docs.blender.org/api/current/bpy.types.BlendDataNodeTrees.html) for material and node-group data blocks
- [`Material.use_nodes`](https://docs.blender.org/api/current/bpy.types.Material.html#bpy.types.Material.use_nodes) and [`ShaderNodeTree`](https://docs.blender.org/api/current/bpy.types.ShaderNodeTree.html) for shader graphs
- [`NodeTree.nodes`](https://docs.blender.org/api/current/bpy.types.NodeTree.html#bpy.types.NodeTree.nodes), [`NodeTree.links`](https://docs.blender.org/api/current/bpy.types.NodeTree.html#bpy.types.NodeTree.links), and [`NodeTreeInterface.new_socket`](https://docs.blender.org/api/current/bpy.types.NodeTreeInterface.html#bpy.types.NodeTreeInterface.new_socket) for graph construction and group interfaces
- [`ShaderNodeMath`](https://docs.blender.org/api/current/bpy.types.ShaderNodeMath.html), [`ShaderNodeVectorMath`](https://docs.blender.org/api/current/bpy.types.ShaderNodeVectorMath.html), and [`ShaderNodeSeparateXYZ`](https://docs.blender.org/api/current/bpy.types.ShaderNodeSeparateXYZ.html) for literal formula translation
- [`ShaderNodeTexCoord`](https://docs.blender.org/api/current/bpy.types.ShaderNodeTexCoord.html) and the [Texture Coordinate node manual](https://docs.blender.org/manual/en/latest/render/shader_nodes/input/texture_coordinate.html) for UV input
- [`ShaderNodeTexNoise`](https://docs.blender.org/api/current/bpy.types.ShaderNodeTexNoise.html), [`ShaderNodeTexVoronoi`](https://docs.blender.org/api/current/bpy.types.ShaderNodeTexVoronoi.html), and [`ShaderNodeTexGabor`](https://docs.blender.org/api/current/bpy.types.ShaderNodeTexGabor.html) for explicitly approximate compact mappings
- [`ShaderNodeBsdfPrincipled`](https://docs.blender.org/api/current/bpy.types.ShaderNodeBsdfPrincipled.html), [`ShaderNodeBump`](https://docs.blender.org/api/current/bpy.types.ShaderNodeBump.html), and [`ShaderNodeOutputMaterial`](https://docs.blender.org/api/current/bpy.types.ShaderNodeOutputMaterial.html) for material routing
- [Blender command-line arguments](https://docs.blender.org/manual/en/latest/advanced/command_line/arguments.html) for background-mode add-on tests
- [Python API gotchas: internal data and Python objects](https://docs.blender.org/api/current/info_gotchas_internal_data_and_python_objects.html) for safe data-block mutation patterns

Use `docs.blender.org/api/current` only as a development convenience. Before release, pin the tested Blender version in `bl_info`, CI, and release notes, and verify every referenced node operation and socket name against that version's API.
