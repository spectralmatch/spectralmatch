The pipeline runs `steps` in the order supplied. To use the footprint processing
functions before generating seamlines, include them after the raster matching
steps:

```python
from spectralmatch import pipeline

result = pipeline(
    shared_input_images="matched_images",
    shared_output_image_path="mosaic.tif",
    steps=(
        "create_footprints",
        "postprocess_footprints",
        "voronoi_center_seamline",
        "mask",
        "merge",
    ),
    create_footprints_band=1,
    create_footprints_eight_connected=True,
    # create_footprints_metadata_csv="image_metadata.csv",
    create_footprints_metadata_image_field_name="image",
    postprocess_footprints_hole_edge_distance=800,
    postprocess_footprints_hole_to_hole_distance=800,
    postprocess_footprints_hole_cut_width="maximum_inscribed_circle",
    postprocess_footprints_hole_cut_method="corridor",
    postprocess_footprints_simplify_smoothing_radius=220,
    postprocess_footprints_simplify_tolerance=120,
    postprocess_footprints_simplify_area_weight=0.5,
    postprocess_footprints_filter_area_rank=1,
)
```

Postprocessing requires a suitable projected CRS. Tune distances in that CRS's
linear units for your imagery; the example values are starting points for the
WorldView example. Postprocessing is opt-in. The default workflow still creates
footprints automatically when Voronoi has no polygon input.

Footprint paths and layer names pass automatically from creation to postprocessing
and then to any seamline method. Explicit `*_input_polygons` and `*_input_layer`
options override these inputs. External polygons default to the `"footprints"`
layer; pass `*_input_layer=None` to read their first layer. When using preceding
footprints, the default `"footprints"` or None inherits their generated layer name.
Set the same `*_image_field_name`
for footprint creation and the selected seamline method when using a custom field.

Replace `voronoi_center_seamline` with `weighted_seamline` and supply
`weighted_seamline_rank_function` to use ranked seamlines. Fields referenced by the
expression must exist in the input polygons; postprocessing preserves them. Existing
polygons can enter through `postprocess_footprints_input_polygons` or the chosen
seamline step's `*_input_polygons` option.

Use `markov_seamline` for triangle-MRF seamlines on registered, color-corrected
rasters in a common projected CRS:

```python
result = pipeline(
    shared_input_images="matched_images",
    shared_output_image_path="mosaic.tif",
    steps=("create_footprints", "markov_seamline", "mask", "merge"),
    markov_seamline_mesh_max_triangle_area=10000,
    markov_seamline_mesh_min_triangle_angle=20,
)
```

Only one of the three seamline methods can be selected per pipeline. Markov uses
the current pipeline rasters and inherits generated footprint paths and layers.
Set `markov_seamline_input_images` or `markov_seamline_input_footprints` to override
those inputs; explicit footprints default to the `"footprints"` layer, or use
`markov_seamline_input_layer=None` for their first layer. Without footprints,
Markov generates them from raster valid-data masks. Its `markov_seamline_*`
options use `Seamline.markov_triangles` defaults; worker, backend, progress, resume,
and cleanup settings use the shared pipeline options. Markov may also be the final
step with a `.gpkg` output. The default pipeline continues to use Voronoi.

Supply `create_footprints_metadata_csv` to join image-level CSV attributes in either
an explicit footprint step or automatic footprint creation before Voronoi.
`create_footprints_metadata_image_field_name` selects the CSV column containing each
full current image basename, including processing suffixes but without its extension,
as a literal, case-insensitive substring. Each
image must match exactly one row. Other columns become footprint attributes and
remain available to ranking expressions after postprocessing.

Each footprint step can also be the final step, in which case
`shared_output_image_path` is its output `.gpkg` file. Intermediate footprint
outputs appear in the result dictionary under `create_footprints` and
`postprocess_footprints`. Both steps use the shared worker, backend, debugging,
resume, and temporary-output cleanup settings.

::: spectralmatch.pipeline
