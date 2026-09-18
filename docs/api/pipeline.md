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
    postprocess_footprints_edge_distance=800,
    postprocess_footprints_hole_to_hole_distance=800,
    postprocess_footprints_cut_width="maximum_inscribed_circle",
    postprocess_footprints_cut_method="corridor",
    postprocess_footprints_smoothing_radius=220,
    postprocess_footprints_simplify_tolerance=120,
    postprocess_footprints_simplify_area_weight=0.5,
    postprocess_footprints_area_rank=1,
)
```

Postprocessing requires a suitable projected CRS. Tune distances in that CRS's
linear units for your imagery; the example values are starting points for the
WorldView example. Postprocessing is opt-in. The default workflow still creates
footprints automatically when Voronoi has no polygon input.

Footprint paths and layer names pass automatically from creation to postprocessing
and then to either seamline method. Explicit `*_input_polygons` and `*_input_layer`
options override these inputs. When supplying an external polygon path without a
layer name, the source's default layer is used. Set the same `*_image_field_name`
for footprint creation and the selected seamline method when using a custom field.

Replace `voronoi_center_seamline` with `weighted_seamline` and supply
`weighted_seamline_rank_function` to use ranked seamlines. Fields referenced by the
expression must exist in the input polygons; postprocessing preserves them. Existing
polygons can enter through `postprocess_footprints_input_polygons` or the chosen
seamline step's `*_input_polygons` option.

Each footprint step can also be the final step, in which case
`shared_output_image_path` is its output `.gpkg` file. Intermediate footprint
outputs appear in the result dictionary under `create_footprints` and
`postprocess_footprints`. Both steps use the shared worker, backend, debugging,
resume, and temporary-output cleanup settings.

::: spectralmatch.pipeline
