# Joint coregistration

## Grid-distributed ORB matching

ORB searches square windows centered on a regular grid inside each pair's overlap. `tie_grid_spacing` sets the distance between search centers in the shared input CRS units, even when `local_model="none"`. An overlap smaller than one grid cell gets a search at its center. `local_grid_spacing` independently controls the local deformation mesh.

`tie_search_radius` is a positive finite number with one unit suffix:

- `"128px"` (default): a radius of 128 pixels at the coarser input resolution, giving approximately 256 × 256 matching pixels before overlap clipping.
- `"100crs"`: a radius of **100 input CRS units**, giving a 200 × 200 CRS-unit window.

Decimal values such as `"12.5crs"` are supported. Bare numbers, `None`, missing or unknown suffixes, and combined units are rejected. Windows and their validity masks are read locally from lazy overlap VRTs; GDAL may fetch surrounding storage blocks. Empty masks, textureless windows, and windows smaller than 63 matching pixels on either axis are skipped.

`tie_orb_max_features` is a positive integer detection cap per image per window, default 100. Descriptor matching keeps mutually best pairs and then the best half by descriptor distance. Displacement and RANSAC filtering remove outliers before final selection.

`tie_max_matches_per_window` is a positive integer cap on selected matches per search window per image pair, default 1. Each match stays with the window that detected it, including when another grid center is closer. Matches nearest their own window center in the reference image are kept first. Set it to `None` to keep all surviving matches. A request larger than the available match count keeps the available matches; there is no minimum-separation rejection between retained points. Identical selected pairs from overlapping windows are stored once.

```python
from spectralmatch import joint_coregistration

joint_coregistration(
    input_images="input/*.tif",
    output_images="coregistered",
    tie_grid_spacing=500.0,
    tie_search_radius="128px",
    tie_orb_max_features=100,
    tie_max_matches_per_window=3,
    tie_load_path="cache/ties.json",
    tie_save_path="cache/ties.json",
    tie_save_crs_path="gis/ties.gpkg",
)
```

## Selected-point cache

JSON saves only selected matches in original zero-based `[column, row]` coordinates, using extension-free image basenames. Matching cached pairs are validated and reused directly, including empty pairs; only absent pairs are calculated. Reloading bypasses ORB, displacement/RANSAC filtering, and per-window selection. The global and local alignment solvers still run with the current settings.

A missing `tie_load_path` file emits a warning and loads nothing, so the example can run before its cache exists. Existing malformed files or out-of-bounds coordinates still raise errors. Changing detection or selection settings does not alter already cached points; omit loading or remove the cache to calculate new points. Regenerate older JSON files containing raw candidates once to obtain selected-point caches.

In `pipeline`, use the same options with the `joint_coregistration_` prefix, including `joint_coregistration_tie_orb_max_features` and `joint_coregistration_tie_max_matches_per_window`.

## GIS point export

Set `tie_save_crs_path` to a vector filename such as `"gis/ties.gpkg"`, `"gis/ties.geojson"`, `"gis/ties.shp"`, or `"gis/ties.fgb"`. The extension selects the vector driver. Export uses only selected ties, including directly reused JSON ties, and writes two point features for each match: one from each image. GeoPackage output uses the `ties` layer, replacing that layer on subsequent exports while preserving other layers.

Each point uses its own original image geotransform and the common input CRS. Conversion includes the pixel-center offset and any rotation or skew. These are the original matched locations before global or local correction. The export streams features without reading raster pixels.

| Field | Meaning |
| --- | --- |
| `tie_id` | Shared, one-based match identifier for the two point features; unique across image pairs in the export. |
| `image` | Extension-free basename of the image containing this point. |
| `match_img` | Extension-free basename of the paired image. |
| `pixel_col` | Original zero-based fractional pixel column. |
| `pixel_row` | Original zero-based fractional pixel row. |

The vector export can be used independently of JSON saving, even with both alignment stages disabled or when raster outputs are reused. No selected ties produces an empty vector layer with the input CRS. Use a path separate from input/output rasters and tie JSON caches. In `pipeline`, the option is `joint_coregistration_tie_save_crs_path`.

## Parameter names

Tie-point options use the `tie_` prefix and appear before the global and local options in the function signature. Update existing calls using these renamed keywords; the previous names are no longer accepted by `joint_coregistration`.

| Previous name | Current name |
| --- | --- |
| `tie_point_feature_method` | `tie_feature_method` |
| `tie_point_grid_spacing` | `tie_grid_spacing` |
| `tie_point_search_radius` | `tie_search_radius` |
| `tie_point_orb_max_features` | `tie_orb_max_features` |
| `tie_point_max_matches_per_window` | `tie_max_matches_per_window` |
| `tie_point_maximum_displacement` | `tie_maximum_displacement` |
| `tie_point_ransac_reprojection_threshold` | `tie_ransac_reprojection_threshold` |
| `robust_loss` | `tie_robust_loss` |
| `robust_loss_scale` | `tie_robust_loss_scale` |
| `tie_point_save_path` | `tie_save_path` |
| `tie_point_load_path` | `tie_load_path` |
| `global_tie_point_alignment_strength` | `global_tie_alignment_strength` |
| `local_tie_point_alignment_strength` | `local_tie_alignment_strength` |
| `global_image_position_preservation_weights` | `global_image_movement_penalty_weights` |
| `local_image_position_preservation_weights` | `local_image_movement_penalty_weights` |

Search-center spacing uses `tie_grid_spacing`, while `local_grid_spacing` controls only the deformation mesh; both default to 500 CRS units. The same renames apply after the `joint_coregistration_` prefix in `pipeline`. PIF cache inputs are now `load_ties` and `pif_load_ties`, including `global_regression_pif_load_ties` in the pipeline. The JSON field remains `"tie_points"`, so existing selected-point caches remain compatible.

## Global and local controls

The `global_image_movement_penalty_weights` and `global_tie_alignment_strength` options control only the image-wide correction stage. Their `local_` counterparts control the additional local deformation. Both stages use the same selected tie points, and the local stage works on the residual mismatch after global correction. `tie_robust_loss` and `tie_robust_loss_scale` apply to both alignment stages.

The movement-penalty options accept dictionaries mapping patterns to positive numbers, or `None` for equal weights. Larger weights resist movement more strongly during the solve. Weights are normalized to mean 1, preserving their ratios. The strength options accept a number from 0 to 1 for all images, or a JSON object string mapping patterns to numbers from 0 to 1. Strengths multiply each image's solved correction: `0.5` applies half, `1` applies all, and `0` applies none. Zero is an ordinary multiplier; it does not lock an image during the joint solve. Different strengths can leave a residual mismatch, and the local stage uses the already scaled global corrections.

Both kinds of mapping use [wcmatch glob patterns](handlers.md#glob-patterns) against case-sensitive, extension-free basenames, with the same enabled syntax as file searches. Rules are processed in insertion order, only against images that remain unmatched. The first matching rule wins, and processing stops when no images remain. Mappings can contain any number of entries. Unmatched images default to 1; patterns matching nothing are ignored. Patterns include `scene_{a,b}`, `@(base|reference)`, `scene_<1-20>`, and `base|reference`. `*` includes leading dots. `*|!base` and `!base` match every name except `base`; excluded names remain available to later rules.

```python
joint_coregistration(
    input_images="input/*.tif",
    output_images="coregistered",
    global_tie_alignment_strength='{"base": 0, "scene_a*": 0.5, "*": 1}',
    local_tie_alignment_strength='{"base": 0, "*": 1}',
    global_image_movement_penalty_weights={"base": 100, "scene_a*": 10, "*": 1},
    local_image_movement_penalty_weights={"base": 100, "*": 1},
    debug_logs=True,
)
```

Debug logging prints the resolved global/local strength and movement-penalty weight for every image, including normalized weights used by the solvers. An unmatched strength of 1 preserves the default full-correction behavior; use 0 to apply no correction in that stage. Empty strength mappings (`'{}'`) also use 1 for every image.

::: spectralmatch.joint_coregistration
