# File Formats and Input Requirements

## Input path patterns

String inputs accept folders or [wcmatch glob patterns](api/handlers.md#glob-patterns), including braces, extglobs, alternatives, exclusions, numeric ranges, hidden names, and recursive `**` / `***`. For example, `input/**/*.{tif,tiff}|!input/**/bad*` searches nested folders for either extension while excluding filenames beginning with `bad`. Lists contain literal paths. Image-weight rules use the same glob syntax against extension-free basenames.

## Input Raster Requirements
Input rasters must meet specific criteria to ensure compatibility during processing. These are checked by _check_raster_requirements():

- Have a valid geotransform
- Share the same coordinate reference system (CRS)
- Have an identical number of bands
- Use consistent nodata values

Additionally, all rasters should:

 - Be a `.tif` file
 - Have overlap which represents the same data in each raster
 - Have a consistent spectral profile 

## Regression Parameters File
Regression parameters can be stored in a `json` file which includes:

 - Adjustments: Per-band scale and offset values applied to each image.
 - Whole Stats: Per-band mean, std, and size representing overall image statistics.
 - Overlap Stats: Per-image pair mean, std, and size for overlapping geometry regions.

The structure is a dictionary keyed by images basenames (no extension) with the following format:

```json
{
  "image_name": {
    "adjustments": {
      "band_0": {"scale": float, "offset": float},
      ...
    },
    "whole_stats": {
      "band_0": {"mean": float, "std": float, "size": int},
      ...
    },
    "overlap_stats": {
      "other_image": {
        "band_0": {"mean": float, "std": float, "size": int},
        ...
      },
      ...
    }
  },
  ...
}
```
This format represents the following: For each image_name there are adjustment, whole_stats and overlap_stats. For each adjustments, for each band, there is scale and offset. For each whole_stats and overlap_stats, for each band, there is mean, std, and size (number of pixels). Each band key follows the format band_0, band_1, etc. Mean and std are floats and size is an integer.

This structure is validated by `_validate_adjustment_model_structure()` before use to ensure consistency and completeness across images and bands. Global regression does not actually use 'adjustments' field because they are recalculated every run.

## Tie-point Adjustments File

`joint_coregistration` saves selected tie points as JSON. Image identifiers are case-sensitive, extension-free basenames, and original zero-based pixel coordinates use `[column, row]` order. Cached pairs are validated and reused directly without ORB, displacement/RANSAC filtering, or grid reselection. The same selected points can be solved with different alignment settings.

```json
{
  "tie_points": [
    {
      "image_1": "image_a",
      "image_2": "image_b",
      "points": [
        [[120.5, 84.0], [117.5, 86.0]],
        [[240.25, 168.0], [237.25, 170.0]]
      ]
    }
  ]
}
```

Each item in `points` is `[[image_1_column, image_1_row], [image_2_column, image_2_row]]`. Partial files are supported. In `joint_coregistration`, loaded pairs that belong to the current overlap network are reused, and missing pairs are calculated. Tie-point thresholds, local-grid spacing, and local falloff distance use the shared input CRS units.

ORB candidates are collected from windows around a grid spaced by `tie_grid_spacing`. `tie_search_radius` is a positive finite number with a unit suffix: `"100crs"` means a 100-CRS-unit half-width; `"128px"` (default) uses 128 pixels at the coarser input resolution. `tie_orb_max_features=100` caps detection per image per window. `tie_max_matches_per_window=1` caps final matches per search window per pair; any positive integer is accepted, and `None` keeps all surviving matches. Each match stays with its originating window. Only selected points are saved; identical selected pairs from overlapping windows are stored once. Cached empty pairs are reused; only absent pairs are calculated. Regenerate older raw-candidate JSON files once to obtain selected-point caches. See [grid-distributed ORB matching](api/joint_coregistration.md#grid-distributed-orb-matching) for details.

`global_regression` accepts the same file through `pif_load_ties` when `pif_method='flood_from_match_points'`. A missing file warns and uses normal feature detection. An existing file must contain every processed overlap pair with at least three usable points; malformed or insufficient pair data still raises an error. Loaded points bypass ORB and RANSAC, while conversion to overlap coordinates and validity checks still apply. The JSON coordinates must describe the exact source pixel grids and case-sensitive basenames passed to `global_regression`.

The equivalent pipeline options are `global_regression_pif_load_ties` and `global_regression_pif_method='flood_from_match_points'`.

## Tie-point GIS export

`joint_coregistration(tie_save_crs_path="ties.gpkg")` exports selected ties as GIS point features in the common input CRS. Each matched pair produces two points sharing a `tie_id`, with `image`, `match_img`, `pixel_col`, and `pixel_row` fields. Coordinates come from each original image's geotransform, including pixel-center offsets and rotation, before alignment corrections. The filename extension selects the vector format; GeoPackage uses the `ties` layer. GeoJSON, Shapefile, and FlatGeobuf are also supported. This export is separate from the JSON cache. The pipeline option is `joint_coregistration_tie_save_crs_path`.

## Missing cache files

Optional cache loads can be enabled before the first run. A missing file emits a `RuntimeWarning`, loads nothing from that path, and lets the normal calculation proceed. Existing malformed files, incompatible block maps, and other read errors still raise errors.

- `joint_coregistration(tie_load_path=...)`: selected tie-point JSON; use the same path with `tie_save_path` to create it on the first run.
- `Match.global_regression(load_adjustments=...)`: statistics JSON; use the same path with `save_adjustments`.
- `Pif.flood_from_match_points(load_ties=...)` and `Match.global_regression(pif_load_ties=...)`: selected tie-point JSON; a missing file enables normal feature detection.
- `Match.local_block_adjustment(load_block_maps=(reference_path, local_paths))`: reference and local GeoTIFF block maps; missing entries warn individually, existing maps are reused, and missing maps are computed. `save_block_maps` writes the cache files.

The corresponding `pipeline` options use the same behavior.

## Block Maps File
Block maps are spatial summaries of raster data, where each block represents the mean values of a group of pixels over a fixed region. They are used to reduce image resolution while preserving local radiometric characteristics, enabling efficient comparison and adjustment across images. Each map is structured as a grid of blocks with values for each spectral band. They can be saved as regular `geotif` files and together store this information: block_local_means, block_reference_mean, num_row, num_col, bounds_canvas_coords. 

There are two types of block maps, although their format is exactly the same:

 - **Local Block Map:** Each block stores the mean value of all pixels within its boundary for a single image.
 - **Reference Block Map:** Each block is the mean of all images means for its boundary; simply the mean of all local block maps.

Both block maps have the shape: `num_row, num_col, num_bands`, however, there are multiple (one for each image) local block maps and only one reference block map. Once a reference block map is created it is unique to its input images and cannot be accurately modified to add additional images. However, images can be 'brought' to a reference block map even if they were not involved in its creation as long as it covers that image.
