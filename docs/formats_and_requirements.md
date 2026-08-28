# File Formats and Input Requirements

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

`joint_coregistration` can save and partially reload raw feature tie points as JSON. Image identifiers are case-sensitive, extension-free basenames, and zero-based pixel coordinates use `[column, row]` order. The file contains no solver parameters, so the same points can be filtered and solved again with different alignment settings.

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

`global_regression` accepts the same file through `pif_load_tie_points` when `pif_method='flood_from_match_points'`. When supplied, every processed overlap pair must exist in the file and retain at least three usable points after validation; malformed, missing, or unusable data raises an error and ORB is not used. The JSON coordinates must describe the exact source pixel grids passed to `global_regression`, with matching case-sensitive basenames. Do not reuse raw points directly with geometrically warped, cropped, resampled, or renamed outputs unless their pixel grids and basenames remain exactly the same.

The equivalent pipeline options are `global_regression_pif_load_tie_points` and `global_regression_pif_method='flood_from_match_points'`.

## Block Maps File
Block maps are spatial summaries of raster data, where each block represents the mean values of a group of pixels over a fixed region. They are used to reduce image resolution while preserving local radiometric characteristics, enabling efficient comparison and adjustment across images. Each map is structured as a grid of blocks with values for each spectral band. They can be saved as regular `geotif` files and together store this information: block_local_means, block_reference_mean, num_row, num_col, bounds_canvas_coords. 

There are two types of block maps, although their format is exactly the same:

 - **Local Block Map:** Each block stores the mean value of all pixels within its boundary for a single image.
 - **Reference Block Map:** Each block is the mean of all images means for its boundary; simply the mean of all local block maps.

Both block maps have the shape: `num_row, num_col, num_bands`, however, there are multiple (one for each image) local block maps and only one reference block map. Once a reference block map is created it is unique to its input images and cannot be accurately modified to add additional images. However, images can be 'brought' to a reference block map even if they were not involved in its creation as long as it covers that image.
