# Reusing exact valid-data footprints

`joint_coregistration`, `global_regression`, `local_block_adjustment`, and
`Pif.flood_from_match_points` accept `input_footprints=None`, a vector filename,
or a list of filenames. Each file's first layer must have a CRS and polygon or
multipolygon geometry, retaining holes and islands. The standard `image` field
identifies its raster (`image_path` is accepted when `image` is absent).
An exact basename wins; otherwise a unique longest basename followed by an
underscore and processing suffix is matched. Missing/ambiguous image IDs fail
explicitly. LiDAR/fixed-reference footprints may be omitted.

`markov_triangles` also accepts `input_footprints` for its single vector file;
`input_polygons` remains a backward-compatible alias, but do not pass both.
Its existing `input_layer` and `image_field_name` options still apply.

These are validity hints about the **current** rasters, not extra clipping
boundaries. No polygon buffering or simplification is performed. A shared
helper builds temporary sparse VRT sources in the worker's temporary directory,
omitting source blocks wholly outside a footprint and coalescing adjacent
blocks into rectangles. Pixel reading, decompression, warping, and statistics
remain in GDAL C++. Original image names, grids, masks, values, and exact or
approximate statistics settings are retained. Empty polygon-overlap pairs are
excluded before matching. The full valid-overlap statistics population is
unchanged; this does not switch PIF statistics to `matched_windows`.

Boundary tiles still use raster validity checks. Rasters without a nodata value
retain the original source reads because band statistics do not consistently
honor explicit mask bands. Omitted footprints preserve existing behavior.

Use input footprints through coverage/masking and joint alignment. Recreate
footprints once from the aligned rasters, then pass that same file through global
and local radiometry. Rebuild again if a later operation changes geometry or
validity. Postprocessed seamline polygons have deliberately changed geometry;
use them for seamline selection, not as exact validity hints for statistics.
The WorldView Python example and VHR WorldView YAML demonstrate this lifecycle.

The separate existing `create_footprints` function polygonizes the GDAL validity
mask in C++; joint alignment does not add an `output_footprints` pass. No claim
of a zero-read output polygonization or fixed speedup is made. Sparse imagery
benefits most; dense footprints may bypass the temporary sparse view entirely.
