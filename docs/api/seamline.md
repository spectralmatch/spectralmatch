Create footprints once, optionally postprocess them, then pass the resulting GeoPackage to a seamline method:

```python
from spectralmatch import (
    create_footprints, postprocess_footprints,
    voronoi_center_seamline, weighted_seamline,
)

footprints = create_footprints(
    "images/*.tif", "footprints.gpkg", image_threads=4,
    metadata_csv="image_metadata.csv", metadata_image_field_name="image",
)
cleaned = postprocess_footprints(
    footprints, "cleaned.gpkg",
    hole_edge_distance=800, hole_to_hole_distance=800,
    hole_cut_width="maximum_inscribed_circle", simplify_smoothing_radius=240,
    simplify_tolerance=120, filter_area_size=None, filter_area_rank=1, image_threads=4,
)
voronoi_center_seamline(cleaned, "voronoi.gpkg")

# CSV attributes remain available after postprocessing.
weighted_seamline(cleaned, "weighted.gpkg", rank_function="{quality_score} - {cloud_cover}")
```

These functions are also available as `Seamline.create_footprints`, `Seamline.postprocess_footprints`, `Seamline.voronoi`, and `Seamline.weighted`, and through the CLI. Voronoi and weighted retain their polygon-first APIs. Both accept optional `input_images` (folder, glob or paths) to match each image basename without extension as a substring of `image_field_name`, using the same field-value convention as `vector_mask`. With no raster list, image names are inferred from the field values. Matching pieces are dissolved and first-match attributes supply ranking fields; an unmatched input image raises an error. Pipeline callers use `voronoi_center_seamline_input_polygons` and `voronoi_center_seamline_input_layer`. When no polygon input is supplied, the default pipeline explicitly calls `create_footprints` on its current images before invoking Voronoi.

`create_footprints` polygonizes the GDAL validity mask of `band` (default 1), including nodata, dataset masks, or alpha masks as provided by GDAL. It retains all islands and holes, with one feature per image and fields `image` (basename without extension) and `image_path`. Images must have unique basenames and the same CRS. A categorical cloud raster must first encode invalid pixels as nodata or a validity mask; its class values alone do not define validity. `eight_connected` controls diagonal connectivity. Polygonization explicitly supplies the raster georeferencing for mask bands through [GDAL's DATASET_FOR_GEOREF option](https://gdal.org/en/stable/api/gdal_alg.html).

Pass `metadata_csv` to attach image-level attributes during footprint creation. `metadata_image_field_name="image"` selects the CSV join column: its value must contain the full current image basename without extension, including any processing suffixes, using the same literal, case-insensitive substring matching as polygon identifiers. For example, `Worldview_20160922_Coregistered_Global_Local` or `Worldview_20160922_Coregistered_Global_Local.tif` matches that LocalMatch raster; `Worldview_20160922` alone does not. The field contains the image basename, as in local matching's `vector_mask` filter. Every input image must match exactly one CSV row; missing or ambiguous matches fail before polygonization. Unused rows are allowed. The join column is excluded from copied attributes; other columns retain inferred numeric or text types, and empty metadata cells become null. Column names must be nonempty and unique ignoring case, and attributes must not conflict with `image_field_name`, `image_path`, `geometry`, or the GeoPackage `fid`. The CSV is read once in the parent process. Its attributes survive postprocessing and can be referenced by `weighted_seamline(rank_function=...)` or `markov_triangles(image_rank_function=...)`. See [ImageMetadata.csv](../examples/data_worldview/Input/ImageMetadata.csv) for illustrative WorldView values; replace them with your own measurements. When joining new metadata into an existing output, use `resume_from_outputs="no"` to regenerate it.

The public functions consistently use `input_layer="footprints"` and `image_field_name="image"` where applicable. Creation and postprocessing write `output_layer="footprints"`; seamline methods write `output_layer="seamlines"`. Pass `input_layer=None` to read the first layer of an external vector file, or supply its actual layer name.

Postprocessing preserves attributes and requires a suitable projected CRS. Distances use that CRS's linear units. Both new functions accept `image_threads`, `concurrent_processing_backend`, `dask_scheduler`, `debug_logs`, and `resume_from_outputs`, following the package's existing execution conventions. Local raster polygonization uses threads, matching the existing raster-to-vector function; local geometry postprocessing uses processes. Dask dispatches both through the shared executor. Workers return geometries; the parent alone writes and flushes each feature as soon as it completes, before logging completion. Parallel outputs are stored in completion order, with attributes attached by input index. Completed features remain readable if another worker fails; an adjacent `.incomplete` marker prevents resume modes from mistaking a partial output for a completed one. Rerunning an incomplete output recomputes the layer.

| Postprocessing parameter | Effect |
| --- | --- |
| `hole_edge_distance=800` | Maximum distance from a hole to its original component's outer ring; zero disables only hole-to-edge cuts. |
| `hole_to_hole_distance=800` | Maximum boundary-to-boundary distance between original holes in the same polygon component. Connects every qualifying pair; zero disables only hole-to-hole cuts. |
| `hole_relative_edge_distance=None` | Optional additional limit on hole-to-edge distance divided by `sqrt(hole_area / pi)`. Does not restrict hole-to-hole cuts. |
| `hole_cut_width="maximum_inscribed_circle"` | Diameter of the largest circle fitting inside each hole, independent of cut direction. Also accepts `"hole_size"` for the largest vertex-to-vertex diameter or a positive integer for a fixed width. Applies to both cut types; hole pairs use the smaller of their two widths. |
| `hole_cut_method="corridor"` | Subtract a shortest connection buffered by half the cut width. With `"buffer"`, expand an edge-selected hole by `distance + width / 2`, or both holes in a pair by `(distance + width) / 2`. |
| `simplify_smoothing_radius=240` | Erode then dilate, intersecting with the cut geometry to prevent expansion or refilling cuts. |
| `simplify_tolerance=120` | Maximum deviation of removed vertices from each shortcut; zero disables simplification. |
| `simplify_area_weight=0.5` | Balance normalized area loss against perimeter reduction; larger values favor retaining area. |
| `filter_area_size=None` | Minimum polygon component area in squared CRS units; None disables the threshold. |
| `filter_area_rank=1` | Keep the largest N components per feature for positive N or smallest abs(N) for negative N; 0 or None keeps all. |

Area filtering runs after smoothing and simplification, first by threshold and then by rank, independently for each feature. Ranking does not remove interior holes from retained components.

Both distance limits use the original geometry, before any cuts or smoothing. Hole pairs are evaluated once, within each polygon component; separate components and features are never paired. A chain of qualifying hole pairs can connect a deep hole to an edge-selected hole, even when the deep hole itself exceeds `hole_edge_distance`. Newly cut boundaries do not make additional pairs or edge cuts eligible. Set both distance parameters to `0` to disable explicit cuts; smoothing can still connect holes by removing narrow strips.

The maximum inscribed circle width measures the hole's thickest interior region: a 1000-by-200 rectangle produces a width of approximately 200 at any rotation. The radius is approximated to a tolerance of `sqrt(hole_area) / 1000`; the width is twice that radius. For an irregular hole with a large lobe and a narrow neck, the large lobe determines the width. Other holes can still be reached by wide cuts, overlapping corridors, or smoothing. Holes are not filled. Empty processed features are omitted; removing every feature raises an error.

The simplifier is an **inward greedy adaptation** of the vertex-restricted shortcut idea in [Shortcut Hulls (Bonerath et al., 2021)](https://arxiv.org/abs/2106.13620). The paper's algorithm constructs outer hulls and optimizes globally; this implementation does neither. It uses a priority queue of vertex removals, cached area and coordinate arithmetic for shortcut costs and displacement, and a spatial index of boundary vertices for local inward-ear checks. It balances normalized area loss and perimeter savings and bounds cumulative shortcut displacement. A final validity and containment check retains the input if numerical degeneracies defeat the local checks. Exterior and hole rings retain at least three vertices. It therefore cannot expand the valid region or fill a cloud hole. It is a local heuristic, and may leave vertices that a global optimizer could remove. Geometry checks can still be costly for very detailed footprints. Smoothing may introduce vertices before simplification; shortcut endpoints come from the geometry entering the simplifier.

`markov_triangles` (also `Seamline.markov_triangles` and the `markov_triangles` CLI command) labels constrained Delaunay triangles using a Markov random field, then merges overlapping quadtree mosaics. Its dependencies are included in the standard `pip install spectralmatch` installation.

```python
from spectralmatch import markov_triangles

markov_triangles(
    input_images="images/*.tif",
    input_polygons="cleaned.gpkg",  # Optional; omission polygonizes band-1 validity masks.
    input_layer="footprints",
    output_mask="seamlines.gpkg",
    image_field_name="image",
    mesh_spacing=10,
    edge_variables_weights_gsds_bands=[
        ["laplacian_difference", 1.0, ["native"], [1]],
        ["laplacian_magnitude", 0.25, [100], [1]],
        # ["costs.tif", 0.5, {"largest": [2, 8]}, [1]],
    ],
    # image_rank_function="{quality} - {cloud_cover}",  # Optional footprint attributes.
    image_quality_weight=1.0,
    image_threads=4,
)
```

`input_images` determines raster paths; footprint `image_path` attributes are not used to locate imagery. Supplied footprints are never regenerated. Their identifier field must contain every input basename without extension, as with the other seamline methods. Matching is case-insensitive and literal; for example `folder/scene_A.tif` matches `scene_A`. Repeated matching features are dissolved and first-match attributes are used for ranking. Choose distinct names carefully: a short name can match multiple field values. Inputs must be registered, color corrected and share a projected CRS. Footprints must lie inside their corresponding rasters.

Each term in `edge_variables_weights_gsds_bands` is `[name_or_path, weight, GSD selection, band selection]`. A single selection must be a plain one-item list, such as `["native"]`, `[100]`, or `[1]`. Two or more values require a reducer dictionary, such as `{"average": ["native", 4]}` or `{"largest": [1, 2, 3]}`. Reducers with only one value (for example `{"average": [1]}`) and multiple values without a reducer are rejected. Weights must be finite and nonnegative; **zero completely disables a term and its raster reads**. An empty list disables all raster costs. Reducers are `average`, `largest`, or `smallest`; band numbers start at 1. Each GSD must be a positive integer or `"native"`; floats and booleans are rejected. Integers are absolute pixel sizes in the projected CRS's linear units, not overview factors or fractions of raster dimensions. For example, `4` means four metres per pixel in a metre-based CRS. For the built-in Laplacian terms, `"native"` uses the smallest native pixel-axis length among all input images, giving the images a common comparison scale. For an external cost raster it uses that raster's own smallest pixel-axis length, or GDAL's suggested pixel size after reprojection to the working CRS. Resolved native sizes can be fractional. Selections such as `{"average": ["native", 4, 16]}` combine native and explicit scales. Repeating a variable in separate entries lets each scale receive its own weight.

| Variable | Cost when adjacent triangle labels differ |
| --- | --- |
| `laplacian_difference` | Absolute difference between the two image Laplacians, reduced over selected bands, then GSDs, then maximized along the shared edge. |
| `laplacian_magnitude` | Absolute Laplacian of each image, reduced over bands, then GSDs, then maximized along the edge; the two image costs are averaged. This penalizes texture even where the images agree. |
| Raster path | Direct raster values reduced over bands, then GSDs, then maximized along the edge. No derivatives or inter-image subtraction are applied. Samples must be finite and nonnegative. External costs may use a different CRS; GDAL reprojects them. |

Costs are summed after multiplication by their weights and are zero between equal labels. Values retain their original numeric units without automatic normalization; choose weights appropriate to the image and cost-raster ranges. The explicit default in the function signature uses difference weight `1` at `"native"` and magnitude weight `0.25` at GSD `100` in CRS units, both using band `1`. These single-value selections do not aggregate. These entries are literal defaults in the function signature; only `"native"` needs resolution metadata. Setting the magnitude weight to zero gives the difference-only objective. Resampling and discrete numerical choices need not reproduce another implementation's exact seam.

GDAL average-resampled VRTs provide a common map-aligned grid for each requested GSD. The five-point Laplacian is `(east + west + north + south - 4 * center) / GSD²`. NumPy computes this stencil and the reductions in compiled array operations. GDAL pixel functions work on raster grids; applying them to child mosaics would require rasterizing their polygon ownership and could change behavior at boundaries, so the shared stencils retain the exact polygon sampling rules. Edge samples are spaced no farther apart than the smallest GSD in that term; these scales do not change the mesh. Raster values and validity masks are read in bounded 256-pixel windows, with up to 64 band windows cached per node. Before reading, requests are grouped by resolved GSD and source. The two built-in variables share one batch of Laplacians per candidate/GSD using the union of requested bands and coordinates; repeated external-raster terms similarly share direct samples. This also handles overlapping band lists, reordered bands, differing scale lists and `"native"` aliases that resolve to an explicit GSD. Each term retains its original sample locations and reducers, so adding another term does not change its cost. Grouping avoids relying on the window cache to share reads between terms, even when their requests span more than 64 windows. Overlapping nodes may still reread the same raster areas. Missing stencil neighbors extend the center value; valid slivers omitted by resampling at rotated edges use the native pixel. Invalid center samples inside footprints raise an error. Only automatic footprint generation polygonizes full raster validity masks.

Parameters use `image_*` for image quality, `edge_*` for seam costs, `foreground_*` for protected polygons, `mesh_*` for triangles, `quadtree_*` for subdivision, and `solver_*` for optimization. Shared input/output, worker, logging and resume parameters retain their library names.

`image_rank_function` uses the same field-expression evaluator as `weighted()`. Raw scores are saved as `quality_score` (null without a ranking expression). Scores are min-max normalized across all inputs into penalties from 0 for best to 1 for worst; equal scores incur zero penalty. `image_rank_descending=True` favors larger scores, and False favors smaller ones. Assigning a triangle to an image adds `image_quality_weight * triangle_area * quality_penalty` to the objective, so better images are encouraged to occupy more area. Composite child mosaics retain the original images' area-weighted penalties. A zero `image_quality_weight` removes this preference while retaining saved scores. Quality affects area ownership, independently of seam texture costs.

`mesh_spacing=10` sets a regular mesh grid in CRS units. Footprint boundaries constrain the triangles, preserving narrow islands and holes. Graph cuts optimize these discrete boundaries; they do not search every original pixel boundary. Alpha expansion finds a move-local minimum, not necessarily a global minimum. `smallest`-reduced differences can violate metric assumptions, so they use alpha-beta swaps. `foreground_path` and optional `foreground_layer` forbid seams through protected interiors by contracting intersecting triangle edges; infeasible protection raises an error.

`solver_max_iterations=50` limits graph-cut sweeps per node and warns when reached. Quadtree defaults are `quadtree_max_images_per_leaf=16`, `quadtree_overlap=0.1`, and `quadtree_max_depth=8`; `quadtree_max_depth=0` optimizes one global mesh. Each completed child acts as a virtual image at its parent. A node above one million grid cells raises an error requesting coarser spacing. `image_threads=None` runs serially, a positive integer or `"cpu"` selects local processes, and `concurrent_processing_backend="dask"` with `dask_scheduler` uses the shared Dask executor. Parallelism is across quadtree nodes; a single root node uses one worker. GDAL reads and warps use one thread within each node to avoid inheriting an unusable thread pool from earlier raster stages. With `debug_logs=True`, mesh construction, adjacency, raster sampling, graph cuts and polygon merging report progress and elapsed time. Files must be accessible to all workers. The parent alone writes the output, and `.incomplete` markers prevent reuse after failures. `resume_from_outputs` accepts `"no"`, `"yes"`, and `"validate"`.

::: spectralmatch.seamline.Seamline
