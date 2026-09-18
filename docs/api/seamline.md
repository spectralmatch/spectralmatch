Create footprints once, optionally postprocess them, then pass the resulting GeoPackage to either seamline method:

```python
from spectralmatch import (
    create_footprints, postprocess_footprints,
    voronoi_center_seamline, weighted_seamline,
)

footprints = create_footprints(
    "images/*.tif", "footprints.gpkg", image_threads=4,
)
cleaned = postprocess_footprints(
    footprints, "cleaned.gpkg",
    edge_distance=800, hole_to_hole_distance=800,
    cut_width="maximum_inscribed_circle", smoothing_radius=240,
    simplify_tolerance=120, area_filter=None, area_rank=1, image_threads=4,
)
voronoi_center_seamline(cleaned, "voronoi.gpkg")

# Add ranking attributes (e.g. quality) to the footprint layer before ranking.
weighted_seamline(cleaned, "weighted.gpkg", rank_function="{quality}")
```

These functions are also available as `Seamline.create_footprints`, `Seamline.postprocess_footprints`, `Seamline.voronoi`, and `Seamline.weighted`, and through the CLI. Voronoi now takes `input_polygons`, `input_layer`, and `image_field_name`; its former `input_images` and substring-based `vector_mask` inputs have been removed. Pipeline callers use `voronoi_center_seamline_input_polygons` and `voronoi_center_seamline_input_layer`. When no polygon input is supplied, the default pipeline explicitly calls `create_footprints` on its current images before invoking Voronoi.

`create_footprints` polygonizes the GDAL validity mask of `band` (default 1), including nodata, dataset masks, or alpha masks as provided by GDAL. It retains all islands and holes, with one feature per image and fields `image` (basename without extension) and `image_path`. Images must have unique basenames and the same CRS. A categorical cloud raster must first encode invalid pixels as nodata or a validity mask; its class values alone do not define validity. `eight_connected` controls diagonal connectivity. Polygonization explicitly supplies the raster georeferencing for mask bands through [GDAL's DATASET_FOR_GEOREF option](https://gdal.org/en/stable/api/gdal_alg.html).

Postprocessing preserves attributes and requires a suitable projected CRS. Distances use that CRS's linear units. Both new functions accept `image_threads`, `concurrent_processing_backend`, `dask_scheduler`, `debug_logs`, and `resume_from_outputs`, following the package's existing execution conventions. Local raster polygonization uses threads, matching the existing raster-to-vector function; local geometry postprocessing uses processes. Dask dispatches both through the shared executor. Workers return geometries; the parent alone writes and flushes each feature as soon as it completes, before logging completion. Parallel outputs are stored in completion order, with attributes attached by input index. Completed features remain readable if another worker fails; an adjacent `.incomplete` marker prevents resume modes from mistaking a partial output for a completed one. Rerunning an incomplete output recomputes the layer.

| Postprocessing parameter | Effect |
| --- | --- |
| `edge_distance=800` | Maximum distance from a hole to its original component's outer ring; zero disables only hole-to-edge cuts. |
| `hole_to_hole_distance=800` | Maximum boundary-to-boundary distance between original holes in the same polygon component. Connects every qualifying pair; zero disables only hole-to-hole cuts. |
| `relative_edge_distance=None` | Optional additional limit on hole-to-edge distance divided by `sqrt(hole_area / pi)`. Does not restrict hole-to-hole cuts. |
| `cut_width="maximum_inscribed_circle"` | Diameter of the largest circle fitting inside each hole, independent of cut direction. Also accepts `"hole_size"` for the largest vertex-to-vertex diameter or a positive integer for a fixed width. Applies to both cut types; hole pairs use the smaller of their two widths. |
| `cut_method="corridor"` | Subtract a shortest connection buffered by half the cut width. With `"buffer"`, expand an edge-selected hole by `distance + width / 2`, or both holes in a pair by `(distance + width) / 2`. |
| `smoothing_radius=240` | Erode then dilate, intersecting with the cut geometry to prevent expansion or refilling cuts. |
| `simplify_tolerance=120` | Maximum deviation of removed vertices from each shortcut; zero disables simplification. |
| `simplify_area_weight=0.5` | Balance normalized area loss against perimeter reduction; larger values favor retaining area. |
| `area_filter=None` | Minimum polygon component area in squared CRS units; None disables the threshold. |
| `area_rank=1` | Keep the largest N components per feature for positive N or smallest abs(N) for negative N; 0 or None keeps all. |

Area filtering runs after smoothing and simplification, first by threshold and then by rank, independently for each feature. Ranking does not remove interior holes from retained components.

Both distance limits use the original geometry, before any cuts or smoothing. Hole pairs are evaluated once, within each polygon component; separate components and features are never paired. A chain of qualifying hole pairs can connect a deep hole to an edge-selected hole, even when the deep hole itself exceeds `edge_distance`. Newly cut boundaries do not make additional pairs or edge cuts eligible. Set both distance parameters to `0` to disable explicit cuts; smoothing can still connect holes by removing narrow strips.

The maximum inscribed circle width measures the hole's thickest interior region: a 1000-by-200 rectangle produces a width of approximately 200 at any rotation. The radius is approximated to a tolerance of `sqrt(hole_area) / 1000`; the width is twice that radius. For an irregular hole with a large lobe and a narrow neck, the large lobe determines the width. Other holes can still be reached by wide cuts, overlapping corridors, or smoothing. Holes are not filled. Empty processed features are omitted; removing every feature raises an error.

The simplifier is an **inward greedy adaptation** of the vertex-restricted shortcut idea in [Shortcut Hulls (Bonerath et al., 2021)](https://arxiv.org/abs/2106.13620). The paper's algorithm constructs outer hulls and optimizes globally; this implementation does neither. It uses a priority queue of vertex removals, cached area and coordinate arithmetic for shortcut costs and displacement, and a spatial index of boundary vertices for local inward-ear checks. It balances normalized area loss and perimeter savings and bounds cumulative shortcut displacement. A final validity and containment check retains the input if numerical degeneracies defeat the local checks. Exterior and hole rings retain at least three vertices. It therefore cannot expand the valid region or fill a cloud hole. It is a local heuristic, and may leave vertices that a global optimizer could remove. Geometry checks can still be costly for very detailed footprints. Smoothing may introduce vertices before simplification; shortcut endpoints come from the geometry entering the simplifier.

::: spectralmatch.seamline.Seamline
