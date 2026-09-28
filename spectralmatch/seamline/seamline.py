import math
import os
from itertools import combinations
from typing import Literal

import fiona
from osgeo import gdal
from pyproj import CRS
from shapely.geometry import LineString, MultiPolygon, Polygon, box, mapping
from shapely.ops import unary_union

from .footprints import (
    _PolygonWriter,
    _footprint_output_is_reusable,
    _validate_polygon_output,
    _prepare_footprints,
    _read_polygons,
    _read_image_metadata,
    _polygon_parts,
)

from ..utils_logging import (
    _print_step_start,
    _print_image_start,
    _print_image_completed,
)

from ..handlers import _existing_outputs_are_reusable, _resolve_paths
from ..utils_multiprocessing import _resolve_parallel_config, _run_image_tasks
from . import footprints as _footprints
from . import markov_triangles as _markov
from ..types_and_validation import Universal, Seamline as SeamlineValidation
from .voronoi_center_seamline import (
    _compute_centerline,
    _mask_by_aoi,
    _save_emp_outlines,
    _save_intersection_points,
    _segment_emp,
)
from .weighted_seamline import weighted_seamline, _rank_polygons

gdal.UseExceptions()


class Seamline:
    @staticmethod
    def create_footprints(
        input_images: Universal.SearchFolderOrListFiles,
        output_polygons: str,
        *,
        metadata_csv: str | None = None,
        metadata_image_field_name: str = "image",
        image_field_name: str = "image",
        output_layer: str = "footprints",
        band: int = 1,
        eight_connected: bool = True,
        image_threads: Universal.Threads = None,
        concurrent_processing_backend: Universal.ConcurrentProcessingBackend = "process_pool",
        dask_scheduler: Universal.DaskScheduler = None,
        debug_logs: Universal.DebugLogs = False,
        resume_from_outputs: Literal["no", "yes", "validate"] = "no",
    ) -> str:
        """Polygonize valid raster masks into a shared seamline GeoPackage, retaining holes and islands.

        Args:
            input_images (str | List[str], required): Defines input files from a glob path, folder, or list of paths. Specify like: "/input/files/*.tif", "/input/folder" (assumes *.tif), ["/input/one.tif", "/input/two.tif"]. Images must have unique basenames and the same CRS.
            output_polygons (str): Output GeoPackage path for the image footprints, including image identifiers and image_path attributes.
            metadata_csv (str | None, optional): CSV attributes to join before polygonization; each image must match exactly one row, the join column is omitted, and other columns retain numeric or text types. Defaults to None.
            metadata_image_field_name (str, optional): CSV column containing the full current image basename, including processing suffixes but excluding the extension, as a literal, case-insensitive substring; missing or multiple matches raise an error. Defaults to "image".
            image_field_name (str, optional): Name of the field containing each image basename without its extension; must differ from geometry and image_path. Defaults to "image".
            output_layer (str, optional): Output GeoPackage layer name. Defaults to "footprints".
            band (int, optional): One-based raster band whose GDAL validity mask defines valid pixels; categorical mask values alone do not define validity. Defaults to 1.
            eight_connected (bool, optional): Use 8-connectedness for polygonization; if False, use 4-connectedness. Defaults to True.
            image_threads (Literal["cpu"] | int | None, optional): Parallelism for per-image operations; "cpu" uses all CPU cores, an integer sets the worker count, and None disables local parallelism. Local GDAL workers use threads. Defaults to None.
            concurrent_processing_backend (Literal["process_pool", "dask"], optional): Use the local execution backend or an existing Dask cluster; local raster polygonization uses threads under the "process_pool" setting. Defaults to "process_pool".
            dask_scheduler (tuple[str, str] | None, optional): Existing Dask scheduler as ("file", path) or ("address", address); required for Dask execution, which requires image_threads=None. Defaults to None.
            debug_logs (Universal.DebugLogs, optional): Enables debug print statements if True; default is False.
            resume_from_outputs (Literal["no", "yes", "validate"], optional): Recompute outputs with "no", reuse existing outputs with "yes", or validate existing outputs before reusing them with "validate"; incomplete streaming outputs are recomputed. Defaults to "no".

        Returns:
            str: Written output GeoPackage path."""
        _print_step_start("create_footprints")
        _validate_polygon_output(output_polygons, output_layer)
        Universal._validate(
            input_images=input_images,
            image_threads=image_threads,
            debug_logs=debug_logs,
            concurrent_processing_backend=concurrent_processing_backend,
            dask_scheduler=dask_scheduler,
        )
        if not isinstance(band, int) or isinstance(band, bool) or band < 1:
            raise ValueError("band must be a positive integer.")
        if not isinstance(eight_connected, bool):
            raise ValueError("eight_connected must be a bool.")
        if (
            not isinstance(image_field_name, str)
            or not image_field_name.strip()
            or image_field_name in {"geometry", "image_path"}
        ):
            raise ValueError(
                "image_field_name must be nonempty and distinct from geometry and image_path."
            )
        if _footprint_output_is_reusable(
            output_polygons,
            resume_mode=resume_from_outputs,
            debug_logs=debug_logs,
            step_name="create_footprints",
            output_layer=output_layer,
            image_field_name=image_field_name,
        ):
            return output_polygons
        paths = _resolve_paths(
            "search", input_images, kwargs={"default_file_pattern": "*.tif"}
        )
        if not paths:
            raise ValueError("No input images found.")
        names = _resolve_paths("name", paths)
        if len(set(names)) != len(names):
            raise ValueError("Input image basenames must be unique.")
        metadata_schema, metadata = _read_image_metadata(
            metadata_csv, metadata_image_field_name, names, image_field_name
        )
        parallel, workers = _resolve_parallel_config(
            image_threads, concurrent_processing_backend, dask_scheduler
        )
        schema = {
            "geometry": "Unknown",
            "properties": {
                image_field_name: "str",
                "image_path": "str",
                **metadata_schema,
            },
        }
        with _PolygonWriter(output_polygons, output_layer, schema, None) as writer:

            def commit(index, result):
                """Validate each raster CRS and commit its geometry and identifier."""
                geometry, crs = result
                crs = CRS.from_user_input(crs)
                if writer.crs is None:
                    writer.crs = crs.to_wkt()
                elif crs != CRS.from_user_input(writer.crs):
                    raise ValueError("Input rasters must use the same CRS.")
                writer.write(
                    geometry,
                    {
                        image_field_name: names[index],
                        "image_path": paths[index],
                        **metadata[index],
                    },
                )

            _run_image_tasks(
                _footprints._footprint_from_image,
                [(path, band, eight_connected) for path in paths],
                input_paths=paths,
                output_paths=[output_polygons] * len(paths),
                parallel=parallel,
                backend="thread",
                workers=workers,
                concurrent_processing_backend=concurrent_processing_backend,
                dask_scheduler=dask_scheduler,
                result_callback=commit,
                collect_results=False,
            )
        return output_polygons

    @staticmethod
    def postprocess_footprints(
        input_polygons: str,
        output_polygons: str,
        *,
        input_layer: str | None = "footprints",
        output_layer: str = "footprints",
        hole_edge_distance: float = 800,
        hole_to_hole_distance: float = 800,
        hole_relative_edge_distance: float | None = None,
        hole_cut_width: (
            int | Literal["hole_size", "maximum_inscribed_circle"]
        ) = "maximum_inscribed_circle",
        hole_cut_method: Literal["corridor", "buffer"] = "corridor",
        simplify_smoothing_radius: float = 240,
        simplify_tolerance: float = 120,
        simplify_area_weight: float = 0.5,
        filter_area_size: float | None = None,
        filter_area_rank: int | None = 1,
        image_threads: Universal.Threads = None,
        concurrent_processing_backend: Universal.ConcurrentProcessingBackend = "process_pool",
        dask_scheduler: Universal.DaskScheduler = None,
        debug_logs: Universal.DebugLogs = False,
        resume_from_outputs: Literal["no", "yes", "validate"] = "no",
    ) -> str:
        """Connect nearby holes and edges, then smooth/simplify inward, preserving attributes.

        Args:
            input_polygons (str): Input polygon layer path with valid, nonempty Polygon or MultiPolygon features in a suitable projected CRS; all distances use that CRS's linear units.
            output_polygons (str): Output GeoPackage path for processed polygons and their original attributes; must differ from input_polygons.
            input_layer (str | None, optional): Input footprint layer name; None selects the first layer. Defaults to "footprints".
            output_layer (str, optional): Output GeoPackage layer name. Defaults to "footprints".
            hole_edge_distance (float, optional): Maximum hole-to-edge distance in CRS units, measured against each component's original outer ring; zero disables only hole-to-edge cuts. Defaults to 800.
            hole_to_hole_distance (float, optional): Maximum boundary-to-boundary distance in CRS units between original holes in the same polygon component; every qualifying pair is connected. Zero disables only hole-to-hole cuts. Defaults to 800.
            hole_relative_edge_distance (float | None, optional): Additional upper limit on hole-to-edge distance divided by sqrt(hole_area / pi); applies only to edge cuts. None disables this size-relative filter. Defaults to None.
            hole_cut_width (int | Literal["hole_size", "maximum_inscribed_circle"], optional): Positive integer width in CRS units, "hole_size" for the largest vertex-to-vertex diameter, or "maximum_inscribed_circle" for the largest circle fitting inside the hole (diameter, approximated with radius tolerance sqrt(hole_area) / 1000). Applies to both edge and hole-pair cuts; pairs use the smaller hole width. Defaults to "maximum_inscribed_circle".
            hole_cut_method (Literal["corridor", "buffer"], optional): Subtract a shortest connection buffered by half the cut width with "corridor". With "buffer", expand an edge-selected hole by distance + width / 2, or both holes in a pair by (distance + width) / 2. Defaults to "corridor".
            simplify_smoothing_radius (float, optional): Nonnegative erosion and dilation radius, followed by intersection with the cut polygon to prevent expansion or refilling cuts; zero disables smoothing. Defaults to 240.
            simplify_tolerance (float, optional): Nonnegative maximum deviation of removed vertices from inward shortcuts, including cumulative removals; zero disables simplification. Defaults to 120.
            simplify_area_weight (float, optional): Weight in [0, 1] balancing normalized area loss against perimeter reduction in the greedy inward shortcut simplifier; larger values favor retaining area. Defaults to 0.5.
            filter_area_size (float | None, optional): Minimum retained component area in squared CRS units, applied after smoothing and simplification and before filter_area_rank; None disables the threshold. Defaults to None.
            filter_area_rank (int | None, optional): Keep the largest N components per feature for positive N, or the smallest abs(N) for negative N; 0 or None keeps all components passing filter_area_size. Defaults to 1.
            image_threads (Literal["cpu"] | int | None, optional): Parallelism for per-feature geometry operations; "cpu" uses all CPU cores, an integer sets the process count, and None disables local parallelism. Defaults to None.
            concurrent_processing_backend (Literal["process_pool", "dask"], optional): Use a local process pool or an existing Dask cluster. Defaults to "process_pool".
            dask_scheduler (tuple[str, str] | None, optional): Existing Dask scheduler as ("file", path) or ("address", address); required for Dask execution, which requires image_threads=None. Defaults to None.
            debug_logs (Universal.DebugLogs, optional): Enables debug print statements if True; default is False.
            resume_from_outputs (Literal["no", "yes", "validate"], optional): Recompute outputs with "no", reuse existing outputs with "yes", or validate existing outputs before reusing them with "validate"; incomplete streaming outputs are recomputed. Defaults to "no".

        Returns:
            str: Written output GeoPackage path; empty processed features are omitted, and removing all features raises ValueError.
        """
        _print_step_start("postprocess_footprints")
        _validate_polygon_output(output_polygons, output_layer)
        Universal._validate(
            image_threads=image_threads,
            debug_logs=debug_logs,
            concurrent_processing_backend=concurrent_processing_backend,
            dask_scheduler=dask_scheduler,
        )
        for name, value in {
            "hole_edge_distance": hole_edge_distance,
            "hole_to_hole_distance": hole_to_hole_distance,
            "simplify_smoothing_radius": simplify_smoothing_radius,
            "simplify_tolerance": simplify_tolerance,
            "simplify_area_weight": simplify_area_weight,
            "hole_relative_edge_distance": (
                0
                if hole_relative_edge_distance is None
                else hole_relative_edge_distance
            ),
        }.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{name} must be finite and nonnegative.")
        if hole_cut_width not in ("hole_size", "maximum_inscribed_circle") and (
            isinstance(hole_cut_width, bool)
            or not isinstance(hole_cut_width, int)
            or hole_cut_width <= 0
        ):
            raise ValueError(
                'hole_cut_width must be a positive integer, "hole_size", or "maximum_inscribed_circle".'
            )
        if simplify_area_weight > 1 or hole_cut_method not in {"corridor", "buffer"}:
            raise ValueError(
                "Require simplify_area_weight in [0, 1] and hole_cut_method corridor or buffer."
            )
        if filter_area_size is not None and (
            isinstance(filter_area_size, bool)
            or not isinstance(filter_area_size, (int, float))
            or not math.isfinite(filter_area_size)
            or filter_area_size < 0
        ):
            raise ValueError(
                "filter_area_size must be a finite nonnegative number or None."
            )
        if filter_area_rank is not None and (
            isinstance(filter_area_rank, bool) or not isinstance(filter_area_rank, int)
        ):
            raise ValueError("filter_area_rank must be an integer or None.")
        if os.path.realpath(input_polygons) == os.path.realpath(output_polygons):
            raise ValueError("Input and output GeoPackages must be different files.")
        if _footprint_output_is_reusable(
            output_polygons,
            resume_mode=resume_from_outputs,
            debug_logs=debug_logs,
            step_name="postprocess_footprints",
        ):
            return output_polygons
        frame = _read_polygons(input_polygons, input_layer)
        if not frame.crs.is_projected:
            raise ValueError(
                "Postprocessing requires a projected CRS; reproject the input first."
            )
        parallel, workers = _resolve_parallel_config(
            image_threads, concurrent_processing_backend, dask_scheduler
        )
        args = [
            (
                geometry,
                hole_edge_distance,
                hole_relative_edge_distance,
                hole_cut_width,
                hole_cut_method,
                simplify_smoothing_radius,
                simplify_tolerance,
                simplify_area_weight,
                filter_area_size,
                filter_area_rank,
                hole_to_hole_distance,
            )
            for geometry in frame.geometry
        ]
        with fiona.open(
            input_polygons, **({"layer": input_layer} if input_layer else {})
        ) as source:
            schema = {
                "geometry": "Unknown",
                "properties": dict(source.schema["properties"]),
            }
        records = frame.drop(columns=frame.geometry.name).to_dict("records")
        with _PolygonWriter(
            output_polygons, output_layer, schema, frame.crs.to_wkt()
        ) as writer:

            def commit(index, geometry):
                """Commit each returned geometry with its original input attributes."""
                writer.write(geometry, records[index])

            _run_image_tasks(
                _footprints._postprocess_polygon,
                args,
                input_paths=[f"{input_polygons}:{index}" for index in frame.index],
                output_paths=[output_polygons] * len(frame),
                parallel=parallel,
                backend="process",
                workers=workers,
                concurrent_processing_backend=concurrent_processing_backend,
                dask_scheduler=dask_scheduler,
                result_callback=commit,
                collect_results=False,
            )
            if writer.count == 0:
                raise ValueError(
                    "Postprocessing removed all polygons; reduce the processing distances or filter_area_size."
                )
        return output_polygons

    @staticmethod
    def markov_triangles(
        input_images: Universal.SearchFolderOrListFiles,
        output_mask: str,
        *,
        input_polygons: str | None = None,
        input_layer: str | None = "footprints",
        output_layer: str = "seamlines",
        image_field_name: str = "image",
        image_rank_function: str | None = None,
        image_rank_descending: bool = True,
        image_quality_weight: float = 1.0,
        edge_variables_weights_gsds_bands: list = [
            ["laplacian_difference", 1.0, ["native"], [1]],
            ["laplacian_magnitude", 0.25, [100], [1]],
        ],
        foreground_path: str | None = None,
        foreground_layer: str | None = None,
        mesh_spacing: float = 10,
        quadtree_max_images_per_leaf: int = 16,
        quadtree_overlap: float = 0.1,
        quadtree_max_depth: int = 8,
        solver_max_iterations: int = 50,
        image_threads: Universal.Threads = None,
        concurrent_processing_backend: Universal.ConcurrentProcessingBackend = "process_pool",
        dask_scheduler: Universal.DaskScheduler = None,
        debug_logs: Universal.DebugLogs = False,
        resume_from_outputs: Literal["no", "yes", "validate"] = "no",
    ) -> str:
        """Optimize triangle-MRF seamlines through overlapping quadtree mosaics, extending Yang et al. (2022; doi:10.1109/LGRS.2022.3166347).

        Requires registered, color-corrected rasters in a common projected CRS.

        Args:
            input_images: Raster folder, glob or path list; basenames without extensions must be unique.
            output_mask: Output GeoPackage path, distinct from input vectors; stores image identifiers, image_path and raw quality_score.
            input_polygons: Optional valid-data footprints; None polygonizes band-1 GDAL masks, while supplied polygons must match every image and are never regenerated.
            input_layer: Input footprint layer name; None selects the first layer. Defaults to "footprints".
            output_layer: Output polygon layer name; defaults to "seamlines".
            image_field_name: Footprint field containing each input basename without extension as a substring; matching pieces are dissolved using first-match attributes; defaults to "image".
            image_rank_function: Optional footprint-attribute expression using weighted() syntax, e.g. "{quality} - {cloud_cover}"; None disables quality preference and saves null scores.
            image_rank_descending: True favors larger quality scores, False favors smaller scores; defaults to True.
            image_quality_weight: Nonnegative multiplier of assigned area times globally min-max-normalized quality penalty; zero removes quality preference; defaults to 1.
            edge_variables_weights_gsds_bands: Terms [name_or_path, nonnegative weight, GSD selection, band selection]; use [value] for one value or {"average" | "largest" | "smallest": [values]} for two or more; GSDs are positive integers in CRS units or "native", bands are one-based integers; defaults weight difference 1 at "native" and magnitude 0.25 at GSD 100, both on band 1; zero disables a term.
            foreground_path: Optional polygon mask forbidding seams through protected interiors; infeasible coverage raises ValueError.
            foreground_layer: Optional layer in foreground_path; None reads the default layer.
            mesh_spacing: Positive mesh-cell width in CRS units, independent of cost GSDs; footprint boundaries constrain triangles; defaults to 10.
            quadtree_max_images_per_leaf: Positive label-count target for quadtree leaves; defaults to 16.
            quadtree_overlap: Child-block overlap fraction in (0, 0.5); defaults to 0.1.
            quadtree_max_depth: Nonnegative quadtree subdivision limit; defaults to 8.
            solver_max_iterations: Positive graph-cut sweep limit per node; defaults to 50; reaching the limit warns.
            image_threads: Local worker count, "cpu" or None for serial execution; defaults to None.
            concurrent_processing_backend: "process_pool" for local workers or "dask" for an existing cluster; defaults to "process_pool".
            dask_scheduler: Dask connection as ("file", path) or ("address", address); requires image_threads=None; defaults to None.
            debug_logs: Print node sizes and processing details; defaults to False.
            resume_from_outputs: "no" recomputes, "yes" reuses existing output, "validate" checks the requested polygon layer; incomplete outputs always recompute.

        Returns:
            str: Written GeoPackage containing disjoint per-image polygons covering the supplied footprints.

        Notes:
            Built-in variables are "laplacian_difference" and "laplacian_magnitude"; an external raster path selects direct costs; defaults use one GSD and band each without aggregation.
            "native" uses the finest input-image pixel size for built-in costs, or the external raster's pixel size in the working CRS; repeated GSD/band requests share sampling and Laplacians within each node.
            GDAL average-resampled VRTs supply windowed samples; vectorized five-point Laplacians divide by GSD squared and extend missing neighbors with the center value.
            Difference costs reduce absolute inter-image Laplacian differences over bands then scales, then take the maximum sample cost along each edge; magnitude reduces each image's absolute Laplacian identically, then averages the two edge values.
            External rasters supply direct nonnegative finite costs, reduced over bands, scales and edge samples without derivatives; all edge terms are zero for equal labels and their weighted sum is minimized alongside quality penalties.
            Repeat a variable with different GSD lists to weight scales separately; costs retain their numeric units without automatic normalization.
            Alpha expansion finds a move-local minimum; smallest-reduced differences use alpha-beta swaps because their costs need not satisfy the triangle inequality.
        """
        _print_step_start("markov_triangles")
        _validate_polygon_output(output_mask, output_layer)
        Universal._validate(
            input_images=input_images,
            image_threads=image_threads,
            debug_logs=debug_logs,
            concurrent_processing_backend=concurrent_processing_backend,
            dask_scheduler=dask_scheduler,
        )
        SeamlineValidation._validate_weighted_seamline(
            image_field_name=image_field_name,
            input_layer=input_layer,
            output_layer=output_layer,
            rank_descending=image_rank_descending,
        )
        if input_polygons is not None:
            SeamlineValidation._validate_weighted_seamline(
                input_polygons=input_polygons
            )
        if image_rank_function is not None:
            SeamlineValidation._validate_weighted_seamline(
                rank_function=image_rank_function
            )
        if isinstance(image_threads, bool):
            raise ValueError(
                "image_threads must be a positive integer, 'cpu', or None."
            )
        for name, value, allow_zero in [
            ("mesh_spacing", mesh_spacing, False),
            ("quadtree_overlap", quadtree_overlap, False),
            ("image_quality_weight", image_quality_weight, True),
        ]:
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
                or (not allow_zero and value == 0)
            ):
                raise ValueError(
                    f"{name} must be finite and {'nonnegative' if allow_zero else 'positive'}."
                )
        for name, value, minimum in [
            ("quadtree_max_images_per_leaf", quadtree_max_images_per_leaf, 1),
            ("quadtree_max_depth", quadtree_max_depth, 0),
            ("solver_max_iterations", solver_max_iterations, 1),
        ]:
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}.")
        if quadtree_overlap >= 0.5:
            raise ValueError("quadtree_overlap must be less than 0.5.")
        _markov._variables(edge_variables_weights_gsds_bands)
        if image_field_name in {"geometry", "image_path", "quality_score"}:
            raise ValueError(
                "image_field_name must differ from geometry, image_path and quality_score."
            )
        if foreground_path is not None and (
            not isinstance(foreground_path, str) or not foreground_path.strip()
        ):
            raise ValueError("foreground_path must be a nonempty string or None.")
        if foreground_layer is not None and (
            not isinstance(foreground_layer, str)
            or not foreground_layer.strip()
            or foreground_path is None
        ):
            raise ValueError(
                "foreground_layer requires foreground_path and a nonempty layer name."
            )
        if resume_from_outputs not in {"no", "yes", "validate"}:
            raise ValueError("resume_from_outputs must be 'no', 'yes', or 'validate'.")
        for path in [input_polygons, foreground_path]:
            if path is not None and os.path.realpath(path) == os.path.realpath(
                output_mask
            ):
                raise ValueError(
                    "Input and output GeoPackages must be different files."
                )
        if _footprint_output_is_reusable(
            output_mask,
            resume_from_outputs,
            debug_logs,
            "markov_triangles",
            output_layer=output_layer,
            image_field_name=image_field_name,
        ):
            return output_mask
        if _markov.maxflow is None or _markov.constrained_delaunay_triangles is None:
            raise ImportError(
                "Triangle seamlines require PyMaxflow>=1.3.2 and Shapely>=2.1.0, included in the standard spectralmatch installation."
            )
        frame = _prepare_footprints(
            input_images,
            input_polygons,
            input_layer,
            image_field_name,
            image_threads,
            concurrent_processing_backend,
            dask_scheduler,
        )
        candidates, paths, terms = _markov._sources(
            frame, image_field_name, edge_variables_weights_gsds_bands
        )
        scores = dict.fromkeys(paths)
        quality = dict.fromkeys(paths, 0.0)
        if image_rank_function is not None:
            ranked = _rank_polygons(
                frame, image_field_name, image_rank_function, image_rank_descending
            )
            scores = dict(zip(ranked[image_field_name], ranked.weighted_score))
            lo, hi = min(scores.values()), max(scores.values())
            if hi > lo:
                scale = max(abs(lo), abs(hi))
                lo, hi = lo / scale, hi / scale
                quality = {
                    name: (
                        (hi - score / scale)
                        if image_rank_descending
                        else (score / scale - lo)
                    )
                    / (hi - lo)
                    for name, score in scores.items()
                }
        foreground = Polygon()
        if foreground_path is not None:
            mask = _read_polygons(foreground_path, foreground_layer)
            if mask.crs != frame.crs:
                raise ValueError(
                    "Foreground mask and footprints must have the same CRS."
                )
            foreground = unary_union(list(mask.geometry))
        levels, children = _markov._quadtree(
            candidates,
            quadtree_max_images_per_leaf,
            quadtree_overlap,
            quadtree_max_depth,
        )
        parallel, workers = _resolve_parallel_config(
            image_threads, concurrent_processing_backend, dask_scheduler
        )
        schema = {
            "geometry": "MultiPolygon",
            "properties": {
                image_field_name: "str",
                "image_path": "str",
                "quality_score": "float",
            },
        }
        with _PolygonWriter(
            output_mask, output_layer, schema, frame.crs.to_wkt()
        ) as writer:
            completed = {}
            for level in reversed(levels):
                args = []
                for key, bounds, source in level:
                    if key in children:
                        source = [completed.pop(child) for child in children[key]]
                    names = {name for candidate in source for name in candidate}
                    args.append(
                        (
                            source,
                            {name: paths[name] for name in names},
                            mesh_spacing,
                            terms,
                            foreground.intersection(box(*bounds)),
                            {name: quality[name] for name in names},
                            image_quality_weight,
                            frame.crs.to_wkt(),
                            solver_max_iterations,
                            debug_logs,
                        )
                    )
                results = _run_image_tasks(
                    _markov._solve_node,
                    args,
                    input_paths=[f"markov_triangles:{node[0]}" for node in level],
                    output_paths=[output_mask] * len(level),
                    parallel=parallel,
                    backend="process",
                    workers=workers,
                    concurrent_processing_backend=concurrent_processing_backend,
                    dask_scheduler=dask_scheduler,
                )
                completed.update(
                    (node[0], result) for node, result in zip(level, results)
                )
            for name, geometry in sorted(completed["root"].items()):
                writer.write(
                    MultiPolygon(_polygon_parts(geometry)),
                    {
                        image_field_name: name,
                        "image_path": paths[name],
                        "quality_score": scores[name],
                    },
                )
            if writer.count == 0:
                raise ValueError("No seamline polygons were generated.")
        return output_mask

    @staticmethod
    def weighted(
        input_polygons: str,
        output_mask: str,
        *,
        input_images: Universal.SearchFolderOrListFiles | None = None,
        rank_function: str,
        image_field_name: str = "image",
        input_layer: str | None = "footprints",
        output_layer: str = "seamlines",
        rank_descending: bool = True,
        debug_logs: Universal.DebugLogs = False,
        resume_from_outputs: Literal["no", "yes", "validate"] = "no",
    ) -> str:
        """Generate seamline polygons by ranking image footprints with a weighted expression.

        Args:
            input_images (str | list[str] | None, optional): Optional raster folder, glob or paths for substring matching; None infers image basenames from the identifier field. Defaults to None.
            input_polygons (str): Input polygon layer path. Each feature should represent an image footprint or a piece of one.
            output_mask (str): Output GeoPackage path for the ranked seamline polygons.
            rank_function (str): Ranking expression using field placeholders like ``{cloud_cover}`` or formulas like ``1 / ({sun_elevation} + 1)``.
            image_field_name (str, optional): Field whose value contains the image basename without extension; matching features are dissolved and first-match attributes are retained. Defaults to "image".
            input_layer (str | None, optional): Input footprint layer name; None selects the first layer. Defaults to "footprints".
            output_layer (str, optional): Output GeoPackage layer name. Defaults to ``"seamlines"``.
            rank_descending (bool, optional): If True, larger scores rank higher and remain on top. Defaults to True.
            debug_logs (bool, optional): If True, prints ranking details. Defaults to False.
            resume_from_outputs (Literal["no", "yes", "validate"], optional): Recompute outputs with "no", reuse existing outputs with "yes", or validate existing outputs before reusing them with "validate"; default is "no".

        Returns:
            str: Written output GeoPackage path."""
        _print_step_start("weighted_seamline")
        SeamlineValidation._validate_weighted_seamline(
            input_polygons=input_polygons,
            output_mask=output_mask,
            rank_function=rank_function,
            image_field_name=image_field_name,
            input_layer=input_layer,
            output_layer=output_layer,
            rank_descending=rank_descending,
        )
        if debug_logs:
            print(f"Input polygons: {input_polygons}")
            print(f"Output mask: {output_mask}")
            print(f"Rank function: {rank_function}")
        if _existing_outputs_are_reusable(
            [output_mask],
            resume_mode=resume_from_outputs,
            debug_logs=debug_logs,
            step_name="weighted_seamline",
        ):
            return output_mask
        _print_image_start(input_polygons, output_mask)
        result = weighted_seamline(
            input_polygons=input_polygons,
            output_mask=output_mask,
            rank_function=rank_function,
            input_images=input_images,
            image_field_name=image_field_name,
            input_layer=input_layer,
            output_layer=output_layer,
            rank_descending=rank_descending,
            debug_logs=debug_logs,
        )
        _print_image_completed(input_polygons, 1, 1)
        return result

    @staticmethod
    def voronoi(
        input_polygons: str,
        output_mask: str,
        *,
        input_images: Universal.SearchFolderOrListFiles | None = None,
        aoi_path: str | None = None,
        input_layer: str | None = "footprints",
        output_layer: str = "seamlines",
        image_field_name: str = "image",
        min_point_spacing: float = 10,
        min_cut_length: float = 0,
        debug_logs: Universal.DebugLogs = False,
        debug_vectors_path: str | None = None,
        resume_from_outputs: Literal["no", "yes", "validate"] = "no",
    ) -> str:
        """Generates a Voronoi-based seamline mask from edge-matching polygons (EMPs) and writes the result to a vector file.

        Args:
            input_images (str | list[str] | None, optional): Optional raster folder, glob or paths for substring matching; None infers image basenames from the identifier field. Defaults to None.
            input_polygons (str): Input polygon layer path. Each feature should represent an image footprint or a piece of one. Features sharing the same image identifier are merged before processing.
            input_layer (str | None, optional): Input footprint layer name; None selects the first layer. Defaults to "footprints".
            output_layer (str, optional): Output GeoPackage layer name. Defaults to ``"seamlines"``.
            output_mask (str): Output path for the final seamline polygon vector file.
            aoi_path (str, optional): Path to an AOI vector file to clip overlapping image polygons; default is None.
            min_point_spacing (float, optional): Minimum spacing between Voronoi seed points; default is 10.
            min_cut_length (float, optional): Minimum cutline segment length to retain; default is 0.
            debug_logs (Universal.DebugLogs, optional): Enables debug print statements if True; default is False.
            image_field_name (str, optional): Field whose value contains the image basename without extension; matching features are dissolved. Defaults to "image".
            debug_vectors_path (str | None, optional): Optional path to save debug layers (cutlines, intersections).
            resume_from_outputs (Literal["no", "yes", "validate"], optional): Recompute outputs with "no", reuse existing outputs with "yes", or validate existing outputs before reusing them with "validate"; default is "no".

        Returns:
            str: Written output GeoPackage path.

        Outputs:
            Saves a polygon seamline layer to `output_mask`, and optionally saves intermediate cutlines to `debug_vectors_path`.
        """
        _print_step_start("voronoi_center_seamline")
        output_dir = os.path.dirname(output_mask)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        if debug_vectors_path:
            debug_dir = os.path.dirname(debug_vectors_path)
            if debug_dir:
                os.makedirs(debug_dir, exist_ok=True)
        if _existing_outputs_are_reusable(
            [output_mask],
            resume_mode=resume_from_outputs,
            debug_logs=debug_logs,
            step_name="voronoi_center_seamline",
        ):
            return output_mask
        SeamlineValidation._validate_voronoi_center_seamline(
            output_mask=output_mask,
            aoi_path=aoi_path,
            image_field_name=image_field_name,
            min_point_spacing=min_point_spacing,
            min_cut_length=min_cut_length,
            debug_vectors_path=debug_vectors_path,
        )
        SeamlineValidation._validate_weighted_seamline(
            input_polygons=input_polygons,
            input_layer=input_layer,
            output_layer=output_layer,
            image_field_name=image_field_name,
        )
        frame = _prepare_footprints(
            input_images, input_polygons, input_layer, image_field_name
        )
        crs = frame.crs.to_wkt()
        emps, input_image_names = [], []
        for image_name, group in frame.groupby(image_field_name, sort=False):
            # Process every island independently; merge by image ID when writing.
            parts = _polygon_parts(unary_union(list(group.geometry)))
            emps.extend(parts)
            input_image_names.extend([str(image_name)] * len(parts))
        input_image_paths = input_image_names
        for image_name in input_image_names:
            _print_image_start(f"{input_polygons}:{image_name}", output_mask)

        image_details = [
            (
                {"Footprint area": f"{emp.area:.2f}", "Bounds": str(emp.bounds)}
                if debug_logs
                else {}
            )
            for emp in emps
        ]

        if debug_vectors_path:
            if os.path.exists(debug_vectors_path):
                os.remove(debug_vectors_path)
            _save_emp_outlines(
                emps,
                input_image_paths,
                debug_vectors_path,
                crs,
                image_field_name=image_field_name,
            )

        cuts: list[LineString] = []
        for i, (left, right) in enumerate(combinations(range(len(emps)), 2)):
            if input_image_names[left] == input_image_names[right]:
                continue
            a, b = emps[left], emps[right]
            ov = a.intersection(b)
            if debug_logs:
                print(f"Overlap {i} area: {ov.area:.2f}")
            if ov.area > 0:
                if debug_vectors_path:
                    _save_intersection_points(a, b, debug_vectors_path, crs, f"{i}")
                cut = _compute_centerline(
                    a,
                    b,
                    min_point_spacing,
                    min_cut_length,
                    debug_logs,
                    crs,
                    debug_vectors_path,
                )
                cuts.append(cut)

        if debug_vectors_path:
            schema = {"geometry": "LineString", "properties": {"pair_id": "str"}}
            with fiona.open(
                debug_vectors_path,
                "w",
                driver="GPKG",
                crs_wkt=crs,
                schema=schema,
                layer="cutlines",
            ) as dst:
                for idx, line in enumerate(cuts):
                    dst.write(
                        {
                            "geometry": mapping(line),
                            "properties": {"pair_id": f"{idx}"},
                        }
                    )

        segmented: list[Polygon] = []
        for idx, emp in enumerate(emps):
            relevant = [cut for cut in cuts if emp.intersects(cut)]
            seg = _segment_emp(emp, relevant, debug_logs)
            if debug_logs:
                image_details[idx].update(
                    {"Cuts": len(relevant), "Segmented area": f"{seg.area:.2f}"}
                )
            segmented.append(seg)

        if aoi_path is not None:
            segmented = _mask_by_aoi(segmented, aoi_path)

        schema = {"geometry": "MultiPolygon", "properties": {image_field_name: "str"}}
        with fiona.open(
            output_mask,
            "w",
            driver="GPKG",
            crs_wkt=crs,
            schema=schema,
            layer=output_layer,
        ) as dst:

            grouped = {}
            for image_name, poly in zip(input_image_names, segmented):
                grouped.setdefault(image_name, []).extend(_polygon_parts(poly))
            for image_name, parts in grouped.items():
                if parts:
                    dst.write(
                        {
                            "geometry": mapping(
                                MultiPolygon(_polygon_parts(unary_union(parts)))
                            ),
                            "properties": {image_field_name: image_name},
                        }
                    )

        for completed, path in enumerate(input_image_paths, 1):
            _print_image_completed(
                path,
                completed,
                len(input_image_paths),
                details=image_details[completed - 1],
            )

        return output_mask


__all__ = ["Seamline"]
