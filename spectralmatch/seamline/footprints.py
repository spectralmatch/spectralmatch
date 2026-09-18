import heapq
import math
import os
from typing import Literal

import fiona
import geopandas as gpd
import pandas as pd
from osgeo import gdal, ogr
from shapely import make_valid
from shapely.strtree import STRtree
from shapely.geometry import LineString, Polygon, MultiPolygon, Point, mapping
from shapely.ops import nearest_points, polylabel, unary_union
from shapely.wkb import loads

from ..handlers import _resolve_paths, _existing_outputs_are_reusable
from ..types_and_validation import Universal
from ..utils_logging import _print_step_start
from ..utils_multiprocessing import _resolve_parallel_config, _run_image_tasks


def _polygon_parts(geometry):
    """Return all nonempty polygon components of a geometry."""
    if geometry.is_empty:
        return []
    if isinstance(geometry, Polygon):
        return [geometry]
    return [
        part
        for child in getattr(geometry, "geoms", [])
        for part in _polygon_parts(child)
    ]


def _read_polygons(path, input_layer=None, image_field_name=None):
    """Read valid polygon features with a CRS and optional image identifiers."""
    frame = gpd.read_file(path, **({"layer": input_layer} if input_layer else {}))
    if frame.empty or frame.crs is None:
        raise ValueError("input_polygons must contain features and have a CRS.")
    if image_field_name is not None:
        if image_field_name not in frame or frame[image_field_name].isna().any():
            raise ValueError(
                f"Input polygons require non-null '{image_field_name}' values."
            )
    for geometry in frame.geometry:
        if (
            geometry is None
            or geometry.is_empty
            or not geometry.is_valid
            or geometry.geom_type not in {"Polygon", "MultiPolygon"}
        ):
            raise ValueError(
                "Input features must be valid, nonempty Polygon or MultiPolygon geometries."
            )
    return frame


def _validate_polygon_output(output_path, output_layer):
    """Validate the GeoPackage destination and layer name."""
    if not isinstance(output_path, str) or not output_path.lower().endswith(".gpkg"):
        raise ValueError("Output path must end in .gpkg.")
    if not isinstance(output_layer, str) or not output_layer.strip():
        raise ValueError("output_layer must be a nonempty string.")


class _PolygonWriter:
    """Commit returned geometries in the parent and mark unfinished outputs as incomplete."""

    def __init__(self, path, layer, schema, crs):
        self.path, self.layer, self.schema, self.crs = path, layer, schema, crs
        self.destination = None
        self.count = 0
        self.marker = path + ".incomplete"

    def __enter__(self):
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        with open(self.marker, "w") as marker:
            marker.write("Footprint processing is incomplete; rerun to recompute.\n")
        return self

    def write(self, geometry, record):
        """Write and flush one feature before the parent reports completion."""
        if geometry.is_empty:
            return
        if self.destination is None:
            # Open only after workers have started, so processes cannot inherit
            # an open GeoPackage writer or its SQLite locks.
            self.destination = fiona.open(
                self.path,
                "w",
                driver="GPKG",
                layer=self.layer,
                schema=self.schema,
                crs_wkt=self.crs,
            )
        properties = {}
        for key, value in record.items():
            if pd.isna(value):
                value = None
            elif hasattr(value, "isoformat"):
                value = value.isoformat()
            properties[key] = value
        self.destination.write(
            {"geometry": mapping(geometry), "properties": properties}
        )
        self.destination.flush()
        self.count += 1

    def __exit__(self, exc_type, exc_value, traceback):
        if self.destination is not None:
            self.destination.close()
        if exc_type is None:
            os.remove(self.marker)


def _footprint_output_is_reusable(output_path, resume_mode, debug_logs, step_name):
    """Never reuse a partial streaming output as a completed GeoPackage."""
    return _existing_outputs_are_reusable(
        [output_path],
        resume_mode=(
            "no" if os.path.exists(output_path + ".incomplete") else resume_mode
        ),
        debug_logs=debug_logs,
        step_name=step_name,
    )


def _footprint_from_image(path, band, eight_connected):
    """Polygonize a raster band's valid-data mask in map coordinates."""
    dataset = gdal.Open(path, gdal.GA_ReadOnly)
    if dataset is None or not 1 <= band <= dataset.RasterCount:
        raise ValueError(f"Cannot read band {band} from {path}.")
    crs = dataset.GetProjectionRef()
    if not crs:
        raise ValueError(f"Raster must have a CRS: {path}")
    mask = dataset.GetRasterBand(band).GetMaskBand()
    vector = ogr.GetDriverByName("MEM").CreateDataSource("")
    layer = vector.CreateLayer("footprint", geom_type=ogr.wkbPolygon)
    # Mask bands may have no parent dataset; explicitly supply georeferencing.
    options = [f"DATASET_FOR_GEOREF={path}"] + (
        ["8CONNECTED=8"] if eight_connected else []
    )
    if gdal.Polygonize(mask, mask, layer, -1, options) != gdal.CE_None:
        raise RuntimeError(f"Cannot polygonize {path}.")
    parts = []
    for feature in layer:
        parts.extend(
            _polygon_parts(
                make_valid(loads(bytes(feature.GetGeometryRef().ExportToWkb())))
            )
        )
    if not parts:
        raise ValueError(f"No valid pixels in {path}.")
    return unary_union(parts), crs


def create_footprints(
    input_images: Universal.SearchFolderOrListFiles,
    output_polygons: str,
    *,
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
    parallel, workers = _resolve_parallel_config(
        image_threads, concurrent_processing_backend, dask_scheduler
    )
    from pyproj import CRS

    schema = {
        "geometry": "Unknown",
        "properties": {image_field_name: "str", "image_path": "str"},
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
                geometry, {image_field_name: names[index], "image_path": paths[index]}
            )

        _run_image_tasks(
            _footprint_from_image,
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


def _simplify_inner(polygon, tolerance, area_weight):
    """Greedily remove vertices using inward shortcuts and a normalized area/perimeter cost."""
    original = polygon
    rings = [list(polygon.exterior.coords)[:-1]] + [
        list(ring.coords)[:-1] for ring in polygon.interiors
    ]
    original_area = original.area
    scale = math.sqrt(original_area)
    tolerance_squared = tolerance * tolerance
    previous = [[(i - 1) % len(ring) for i in range(len(ring))] for ring in rings]
    following = [[(i + 1) % len(ring) for i in range(len(ring))] for ring in rings]
    active = [set(range(len(ring))) for ring in rings]
    chains = [[[] for _ in ring] for ring in rings]
    versions = [[0] * len(ring) for ring in rings]
    queue = []
    vertex_keys = [(r, i) for r, ring in enumerate(rings) for i in range(len(ring))]
    vertices = STRtree([Point(rings[r][i]) for r, i in vertex_keys])
    ring_signs = [
        1 if ring.is_ccw else -1 for ring in [original.exterior, *original.interiors]
    ]

    def enqueue(r, i):
        """Queue a local shortcut with its area/perimeter cost and displacement bound."""
        versions[r][i] += 1
        if i not in active[r] or len(active[r]) <= 3:
            return
        p, n = previous[r][i], following[r][i]
        a, v, b = rings[r][p], rings[r][i], rings[r][n]
        dx, dy = b[0] - a[0], b[1] - a[1]
        length_squared = dx * dx + dy * dy
        removed = chains[r][p] + [v] + chains[r][i]
        for x, y in removed:
            t = (
                max(0.0, min(1.0, ((x - a[0]) * dx + (y - a[1]) * dy) / length_squared))
                if length_squared
                else 0.0
            )
            if (x - a[0] - t * dx) ** 2 + (y - a[1] - t * dy) ** 2 > tolerance_squared:
                return
        loss = abs((v[0] - a[0]) * dy - (v[1] - a[1]) * dx) / (2 * original_area)
        saving = (math.dist(a, v) + math.dist(v, b) - math.dist(a, b)) / scale
        cost = area_weight * loss - (1 - area_weight) * saving
        if cost <= 0:
            heapq.heappush(queue, (cost, r, i, versions[r][i]))

    for r, ring in enumerate(rings):
        for i in range(len(ring)):
            enqueue(r, i)
    while queue:
        _, r, i, version = heapq.heappop(queue)
        if i not in active[r] or versions[r][i] != version or len(active[r]) <= 3:
            continue
        p, n = previous[r][i], following[r][i]
        triangle = Polygon([rings[r][p], rings[r][i], rings[r][n]])
        if triangle.area:
            a, v, b = rings[r][p], rings[r][i], rings[r][n]
            turn = (v[0] - a[0]) * (b[1] - v[1]) - (v[1] - a[1]) * (b[0] - v[0])
            if turn * ring_signs[r] * (1 if r == 0 else -1) <= 0:
                continue
            # The inward turn and absence of other active boundary vertices
            # establish a removable ear, including when the polygon has holes.
            # A crossing edge would have to cross an existing adjacent edge or
            # have an endpoint in this ear; both are excluded for a valid ring.
            adjacent = {(r, p), (r, i), (r, n)}
            if any(
                vertex_keys[index] not in adjacent
                and vertex_keys[index][1] in active[vertex_keys[index][0]]
                for index in vertices.query(triangle, predicate="intersects")
            ):
                continue
        chains[r][p] += [rings[r][i]] + chains[r][i]
        following[r][p], previous[r][n] = n, p
        active[r].remove(i)
        enqueue(r, p)
        enqueue(r, n)
    candidate_rings = [
        [point for i, point in enumerate(ring) if i in active[r]]
        for r, ring in enumerate(rings)
    ]
    candidate = Polygon(candidate_rings[0], candidate_rings[1:])
    # Retain the input if numerical degeneracies defeat the local checks.
    return candidate if candidate.is_valid and original.covers(candidate) else original


def _filter_polygon_area(geometry, area_filter=None, area_rank=1):
    """Filter components by minimum area, then retain the largest or smallest requested count."""
    parts = [
        part
        for part in _polygon_parts(geometry)
        if area_filter is None or part.area >= area_filter
    ]
    if area_rank:
        parts = sorted(parts, key=lambda part: part.area, reverse=area_rank > 0)[
            : abs(area_rank)
        ]
    return unary_union(parts) if parts else MultiPolygon()


def _hole_diameter(hole):
    """Return the maximum vertex-to-vertex distance using convex-hull rotating calipers."""
    points = list(hole.convex_hull.exterior.coords)[:-1]
    count = len(points)
    opposite = 1
    diameter_squared = 0.0

    def cross_area(a, b, point):
        """Return twice the unsigned area between an edge and a point."""
        return abs(
            (b[0] - a[0]) * (point[1] - a[1]) - (b[1] - a[1]) * (point[0] - a[0])
        )

    for index, a in enumerate(points):
        b = points[(index + 1) % count]
        while cross_area(a, b, points[(opposite + 1) % count]) > cross_area(
            a, b, points[opposite]
        ):
            opposite = (opposite + 1) % count
        # Include the next antipodal vertex to cover parallel-edge ties.
        for endpoint in (a, b):
            for candidate in (points[opposite], points[(opposite + 1) % count]):
                diameter_squared = max(
                    diameter_squared,
                    (endpoint[0] - candidate[0]) ** 2
                    + (endpoint[1] - candidate[1]) ** 2,
                )
    return math.sqrt(diameter_squared)


def _hole_cut_width(hole, cut_width):
    """Resolve an angle-independent width from the original hole geometry."""
    if cut_width == "maximum_inscribed_circle":
        # polylabel uses the native MIC implementation on Shapely 2.1+, and
        # provides the same search on 2.0. Use a rotation-invariant tolerance.
        center = polylabel(hole, tolerance=math.sqrt(hole.area) / 1000)
        return 2 * center.distance(hole.boundary)
    if cut_width == "hole_size":
        return _hole_diameter(hole)
    return cut_width


def _postprocess_polygon(
    geometry,
    edge_distance,
    relative_edge_distance,
    cut_width,
    cut_method,
    smoothing_radius,
    simplify_tolerance,
    simplify_area_weight,
    area_filter=None,
    area_rank=1,
    hole_to_hole_distance=800,
):
    """Connect selected holes to edges/holes, then smooth and simplify inward."""
    results = []
    for original in _polygon_parts(geometry):
        cuts = []
        holes = [Polygon(ring) for ring in original.interiors]
        widths = {}

        def width_for(index):
            if index not in widths:
                widths[index] = _hole_cut_width(holes[index], cut_width)
            return widths[index]

        if edge_distance > 0:
            for index, hole in enumerate(holes):
                distance = hole.distance(original.exterior)
                if distance > edge_distance:
                    continue
                if (
                    relative_edge_distance is not None
                    and distance / math.sqrt(hole.area / math.pi)
                    > relative_edge_distance
                ):
                    continue
                width = width_for(index)
                if cut_method == "corridor":
                    cuts.append(
                        LineString(nearest_points(hole, original.exterior)).buffer(
                            width / 2
                        )
                    )
                else:
                    cuts.append(hole.buffer(distance + width / 2))
        if hole_to_hole_distance > 0 and len(holes) > 1:
            tree = STRtree(holes)
            for index, hole in enumerate(holes):
                # Query original holes and visit each unordered pair once;
                # newly cut boundaries never change distance eligibility.
                neighbors = tree.query(
                    hole, predicate="dwithin", distance=hole_to_hole_distance
                )
                for other_index in sorted(i for i in neighbors if i > index):
                    other = holes[other_index]
                    width = min(width_for(index), width_for(other_index))
                    if cut_method == "corridor":
                        cuts.append(
                            LineString(nearest_points(hole, other)).buffer(width / 2)
                        )
                    else:
                        radius = (hole.distance(other) + width) / 2
                        cuts.extend([hole.buffer(radius), other.buffer(radius)])
        cut = original.difference(unary_union(cuts)) if cuts else original
        if smoothing_radius:
            cut = (
                cut.buffer(-smoothing_radius).buffer(smoothing_radius).intersection(cut)
            )
        for part in _polygon_parts(cut):
            if simplify_tolerance:
                part = _simplify_inner(part, simplify_tolerance, simplify_area_weight)
            results.append(part)
    geometry = unary_union(results) if results else MultiPolygon()
    return _filter_polygon_area(geometry, area_filter, area_rank)


def postprocess_footprints(
    input_polygons: str,
    output_polygons: str,
    *,
    input_layer: str | None = None,
    output_layer: str = "footprints",
    edge_distance: float = 800,
    hole_to_hole_distance: float = 800,
    relative_edge_distance: float | None = None,
    cut_width: (
        int | Literal["hole_size", "maximum_inscribed_circle"]
    ) = "maximum_inscribed_circle",
    cut_method: Literal["corridor", "buffer"] = "corridor",
    smoothing_radius: float = 240,
    simplify_tolerance: float = 120,
    simplify_area_weight: float = 0.5,
    area_filter: float | None = None,
    area_rank: int | None = 1,
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
        input_layer (str | None, optional): Optional input layer name when reading multi-layer vector sources. Defaults to None.
        output_layer (str, optional): Output GeoPackage layer name. Defaults to "footprints".
        edge_distance (float, optional): Maximum hole-to-edge distance in CRS units, measured against each component's original outer ring; zero disables only hole-to-edge cuts. Defaults to 800.
        hole_to_hole_distance (float, optional): Maximum boundary-to-boundary distance in CRS units between original holes in the same polygon component; every qualifying pair is connected. Zero disables only hole-to-hole cuts. Defaults to 800.
        relative_edge_distance (float | None, optional): Additional upper limit on hole-to-edge distance divided by sqrt(hole_area / pi); applies only to edge cuts. None disables this size-relative filter. Defaults to None.
        cut_width (int | Literal["hole_size", "maximum_inscribed_circle"], optional): Positive integer width in CRS units, "hole_size" for the largest vertex-to-vertex diameter, or "maximum_inscribed_circle" for the largest circle fitting inside the hole (diameter, approximated with radius tolerance sqrt(hole_area) / 1000). Applies to both edge and hole-pair cuts; pairs use the smaller hole width. Defaults to "maximum_inscribed_circle".
        cut_method (Literal["corridor", "buffer"], optional): Subtract a shortest connection buffered by half the cut width with "corridor". With "buffer", expand an edge-selected hole by distance + width / 2, or both holes in a pair by (distance + width) / 2. Defaults to "corridor".
        smoothing_radius (float, optional): Nonnegative erosion and dilation radius, followed by intersection with the cut polygon to prevent expansion or refilling cuts; zero disables smoothing. Defaults to 240.
        simplify_tolerance (float, optional): Nonnegative maximum deviation of removed vertices from inward shortcuts, including cumulative removals; zero disables simplification. Defaults to 120.
        simplify_area_weight (float, optional): Weight in [0, 1] balancing normalized area loss against perimeter reduction in the greedy inward shortcut simplifier; larger values favor retaining area. Defaults to 0.5.
        area_filter (float | None, optional): Minimum retained component area in squared CRS units, applied after smoothing and simplification and before area_rank; None disables the threshold. Defaults to None.
        area_rank (int | None, optional): Keep the largest N components per feature for positive N, or the smallest abs(N) for negative N; 0 or None keeps all components passing area_filter. Defaults to 1.
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
        "edge_distance": edge_distance,
        "hole_to_hole_distance": hole_to_hole_distance,
        "smoothing_radius": smoothing_radius,
        "simplify_tolerance": simplify_tolerance,
        "simplify_area_weight": simplify_area_weight,
        "relative_edge_distance": (
            0 if relative_edge_distance is None else relative_edge_distance
        ),
    }.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise ValueError(f"{name} must be finite and nonnegative.")
    if cut_width not in ("hole_size", "maximum_inscribed_circle") and (
        isinstance(cut_width, bool) or not isinstance(cut_width, int) or cut_width <= 0
    ):
        raise ValueError(
            'cut_width must be a positive integer, "hole_size", or "maximum_inscribed_circle".'
        )
    if simplify_area_weight > 1 or cut_method not in {"corridor", "buffer"}:
        raise ValueError(
            "Require simplify_area_weight in [0, 1] and cut_method corridor or buffer."
        )
    if area_filter is not None and (
        isinstance(area_filter, bool)
        or not isinstance(area_filter, (int, float))
        or not math.isfinite(area_filter)
        or area_filter < 0
    ):
        raise ValueError("area_filter must be a finite nonnegative number or None.")
    if area_rank is not None and (
        isinstance(area_rank, bool) or not isinstance(area_rank, int)
    ):
        raise ValueError("area_rank must be an integer or None.")
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
            edge_distance,
            relative_edge_distance,
            cut_width,
            cut_method,
            smoothing_radius,
            simplify_tolerance,
            simplify_area_weight,
            area_filter,
            area_rank,
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
            _postprocess_polygon,
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
                "Postprocessing removed all polygons; reduce the processing distances or area_filter."
            )
    return output_polygons
