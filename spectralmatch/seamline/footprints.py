import csv
import heapq
import math
import os

import fiona
import geopandas as gpd
import pandas as pd
from osgeo import gdal, ogr
from pyproj import CRS
from shapely import make_valid
from shapely.strtree import STRtree
from shapely.geometry import LineString, Polygon, MultiPolygon, Point, mapping
from shapely.ops import nearest_points, polylabel, unary_union
from shapely.wkb import loads

from ..handlers import _resolve_paths, _existing_outputs_are_reusable
from ..types_and_validation import Universal
from ..utils_multiprocessing import _resolve_parallel_config, _run_image_tasks


def _matching_image_rows(frame, field, name):
    """Select rows whose field literally contains an image basename, ignoring case."""
    return frame[frame[field].astype(str).str.contains(name, regex=False, case=False)]


def _read_image_metadata(path, join_field, names, image_field_name):
    """Read typed CSV attributes and require one basename-substring match per image before polygonization."""
    if not isinstance(join_field, str) or not join_field.strip():
        raise ValueError("metadata_image_field_name must be a nonempty string.")
    if path is None:
        return {}, [{} for _ in names]
    with open(path, newline="", encoding="utf-8-sig") as source:
        columns = next(csv.reader(source), [])
        if not columns or any(not column.strip() for column in columns):
            raise ValueError("metadata_csv requires nonempty column names.")
        if len({column.casefold() for column in columns}) != len(columns):
            raise ValueError("metadata_csv column names must be unique ignoring case.")
        if join_field not in columns:
            raise ValueError(f"metadata_csv has no join column {join_field!r}.")
        source.seek(0)
        frame = pd.read_csv(
            source,
            dtype={join_field: str},
            keep_default_na=False,
            na_values={column: [""] for column in columns if column != join_field},
        )
    if frame[join_field].str.strip().eq("").any():
        raise ValueError(f"metadata_csv join column {join_field!r} has empty values.")
    attributes = frame.drop(columns=join_field)
    reserved = {image_field_name.casefold(), "image_path", "geometry", "fid"}
    collisions = [column for column in attributes if column.casefold() in reserved]
    if collisions:
        raise ValueError(
            f"metadata_csv columns conflict with output fields: {collisions}."
        )
    schema = {
        column: (
            "int64"
            if pd.api.types.is_integer_dtype(dtype) or pd.api.types.is_bool_dtype(dtype)
            else "float" if pd.api.types.is_float_dtype(dtype) else "str"
        )
        for column, dtype in attributes.dtypes.items()
    }
    attributes = attributes.to_dict("index")
    records = []
    for name in names:
        matches = _matching_image_rows(frame, join_field, name)
        if len(matches) != 1:
            raise ValueError(
                f"metadata_csv requires exactly one row in {join_field!r} containing "
                f"input image name {name!r}; found {len(matches)}. "
                "The CSV value must contain the full current raster basename, "
                "including processing suffixes; the file extension is optional."
            )
        records.append(attributes[matches.index[0]])
    return schema, records


def _match_footprints(frame, image_field_name, paths=None):
    """Dissolve features whose field contains each image basename without extension, preserving first-match attributes."""
    names = (
        _resolve_paths("name", paths)
        if paths is not None
        else list(
            dict.fromkeys(
                _resolve_paths("name", frame[image_field_name].astype(str).tolist())
            )
        )
    )
    if paths is not None and len(set(names)) != len(names):
        raise ValueError("Input image basenames must be unique.")
    records = []
    for index, name in enumerate(names):
        matches = _matching_image_rows(frame, image_field_name, name)
        if matches.empty:
            raise ValueError(
                f"No footprint in '{image_field_name}' contains input image name {name!r}."
            )
        record = matches.iloc[0].copy()
        record.geometry = unary_union(list(matches.geometry))
        record[image_field_name] = name
        if paths is not None:
            record["image_path"] = paths[index]
        records.append(record)
    return gpd.GeoDataFrame(records, geometry="geometry", crs=frame.crs).reset_index(
        drop=True
    )


def _prepare_footprints(
    input_images,
    input_polygons,
    input_layer,
    image_field_name,
    image_threads=None,
    concurrent_processing_backend="process_pool",
    dask_scheduler=None,
):
    """Match supplied footprints to every raster, or polygonize valid masks only when polygons are omitted."""
    paths = None
    if input_images is not None:
        Universal._validate(input_images=input_images)
        paths = _resolve_paths(
            "search", input_images, kwargs={"default_file_pattern": "*.tif"}
        )
        if not paths:
            raise ValueError("No input images found.")
        paths = [os.path.abspath(path) for path in paths]
    if input_polygons is not None:
        return _match_footprints(
            _read_polygons(input_polygons, input_layer, image_field_name),
            image_field_name,
            paths,
        )
    if paths is None:
        raise ValueError("input_images or input_polygons is required.")
    names = _resolve_paths("name", paths)
    if len(set(names)) != len(names):
        raise ValueError("Input image basenames must be unique.")
    parallel, workers = _resolve_parallel_config(
        image_threads, concurrent_processing_backend, dask_scheduler
    )
    results = _run_image_tasks(
        _footprint_from_image,
        [(path, 1, True) for path in paths],
        input_paths=paths,
        output_paths=["valid-data footprint"] * len(paths),
        parallel=parallel,
        backend="thread",
        workers=workers,
        concurrent_processing_backend=concurrent_processing_backend,
        dask_scheduler=dask_scheduler,
    )
    crs = CRS(results[0][1])
    if any(CRS(result[1]) != crs for result in results):
        raise ValueError("Input rasters must use the same CRS.")
    return gpd.GeoDataFrame(
        {image_field_name: names, "image_path": paths},
        geometry=[result[0] for result in results],
        crs=crs,
    )


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
            if os.path.isfile(self.path):
                try:
                    fiona.listlayers(self.path)
                except fiona.errors.DriverError:
                    os.remove(self.path)
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


def _footprint_output_is_reusable(
    output_path,
    resume_mode,
    debug_logs,
    step_name,
    *,
    output_layer=None,
    image_field_name=None,
):
    """Reject incomplete outputs, and inspect polygon data in validate mode."""
    reusable = _existing_outputs_are_reusable(
        [output_path],
        resume_mode=(
            "no" if os.path.exists(output_path + ".incomplete") else resume_mode
        ),
        debug_logs=debug_logs,
        step_name=step_name,
    )
    if reusable and resume_mode == "validate":
        try:
            _read_polygons(output_path, output_layer, image_field_name)
        except (ValueError, OSError, RuntimeError, fiona.errors.FionaError) as exc:
            if debug_logs:
                print(f"Existing polygon output invalid; recomputing ({exc})")
            return False
    return reusable


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


def _filter_polygon_area(geometry, filter_area_size=None, filter_area_rank=1):
    """Filter components by minimum area, then retain the largest or smallest requested count."""
    parts = [
        part
        for part in _polygon_parts(geometry)
        if filter_area_size is None or part.area >= filter_area_size
    ]
    if filter_area_rank:
        parts = sorted(parts, key=lambda part: part.area, reverse=filter_area_rank > 0)[
            : abs(filter_area_rank)
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


def _hole_cut_width(hole, hole_cut_width):
    """Resolve an angle-independent width from the original hole geometry."""
    if hole_cut_width == "maximum_inscribed_circle":
        # polylabel uses the native MIC implementation on Shapely 2.1+, and
        # provides the same search on 2.0. Use a rotation-invariant tolerance.
        center = polylabel(hole, tolerance=math.sqrt(hole.area) / 1000)
        return 2 * center.distance(hole.boundary)
    if hole_cut_width == "hole_size":
        return _hole_diameter(hole)
    return hole_cut_width


def _postprocess_polygon(
    geometry,
    hole_edge_distance,
    hole_relative_edge_distance,
    hole_cut_width,
    hole_cut_method,
    simplify_smoothing_radius,
    simplify_tolerance,
    simplify_area_weight,
    filter_area_size=None,
    filter_area_rank=1,
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
                widths[index] = _hole_cut_width(holes[index], hole_cut_width)
            return widths[index]

        if hole_edge_distance > 0:
            for index, hole in enumerate(holes):
                distance = hole.distance(original.exterior)
                if distance > hole_edge_distance:
                    continue
                if (
                    hole_relative_edge_distance is not None
                    and distance / math.sqrt(hole.area / math.pi)
                    > hole_relative_edge_distance
                ):
                    continue
                width = width_for(index)
                if hole_cut_method == "corridor":
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
                    if hole_cut_method == "corridor":
                        cuts.append(
                            LineString(nearest_points(hole, other)).buffer(width / 2)
                        )
                    else:
                        radius = (hole.distance(other) + width) / 2
                        cuts.extend([hole.buffer(radius), other.buffer(radius)])
        cut = original.difference(unary_union(cuts)) if cuts else original
        if simplify_smoothing_radius:
            cut = (
                cut.buffer(-simplify_smoothing_radius)
                .buffer(simplify_smoothing_radius)
                .intersection(cut)
            )
        for part in _polygon_parts(cut):
            if simplify_tolerance:
                part = _simplify_inner(part, simplify_tolerance, simplify_area_weight)
            results.append(part)
    geometry = unary_union(results) if results else MultiPolygon()
    return _filter_polygon_area(geometry, filter_area_size, filter_area_rank)
