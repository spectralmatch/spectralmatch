"""Triangle meshes, sampled raster costs and hierarchical Markov optimization."""

import math
import warnings
from collections import OrderedDict
from itertools import combinations
from time import perf_counter

import numpy as np
from osgeo import gdal
from pyproj import CRS
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from shapely import contains_xy, intersects_xy
from shapely.geometry import LineString, Polygon, box
from shapely.ops import polygonize, unary_union
from shapely.strtree import STRtree

try:
    import maxflow
except ImportError:
    maxflow = None
try:
    from shapely import constrained_delaunay_triangles
except ImportError:
    constrained_delaunay_triangles = None

from .footprints import _polygon_parts
from ..utils_logging import _print_line

_LAPLACIAN_VARIABLES = {"laplacian_difference", "laplacian_magnitude"}


def _mesh(coverages, spacing):
    """Constrain Delaunay triangles to grid cells and all footprint boundaries, retaining holes and islands."""
    extent = unary_union(coverages)
    xmin, ymin, xmax, ymax = extent.bounds
    nx, ny = math.ceil((xmax - xmin) / spacing), math.ceil((ymax - ymin) / spacing)
    if nx * ny > 1_000_000:
        raise ValueError("Mesh exceeds one million cells; increase mesh_spacing.")
    lines = [geometry.boundary for geometry in coverages]
    lines.extend(
        LineString([(x, ymin), (x, ymax)]) for x in np.linspace(xmin, xmax, nx + 1)
    )
    lines.extend(
        LineString([(xmin, y), (xmax, y)]) for y in np.linspace(ymin, ymax, ny + 1)
    )
    tree = STRtree(coverages)
    faces, visibility = [], []
    for cell in polygonize(unary_union(lines)):
        owners = tree.query(cell.representative_point(), predicate="covered_by")
        if len(owners):
            triangles = list(constrained_delaunay_triangles(cell).geoms)
            faces.extend(triangles)
            visible = np.zeros(len(coverages), dtype=bool)
            visible[owners] = True
            visibility.extend([visible] * len(triangles))
    return faces, np.asarray(visibility)


def _adjacency(faces):
    """Find full or partial shared edges, excluding point-only contacts."""
    tree = STRtree(faces)
    pairs = tree.query(faces, predicate="intersects").T
    edges, segments = [], []
    for a, b in pairs[pairs[:, 0] < pairs[:, 1]]:
        shared = faces[a].intersection(faces[b])
        if shared.geom_type == "LineString" and shared.length > 0:
            edges.append((a, b))
            segments.append(shared)
    return np.asarray(edges, dtype=int).reshape(-1, 2), segments


def _clip_candidates(candidates, bounds):
    region = box(*bounds)
    result = []
    for candidate in candidates:
        clipped = {
            name: unary_union(_polygon_parts(geometry.intersection(region)))
            for name, geometry in candidate.items()
            if geometry.intersects(region)
        }
        clipped = {
            name: geometry
            for name, geometry in clipped.items()
            if not geometry.is_empty
        }
        if clipped:
            result.append(clipped)
    return result


def _quadtree(candidates, max_images, overlap, max_depth):
    """Plan overlapping nodes; stop when subdivision cannot reduce label count."""
    bounds = unary_union([g for c in candidates for g in c.values()]).bounds
    levels = [[("root", bounds, candidates)]]
    children = {}
    for _ in range(max_depth):
        following = []
        for key, (xmin, ymin, xmax, ymax), source in levels[-1]:
            if len(source) <= max_images:
                continue
            xm, ym = (xmin + xmax) / 2, (ymin + ymax) / 2
            dx, dy = (xmax - xmin) * overlap / 2, (ymax - ymin) * overlap / 2
            boxes = [
                (xmin, ymin, xm + dx, ym + dy),
                (xm - dx, ymin, xmax, ym + dy),
                (xmin, ym - dy, xm + dx, ymax),
                (xm - dx, ym - dy, xmax, ymax),
            ]
            nodes = [
                (f"{key}.{i}", bounds, _clip_candidates(source, bounds))
                for i, bounds in enumerate(boxes)
            ]
            nodes = [node for node in nodes if node[2]]
            if len(nodes) < 2 or all(len(node[2]) == len(source) for node in nodes):
                continue
            children[key] = [node[0] for node in nodes]
            following.extend(nodes)
        if not following:
            break
        levels.append(following)
    return levels, children


def _reduce(values, reducer, axis=0):
    """Select a single band or scale directly, or aggregate multiple values with the requested reducer."""
    if reducer is None:
        return np.take(values, 0, axis=axis)
    return {"average": np.mean, "largest": np.max, "smallest": np.min}[reducer](
        values, axis=axis
    )


def _variables(specification):
    """Validate weighted terms without modifying the specification or opening disabled rasters."""
    if not isinstance(specification, (list, tuple)):
        raise ValueError(
            "edge_variables_weights_gsds_bands must be a list of four-item terms."
        )
    terms = []
    for term in specification:
        if not isinstance(term, (list, tuple)) or len(term) != 4:
            raise ValueError(
                "Each variable requires [name_or_path, weight, GSD selection, band selection]."
            )
        variable, weight, scales, channels = term
        if not isinstance(variable, str) or not variable.strip():
            raise ValueError(
                "Variable must be laplacian_difference, laplacian_magnitude, or a raster path."
            )
        if (
            isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not math.isfinite(weight)
            or weight < 0
        ):
            raise ValueError("Variable weights must be finite and nonnegative.")
        selections = []
        for selection, kind in [(scales, "GSDs"), (channels, "bands")]:
            if isinstance(selection, dict):
                if len(selection) != 1 or next(iter(selection)) not in {
                    "average",
                    "largest",
                    "smallest",
                }:
                    raise ValueError(
                        f"{kind} require one reducer: average, largest or smallest."
                    )
                reducer, values = next(iter(selection.items()))
            else:
                reducer, values = None, selection
            if not isinstance(values, (tuple, list)) or not values:
                raise ValueError(f"{kind} must be a nonempty list.")
            if reducer is not None and len(values) < 2:
                raise ValueError(
                    f"{kind} aggregation requires at least two values; use a plain one-item list for a single value."
                )
            if reducer is None and len(values) != 1:
                raise ValueError(
                    f"Multiple {kind} require an average, largest or smallest reducer."
                )
            for value in values:
                if not (
                    (type(value) is int and value > 0)
                    or (kind == "GSDs" and isinstance(value, str) and value == "native")
                ):
                    raise ValueError(
                        f"{kind} must contain positive integers{' or native' if kind == 'GSDs' else ''}."
                    )
            if len(set(values)) != len(values):
                raise ValueError(f"{kind} must not contain duplicates.")
            selections.extend([reducer, tuple(values)])
        if weight:
            terms.append((variable, float(weight), *selections))
    return terms


def _native_resolution(dataset, crs):
    """Return the smallest pixel-axis length in the working CRS, using GDAL's suggested grid when reprojection is needed."""
    if CRS(dataset.GetProjection()) != crs:
        dataset = gdal.AutoCreateWarpedVRT(dataset, None, crs.to_wkt())
    transform = dataset.GetGeoTransform()
    resolution = min(
        math.hypot(transform[1], transform[4]), math.hypot(transform[2], transform[5])
    )
    if not math.isfinite(resolution) or resolution <= 0:
        raise ValueError("Raster requires a finite positive native resolution.")
    return resolution


def _sources(frame, image_field_name, specification):
    """Validate source footprints, projected georeferencing, selected bands and external cost rasters."""
    if not frame.crs.is_projected:
        raise ValueError(
            "Triangle seamlines require a projected CRS; reproject inputs first."
        )
    paths = dict(zip(frame[image_field_name], frame.image_path))
    band_counts, resolutions = [], []
    for name, path in paths.items():
        try:
            with gdal.Open(path) as dataset:
                if (
                    not dataset.GetProjection()
                    or CRS(dataset.GetProjection()) != frame.crs
                ):
                    raise ValueError(
                        f"Raster and footprints must have the same CRS: {path}"
                    )
                transform = dataset.GetGeoTransform(can_return_null=True)
                if transform is None or gdal.InvGeoTransform(transform) is None:
                    raise ValueError(
                        f"Raster requires an invertible geotransform: {path}"
                    )
                extent = Polygon(
                    [
                        gdal.ApplyGeoTransform(transform, col, row)
                        for col, row in [
                            (0, 0),
                            (dataset.RasterXSize, 0),
                            (dataset.RasterXSize, dataset.RasterYSize),
                            (0, dataset.RasterYSize),
                        ]
                    ]
                )
                geometry = frame.loc[frame[image_field_name] == name].geometry.iloc[0]
                if geometry.difference(extent).area > 1e-9 * max(1, geometry.area):
                    raise ValueError(f"Footprint extends outside its raster: {path}")
                band_counts.append(dataset.RasterCount)
                resolutions.append(_native_resolution(dataset, frame.crs))
        except RuntimeError as exc:
            raise ValueError(f"Cannot read raster {path}: {exc}") from exc
    terms, external = [], {}
    for variable, weight, scale_reducer, scales, band_reducer, bands in _variables(
        specification
    ):
        if variable in _LAPLACIAN_VARIABLES:
            count = min(band_counts)
            resolution = min(resolutions)
        else:
            if variable not in external:
                try:
                    with gdal.Open(variable) as dataset:
                        if (
                            not dataset.GetProjection()
                            or dataset.GetGeoTransform(can_return_null=True) is None
                            or gdal.InvGeoTransform(dataset.GetGeoTransform()) is None
                        ):
                            raise ValueError(
                                f"External cost raster requires a CRS and invertible geotransform: {variable}"
                            )
                        external[variable] = (
                            dataset.RasterCount,
                            _native_resolution(dataset, frame.crs),
                        )
                except RuntimeError as exc:
                    raise ValueError(
                        f"Cannot read cost raster {variable}: {exc}"
                    ) from exc
            count, resolution = external[variable]
        if max(bands) > count:
            raise ValueError(f"Selected bands are unavailable for {variable}.")
        terms.append(
            (
                variable,
                weight,
                scale_reducer,
                tuple(resolution if gsd == "native" else gsd for gsd in scales),
                band_reducer,
                bands,
            )
        )
    return (
        [{row[image_field_name]: row.geometry} for _, row in frame.iterrows()],
        paths,
        terms,
    )


class _RasterSampler:
    """Cache GDAL average-resampled VRTs and bounded pixel windows for one optimization node."""

    def __init__(self, crs):
        self.crs = crs
        self.datasets = {}
        self.tiles = OrderedDict()

    def close(self):
        """Release cached windows and GDAL datasets before a worker returns."""
        self.tiles.clear()
        self.datasets.clear()

    def sample(self, path, coordinates, bands, gsd, native_fallback=False):
        """Read values and validity masks in 256-pixel windows on a map-aligned GSD grid."""
        key = (path, gsd)
        if key not in self.datasets:
            self.datasets[key] = (
                gdal.Open(path)
                if gsd is None
                else gdal.Warp(
                    "",
                    path,
                    format="VRT",
                    dstSRS=self.crs,
                    xRes=gsd,
                    yRes=gsd,
                    targetAlignedPixels=True,
                    resampleAlg="average",
                    outputType=gdal.GDT_Float64,
                    dstNodata=float("nan"),
                )
            )
        dataset = self.datasets[key]
        inverse = gdal.InvGeoTransform(dataset.GetGeoTransform())
        x, y = coordinates.T
        col = inverse[0] + inverse[1] * x + inverse[2] * y
        row = inverse[3] + inverse[4] * x + inverse[5] * y
        for values, size in [(col, dataset.RasterXSize), (row, dataset.RasterYSize)]:
            values[np.isclose(values, size, rtol=0, atol=1e-7)] = size - 1e-7
            values[np.isclose(values, 0, rtol=0, atol=1e-7)] = 0
        cols, rows = np.floor(col).astype(int), np.floor(row).astype(int)
        inside = np.flatnonzero(
            (cols >= 0)
            & (cols < dataset.RasterXSize)
            & (rows >= 0)
            & (rows < dataset.RasterYSize)
        )
        result = np.full((len(coordinates), len(bands)), np.nan)
        unique, groups = np.unique(
            np.column_stack((cols[inside] // 256, rows[inside] // 256)),
            axis=0,
            return_inverse=True,
        )
        order = np.argsort(groups, kind="stable")
        starts = np.r_[0, np.cumsum(np.bincount(groups, minlength=len(unique)))]
        for tile, (tx, ty) in enumerate(unique):
            selected = inside[order[starts[tile] : starts[tile + 1]]]
            xoff, yoff = int(tx * 256), int(ty * 256)
            width, height = min(256, dataset.RasterXSize - xoff), min(
                256, dataset.RasterYSize - yoff
            )
            for channel, number in enumerate(bands):
                tile_key = (path, gsd, number, int(tx), int(ty))
                if tile_key not in self.tiles:
                    band = dataset.GetRasterBand(number)
                    data = band.ReadAsArray(xoff, yoff, width, height)
                    valid = (
                        band.GetMaskBand().ReadAsArray(xoff, yoff, width, height) != 0
                    )
                    self.tiles[tile_key] = np.where(valid, data, np.nan)
                    if len(self.tiles) > 64:
                        self.tiles.popitem(last=False)
                self.tiles.move_to_end(tile_key)
                result[selected, channel] = self.tiles[tile_key][
                    rows[selected] - yoff, cols[selected] - xoff
                ]
        # Warped cells centered outside rotated rasters can omit valid edge
        # slivers; use the original pixel only where the VRT has no value.
        missing = ~np.isfinite(result).all(axis=1)
        if native_fallback and missing.any():
            native = self.sample(path, coordinates[missing], bands, None)
            result[missing] = np.where(
                np.isfinite(result[missing]), result[missing], native
            )
        return result


def _laplacian(candidate, paths, coordinates, bands, gsd, sampler):
    """Evaluate a five-point map-space Laplacian of an image or child mosaic, extending missing neighbors by the center."""
    offsets = np.array([(0, 0), (gsd, 0), (-gsd, 0), (0, gsd), (0, -gsd)])
    stencil = (coordinates[:, None, :] + offsets).reshape(-1, 2)
    pixels = np.full((len(stencil), len(bands)), np.nan)
    covered = np.zeros(len(stencil), dtype=bool)
    for name, geometry in candidate.items():
        selected = intersects_xy(geometry, *stencil.T) & ~covered
        if not selected.any():
            continue
        points = stencil[selected]
        values = sampler.sample(paths[name], points, bands, gsd, native_fallback=True)
        boundary = ~contains_xy(geometry, *points.T)
        for dx, dy in [
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1),
        ]:
            missing = boundary & ~np.isfinite(values).all(axis=1)
            if not missing.any():
                break
            inward = points[missing] + np.array([dx, dy]) * gsd * 1e-6
            inside = contains_xy(geometry, *inward.T)
            indices = np.flatnonzero(missing)[inside]
            values[indices] = np.where(
                np.isfinite(values[indices]),
                values[indices],
                sampler.sample(
                    paths[name], inward[inside], bands, gsd, native_fallback=True
                ),
            )
        pixels[selected] = values
        covered[selected] = True
    pixels = pixels.reshape(len(coordinates), 5, len(bands))
    centers = pixels[:, 0]
    if np.any(covered[::5] & ~np.isfinite(centers).all(axis=1)):
        raise ValueError(
            "A footprint includes nodata or nonfinite raster samples; regenerate valid-data footprints for the selected bands."
        )
    centers = np.nan_to_num(centers, nan=0)
    neighbors = np.where(np.isfinite(pixels[:, 1:]), pixels[:, 1:], centers[:, None])
    return (neighbors.sum(axis=1) - 4 * centers) / gsd**2


def _edge_samples(segments, spacing):
    """Place evenly spaced midpoint samples on every edge and retain their edge indices."""
    sizes = np.array([max(1, math.ceil(line.length / spacing)) for line in segments])
    coordinates = np.concatenate(
        [
            np.asarray(line.coords)[0]
            + ((np.arange(size) + 0.5) / size)[:, None]
            * (np.asarray(line.coords)[-1] - np.asarray(line.coords)[0])
            for line, size in zip(segments, sizes)
        ]
    )
    return (
        np.r_[0, sizes.cumsum()[:-1]],
        np.repeat(np.arange(len(segments)), sizes),
        coordinates,
    )


# Earlier raster stages can initialize a GDAL thread pool before node workers fork.
# Keep reads and warps serial inside each node; independent nodes still run in parallel.
@gdal.config_option("GDAL_NUM_THREADS", "1")
def _shared_samples(candidates, paths, plans, terms, crs):
    """Batch all requested bands and coordinates per raster/GSD, sharing built-in Laplacians and repeated external reads."""
    groups = {}
    for variable, _, _, scales, _, bands in terms:
        source = None if variable in _LAPLACIAN_VARIABLES else variable
        for gsd in scales:
            group = groups.setdefault(
                (source, gsd), {"bands": set(), "spacings": set()}
            )
            group["bands"].update(bands)
            group["spacings"].add(min(scales))
    sampler = _RasterSampler(crs)
    try:
        for (source, gsd), group in groups.items():
            bands = tuple(sorted(group["bands"]))
            spacings = sorted(group.pop("spacings"))
            # Different terms keep their own sample locations and reducers.
            # Deduplicate their union, then restore each term's order by index.
            coordinates, inverse = np.unique(
                np.concatenate([plans[spacing][2] for spacing in spacings]),
                axis=0,
                return_inverse=True,
            )
            offsets = np.r_[
                0, np.cumsum([len(plans[spacing][2]) for spacing in spacings])
            ]
            group["indices"] = {
                spacing: inverse[offsets[i] : offsets[i + 1]]
                for i, spacing in enumerate(spacings)
            }
            group["bands"] = {band: i for i, band in enumerate(bands)}
            if source is None:
                group["values"] = np.asarray(
                    [
                        _laplacian(candidate, paths, coordinates, bands, gsd, sampler)
                        for candidate in candidates
                    ]
                )
            else:
                values = sampler.sample(source, coordinates, bands, gsd)
                if not np.isfinite(values).all() or np.any(values < 0):
                    raise ValueError(
                        f"External cost raster must provide finite nonnegative values at every seam sample: {source}"
                    )
                group["values"] = values
    finally:
        sampler.close()
    return groups


def _edge_costs(candidates, paths, segments, terms, crs):
    """Build independent weighted costs from shared samples, reducing bands then scales then maximizing along edges."""
    prepared, upper_bound = [], 0.0
    plans = {
        spacing: _edge_samples(segments, spacing)
        for spacing in {min(term[3]) for term in terms}
    }
    groups = _shared_samples(candidates, paths, plans, terms, crs)
    for variable, weight, scale_reducer, scales, band_reducer, bands in terms:
        source = None if variable in _LAPLACIAN_VARIABLES else variable
        spacing = min(scales)
        starts, sample_edges, _ = plans[spacing]
        values = []
        for gsd in scales:
            group = groups[source, gsd]
            indices = group["indices"][spacing]
            channels = [group["bands"][band] for band in bands]
            values.append(group["values"][..., indices[:, None], channels])
        values = np.asarray(values)
        if variable == "laplacian_magnitude":
            values = _reduce(
                _reduce(np.abs(values), band_reducer, axis=3), scale_reducer, axis=0
            )
            values = np.maximum.reduceat(values, starts, axis=1)
            bound = values.max(axis=0).sum()
        elif variable == "laplacian_difference":
            bound = np.maximum.reduceat(
                np.ptp(values, axis=1).max(axis=(0, 2)), starts
            ).sum()
        else:
            values = np.maximum.reduceat(
                _reduce(_reduce(values, band_reducer, axis=2), scale_reducer, axis=0),
                starts,
            )
            bound = values.sum()
        prepared.append(
            (
                variable,
                weight,
                scale_reducer,
                band_reducer,
                values,
                starts,
                sample_edges,
            )
        )
        upper_bound += weight * bound
    if not math.isfinite(upper_bound):
        raise ValueError("Nonfinite seam costs; check raster values, weights and GSDs.")

    def cost(left, right):
        total = np.zeros(len(segments))
        for (
            variable,
            weight,
            scale_reducer,
            band_reducer,
            values,
            starts,
            sample_edges,
        ) in prepared:
            if variable == "laplacian_difference":
                samples = np.arange(len(sample_edges))
                difference = np.abs(
                    values[:, left[sample_edges], samples]
                    - values[:, right[sample_edges], samples]
                )
                reduced = _reduce(
                    _reduce(difference, band_reducer, axis=2), scale_reducer, axis=0
                )
                term = np.maximum.reduceat(reduced, starts)
            elif variable == "laplacian_magnitude":
                term = (
                    values[left, np.arange(len(segments))]
                    + values[right, np.arange(len(segments))]
                ) / 2
            else:
                term = values
            total += weight * term * (left != right)
        return total

    nonmetric = any(
        v == "laplacian_difference" and "smallest" in (sr, br)
        for v, _, sr, _, br, _ in terms
    )
    return cost, upper_bound, nonmetric


def _binary_cut(unary, edges, weights):
    """Solve a submodular binary energy and return the nodes assigned to its second state."""
    unary = unary - unary.min(axis=1, keepdims=True)
    graph = maxflow.Graph[float](len(unary), len(edges))
    graph.add_nodes(len(unary))
    for node, (keep, switch) in enumerate(unary):
        graph.add_tedge(node, float(switch), float(keep))
    for (a, b), weight in zip(edges, weights):
        graph.add_edge(int(a), int(b), float(weight), float(weight))
    graph.maxflow()
    return graph.get_grid_segments(np.arange(len(unary)))


def _alpha_expansion(
    visible, edges, cost, upper_bound, max_iterations, unary=None, swap=False
):
    """Minimize area unaries and pair costs by alpha expansion, or alpha-beta swaps for nonmetric reducers; convergence is move-local."""
    if not visible.any(axis=1).all():
        raise ValueError("No image covers a mesh face or foreground component.")
    unary = np.zeros(visible.shape) if unary is None else unary
    labels = np.where(visible, unary, np.inf).argmin(axis=1)
    nodes = np.arange(len(labels))
    left, right = edges.T
    energy = float(unary[nodes, labels].sum() + cost(labels[left], labels[right]).sum())
    hard = 4 * (upper_bound + unary.max(axis=1).sum()) + 1
    if not np.isfinite(hard):
        raise ValueError(
            "Graph capacities overflow; reduce weights or rescale raster costs."
        )
    for _ in range(max_iterations):
        changed = False
        moves = (
            combinations(range(visible.shape[1]), 2)
            if swap
            else ((a, None) for a in range(visible.shape[1]))
        )
        for alpha, beta in moves:
            if swap:
                active = (labels == alpha) | (labels == beta)
                first = np.where(active, alpha, labels)
                second = np.where(active, beta, labels)
            else:
                first, second = labels, np.full(len(labels), alpha)
            e00 = cost(first[left], first[right])
            e01 = cost(first[left], second[right])
            e10 = cost(second[left], first[right])
            e11 = cost(second[left], second[right])
            weights = (e01 + e10 - e00 - e11) / 2
            if np.any(weights < -1e-8 * max(1, upper_bound)):
                raise ValueError(
                    "Alpha expansion requires metric costs; use swap moves for minimum reducers."
                )
            weights = np.maximum(weights, 0)
            terminals = np.column_stack((unary[nodes, first], unary[nodes, second]))
            np.add.at(terminals[:, 1], left, e10 - e00 - weights)
            np.add.at(terminals[:, 1], right, e01 - e00 - weights)
            terminals[~visible[nodes, first], 0] += hard
            terminals[~visible[nodes, second], 1] += hard
            proposal = np.where(_binary_cut(terminals, edges, weights), second, first)
            if not visible[nodes, proposal].all():
                raise RuntimeError("Graph cut violated an image visibility constraint.")
            candidate = float(
                unary[nodes, proposal].sum()
                + cost(proposal[left], proposal[right]).sum()
            )
            if candidate < energy - 1e-12 * max(1, abs(energy)):
                labels, energy, changed = proposal, candidate, True
        if not changed:
            return labels
    warnings.warn(
        "Graph optimization reached solver_max_iterations; increase it to reach convergence.",
        RuntimeWarning,
        stacklevel=2,
    )
    return labels


def _solve_node(
    candidates,
    paths,
    mesh_spacing,
    terms,
    foreground,
    quality,
    quality_weight,
    crs,
    max_iterations,
    debug_logs,
):
    """Optimize triangle labels with raster edge costs, protected regions and area-weighted image quality."""
    if len(candidates) == 1:
        return candidates[0]
    started = perf_counter()

    def progress(message):
        """Flush stage progress with elapsed time when debugging is enabled."""
        if debug_logs:
            _print_line(
                f"Triangle MRF: {message} | elapsed={perf_counter() - started:.1f}s"
            )

    progress(f"building mesh for {len(candidates)} candidate images/mosaics")
    coverages = [unary_union(list(candidate.values())) for candidate in candidates]
    faces, visible = _mesh(coverages, mesh_spacing)
    progress(f"{len(faces)} triangles; finding shared edges")
    edges, segments = _adjacency(faces)
    progress(f"{len(edges)} shared edges; applying constraints and image quality")
    blocked = np.array(
        [
            not foreground.is_empty and segment.intersection(foreground).length > 0
            for segment in segments
        ],
        dtype=bool,
    )
    links = edges[blocked]
    graph = coo_matrix(
        (np.ones(len(links)), (links[:, 0], links[:, 1])),
        shape=(len(faces), len(faces)),
    ).tocsr()
    count, components = connected_components(graph, directed=False)
    allowed = np.ones((count, len(candidates)), dtype=bool)
    np.logical_and.at(allowed, components, visible)
    if not allowed.any(axis=1).all():
        raise ValueError(
            "Foreground constraints are infeasible: no single image or child mosaic covers a connected protected region."
        )
    unary = np.zeros(allowed.shape)
    if quality_weight and any(quality.values()):
        for label, candidate in enumerate(candidates):
            for face_index in np.flatnonzero(visible[:, label]):
                penalty = sum(
                    faces[face_index].intersection(geometry).area * quality[name]
                    for name, geometry in candidate.items()
                )
                unary[components[face_index], label] += quality_weight * penalty
    keep = components[edges[:, 0]] != components[edges[:, 1]]
    edges = components[edges[keep]]
    segments = [segment for segment, retain in zip(segments, keep) if retain]
    if len(edges) and terms:
        progress(
            f"sampling raster costs for {len(edges)} edges and {len(terms)} variables"
        )
        cost, bound, swap = _edge_costs(candidates, paths, segments, terms, crs)
        progress("optimizing labels with graph cuts")
        labels = _alpha_expansion(
            allowed, edges, cost, bound, max_iterations, unary=unary, swap=swap
        )[components]
    else:
        labels = np.where(allowed, unary, np.inf).argmin(axis=1)[components]
    progress("merging labeled triangles into image polygons")
    pieces = {}
    for label, candidate in enumerate(candidates):
        selected = unary_union(
            [face for face, owner in zip(faces, labels) if owner == label]
        )
        for name, geometry in candidate.items():
            part = selected.intersection(geometry)
            if part.area > 0:
                pieces.setdefault(name, []).extend(_polygon_parts(part))
    result = {name: unary_union(parts) for name, parts in pieces.items()}
    progress(f"completed {len(result)} image polygons")
    return result
