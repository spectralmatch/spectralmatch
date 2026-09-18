import math
import os
import networkx as nx
import fiona

from fiona.errors import DriverError
from shapely.geometry import (
    Polygon,
    LineString,
    mapping,
    GeometryCollection,
    Point,
    MultiPoint,
    shape
)
from fiona import open as fopen
from shapely.ops import split, voronoi_diagram, unary_union
from itertools import combinations
from typing import List


def _densify_polygon(
    poly,
    target_spacing: float,
):
    """
    Return evenly spaced points along a polygon's exterior by arc-length.

    Args:
        poly: shapely Polygon
        target_spacing: target arc-length spacing between seeds (density goal)

    Returns:
        list[(x, y)]  -- open list (first point not repeated at end)
    """
    if target_spacing <= 0:
        raise ValueError("target_spacing must be > 0")

    if hasattr(poly, "geoms"):
        return [point for part in poly.geoms if isinstance(part, Polygon) for point in _densify_polygon(part, target_spacing)]
    coords = list(poly.exterior.coords)
    if len(coords) < 2:
        return coords

    # drop the duplicate closing vertex
    coords = coords[:-1]

    pts = []
    # seed with the very first vertex once
    pts.append(tuple(coords[0]))

    for (x0, y0), (x1, y1) in zip(coords, coords[1:] + coords[:1]):
        dx, dy = (x1 - x0), (y1 - y0)
        L = math.hypot(dx, dy)
        if L == 0:
            continue

        # choose N to minimize |L - N*target_spacing|
        n_floor = int(L // target_spacing)
        if n_floor < 1:
            n_floor = 1
        n_ceil = n_floor + 1

        # pick the better of {n_floor, n_ceil}
        N = n_floor if abs(L - n_floor * target_spacing) <= abs(L - n_ceil * target_spacing) else n_ceil

        # actual spacing along this edge
        for k in range(1, N):
            t = k / N  # fraction along this edge
            x = x0 + dx * t
            y = y0 + dy * t
            pts.append((x, y))

        # do NOT append (x1, y1); next edge starts from it

    return pts


def _polygonal_intersection(a: Polygon, b: Polygon, buffer_eps: float = 1e-8):
    """
    Returns only the polygonal portion of a ∩ b. If the intersection is line-like or point-like, it buffers slightly to form a polygon.

    Args:
        a (Polygon): Input geometries.
        b (Polygon): Input geometries.
        buffer_eps (float): Small buffer distance to 'inflate' line/point intersections.

    Returns:
        Polygon or MultiPolygon
    """
    inter = a.intersection(b)

    # Case 1: already polygonal
    if inter.geom_type in ("Polygon", "MultiPolygon"):
        return inter

    # Case 2: geometry collection — extract polygonal parts
    if inter.geom_type == "GeometryCollection":
        polys = [g for g in inter.geoms if g.geom_type in ("Polygon", "MultiPolygon")]
        if polys:
            return unary_union(polys)

    # Case 3: line or point — buffer to small polygon
    if not inter.is_empty:
        # use small fraction of bbox size as buffer
        eps = max(a.bounds[2] - a.bounds[0], a.bounds[3] - a.bounds[1]) * buffer_eps
        return inter.buffer(eps).buffer(0)

    # Case 4: no overlap
    return Polygon()

def _compute_centerline(
    a: Polygon,
    b: Polygon,
    min_point_spacing: float,
    min_cut_length: float,
    debug_logs: bool = False,
    crs=None,
    debug_vectors_path=None,
) -> LineString:
    """
    Computes a Voronoi-based centerline between two overlapping polygons.

    Args:
        a (Polygon): First polygon.
        b (Polygon): Second polygon.
        min_point_spacing (float): Minimum spacing between seed points for Voronoi generation.
        min_cut_length (float): Minimum segment length to include in the centerline graph.
        debug_logs (bool, optional): If True, prints debug information; default is False.
        crs (optional): Coordinate reference system used for optional debug output.
        debug_vectors_path (optional): Path to save debug Voronoi cells; if None, skips saving.

    Returns:
        LineString: Shortest centerline path computed through the Voronoi diagram of the overlap.
    """

    voa = _polygonal_intersection(a, b)
    pts = _densify_polygon(voa, min_point_spacing)

    # Compute intersection and extract both Voronoi and anchor points
    boundary_pts = a.boundary.intersection(b.boundary)
    coords = []

    if isinstance(boundary_pts, Point):
        pt = (boundary_pts.x, boundary_pts.y)
        pts.append(pt)
        coords.append(pt)
    elif isinstance(boundary_pts, LineString):
        mid = boundary_pts.interpolate(0.5, normalized=True)
        pt = (mid.x, mid.y)
        pts.append(pt)
        coords.append(pt)
    elif hasattr(boundary_pts, "geoms"):
        for geom in boundary_pts.geoms:
            if isinstance(geom, Point):
                pt = (geom.x, geom.y)
            elif isinstance(geom, LineString):
                mid = geom.interpolate(0.5, normalized=True)
                pt = (mid.x, mid.y)
            else:
                continue
            pts.append(pt)
            coords.append(pt)

    if debug_logs:
        print(f"Densified {len(pts)} points")
        print(f"Convex hull area: {MultiPoint(pts).convex_hull.area}")

    minx, miny, maxx, maxy = voa.bounds
    w, h = maxx - minx, maxy - miny
    eps = max(w, h) * 1e-9                 # very small offset
    pts = [(x + (i & 1) * eps, y + ((i >> 1) & 1) * eps)
           for i, (x, y) in enumerate(pts)]
    if debug_logs:
        print(f"Applied jitter of {eps} to {len(pts)} points")

    if debug_vectors_path:
        _save_seed_points(pts, debug_vectors_path, crs)

    multi = voronoi_diagram(GeometryCollection([Point(p) for p in pts]), edges=False)

    if debug_vectors_path:
        _save_voronoi_cells(
            multi, debug_vectors_path, crs, layer_name=f"voronoi_{int(voa.area)}"
        )

    G = nx.Graph()
    for poly in multi.geoms:
        if isinstance(poly, Polygon):
            coords_poly = list(poly.exterior.coords)
            for i in range(len(coords_poly) - 1):
                p1, p2 = coords_poly[i], coords_poly[i + 1]
                seg = LineString([p1, p2])
                if seg.length >= min_cut_length:
                    G.add_edge(p1, p2, weight=seg.length)

    if debug_logs:
        print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    if len(coords) >= 2:
        u, v = coords[0], coords[1]
    else:
        u, v = max(
            combinations(pts, 2),
            key=lambda p: (p[0][0] - p[1][0]) ** 2 + (p[0][1] - p[1][1]) ** 2,
        )

    nodes = list(G.nodes())
    if not nodes:
        raise ValueError(
            "Empty Voronoi graph: no centerline could be computed for the overlap"
        )

    start = min(nodes, key=lambda n: (n[0] - u[0]) ** 2 + (n[1] - u[1]) ** 2)
    end = min(nodes, key=lambda n: (n[0] - v[0]) ** 2 + (n[1] - v[1]) ** 2)
    if debug_logs:
        print(f"Snapped start={start}, end={end}")

    path = nx.shortest_path(G, source=start, target=end, weight="weight")
    return LineString([u] + path + [v])


def _segment_emp(
    emp: Polygon, cuts: List[LineString], debug_logs: bool = False
) -> Polygon:
    """
    Segments an EMP polygon by sequentially applying centerline cuts, retaining the piece containing the centroid.

    Args:
        emp (Polygon): The original EMP polygon to segment.
        cuts (List[LineString]): List of cutlines to apply.
        debug_logs (bool, optional): If True, prints debug info; default is False.

    Returns:
        Polygon: The segmented portion of the EMP containing the original centroid.
    """

    # sequentially cut EMP by each centerline, choosing the segment containing the EMP centroid
    for i, ln in enumerate(cuts):
        if not emp.intersects(ln):
            # Force cut if it's close enough (e.g., < 1 unit)
            dist = emp.distance(ln)
            if dist > 1e-3:
                if debug_logs:
                    print(f"Cut {i} too far (distance={dist:.4f}), skipping")
                continue
            if debug_logs:
                print(f"Cut {i} near EMP (distance={dist:.4f}), forcing split")

        pieces = list(split(emp, ln).geoms)
        if not pieces:
            continue

        # choose the piece that contains the original centroid
        centroid = emp.centroid
        chosen = None
        for p in pieces:
            if p.contains(centroid):
                chosen = p
                break
        if chosen is None:
            # fallback to largest area if centroid-based selection fails
            chosen = max(pieces, key=lambda p: p.area)

        emp = chosen
        if debug_logs:
            print(
                f"After cut {i}: {len(pieces)} pieces, selected piece area={emp.area:.2f}"
            )

    return emp


def _save_intersection_points(
    a: Polygon,
    b: Polygon,
    path: str,
    crs,
    pair_id: str,
) -> None:
    """
    Saves intersection points between the boundaries of two polygons to a GeoPackage layer.

    Args:
        a (Polygon): First polygon.
        b (Polygon): Second polygon.
        path (str): Path to the output GeoPackage file.
        crs: Coordinate reference system for the output.
        pair_id (str): Identifier for the polygon pair, saved as an attribute.

    Returns:
        None
    """

    inter = a.boundary.intersection(b.boundary)
    if isinstance(inter, Point):
        points = [inter]
    elif hasattr(inter, "geoms"):
        points = [g for g in inter.geoms if isinstance(g, Point)]
    else:
        points = []

    if not points:
        return

    schema = {"geometry": "Point", "properties": {"pair_id": "str"}}
    layer_name = "intersections"

    mode = "a"
    if not os.path.exists(path) or layer_name not in fiona.listlayers(path):
        mode = "w"

    with fiona.open(
        path, mode=mode, driver="GPKG", crs_wkt=crs, schema=schema, layer=layer_name
    ) as dst:
        for pt in points:
            dst.write(
                {
                    "geometry": mapping(pt),
                    "properties": {"pair_id": pair_id},
                }
            )


def _save_voronoi_cells(
    voronoi_cells: GeometryCollection, path: str, crs, layer_name: str = "voronoi_cells"
) -> None:
    """
    Saves Voronoi polygon geometries to a specified GeoPackage layer.

    Args:
        voronoi_cells (GeometryCollection): Collection of Voronoi polygon geometries.
        path (str): Path to the output GeoPackage file.
        crs: Coordinate reference system for the output layer.
        layer_name (str, optional): Name of the layer to write; default is "voronoi_cells".

    Returns:
        None
    """

    schema = {"geometry": "Polygon", "properties": {}}

    # Determine if file and layer exist
    layer_exists = False
    if os.path.exists(path):
        try:
            with fiona.open(path, mode="r", driver="GPKG") as src:
                layer_exists = (
                    layer_name in src.listlayers()
                    if hasattr(src, "listlayers")
                    else False
                )
        except DriverError:
            pass

    mode = "a" if layer_exists else "w"

    with fiona.open(
        path, mode=mode, driver="GPKG", crs_wkt=crs, schema=schema, layer=layer_name
    ) as dst:
        for geom in voronoi_cells.geoms:
            if isinstance(geom, Polygon):
                dst.write(
                    {
                        "geometry": mapping(geom),
                        "properties": {},
                    }
                )


def _save_emp_outlines(
    emps: List[Polygon],
    image_paths: List[str],
    path: str,
    crs,
    image_field_name: str = "image",
    layer_name: str = "emp_outline",
) -> None:
    """
    Save initial EMP polygons (one per image) to a GPKG layer.
    """
    schema = {"geometry": "Polygon", "properties": {image_field_name: "str"}}

    # decide append vs create layer
    mode = "a"
    if not os.path.exists(path) or layer_name not in fiona.listlayers(path):
        mode = "w"

    with fiona.open(
        path, mode=mode, driver="GPKG", crs_wkt=crs, schema=schema, layer=layer_name
    ) as dst:
        for emp, img in zip(emps, image_paths):
            dst.write(
                {
                    "geometry": mapping(emp),
                    "properties": {
                        image_field_name: os.path.splitext(os.path.basename(img))[0]
                    },
                }
            )


def _save_seed_points(
    pts: list[tuple[float, float]],
    path: str,
    crs,
    layer_name: str = "voronoi_seeds",
) -> None:
    """
    Saves Voronoi seed points to a GeoPackage layer.

    Args:
        pts (list[tuple]): List of (x, y) seed coordinates.
        path (str): Path to the output GeoPackage.
        crs: Coordinate reference system.
        layer_name (str, optional): Layer name. Defaults to 'voronoi_seeds'.
    """
    schema = {"geometry": "Point", "properties": {}}

    mode = "a"
    if not os.path.exists(path) or layer_name not in fiona.listlayers(path):
        mode = "w"

    with fiona.open(path, mode=mode, driver="GPKG", crs_wkt=crs, schema=schema, layer=layer_name) as dst:
        for x, y in pts:
            dst.write({"geometry": mapping(Point(x, y)), "properties": {}})


def _mask_by_aoi(polygons: list[Polygon], aoi_path: str) -> list[Polygon]:
    """Clip polygons by an AOI layer from file.

    Args:
        polygons (list[Polygon]): Input seamline polygons.
        aoi_path (str): Path to vector file containing AOI polygon(s).

    Returns:
        list[Polygon]: List of clipped polygons (empties dropped).
    """
    with fopen(aoi_path, "r") as src:
        aoi = unary_union([shape(feat["geometry"]) for feat in src])

    return [poly.intersection(aoi) for poly in polygons]
