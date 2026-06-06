import os
from itertools import combinations

import fiona
from osgeo import gdal
from shapely.geometry import LineString, Polygon, mapping

from ..handlers import _resolve_paths
from ..types_and_validation import Universal, Seamline as SeamlineValidation
from .voronoi_center_seamline import (
    _compute_centerline,
    _emp_polygon_from_image,
    _load_emp_polygons_from_vector,
    _mask_by_aoi,
    _save_emp_outlines,
    _save_intersection_points,
    _segment_emp,
)
from .weighted_seamline import weighted_seamline

gdal.UseExceptions()


class Seamline:
    @staticmethod
    def weighted(
        input_polygons: str,
        output_mask: str,
        *,
        rank_function: str,
        image_field_name: str = "image",
        input_layer: str | None = None,
        output_layer: str = "seamlines",
        rank_descending: bool = True,
        debug_logs: Universal.DebugLogs = False,
    ) -> str:
        """Generate seamline polygons by ranking image footprints with a weighted expression.

Args:
    input_polygons (str): Input polygon layer path. Each feature should represent an image footprint or a piece of one.
    output_mask (str): Output GeoPackage path for the ranked seamline polygons.
    rank_function (str): Ranking expression using field placeholders like ``{cloud_cover}`` or formulas like ``1 / ({sun_elevation} + 1)``.
    image_field_name (str, optional): Field containing the image identifier. Features sharing the same value are merged before ranking. Defaults to ``"image"``.
    input_layer (str | None, optional): Optional input layer name when reading multi-layer vector sources. Defaults to None.
    output_layer (str, optional): Output GeoPackage layer name. Defaults to ``"seamlines"``.
    rank_descending (bool, optional): If True, larger scores rank higher and remain on top. Defaults to True.
    debug_logs (bool, optional): If True, prints ranking details. Defaults to False.

Returns:
    str: Written output GeoPackage path.
"""
        print("Start weighted seamline")
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
        return weighted_seamline(
            input_polygons=input_polygons,
            output_mask=output_mask,
            rank_function=rank_function,
            image_field_name=image_field_name,
            input_layer=input_layer,
            output_layer=output_layer,
            rank_descending=rank_descending,
            debug_logs=debug_logs,
        )

    @staticmethod
    def voronoi(
        input_images: Universal.CreateInFolderOrListFiles,
        output_mask: str,
        *,
        aoi_path: str | None = None,
        vector_mask: tuple[str, str] | None = None,
        image_field_name: str = "image",
        min_point_spacing: float = 10,
        min_cut_length: float = 0,
        debug_logs: Universal.DebugLogs = False,
        debug_vectors_path: str | None = None,
    ) -> None:
        """Generates a Voronoi-based seamline mask from edge-matching polygons (EMPs) and writes the result to a vector file.

Args:
    input_images (str | List[str], required): Defines input files from a glob path, folder, or list of paths. Specify like: "/input/files/*.tif", "/input/folder" (assumes *.tif), ["/input/one.tif", "/input/two.tif"].
    output_mask (str): Output path for the final seamline polygon vector file.
    aoi_path (str, optional): Path to an AOI vector file to clip overlapping image polygons; default is None.
    vector_mask (Tuple[str, str] | None, optional): Optional polygon source to use instead of extracting EMPs from rasters. The tuple is (vector_path, field_name). For each input image, polygons are selected when the field value is included anywhere in the image name. Matching polygons for the same image are unioned together.
    min_point_spacing (float, optional): Minimum spacing between Voronoi seed points; default is 10.
    min_cut_length (float, optional): Minimum cutline segment length to retain; default is 0.
    debug_logs (Universal.DebugLogs, optional): Enables debug print statements if True; default is False.
    image_field_name (str, optional): Name of the attribute field for image ID in output; default is 'image'.
    debug_vectors_path (str | None, optional): Optional path to save debug layers (cutlines, intersections).

Outputs:
    Saves a polygon seamline layer to `output_mask`, and optionally saves intermediate cutlines to `debug_vectors_path`."""
        print("Start voronoi center seamline")
        output_dir = os.path.dirname(output_mask)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        if debug_vectors_path:
            debug_dir = os.path.dirname(debug_vectors_path)
            if debug_dir:
                os.makedirs(debug_dir, exist_ok=True)

        Universal._validate(
            input_images=input_images,
        )
        SeamlineValidation._validate_voronoi_center_seamline(
            output_mask=output_mask,
            aoi_path=aoi_path,
            vector_mask=vector_mask,
            image_field_name=image_field_name,
            min_point_spacing=min_point_spacing,
            min_cut_length=min_cut_length,
            debug_vectors_path=debug_vectors_path,
        )
        input_image_paths = _resolve_paths(
            "search", input_images, kwargs={"default_file_pattern": "*.tif"}
        )
        input_image_names = _resolve_paths("name", input_image_paths)

        if vector_mask is None:
            emps = []
            crs = None
            for path in input_image_paths:
                emp = _emp_polygon_from_image(path)
                emps.append(emp)
                if crs is None:
                    ds = gdal.Open(path, gdal.GA_ReadOnly)
                    crs = ds.GetProjectionRef()
                    ds = None
        else:
            emps, crs = _load_emp_polygons_from_vector(
                input_image_paths=input_image_paths,
                input_image_names=input_image_names,
                vector_mask=vector_mask,
                debug_logs=debug_logs,
            )

        for i, emp in enumerate(emps):
            if debug_logs:
                print(f"EMP{i}: area={emp.area:.2f}, bounds={emp.bounds}")

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
        for i, (a, b) in enumerate(combinations(emps, 2)):
            ov = a.intersection(b)
            if debug_logs:
                print(f"Overlap {i} area: {ov.area:.2f}")
            if not ov.is_empty:
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
                print(
                    f"EMP{idx} has {len(relevant)} intersecting cuts and {seg.area:.2f} segmented area"
                )
            segmented.append(seg)

        if aoi_path is not None:
            segmented = _mask_by_aoi(segmented, aoi_path)

        schema = {"geometry": "Polygon", "properties": {image_field_name: "str"}}
        with fiona.open(
            output_mask,
            "w",
            driver="GPKG",
            crs_wkt=crs,
            schema=schema,
            layer="seamlines",
        ) as dst:
            for image_name, poly in zip(input_image_names, segmented):
                dst.write(
                    {
                        "geometry": mapping(poly),
                        "properties": {image_field_name: image_name},
                    }
                )


__all__ = ["Seamline"]
