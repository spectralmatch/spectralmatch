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

gdal.UseExceptions()


class Seamline:
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
        print("Start voronoi center seamline")
        output_dir = os.path.dirname(output_mask)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        if debug_vectors_path:
            debug_dir = os.path.dirname(debug_vectors_path)
            if debug_dir:
                os.makedirs(debug_dir, exist_ok=True)

        Universal.validate(
            input_images=input_images,
        )
        SeamlineValidation.validate_voronoi_center_seamline(
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
