import os
from itertools import combinations
from typing import Literal

import fiona
from osgeo import gdal
from shapely.geometry import LineString, Polygon, mapping
from shapely.ops import unary_union

from .footprints import (
    create_footprints,
    postprocess_footprints,
    _read_polygons,
    _polygon_parts,
)

from ..utils_logging import (
    _print_step_start,
    _print_image_start,
    _print_image_completed,
)

from ..handlers import _existing_outputs_are_reusable
from ..types_and_validation import Universal, Seamline as SeamlineValidation
from .voronoi_center_seamline import (
    _compute_centerline,
    _mask_by_aoi,
    _save_emp_outlines,
    _save_intersection_points,
    _segment_emp,
)
from .weighted_seamline import weighted_seamline

gdal.UseExceptions()


class Seamline:
    create_footprints = staticmethod(create_footprints)
    postprocess_footprints = staticmethod(postprocess_footprints)

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
        resume_from_outputs: Literal["no", "yes", "validate"] = "no",
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
        aoi_path: str | None = None,
        input_layer: str | None = None,
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
    input_polygons (str): Input polygon layer path. Each feature should represent an image footprint or a piece of one. Features sharing the same image identifier are merged before processing.
    input_layer (str | None, optional): Optional input layer name when reading multi-layer vector sources. Defaults to None.
    output_layer (str, optional): Output GeoPackage layer name. Defaults to ``"seamlines"``.
    output_mask (str): Output path for the final seamline polygon vector file.
    aoi_path (str, optional): Path to an AOI vector file to clip overlapping image polygons; default is None.
    min_point_spacing (float, optional): Minimum spacing between Voronoi seed points; default is 10.
    min_cut_length (float, optional): Minimum cutline segment length to retain; default is 0.
    debug_logs (Universal.DebugLogs, optional): Enables debug print statements if True; default is False.
    image_field_name (str, optional): Name of the attribute field for image ID in output; default is 'image'.
    debug_vectors_path (str | None, optional): Optional path to save debug layers (cutlines, intersections).

    resume_from_outputs (Literal["no", "yes", "validate"], optional): Recompute outputs with "no", reuse existing outputs with "yes", or validate existing outputs before reusing them with "validate"; default is "no".

Returns:
    str: Written output GeoPackage path.

Outputs:
    Saves a polygon seamline layer to `output_mask`, and optionally saves intermediate cutlines to `debug_vectors_path`."""
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
        frame = _read_polygons(input_polygons, input_layer, image_field_name)
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
            from shapely.geometry import MultiPolygon

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
