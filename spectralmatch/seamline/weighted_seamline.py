from ..utils_progress import reports_progress
import math
import os
import re

import geopandas as gpd

from .footprints import _prepare_footprints
from ..types_and_validation import Universal

_FIELD_PATTERN = re.compile(r"\{([^{}]+)\}")
_SAFE_FUNCTIONS = {
    "abs": abs,
    "min": min,
    "max": max,
    "round": round,
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
}


@reports_progress
def weighted_seamline(
    input_polygons: str,
    output_mask: str,
    *,
    rank_function: str,
    input_images: Universal.SearchFolderOrListFiles | None = None,
    image_field_name: str = "image",
    input_layer: str | None = "footprints",
    output_layer: str = "seamlines",
    rank_descending: bool = True,
    debug_logs: bool = False,
) -> str:
    """Generate ranked seamline polygons from input polygons and a score expression.

    Rank function placeholders support field references like ``{cloud_cover}``.

    Args:
        input_polygons (str): Input polygon layer path. Each feature should represent an image footprint or a piece of one.
        input_images (str | list[str] | None, optional): Optional raster folder, glob or paths for substring matching; None infers basenames from field values.
        output_mask (str): Output GeoPackage path for the ranked seamline polygons.
        rank_function (str): Ranking expression using field placeholders like ``{cloud_cover}`` or formulas like ``1 / ({sun_elevation} + 1)``.
        image_field_name (str, optional): Field containing each image basename without extension as a substring; matches are dissolved and first-match attributes are ranked. Defaults to "image".
        input_layer (str | None, optional): Input footprint layer name; None selects the first layer. Defaults to "footprints".
        output_layer (str, optional): Output GeoPackage layer name. Defaults to ``"seamlines"``.
        rank_descending (bool, optional): If True, larger scores rank higher and remain on top. Defaults to True.
        debug_logs (bool, optional): If True, prints ranking details. Defaults to False.

    Returns:
        str: Written output GeoPackage path."""
    frame = _prepare_footprints(
        input_images, input_polygons, input_layer, image_field_name
    )
    ranked_gdf = _rank_polygons(frame, image_field_name, rank_function, rank_descending)

    output_records = []
    painted_geometry = None
    for _, row in ranked_gdf.iterrows():
        visible_geometry = row.geometry
        if painted_geometry is not None:
            visible_geometry = visible_geometry.difference(painted_geometry)
        if visible_geometry.is_empty:
            continue
        output_row = row.copy()
        output_row.geometry = visible_geometry
        output_records.append(output_row)
        painted_geometry = (
            visible_geometry
            if painted_geometry is None
            else painted_geometry.union(visible_geometry)
        )
        if debug_logs:
            print(
                f"Rank {int(row['weighted_rank'])}: {row[image_field_name]} "
                f"score={row['weighted_score']}"
            )

    if not output_records:
        raise ValueError("No seamline polygons were produced from the ranked inputs.")

    output_gdf = gpd.GeoDataFrame(
        output_records, geometry="geometry", crs=ranked_gdf.crs
    )
    output_dir = os.path.dirname(output_mask)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    if os.path.exists(output_mask):
        os.remove(output_mask)
    output_gdf.to_file(output_mask, layer=output_layer, driver="GPKG")
    return output_mask


def _rank_polygons(frame, image_field_name, rank_function, rank_descending):
    """Score footprint attributes with the shared expression evaluator and order images deterministically."""
    ranked = frame.copy()
    ranked["weighted_score"] = ranked.apply(
        lambda row: _evaluate_weighted_expression(row, rank_function), axis=1
    )
    ranked = ranked.sort_values(
        by=["weighted_score", image_field_name],
        ascending=[not rank_descending, True],
        kind="mergesort",
    ).reset_index(drop=True)
    ranked["weighted_rank"] = ranked.index + 1
    return ranked


def _evaluate_weighted_expression(row, expression: str) -> float:
    placeholders = _FIELD_PATTERN.findall(expression)
    if not placeholders:
        raise ValueError(
            "rank_function must reference at least one field placeholder like {cloud_cover}."
        )

    rewritten_expression = expression
    local_values = {}
    for field_name in dict.fromkeys(placeholders):
        if field_name not in row.index:
            raise ValueError(
                f"Field '{field_name}' referenced in rank_function was not found in the input polygons."
            )
        variable_name = f"field_{field_name}"
        rewritten_expression = rewritten_expression.replace(
            f"{{{field_name}}}",
            variable_name,
        )
        local_values[variable_name] = row[field_name]

    try:
        result = eval(
            rewritten_expression,
            {"__builtins__": {}},
            {**_SAFE_FUNCTIONS, **local_values},
        )
    except Exception as exc:
        raise ValueError(
            f"Could not evaluate rank_function '{expression}': {exc}"
        ) from exc

    try:
        result = float(result)
        if not math.isfinite(result):
            raise ValueError("Score must be finite.")
        return result
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"rank_function '{expression}' did not produce a finite numeric result."
        ) from exc
