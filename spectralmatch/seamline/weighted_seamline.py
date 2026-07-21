import math
import os
import re

import geopandas as gpd


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


def weighted_seamline(
    input_polygons: str,
    output_mask: str,
    *,
    rank_function: str,
    image_field_name: str = "image",
    input_layer: str | None = None,
    output_layer: str = "seamlines",
    rank_descending: bool = True,
    debug_logs: bool = False,
) -> str:
    """Generate ranked seamline polygons from input polygons and a score expression.

    Rank function placeholders support field references like ``{cloud_cover}``.
    """
    read_kwargs = {}
    if input_layer is not None:
        read_kwargs["layer"] = input_layer
    gdf = gpd.read_file(input_polygons, **read_kwargs)
    if gdf.empty:
        raise ValueError("input_polygons contains no features.")
    if image_field_name not in gdf.columns:
        raise ValueError(
            f"Field '{image_field_name}' was not found in input_polygons."
        )
    if gdf.crs is None:
        raise ValueError("input_polygons must have a CRS.")

    grouped_records = []
    for _, group in gdf.groupby(image_field_name, sort=False, dropna=False):
        record = group.iloc[0].copy()
        geometries = group.geometry[group.geometry.notnull()]
        if geometries.empty:
            continue
        if hasattr(geometries, "union_all"):
            record.geometry = geometries.union_all()
        else:
            record.geometry = geometries.unary_union
        grouped_records.append(record)

    if not grouped_records:
        raise ValueError("input_polygons contains no valid polygon geometry.")

    ranked_gdf = gpd.GeoDataFrame(grouped_records, geometry="geometry", crs=gdf.crs)
    ranked_gdf["weighted_score"] = ranked_gdf.apply(
        lambda row: _evaluate_weighted_expression(row, rank_function),
        axis=1,
    )
    ranked_gdf = ranked_gdf.sort_values(
        by=["weighted_score", image_field_name],
        ascending=[not rank_descending, True],
        kind="mergesort",
    ).reset_index(drop=True)
    ranked_gdf["weighted_rank"] = ranked_gdf.index + 1

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

    output_gdf = gpd.GeoDataFrame(output_records, geometry="geometry", crs=ranked_gdf.crs)
    output_dir = os.path.dirname(output_mask)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    if os.path.exists(output_mask):
        os.remove(output_mask)
    output_gdf.to_file(output_mask, layer=output_layer, driver="GPKG")
    return output_mask


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
        raise ValueError(f"Could not evaluate rank_function '{expression}': {exc}") from exc

    try:
        return float(result)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"rank_function '{expression}' did not produce a numeric result."
        ) from exc
