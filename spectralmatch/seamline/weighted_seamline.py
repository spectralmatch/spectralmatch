from ..utils_progress import reports_progress
import ast
import math
import os
import re
import statistics

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
_SUMMARY_FUNCTIONS = {
    "mean": statistics.mean,
    "std": statistics.pstdev,
    "variance": statistics.pvariance,
    "min": min,
    "max": max,
    "median": statistics.median,
    "sum": math.fsum,
    "count": len,
    "range": lambda values: max(values) - min(values),
}
_EXPRESSION_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Load, ast.Constant,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.UAdd, ast.USub, ast.Not, ast.BoolOp, ast.And, ast.Or, ast.IfExp,
    ast.Compare, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.keyword,
)


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

    Higher expression values rank on top by default (rank_descending=True).
    Negate lower-is-better quantities in the expression.

    Use ``{var:field}`` for image attributes and ``{mean:field}`` or
    ``{std:field}`` for summaries across the candidate images. ``{stat:area_crs}``
    calculates each connected polygon's area in squared CRS units and ranks
    components separately without adding a table field.

    Args:
        input_polygons (str): Input polygon layer path. Each feature should represent an image footprint or a piece of one.
        input_images (str | list[str] | None, optional): Optional raster folder, glob or paths for substring matching; None infers basenames from field values.
        output_mask (str): Output GeoPackage path for the ranked seamline polygons.
        rank_function (str): Arithmetic with ``{var:field}`` and ``{summary:field}`` placeholders. Summaries are mean, std, variance, min, max, median, sum, count and range; std/variance use the population. Legacy ``{field}`` placeholders remain supported.
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


def _rank_polygons(frame, image_field_name, rank_function, rank_descending, *, split_components=True):
    """Rank table attributes and calculated geometry statistics without adding stat columns."""
    code, variables, summaries, geometry_stats = _parse_rank_expression(rank_function)
    for field in variables.values():
        if field not in frame.columns:
            raise ValueError(f"Field '{field}' referenced in rank_function was not found in the input polygons.")
    # Existing summaries remain image-level, even when components are ranked separately.
    summary_values = _rank_summary_values(frame, summaries)
    ranked = frame.copy()
    if geometry_stats:
        if split_components:
            ranked = ranked.explode(index_parts=False, ignore_index=True)
        elif any(geometry.geom_type == "MultiPolygon" and len(geometry.geoms) > 1
                 for geometry in ranked.geometry):
            raise ValueError("stat:area_crs ranks individual polygon components; image-level quality ranking cannot assign separate scores to a MultiPolygon.")
    scores = []
    for _, row in ranked.iterrows():
        calculated = {symbol: float(row.geometry.area) for symbol in geometry_stats}
        try:
            scores.append(_evaluate_weighted_expression(
                row, code, variables, {**summary_values, **calculated}, rank_function))
        except ValueError as exc:
            raise ValueError(f"Cannot rank image {row[image_field_name]!r}: {exc}") from exc
    ranked["weighted_score"] = scores
    ranked = ranked.sort_values(
        by=["weighted_score", image_field_name],
        ascending=[not rank_descending, True],
        kind="mergesort",
    ).reset_index(drop=True)
    ranked["weighted_rank"] = ranked.index + 1
    return ranked


def _parse_rank_expression(expression):
    """Compile placeholders once, using generated names rather than metadata as code."""
    if not isinstance(expression, str) or not expression.strip():
        raise ValueError("rank_function must be a non-empty string.")
    placeholders = list(dict.fromkeys(_FIELD_PATTERN.findall(expression)))
    if not placeholders:
        if "{" in expression or "}" in expression:
            raise ValueError("Invalid rank_function: unmatched placeholder braces.")
        raise ValueError("rank_function must reference a placeholder like {var:view_zenith} or {mean:view_zenith}.")
    variables, summaries, geometry_stats = {}, {}, {}
    rewritten = expression
    for index, placeholder in enumerate(placeholders):
        symbol = f"_rank_value_{index}"
        parts = [part.strip() for part in placeholder.split(":", 1)]
        if len(parts) == 1 and parts[0]:
            variables[symbol] = parts[0]  # Backward-compatible {field} spelling.
        elif len(parts) == 2 and parts[1]:
            if parts[0] == "var":
                variables[symbol] = parts[1]
            elif parts[0] == "stat":
                if parts[1] != "area_crs":
                    raise ValueError(f"Unknown geometry statistic {parts[1]!r}; choose area_crs.")
                geometry_stats[symbol] = parts[1]
            elif parts[0] in _SUMMARY_FUNCTIONS:
                summaries[symbol] = (parts[0], parts[1])
            else:
                raise ValueError(f"Unknown summary statistic {parts[0]!r}; choose {', '.join(_SUMMARY_FUNCTIONS)}.")
        else:
            raise ValueError(f"Invalid ranking placeholder {{{placeholder}}}; use {{var:field}}, {{summary:field}}, or {{stat:area_crs}}.")
        rewritten = rewritten.replace("{" + placeholder + "}", symbol)
    try:
        tree = ast.parse(rewritten, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid rank_function expression: {exc.msg}") from exc
    names = {*variables, *summaries, *geometry_stats, *_SAFE_FUNCTIONS}
    for node in ast.walk(tree):
        if not isinstance(node, _EXPRESSION_NODES):
            raise ValueError("rank_function supports arithmetic, comparisons and the documented math functions only.")
        if isinstance(node, ast.Name) and node.id not in names:
            raise ValueError(f"Unknown name {node.id!r} in rank_function; use a field placeholder.")
        if isinstance(node, ast.Call) and (not isinstance(node.func, ast.Name) or node.func.id not in _SAFE_FUNCTIONS):
            raise ValueError("rank_function may only call the documented math functions.")
    return compile(tree, "<rank_function>", "eval"), variables, summaries, geometry_stats


def _rank_summary_values(frame, summaries):
    """Compute each requested finite numeric image-level summary once per ranking."""
    fields, results, values = {}, {}, {}
    for symbol, (statistic, field) in summaries.items():
        if field not in fields:
            if field not in frame.columns:
                raise ValueError(f"Field '{field}' referenced in rank_function was not found in the input polygons.")
            try:
                fields[field] = [float(value) for value in frame[field]]
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"Summary field '{field}' must contain finite numeric values for every candidate image.") from exc
            if not fields[field] or not all(math.isfinite(value) for value in fields[field]):
                raise ValueError(f"Summary field '{field}' must contain finite numeric values for every candidate image.")
        key = (statistic, field)
        if key not in results:
            try:
                results[key] = float(_SUMMARY_FUNCTIONS[statistic](fields[field]))
            except (OverflowError, ValueError) as exc:
                raise ValueError(f"Summary statistic '{statistic}' for '{field}' must be finite.") from exc
            if not math.isfinite(results[key]):
                raise ValueError(f"Summary statistic '{statistic}' for '{field}' must be finite.")
        values[symbol] = results[key]
    return values


def _evaluate_weighted_expression(row, code, variables, summaries, expression) -> float:
    local_values = {name: row[field] for name, field in variables.items()}

    try:
        result = eval(
            code,
            {"__builtins__": {}},
            {**_SAFE_FUNCTIONS, **summaries, **local_values},
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
