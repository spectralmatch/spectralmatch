import os
import tempfile
import numpy as np
import re

from concurrent.futures import as_completed
from osgeo import gdal, osr, ogr

from .mask import _resolve_percentile_expressions, _write_expression_vrt, _write_expression_raster
from ..utils_multiprocessing import _get_executor, _resolve_parallel_config
from ..handlers import _resolve_paths, _resolve_nodata_value
from ..types_and_validation import Universal
from ..utils import _set_gdal_cache, _set_gdal_workers


def process_raster_values_to_vector_polygons(
    input_images: Universal.SearchFolderOrListFiles,
    output_vectors: Universal.CreateInFolderOrListFiles,
    *,
    extraction_expression: str,
    custom_nodata_value: Universal.CustomNodataValue = None,
    custom_output_dtype: Universal.CustomOutputDtype = None,
    cache: Universal.Cache = None,
    image_threads: Universal.Threads = None,
    io_threads: Universal.Threads = None,
    tile_threads: Universal.Threads = None,
    debug_logs: Universal.DebugLogs = False,
    filter_by_polygon_size: str | None = None,
    polygon_buffer: float = 0.0,
    value_mapping: dict | None = None,
    estimate_statistics: bool = True,
):
    """
    Converts raster values into vector polygons based on an expression.

    Args:
        input_images (str | List[str], required): Defines input files from a glob path, folder, or list of paths. Specify like: "/input/files/*.tif", "/input/folder" (assumes *.tif), ["/input/one.tif", "/input/two.tif"].
        output_vectors (str | List[str], required): Defines output vector files from a template path, folder, or list of paths (with the same length as the input). Specify like: "/input/files/$.gpkg", "/input/folder" (assumes $_Vectorized.gpkg), ["/input/one.gpkg", "/input/two.gpkg"].
        extraction_expression (str): A muparser‑compatible expression applied to the raster bands, see https://github.com/beltoforion/muparser. Bands are referenced as B1, B2, … and you can use C‑style comparison and logical operators (such as >, <, >=, <=, ==, !=, &&, ||, !) along with parentheses and ternary ? : constructs. Percentile thresholds are supported in both `5%B1` form and computed-expression form like `10%( ((B7 + B5) != 0) ? ((B7 - B5) / (B7 + B5)) : 0 )`.
        custom_nodata_value (Universal.CustomNodataValue, optional): Custom NoData value to override the default from the raster metadata.
        custom_output_dtype (Universal.CustomOutputDtype, optional): Reserved for compatibility; vector output is always produced.
        cache (float | None): Controls GDAL cache size in GB. Defaults to preset cache size. Applied via GDAL_CACHEMAX.
        image_threads (Literal["cpu"] | int | None): Parallelism for per-image operations. "cpu" to get number of cores, int to assign number, and None to disable image level parallelism.
        io_threads (Literal["cpu"] | int | None): Parallelism for IO operations. "cpu" to get number of cores, int to assign number, and None to disable io level parallelism.
        tile_threads (Literal["cpu"] | int | None): "cpu" to get number of cores, int to assign number, and None to disable tile level parallelism.
        debug_logs (Universal.DebugLogs, optional): Whether to print debug logs to the console.
        filter_by_polygon_size (str, optional): Area filter for resulting polygons. Can be a number (e.g., ">100") or percentile (e.g., ">95%").
        polygon_buffer (float, optional): Distance in coordinate units to buffer the resulting polygons. Default is 0.
        value_mapping (dict, optional): Mapping from original raster values to new values. Use `None` to convert to NoData.
        estimate_statistics (bool, optional): Whether to estimate statistics for percentile thresholds. Defaults to True.
    """

    print("Start raster value extraction to polygons")

    Universal.validate(
        input_images=input_images,
        output_images=output_vectors,
        custom_nodata_value=custom_nodata_value,
        custom_output_dtype=custom_output_dtype,
        cache=cache,
        image_threads=image_threads,
        io_threads=io_threads,
        tile_threads=tile_threads,
        debug_logs=debug_logs,
    )

    # Set gdal params
    _set_gdal_cache(cache, debug_logs)
    _set_gdal_workers(io_threads, debug_logs)

    input_image_paths = _resolve_paths(
        "search", input_images, kwargs={"default_file_pattern": "*.tif"}
    )
    output_image_paths = _resolve_paths(
        "create",
        output_vectors,
        kwargs={
            "paths_or_bases": input_image_paths,
            "default_file_pattern": "$_Vectorized.gpkg",
        },
    )

    if debug_logs:
        print(f"Input: {input_image_paths}")
        print(f"Output: {output_image_paths}")

    # Determine multiprocessing and worker count
    image_backend = "thread" # "thread" or "process"
    image_threads_on, image_thread_workers = _resolve_parallel_config(image_threads)
    tile_thread_on, tile_thread_workers = _resolve_parallel_config(tile_threads)

    image_args = [
        (
            in_path,
            out_path,
            extraction_expression,
            filter_by_polygon_size,
            polygon_buffer,
            value_mapping,
            custom_nodata_value,
            debug_logs,
            estimate_statistics,
            tile_thread_on,
            tile_thread_workers,
        )
        for in_path, out_path in zip(input_image_paths, output_image_paths)
    ]

    if image_threads_on:
        with _get_executor(image_backend, image_thread_workers) as executor:
            futures = [
                executor.submit(_process_image_to_polygons, *args)
                for args in image_args
            ]
            for future in as_completed(futures):
                future.result()
    else:
        for args in image_args:
            _process_image_to_polygons(*args)


def _process_image_to_polygons(
    input_image_path,
    output_vector_path,
    extraction_expression,
    filter_by_polygon_size,
    polygon_buffer,
    value_mapping,
    custom_nodata_value,
    debug_logs,
    estimate_statistics,
    tile_thread_on,
    tile_thread_workers,

):
    """
    Processes a single raster file into polygons based on an expression.

    Args:
        input_image_path (str): Path to the input raster image.
        output_vector_path (str): Output file path for the resulting vector file.
        extraction_expression (str): Logical expression using band indices (e.g., "b1 > 5 & b2 < 10").
        filter_by_polygon_size (str): Area filter for polygons. Supports direct comparisons (">100") or percentiles ("90%").
        polygon_buffer (float): Amount of buffer to apply to polygons in projection units.
        value_mapping (dict): Dictionary mapping original raster values to new ones. Set value to `None` to mark as NoData.
        custom_nodata_value: Custom NoData value to use during processing.
        debug_logs (bool): Whether to print debug logging information.
        estimate_statistics (bool): Whether to estimate statistics for percentile thresholds.
        tile_thread_on (bool): Whether to use tiled processing.
        tile_thread_workers (int): Number of worker threads to use for tiled processing.
    """

    if debug_logs:
        print(f"Processing {input_image_path}")

    try:
        ds = gdal.Open(input_image_path, gdal.GA_ReadOnly)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Could not open input raster '{input_image_path}'. "
            "Check that the glob only matches raster files and that they are not mislabeled vector containers."
        ) from exc
    if ds is None:
        raise RuntimeError(f"Could not open input raster '{input_image_path}'.")

    xsize, ysize = ds.RasterXSize, ds.RasterYSize
    gt, srs_wkt = ds.GetGeoTransform(), ds.GetProjectionRef() or ""
    num_bands = ds.RasterCount
    nodata_value = _resolve_nodata_value(input_image_path, custom_nodata_value)
    with tempfile.TemporaryDirectory(prefix="poly_") as tmpdir:
        expr_eval = _resolve_percentile_expressions(
            extraction_expression,
            input_image_path,
            xsize,
            ysize,
            gt,
            srs_wkt,
            num_bands,
            nodata_value,
            debug_logs,
            estimate_statistics,
            tmpdir,
        )

        expr_vrt = os.path.join(tmpdir, "expression.vrt")
        _write_expression_vrt(
            expr_vrt,
            input_image_path,
            xsize,
            ysize,
            gt,
            srs_wkt,
            num_bands,
            expr_eval,
            nodata_value,
            "Float32",
        )

        mask_tif = os.path.join(tmpdir, "mask.tif")
        _write_expression_raster(
            expr_vrt_path=expr_vrt,
            output_path=mask_tif,
            nodata_value=nodata_value,
            output_dtype="Float32",
            tile_thread_on=tile_thread_on,
            tile_thread_workers=tile_thread_workers,
            window_size=None,
            reference_image_path=input_image_path,
            debug_logs=debug_logs,
        )
        _polygonize_expression_raster(
            mask_tif,
            output_vector_path,
            srs_wkt,
            filter_by_polygon_size,
            polygon_buffer,
            value_mapping,
        )

    ds = None
    if debug_logs:
        print(f"Wrote: {output_vector_path}")

def _polygonize_expression_raster(
    raster_path: str,
    output_vector_path: str,
    srs_wkt: str,
    filter_by_polygon_size,
    polygon_buffer: float,
    value_mapping: dict | None,
) -> None:
    if os.path.exists(output_vector_path):
        try:
            os.remove(output_vector_path)
        except FileNotFoundError:
            pass

    driver_name = _get_vector_driver_name(output_vector_path)
    drv = ogr.GetDriverByName(driver_name)
    if drv is None:
        raise ValueError(f"Unsupported vector output format: {output_vector_path}")

    vds = drv.CreateDataSource(output_vector_path)
    srs = osr.SpatialReference()
    if srs_wkt:
        srs.ImportFromWkt(srs_wkt)
    layer_name = os.path.splitext(os.path.basename(output_vector_path))[0] or "mask"
    layer = vds.CreateLayer(layer_name, srs=srs, geom_type=ogr.wkbPolygon)
    layer.CreateField(ogr.FieldDefn("value", ogr.OFTReal))

    rds = gdal.Open(raster_path, gdal.GA_ReadOnly)
    band = rds.GetRasterBand(1)
    polygonize_fn = getattr(gdal, "FPolygonize", gdal.Polygonize)
    polygonize_fn(
        srcBand=band,
        maskBand=None,
        outLayer=layer,
        iPixValField=layer.GetLayerDefn().GetFieldIndex("value"),
    )

    def _apply_mapping_feat(feat):
        if not value_mapping:
            return True
        val = feat.GetField("value")
        if val in value_mapping:
            new = value_mapping[val]
            if new is None:
                return False
            feat.SetField("value", float(new))
        return True

    def _area_ok(geom_area, areas):
        if not filter_by_polygon_size:
            return True
        m = re.match(r"([<>]=?|==|!=)\s*(\d+(?:\.\d+)?%?)", filter_by_polygon_size.strip())
        if not m:
            raise ValueError(f"Invalid filter_by_polygon_size: {filter_by_polygon_size}")
        op, raw = m.groups()
        thr = np.percentile(areas, float(raw[:-1])) if raw.endswith("%") else float(raw)
        return {
            "<": lambda a: a < thr, "<=": lambda a: a <= thr,
            ">": lambda a: a > thr, ">=": lambda a: a >= thr,
            "==": lambda a: a == thr, "!=": lambda a: a != thr,
        }[op](geom_area)

    areas = []
    if filter_by_polygon_size:
        for feat in layer:
            geom = feat.GetGeometryRef()
            if geom:
                areas.append(geom.GetArea())
        layer.ResetReading()

    delete_ids = []
    for feat in layer:
        geom = feat.GetGeometryRef()
        if geom is None:
            delete_ids.append(feat.GetFID())
            continue
        if feat.GetField("value") == 0:
            delete_ids.append(feat.GetFID())
            continue
        if not _apply_mapping_feat(feat):
            delete_ids.append(feat.GetFID())
            continue
        if filter_by_polygon_size and not _area_ok(geom.GetArea(), areas):
            delete_ids.append(feat.GetFID())
            continue
        if polygon_buffer:
            buffered = geom.Buffer(polygon_buffer)
            if buffered is None or buffered.IsEmpty():
                delete_ids.append(feat.GetFID())
                continue
            feat.SetGeometry(buffered)
        layer.SetFeature(feat)
    layer.ResetReading()
    for fid in delete_ids:
        layer.DeleteFeature(fid)

    rds = None
    vds = None

def _get_vector_driver_name(path: str) -> str:
    extension = os.path.splitext(path)[1].lower()
    if extension == ".gpkg":
        return "GPKG"
    if extension == ".shp":
        return "ESRI Shapefile"
    if extension in {".geojson", ".json"}:
        return "GeoJSON"
    raise ValueError(f"Unsupported vector output format: {path}")
