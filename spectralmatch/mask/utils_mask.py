import os
import tempfile
import shutil
import numpy as np
import re

from concurrent.futures import as_completed
from osgeo import gdal, osr, ogr

from .mask import _calculate_threshold_from_percent
from ..utils_multiprocessing import _get_executor, _resolve_parallel_config
from ..handlers import _resolve_paths
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
    concurrent_processing_backend: Universal.ConcurrentProcessingBackend = "process_pool",
    dask_scheduler: Universal.DaskScheduler = None,
    debug_logs: Universal.DebugLogs = False,
    filter_by_polygon_size: str | None = None,
    polygon_buffer: float = 0.0,
    value_mapping: dict | None = None,
    estimate_statistics: bool = True,
):
    """
    Converts raster values into vector polygons based on an expression and optional filtering logic.

    Args:
        input_images (str | List[str], required): Defines input files from a glob path, folder, or list of paths. Specify like: "/input/files/*.tif", "/input/folder" (assumes *.tif), ["/input/one.tif", "/input/two.tif"].
        output_vectors (str | List[str], required): Defines output files from a template path, folder, or list of paths (with the same length as the input). Specify like: "/input/files/$.gpkg", "/input/folder" (assumes $_Vectorized.gpkg), ["/input/one.gpkg", "/input/two.gpkg"].
        extraction_expression (str): A muparser‑compatible expression applied to the raster bands, see https://github.com/beltoforion/muparser. Bands are referenced as B1, B2, … and you can use C‑style comparison and logical operators (such as >, <, >=, <=, ==, !=, &&, ||, !) along with parentheses and ternary ? : constructs—for example, ((B1 > 5) && (B2 < 10)) ? 1 : 0. Percentile‑based thresholds are supported: write 5%B1 to substitute the 5th‑percentile value of band 1 into the expression before evaluation.
        custom_nodata_value (Universal.CustomNodataValue, optional): Custom NoData value to override the default from the raster metadata.
        custom_output_dtype (Universal.CustomOutputDtype, optional): Desired output data type. If not set, defaults to raster’s dtype.
        cache (float | None): Controls GDAL cache size in GB. Defaults to preset cache size. Applied via GDAL_CACHEMAX.
        image_threads (Literal["cpu"] | int | None): Parallelism for per-image operations. "cpu" to get number of cores, int to assign number, and None to disable image level parallelism.
        io_threads (Literal["cpu"] | int | None): Parallelism for IO operations. "cpu" to get number of cores, int to assign number, and None to disable io level parallelism.
        tile_threads (Literal["cpu"] | int | None): "cpu" to get number of cores, int to assign number, and None to disable tile level parallelism.
        concurrent_processing_backend: Use a local process pool or an existing Dask cluster.
        dask_scheduler: Existing Dask scheduler as ("file", path) or ("address", address).
        debug_logs (Universal.DebugLogs, optional): Whether to print debug logs to the console.
        filter_by_polygon_size (str, optional): Area filter for resulting polygons. Can be a number (e.g., ">100") or percentile (e.g., ">95%").
        polygon_buffer (float, optional): Distance in coordinate units to buffer the resulting polygons. Default is 0.
        value_mapping (dict, optional): Mapping from original raster values to new values. Use `None` to convert to NoData.
        estimate_statistics (bool, optional): Whether to estimate statistics for percentile thresholds. Defaults to True.
    """

    print("Start raster value extraction to polygons")

    Universal._validate(
        input_images=input_images,
        output_images=output_vectors,
        custom_nodata_value=custom_nodata_value,
        custom_output_dtype=custom_output_dtype,
        cache=cache,
        image_threads=image_threads,
        io_threads=io_threads,
        tile_threads=tile_threads,
        debug_logs=debug_logs,
        concurrent_processing_backend=concurrent_processing_backend,
        dask_scheduler=dask_scheduler,
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
    image_threads_on, image_thread_workers = _resolve_parallel_config(
        image_threads, concurrent_processing_backend, dask_scheduler
    )
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
        with _get_executor(
            image_backend,
            image_thread_workers,
            concurrent_processing_backend=concurrent_processing_backend,
            dask_scheduler=dask_scheduler,
        ) as executor:
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
    Processes a single raster file and extracts polygons based on logical expressions and optional filters.

    Args:
        input_image_path (str): Path to the input raster image.
        output_vector_path (str): Output file path for the resulting vector file (GeoPackage format).
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

    # Open dataset
    ds = gdal.Open(input_image_path, gdal.GA_ReadOnly)
    xsize, ysize = ds.RasterXSize, ds.RasterYSize
    gt, srs_wkt = ds.GetGeoTransform(), ds.GetProjectionRef() or ""
    num_bands = ds.RasterCount

    # Percent substitution: e.g. "5%B1" -> numeric threshold
    patt = re.compile(r"(\d+(?:\.\d+)?)%B(\d+)")
    def _sub(m):
        v = _calculate_threshold_from_percent(
            input_image_path=input_image_path,
            threshold=float(m.group(1)),
            band_index=int(m.group(2)),
            debug_logs=debug_logs,
            nodata_value=custom_nodata_value,
            estimate_statistics=estimate_statistics,
        )
        return str(v)
    expr_eval = patt.sub(_sub, extraction_expression)

    # VRT with muparser expression
    xml_expr = (expr_eval.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;"))
    sources = "\n".join(
        f"    <SimpleSource>\n"
        f"      <SourceFilename relativeToVRT=\"0\">{input_image_path}</SourceFilename>\n"
        f"      <SourceBand>{i}</SourceBand>\n"
        f"      <SrcRect xOff=\"0\" yOff=\"0\" xSize=\"{xsize}\" ySize=\"{ysize}\"/>\n"
        f"      <DstRect xOff=\"0\" yOff=\"0\" xSize=\"{xsize}\" ySize=\"{ysize}\"/>\n"
        f"    </SimpleSource>"
        for i in range(1, num_bands+1)
    )
    vrt_xml = (
        f"<VRTDataset rasterXSize=\"{xsize}\" rasterYSize=\"{ysize}\">\n"
        f"  <SRS>{srs_wkt}</SRS>\n"
        f"  <GeoTransform>{', '.join(str(v) for v in gt)}</GeoTransform>\n"
        f"  <VRTRasterBand dataType=\"Byte\" band=\"1\" subClass=\"VRTDerivedRasterBand\">\n"
        + (f"    <NoDataValue>{custom_nodata_value}</NoDataValue>\n" if custom_nodata_value is not None else "")
        + "    <PixelFunctionType>expression</PixelFunctionType>\n"
        + f"    <PixelFunctionArguments dialect=\"muparser\" expression=\"{xml_expr}\"/>\n"
        + sources + "\n  </VRTRasterBand>\n</VRTDataset>\n"
    )

    tmpdir = tempfile.mkdtemp(prefix="poly_")
    vrt_path = os.path.join(tmpdir, "mask.vrt")
    with open(vrt_path, "w", encoding="utf-8") as f: f.write(vrt_xml)

    # Translate mask
    co = ["TILED=YES", "COMPRESS=DEFLATE", "PREDICTOR=2", "ZLEVEL=6", "BIGTIFF=IF_SAFER"]
    if tile_thread_on:
        co.append(f"NUM_THREADS={tile_thread_workers}")

    mask_tif = os.path.join(tmpdir, "mask.tif")
    if os.path.exists(mask_tif):
        try:
            gdal.Unlink(mask_tif)
        except:
            pass

    out_ds = gdal.Translate(
        mask_tif, vrt_path,
        options=gdal.TranslateOptions(
            format="GTiff",
            creationOptions=co,
            bandList=[1],
            noData=0 if custom_nodata_value is None else custom_nodata_value,
            outputType=gdal.GDT_Byte,
        ),
    )

    if out_ds is None: raise RuntimeError("Failed to materialize mask")
    out_ds = None

    # Polygonize non-zero values; write to GPKG
    if os.path.exists(output_vector_path):
        try: os.remove(output_vector_path)
        except: pass

    drv = ogr.GetDriverByName("GPKG")
    vds = drv.CreateDataSource(output_vector_path)
    srs = osr.SpatialReference()
    if srs_wkt: srs.ImportFromWkt(srs_wkt)
    layer = vds.CreateLayer("mask", srs=srs, geom_type=ogr.wkbPolygon)
    layer.CreateField(ogr.FieldDefn("value", ogr.OFTInteger))

    rds = gdal.Open(mask_tif, gdal.GA_ReadOnly)
    band = rds.GetRasterBand(1)
    gdal.Polygonize(srcBand=band, maskBand=None, outLayer=layer, iPixValField=layer.GetLayerDefn().GetFieldIndex("value"))

    # Area filter
    def _apply_mapping_feat(feat):
        if not value_mapping: return True
        val = feat.GetField("value")
        if val in value_mapping:
            new = value_mapping[val]
            if new is None: return False
            feat.SetField("value", int(new))
        return True

    def _area_ok(geom_area, areas):
        if not filter_by_polygon_size: return True
        m = re.match(r"([<>]=?|==|!=)\s*(\d+(?:\.\d+)?%?)", filter_by_polygon_size.strip())
        if not m: raise ValueError(f"Invalid filter_by_polygon_size: {filter_by_polygon_size}")
        op, raw = m.groups()
        thr = np.percentile(areas, float(raw[:-1])) if raw.endswith("%") else float(raw)
        return {
            "<":  lambda a: a <  thr, "<=": lambda a: a <= thr,
            ">":  lambda a: a >  thr, ">=": lambda a: a >= thr,
            "==": lambda a: a == thr, "!=": lambda a: a != thr,
        }[op](geom_area)

    # Precompute areas for percentile filter
    areas = []
    if filter_by_polygon_size:
        for f in layer:
            g = f.GetGeometryRef()
            if g: areas.append(g.GetArea())
        layer.ResetReading()

    del_ids = []
    for f in layer:
        g = f.GetGeometryRef()
        if g is None: del_ids.append(f.GetFID()); continue
        # drop zeros
        if f.GetField("value") == 0: del_ids.append(f.GetFID()); continue
        if not _apply_mapping_feat(f): del_ids.append(f.GetFID()); continue
        if filter_by_polygon_size and not _area_ok(g.GetArea(), areas): del_ids.append(f.GetFID()); continue
        if polygon_buffer:
            buf = g.Buffer(polygon_buffer)
            if buf is None or buf.IsEmpty(): del_ids.append(f.GetFID()); continue
            f.SetGeometry(buf)
        layer.SetFeature(f)
    layer.ResetReading()
    for fid in del_ids: layer.DeleteFeature(fid)

    rds = None; vds = None; ds = None
    shutil.rmtree(tmpdir, ignore_errors=True)
    if debug_logs: print(f"Wrote: {output_vector_path}")
