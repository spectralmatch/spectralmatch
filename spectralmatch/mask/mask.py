import tempfile
import re
import os
import shutil
import numpy as np

from typing import List
from omnicloudmask import predict_from_array
from concurrent.futures import as_completed
from osgeo import gdal
gdal.UseExceptions()

from ..types_and_validation import Universal
from ..handlers import _resolve_paths, _resolve_nodata_value, _check_raster_requirements
from ..utils_multiprocessing import _resolve_parallel_config, _get_executor
from ..utils import _set_gdal_cache, _set_gdal_workers, _resolve_gdal_dtype, _resolve_window_size


def create_cloud_mask_with_omnicloudmask(
    input_images: Universal.SearchFolderOrListFiles,
    output_images: Universal.CreateInFolderOrListFiles,
    red_band_index: int,
    green_band_index: int,
    nir_band_index: int,
    *,
    down_sample_m: float = None,
    debug_logs: Universal.DebugLogs = False,
    image_threads: Universal.Threads = None,
    image_processing_backend: Universal.ImageProcessingBackend = "local",
    dask_scheduler: Universal.DaskScheduler = None,
    omnicloud_kwargs: dict | None = None,
) -> list[str]:
    """
    Generates cloud masks from input images using OmniCloudMask, with optional downsampling and multiprocessing.

    Args:
        input_images (str | List[str], required): Defines input files from a glob path, folder, or list of paths. Specify like: "/input/files/*.tif", "/input/folder" (assumes *.tif), ["/input/one.tif", "/input/two.tif"].
        output_images (str | List[str], required): Defines output files from a template path, folder, or list of paths (with the same length as the input). Specify like: "/input/files/$.tif", "/input/folder" (assumes $_CloudMask.tif), ["/input/one.tif", "/input/two.tif"].
        red_band_index (int): Index of red band in the image.
        green_band_index (int): Index of green band in the image.
        nir_band_index (int): Index of NIR band in the image.
        down_sample_m (float, optional): If set, resamples input to this resolution in meters. Recommended to use a target resolution of 10 m or lower.
        debug_logs (bool, optional): If True, prints progress and debug info.
        image_threads (Literal["cpu"] | int | None): Enables parallel execution. Note: "process" does not work on macOS due to PyTorch MPS limitations.
        image_processing_backend: Use local threads or an existing Dask cluster.
        dask_scheduler: Existing Dask scheduler as ("file", path) or ("address", address).
        omnicloud_kwargs (dict | None): Additional arguments forwarded to predict_from_array.

    Raises:
        Exception: Propagates any error from processing individual images.
    """

    print("Start omnicloudmask")
    Universal._validate(
        input_images=input_images,
        output_images=output_images,
        debug_logs=debug_logs,
        image_threads=image_threads,
        image_processing_backend=image_processing_backend,
        dask_scheduler=dask_scheduler,
    )

    input_image_paths = _resolve_paths(
        "search", input_images, kwargs={"default_file_pattern": "*.tif"}
    )
    output_image_paths = _resolve_paths(
        "create",
        output_images,
        kwargs={
            "paths_or_bases": input_image_paths,
            "default_file_pattern": "$_CloudClip.tif",
        },
    )

    # Determine multiprocessing and worker count
    image_backend = "thread" # "thread" or "process"
    image_threads_on, image_thread_workers = _resolve_parallel_config(
        image_threads, image_processing_backend, dask_scheduler
    )


    if debug_logs:
        print(f"Input images: {input_image_paths}")
        print(f"Output images: {output_image_paths}")

    image_args = [
        (
            input_path,
            output_path,
            red_band_index,
            green_band_index,
            nir_band_index,
            down_sample_m,
            debug_logs,
            omnicloud_kwargs,
        )
        for input_path, output_path in zip(input_image_paths, output_image_paths)
    ]

    if image_threads_on:
        with _get_executor(
            image_backend,
            image_thread_workers,
            image_processing_backend=image_processing_backend,
            dask_scheduler=dask_scheduler,
        ) as executor:
            futures = [
                executor.submit(_process_cloud_mask_image, *args) for args in image_args
            ]
            for future in as_completed(futures):
                future.result()
    else:
        for args in image_args:
            _process_cloud_mask_image(*args)
    return output_image_paths


def _process_cloud_mask_image(
    input_image_path: str,
    output_mask_path: str,
    red_band_index: int,
    green_band_index: int,
    nir_band_index: int,
    down_sample_m: float,
    debug_logs: bool,
    omnicloud_kwargs: dict | None,
):
    """
    Processes a single image to generate a cloud mask using OmniCloudMask.

    Args:
        input_image_path (str): Path to input image.
        output_mask_path (str): Path to save output mask.
        red_band_index (int): Index of red band.
        green_band_index (int): Index of green band.
        nir_band_index (int): Index of NIR band.
        down_sample_m (float): Target resolution (if resampling).
        debug_logs (bool): If True, print progress info.
        omnicloud_kwargs (dict | None): Passed to predict_from_array.

    Raises:
        Exception: If any step in reading, prediction, or writing fails.
    """
    if omnicloud_kwargs is None:
        omnicloud_kwargs = {}

    ds = gdal.Open(input_image_path, gdal.GA_ReadOnly)
    gt = ds.GetGeoTransform()
    left, top = gt[0], gt[3]
    px_w, px_h = abs(gt[1]), abs(gt[5])  # pixel sizes
    width, height = ds.RasterXSize, ds.RasterYSize

    # Compute output dimensions and transform if downsampling
    if down_sample_m:
        right = left + width * gt[1]
        bottom = top + height * gt[5]
        new_width = max(1, int((right - left) / down_sample_m))
        new_height = max(1, int((top - bottom) / down_sample_m))
        out_gt = (left, down_sample_m, 0.0, top, 0.0, -down_sample_m)
        # Read and resample each band
        read_kwargs = {"buf_xsize": new_width, "buf_ysize": new_height}
        red = ds.GetRasterBand(red_band_index).ReadAsArray(**read_kwargs)
        green = ds.GetRasterBand(green_band_index).ReadAsArray(**read_kwargs)
        nir = ds.GetRasterBand(nir_band_index).ReadAsArray(**read_kwargs)
    else:
        out_gt = gt
        new_width, new_height = width, height
        red = ds.GetRasterBand(red_band_index).ReadAsArray()
        green = ds.GetRasterBand(green_band_index).ReadAsArray()
        nir = ds.GetRasterBand(nir_band_index).ReadAsArray()

    # Run the OmniCloudMask model
    band_array = np.stack([red, green, nir], axis=0)
    pred_mask = predict_from_array(band_array, **omnicloud_kwargs).squeeze().astype(np.uint8)

    # Write the mask using GDAL
    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(output_mask_path, new_width, new_height, 1, gdal.GDT_Byte)
    if out_ds is None:
        raise RuntimeError(f"Unable to create output: {output_mask_path}")
    out_ds.SetGeoTransform(out_gt)
    out_ds.SetProjection(ds.GetProjection())
    out_band = out_ds.GetRasterBand(1)
    out_band.SetNoDataValue(0)
    out_band.WriteArray(pred_mask)
    out_band.FlushCache()
    out_ds = None
    ds = None

    if debug_logs:
        print(f"Wrote mask: {output_mask_path}")


def band_math(
    input_images: Universal.SearchFolderOrListFiles,
    output_images: Universal.CreateInFolderOrListFiles,
    threshold_math: str,
    *,
    debug_logs: Universal.DebugLogs = False,
    custom_nodata_value: Universal.CustomNodataValue = None,
    cache: Universal.Cache = None,
    image_threads: Universal.Threads = None,
    io_threads: Universal.Threads = None,
    tile_threads: Universal.Threads = None,
    image_processing_backend: Universal.ImageProcessingBackend = "local",
    dask_scheduler: Universal.DaskScheduler = None,
    window_size: Universal.WindowSize = None,
    custom_output_dtype: Universal.CustomOutputDtype = None,
    calculation_dtype: Universal.CalculationDtype = "float32",
) -> List[str]:
    """
    Applies a thresholding operation to input raster images using a mathematical expression string.

    Args:
        input_images (str | List[str], required): Defines input files from a glob path, folder, or list of paths. Specify like: "/input/files/*.tif", "/input/folder" (assumes *.tif), ["/input/one.tif", "/input/two.tif"].
        output_images (str | List[str], required): Defines output files from a template path, folder, or list of paths (with the same length as the input). Specify like: "/input/files/$.tif", "/input/folder" (assumes $_Threshold.tif), ["/input/one.tif", "/input/two.tif"].
        threshold_math (str): A muparser‑compatible expression applied to the raster bands, see https://github.com/beltoforion/muparser. Bands are referenced as B1, B2, … and you can use C‑style comparison and logical operators (such as >, <, >=, <=, ==, !=, &&, ||, !) along with parentheses and ternary ? : constructs—for example, ((B1 > 5) && (B2 < 10)) ? 1 : 0. Percentile‑based thresholds are supported: write 5%B1 to substitute the 5th‑percentile value of band 1 into the expression before evaluation.
        debug_logs (bool, optional): If True, prints debug messages.
        custom_nodata_value (float | int | None, optional): Override the dataset's nodata value.
        cache (float | None): Controls GDAL cache size in GB. Defaults to preset cache size. Applied via GDAL_CACHEMAX.
        image_threads (Literal["cpu"] | int | None): Parallelism for per-image operations. "cpu" to get number of cores, int to assign number, and None to disable image level parallelism.
        io_threads (Literal["cpu"] | int | None): Parallelism for IO operations. "cpu" to get number of cores, int to assign number, and None to disable io level parallelism.
        tile_threads (Literal["cpu"] | int | None): "cpu" to get number of cores, int to assign number, and None to disable tile level parallelism.
        image_processing_backend: Use local threads or an existing Dask cluster.
        dask_scheduler: Existing Dask scheduler as ("file", path) or ("address", address).
        window_size (WindowSize, optional): Window tiling strategy for memory-efficient processing.
        custom_output_dtype (CustomOutputDtype, optional): Output data type override.
        calculation_dtype (CalculationDtype, optional): Internal computation dtype.

    Returns:
        List[str]: Paths to the thresholded output images.
    """

    Universal._validate(
        input_images=input_images,
        output_images=output_images,
        debug_logs=debug_logs,
        custom_nodata_value=custom_nodata_value,
        image_threads=image_threads,
        io_threads=io_threads,
        tile_threads=tile_threads,
        window_size=window_size,
        custom_output_dtype=custom_output_dtype,
        calculation_dtype=calculation_dtype,
        image_processing_backend=image_processing_backend,
        dask_scheduler=dask_scheduler,
    )

    # Set gdal params
    _set_gdal_cache(cache, debug_logs)
    _set_gdal_workers(io_threads, debug_logs)

    # Resolve input and output paths based on the provided specification (folder, glob or list).
    input_image_paths = _resolve_paths(
        "search", input_images, kwargs={"default_file_pattern": "*.tif"}
    )
    output_image_paths = _resolve_paths(
        "create",
        output_images,
        kwargs={
            "paths_or_bases": input_image_paths,
            "default_file_pattern": "$_Threshold.tif",
        },
    )
    image_names = _resolve_paths("name", input_image_paths)

    if debug_logs:
        print(f"Input images: {input_image_paths}")
        print(f"Output images: {output_image_paths}")

    # Dtype and nodata
    nodata_value = _resolve_nodata_value(input_image_paths[0], custom_nodata_value)
    output_dtype = _resolve_gdal_dtype(custom_output_dtype, input_image_paths[0], debug_logs)

    # Check raster requirements
    _check_raster_requirements(
        input_image_paths,
        debug_logs,
        check_geotransform=True,
        check_crs=True,
        check_bands=True,
        check_nodata=True,
    )

    # Determine multiprocessing and worker count
    image_backend = "thread" # "thread" or "process"
    image_threads_on, image_thread_workers = _resolve_parallel_config(
        image_threads, image_processing_backend, dask_scheduler
    )
    tile_thread_on, tile_thread_workers = _resolve_parallel_config(tile_threads)

    # Process each image
    if debug_logs:
        print(f"Thresholding and saving results for:")
    image_args = [
        (
            in_path,
            out_path,
            name,
            threshold_math,
            debug_logs,
            nodata_value,
            tile_thread_on,
            tile_thread_workers,
            window_size,
            output_dtype,
            calculation_dtype,
        )
        for in_path, out_path, name in zip(input_image_paths, output_image_paths, image_names)
    ]

    if image_threads_on:
        with _get_executor(
            image_backend,
            image_thread_workers,
            image_processing_backend=image_processing_backend,
            dask_scheduler=dask_scheduler,
        ) as executor:
            futures = [executor.submit(_band_math_process_image, *args) for args in image_args]
            for future in as_completed(futures):
                future.result()
    else:
        for args in image_args:
            _band_math_process_image(*args)

    return output_image_paths


def _band_math_process_image(
    input_image_path: str,
    output_image_path: str,
    name: str,
    threshold_math: str,
    debug_logs: bool,
    nodata_value,
    tile_threads_on: bool,
    tile_thread_workers: int,
    window_size,
    output_dtype: str,
    calculation_dtype: str,
) -> None:
    """
    Processes a single input raster image using a threshold expression and writes the result to disk.

    Args:
        input_image_path (str): Path to input raster image.
        output_image_path (str): Path to save the output thresholded image.
        name (str): Image name for worker context.
        threshold_math (str): Expression string to evaluate pixel-wise conditions.
        debug_logs (bool): Enable debug logging.
        nodata_value (float | int | None): Value considered as nodata.
        tile_threads_on (bool): Enable GDAL multithreaded tiling if ``True``.
        tile_thread_workers (int): Number of worker threads for GDAL tiling.
        window_size: Window tiling size for memory efficiency.
        output_dtype: Output raster data type.
        calculation_dtype: Data type used for internal calculations.

    Returns:
        None
    """
    if debug_logs:
        print(f"    Processing: {input_image_path}")

    ds = gdal.Open(input_image_path, gdal.GA_ReadOnly)

    # Metadata
    xsize = ds.RasterXSize
    ysize = ds.RasterYSize
    num_bands = ds.RasterCount
    geotransform = ds.GetGeoTransform()
    projection = ds.GetProjectionRef() or ""

    # Substitute percentile references (e.g. 5%b1) with numeric thresholds
    percent_pattern = re.compile(r"(\d+(\.\d+)?)%B(\d+)")

    def replace_percent_with_threshold(match):
        percent, _, band_num = match.groups()
        value = _calculate_threshold_from_percent(
            input_image_path=input_image_path,
            threshold=percent,
            band_index=int(band_num),
            debug_logs=debug_logs,
            nodata_value=nodata_value,
        )
        return str(value)

    evaluated_threshold_math = percent_pattern.sub(
        replace_percent_with_threshold, threshold_math
    )

    # The expression is already in muparser syntax – use it directly
    expr = evaluated_threshold_math

    # Optionally wrap with a nodata guard
    if nodata_value is not None:
        nodata_checks = " || ".join([f"(B{i}=={nodata_value})" for i in range(1, num_bands + 1)])
        final_expr = f"({nodata_checks}) ? {nodata_value} : ({expr})"
    else:
        final_expr = expr

    # Escape XML characters when embedding in the VRT
    xml_final_expr = final_expr.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    # Create a temporary VRT
    tmpdir = tempfile.mkdtemp(prefix="threshold_")
    vrt_path = os.path.join(tmpdir, f"{name}_threshold.vrt")

    # Build the list of SimpleSource entries in one shot
    vrt_xml = (
            f"<VRTDataset rasterXSize=\"{xsize}\" rasterYSize=\"{ysize}\">\n"
            f"  <SRS>{projection}</SRS>\n"
            f"  <GeoTransform>{', '.join(str(v) for v in geotransform)}</GeoTransform>\n"
            f"  <VRTRasterBand dataType=\"{calculation_dtype}\" band=\"1\" subClass=\"VRTDerivedRasterBand\">\n"
            + (f"    <NoDataValue>{nodata_value}</NoDataValue>\n" if nodata_value is not None else "")
            + "    <PixelFunctionType>expression</PixelFunctionType>\n"
            + f"    <PixelFunctionArguments dialect=\"muparser\" expression=\"{xml_final_expr}\"/>\n"
            + "\n".join(
        f"    <SimpleSource>\n"
        f"      <SourceFilename relativeToVRT=\"0\">{input_image_path}</SourceFilename>\n"
        f"      <SourceBand>{i}</SourceBand>\n"
        f"      <SrcRect xOff=\"0\" yOff=\"0\" xSize=\"{xsize}\" ySize=\"{ysize}\"/>\n"
        f"      <DstRect xOff=\"0\" yOff=\"0\" xSize=\"{xsize}\" ySize=\"{ysize}\"/>\n"
        f"    </SimpleSource>"
        for i in range(1, num_bands + 1)
    ) + "\n"
            + "  </VRTRasterBand>\n"
            + "</VRTDataset>\n"
    )

    # Write the VRT to disk
    with open(vrt_path, "w", encoding="utf-8") as f:
        f.write(vrt_xml)

    # Determine block size for output tiling
    window_size = _resolve_window_size(window_size, input_image_path, debug_logs)

    # Assemble creation options and include block sizes and thread count if needed
    creation_options = [
        "TILED=YES",
        "COMPRESS=DEFLATE",
        "PREDICTOR=2",
        "ZLEVEL=6",
        "BIGTIFF=IF_SAFER",
    ]
    if window_size:
        creation_options += [f"BLOCKXSIZE={window_size}", f"BLOCKYSIZE={window_size}"]
    if tile_threads_on:
        creation_options.append(f"NUM_THREADS={tile_thread_workers}")

    # Remove pre‑existing output file, if any
    if os.path.exists(output_image_path):
        try:
            gdal.Unlink(output_image_path)
        except Exception:
            pass

    # Translate VRT to a tiled GeoTIFF
    out_ds = gdal.Translate(
        output_image_path,
        vrt_path,
        options=gdal.TranslateOptions(
            format="GTiff",
            creationOptions=creation_options,
            bandList=[1],
            noData=nodata_value if nodata_value is not None else None,
            outputType=gdal.GetDataTypeByName(output_dtype),
        )
    )

    out_ds = None
    ds = None
    shutil.rmtree(tmpdir, ignore_errors=True)

    if debug_logs:
        print(f"    Wrote: {output_image_path}")


def _calculate_threshold_from_percent(
    input_image_path: str,
    threshold: int | float,
    band_index: int,
    debug_logs: bool = False,
    nodata_value=None,
    bins: int = 1000,
    estimate_statistics: bool = True,
) -> float:
    """
    Compute a percentile value for a raster band using GDAL.

    Args:
        input_image_path: Path to the input raster file.
        threshold: Desired percentile (e.g., 95 for the 95th percentile).
        band_index: 1-based index of the band to process.
        debug_logs: If True, print debug information.
        nodata_value: Value to be treated as nodata.
        bins: Number of histogram bins (default is 1000).
        estimate_statistics: If True, allow GDAL to approximate min/max and histogram for speed.

    Returns:
        float: The pixel value corresponding to the requested percentile.
    """

    ds = gdal.Open(input_image_path, gdal.GA_ReadOnly)
    band = ds.GetRasterBand(band_index)

    # Nodata value
    if nodata_value is not None:
        band.SetNoDataValue(nodata_value)

    # Compute min/max
    min_val, max_val, _, _ = band.GetStatistics(approx_ok=estimate_statistics, force=True)

    # Build a histogram
    hist = band.GetHistogram(
        min_val, max_val, bins,
        include_out_of_range=False,
        approx_ok=estimate_statistics,
    )
    if hist is None: raise RuntimeError("GDAL failed to compute histogram.")

    # Compute the percentile from the cumulative histogram
    cumsum = np.cumsum(hist)
    cutoff = (threshold / 100.0) * float(cumsum[-1])
    bin_index = int(np.searchsorted(cumsum, cutoff))
    bin_index = min(bin_index, bins - 1)
    bin_edges = np.linspace(min_val, max_val, bins + 1)
    value = float(bin_edges[bin_index])

    if debug_logs:
        print(
            f"Threshold: {threshold} → {value:.4f} using {bins} bins in range ({min_val:.4f}, {max_val:.4f})"
        )

    ds = None
    return value
