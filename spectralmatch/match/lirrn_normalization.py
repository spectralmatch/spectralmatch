import numpy as np

from osgeo import gdal
gdal.UseExceptions()

from typing import List, Tuple
from scipy.spatial.distance import cdist
from skimage.filters import threshold_multiotsu

from ..types_and_validation import Universal
from ..handlers import _resolve_paths, _resolve_nodata_value, _check_raster_requirements
from ..utils import (
    create_masked_vrts,
    _set_gdal_cache,
    _set_gdal_workers,
    _resolve_gdal_dtype,
    _resolve_window_size,
)
from ..utils_multiprocessing import _resolve_parallel_config, _get_executor


def lirrn_normalization(
    input_images: Universal.SearchFolderOrListFiles,
    output_images: Universal.CreateInFolderOrListFiles,
    *,
    calculation_dtype: Universal.CalculationDtype = "float32",
    output_dtype: Universal.CustomOutputDtype = None,
    vector_mask: Universal.VectorMask = None,
    debug_logs: Universal.DebugLogs = False,
    custom_nodata_value: Universal.CustomNodataValue = None,
    cache: Universal.Cache = None,
    image_threads: Universal.Threads = None,
    io_threads: Universal.Threads = None,
    tile_threads: Universal.Threads = None,
    estimate_stats: bool = True,
    window_size: Universal.WindowSize = None,
    save_as_cog: Universal.SaveAsCog = False,
) -> List[str]:
    """
    Location-Independent Relative Radiometric Normalization (LIRRN).

    Each input image is normalized to the first image in the input list,
    which acts as the reference image.
    """

    Universal.validate(
        input_images=input_images,
        output_images=output_images,
        save_as_cog=save_as_cog,
        debug_logs=debug_logs,
        vector_mask=vector_mask,
        window_size=window_size,
        custom_nodata_value=custom_nodata_value,
        calculation_dtype=calculation_dtype,
        output_dtype=output_dtype,
        cache=cache,
        image_threads=image_threads,
        io_threads=io_threads,
        tile_threads=tile_threads,
        estimate_stats=estimate_stats,
    )

    _set_gdal_cache(cache, debug_logs)
    _set_gdal_workers(io_threads, debug_logs)

    input_paths = _resolve_paths(
        "search", input_images, kwargs={"default_file_pattern": "*.tif"}
    )
    output_paths = _resolve_paths(
        "create",
        output_images,
        kwargs={
            "paths_or_bases": input_paths,
            "default_file_pattern": "$_LIRRN.tif",
        },
    )
    names = _resolve_paths("name", input_paths)

    _check_raster_requirements(
        input_paths,
        debug_logs,
        check_geotransform=True,
        check_crs=True,
        check_bands=True,
        check_nodata=True,
    )

    output_dtype = _resolve_gdal_dtype(output_dtype, input_paths[0], debug_logs)
    nodata_val = _resolve_nodata_value(input_paths[0], custom_nodata_value)

    # masked_paths = create_masked_vrts(
    #     dict(zip(names, input_paths)),
    #     vector_mask=vector_mask,
    #     debug_logs=debug_logs,
    # )
    masked_paths = dict(zip(names, input_paths))

    ref_name = names[0]
    ref_ds = gdal.Open(masked_paths[ref_name], gdal.GA_ReadOnly)
    num_bands = ref_ds.RasterCount
    ref_arr = _read_as_array(ref_ds, calculation_dtype)
    ref_ds = None

    image_backend = "thread"
    image_threads_on, image_workers = _resolve_parallel_config(image_threads)

    tasks = [
        (
            name,
            masked_paths[name],
            output_paths[i],
            ref_arr,
            num_bands,
            nodata_val,
            window_size,
            calculation_dtype,
            output_dtype,
            save_as_cog,
            debug_logs,
        )
        for i, name in enumerate(names)
        if name != ref_name
    ]

    if image_threads_on:
        with _get_executor(image_backend, image_workers) as ex:
            futures = [ex.submit(_process_image, *args) for args in tasks]
            for f in futures:
                f.result()
    else:
        for args in tasks:
            _process_image(*args)

    return output_paths


def _process_image(
    image_name: str,
    input_path: str,
    output_path: str,
    ref_arr: np.ndarray,
    num_bands: int,
    nodata_val,
    window_size,
    calculation_dtype,
    output_dtype,
    save_as_cog,
    debug_logs: bool,
):
    if debug_logs:
        print(f"Normalizing {image_name}")

    ds = gdal.Open(input_path, gdal.GA_ReadOnly)
    sub_arr = _read_as_array(ds, calculation_dtype)
    gt = ds.GetGeoTransform()
    proj = ds.GetProjection()
    ds = None

    norm = np.zeros_like(sub_arr)
    for b in range(num_bands -1):
        a, b0 = _lirrn_band(sub_arr[..., b], ref_arr[..., b])
        norm[..., b] = a * sub_arr[..., b] + b0

    _write_output(
        norm,
        output_path,
        gt,
        proj,
        nodata_val,
        window_size,
        output_dtype,
        save_as_cog,
    )


def _lirrn_band(sub: np.ndarray, ref: np.ndarray) -> Tuple[float, float]:
    """
    Paper-faithful LIRRN band normalization:
      1) 3-class Otsu segmentation
      2) For each class: select samples near (min, mean, max)
      3) Random 10% subset
      4) Nearest-distance matching per statistic group
      5) Concatenate → linear regression
    """

    sub = np.nan_to_num(sub).ravel()
    ref = np.nan_to_num(ref).ravel()

    thresh_s = threshold_multiotsu(sub, classes=3)
    thresh_r = threshold_multiotsu(ref, classes=3)

    labels_s = np.digitize(sub, thresh_s)
    labels_r = np.digitize(ref, thresh_r)

    all_sub_matches = []
    all_ref_matches = []

    N = 1000
    keep_frac = 0.1

    for c in range(3):

        s = sub[labels_s == c]
        r = ref[labels_r == c]

        if s.size == 0 or r.size == 0:
            continue

        stats_s = [s.min(), s.mean(), s.max()]
        stats_r = [r.min(), r.mean(), r.max()]

        for stat_s, stat_r in zip(stats_s, stats_r):

            # distance to statistic
            ds = np.abs(s - stat_s)
            dr = np.abs(r - stat_r)

            # take N closest to statistic
            s0 = s[np.argsort(ds)[:min(N, len(s))]]
            r0 = r[np.argsort(dr)[:min(N, len(r))]]

            # random 10%
            k_s = max(1, int(len(s0) * keep_frac))
            k_r = max(1, int(len(r0) * keep_frac))

            s_sel = s0[np.random.choice(len(s0), k_s, replace=False)]
            r_sel = r0[np.random.choice(len(r0), k_r, replace=False)]

            s_match, r_match = _sample_selection(len(s_sel), s_sel, r_sel)

            if s_match.size:
                all_sub_matches.append(s_match)
                all_ref_matches.append(r_match)

    if not all_sub_matches:
        return 1.0, 0.0

    all_sub = np.concatenate(all_sub_matches)
    all_ref = np.concatenate(all_ref_matches)

    return np.polyfit(all_sub, all_ref, 1)


def _sample_selection(
    n: int, a: np.ndarray, b: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    a = a.ravel()
    b = b.ravel()
    if len(a) == 0 or len(b) == 0:
        return np.zeros(0), np.zeros(0)

    a = a[np.random.choice(len(a), min(n, len(a)), replace=False)]
    b = b[np.random.choice(len(b), min(n, len(b)), replace=False)]

    dist = cdist(a[:, None], b[:, None])
    ai, bi = [], []
    while len(ai) < min(len(a), len(b)):
        i, j = np.unravel_index(np.argmin(dist), dist.shape)
        ai.append(i)
        bi.append(j)
        dist[i, :] = np.inf
        dist[:, j] = np.inf

    return a[ai], b[bi]


def _read_as_array(ds, dtype):
    arr = np.stack(
        [ds.GetRasterBand(i + 1).ReadAsArray() for i in range(ds.RasterCount)],
        axis=-1,
    )
    return arr.astype(dtype)


def _write_output(
    arr,
    path,
    gt,
    proj,
    nodata,
    window_size,
    output_dtype,
    save_as_cog,
):
    driver = gdal.GetDriverByName("COG" if save_as_cog else "GTiff")
    y, x, bands = arr.shape
    ds = driver.Create(
        path,
        x,
        y,
        bands,
        gdal.GetDataTypeByName(output_dtype),
    )
    ds.SetGeoTransform(gt)
    ds.SetProjection(proj)
    for i in range(bands):
        rb = ds.GetRasterBand(i + 1)
        if nodata is not None:
            rb.SetNoDataValue(nodata)
        rb.WriteArray(arr[..., i])
    ds = None