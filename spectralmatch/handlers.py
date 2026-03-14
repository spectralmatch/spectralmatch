import os
import warnings
import re
import glob

from osgeo import gdal
gdal.UseExceptions()
from typing import List, Optional, Literal, Tuple

from spectralmatch.types_and_validation import Universal


def _resolve_paths(
    mode: Literal["search", "create", "match", "name"],
    input: Universal.SearchFolderOrListFiles | Universal.CreateInFolderOrListFiles,
    *,
    kwargs: dict | None = None,
) -> List[str]:
    """
    Resolves a list of input based on the mode and input format.

    Args:
        mode (Literal["search", "create", "match", "name"]): Type of operation to perform.
        input (str | List[str]): Either a list of file input or a folder/template string.
        kwargs (dict, optional): Additional keyword arguments passed to the resolved function.

    Returns:
        List[str]: List of resolved input.
    """
    kwargs = kwargs or {}

    if mode == "name":
        resolved = [os.path.splitext(os.path.basename(p))[0] for p in input]
    elif isinstance(input, list):
        resolved = input
    elif mode == "search":
        resolved = search_paths(input, **kwargs)
    elif mode == "create":
        resolved = create_paths(input, **kwargs)
    elif mode == "match":
        resolved = match_paths(**kwargs)
    else:
        raise ValueError(f"Invalid mode: {mode}")

    if len(resolved) == 0:
        warnings.warn("No results found for paths.", RuntimeWarning)

    return resolved


def search_paths(
    search_pattern: str,
    *,
    default_file_pattern: str | None = None,
    recursive: bool = False,
    match_to_paths: Tuple[List[str], str] | None = None,
    debug_logs: bool = False,
) -> List[str]:
    """
    Search for files using a glob pattern, or a folder with a default file pattern.

    Args:
        search_pattern (str, required): Defines input files from a glob path or folder. Specify like: "/input/files/*.tif" or "/input/folder" (while passing default_file_pattern like: '*.tif')
        default_file_pattern (str, optional): Used when `pattern` is a directory. If not set and `pattern` is a folder, raises an error.
        recursive (bool, optional): Whether to search recursively.
        match_to_paths (Tuple[List[str], str], optional): Matches input files to a reference list using a regex.
        debug_logs (bool, optional): Whether to print matched paths.

    Returns:
        List[str]: Sorted list of matched file paths.

    Raises:
        ValueError: If `search_pattern` is a directory and `default_file_pattern` is not provided.
    """
    if not os.path.basename(search_pattern).count("."):
        if not default_file_pattern:
            raise ValueError(
                "Pattern is a directory, but no default_file_pattern was provided."
            )
        search_pattern = os.path.join(search_pattern, default_file_pattern)

    input_paths = sorted(glob.glob(search_pattern, recursive=recursive))

    if debug_logs:
        print(f"Found {len(input_paths)} file(s) matching: {search_pattern}")

    if match_to_paths:
        input_paths = match_paths(input_paths, *match_to_paths)

    return input_paths


def create_paths(
    template_pattern: str,
    paths_or_bases: List[str],
    *,
    default_file_pattern: str | None = None,
    debug_logs: bool = False,
    replace_symbol: str = "$",
    create_folders: bool = True,
) -> List[str]:
    """
    Create output paths using a filename template_pattern and a list of reference paths or names.

    Args:
        template_pattern (str, required): Defines output files from a glob path or folder to match input paths or names. Specify like: "/input/files/$.tif" or "/input/folder" (while passing default_file_pattern like: '$.tif')
        paths_or_bases (List[str]): List of full paths or base names to derive the replace_symbol from.
        default_file_pattern (str, optional): Used if `template_pattern` is a directory.
        debug_logs (bool): Whether to print the created paths.
        replace_symbol (str): Placeholder symbol in the template to replace with base names.
        create_folders (bool): Whether to create output folders if they don't exist.

    Returns:
        List[str]: List of constructed file paths.

    Raises:
        ValueError: If `template_pattern` is a directory and `default_file_pattern` is not provided.
    """
    if not os.path.basename(template_pattern).count("."):
        if not default_file_pattern:
            raise ValueError(
                "Template is a directory, but no default_file_pattern was provided."
            )
        template_pattern = os.path.join(template_pattern, default_file_pattern)

    output_paths = []
    for ref in paths_or_bases:
        base = (
            os.path.splitext(os.path.basename(ref))[0]
            if ("/" in ref or "\\" in ref)
            else os.path.splitext(ref)[0]
        )
        filename = template_pattern.replace(replace_symbol, base)
        output_paths.append(filename)

    if create_folders:
        for path in output_paths:
            os.makedirs(os.path.dirname(path), exist_ok=True)

    if debug_logs:
        print(f"Created {len(output_paths)} paths:")
        for p in output_paths:
            print(f"  {p}")

    return output_paths


def match_paths(
    input_match_paths: List[str],
    reference_paths: List[str],
    match_regex: str,
    debug_logs: bool = False,
) -> List[Optional[str]]:
    """
    Match `reference_paths` to `input_match_paths` using a regex applied to the basenames of `input_match_paths`. The extracted key must be a substring of the reference filename.

    Args:
        input_match_paths (List[str]): List of candidate paths to extract keys from.
        reference_paths (List[str]): List of reference paths to align to.
        match_regex (str): Regex applied to basenames of input_match_paths to extract a key to match via *inclusion* in reference_paths (e.g. "(.*)_LocalMatch\\.gpkg$" (without one of the backslashes)).
        debug_logs (bool): If True, print matched and unmatched file basenames.

    Returns:
        List[Optional[str]]: A list the same length as `reference_paths` where each
        element is the matched path from `input_match_paths` or None.

    Raises:
        ValueError: If output list length does not match reference_paths length.
    """
    pattern = re.compile(match_regex)
    match_keys = {}
    used_matches = set()

    # Extract keys from input_match_paths
    for mpath in input_match_paths:
        basename = os.path.basename(mpath)
        match = pattern.search(basename)
        if not match:
            continue
        key = match.group(1) if match.groups() else match.group(0)
        match_keys[key] = mpath

    # Match each reference path
    matched_list: List[Optional[str]] = []
    for rpath in reference_paths:
        rbase = os.path.basename(rpath)
        matched = None
        for key, mpath in match_keys.items():
            if key in rbase:
                matched = mpath
                used_matches.add(mpath)
                break
        matched_list.append(matched)

    # Validate output length
    if len(matched_list) != len(reference_paths):
        raise ValueError("Matched list length does not match reference_paths length.")

    return matched_list


def _check_raster_requirements(
    input_image_paths: list,
    debug_logs: bool,
    check_geotransform: bool = False,
    check_crs: bool = False,
    check_bands: bool = False,
    check_nodata: bool = False,
    check_resolution: bool = False,
) -> bool:
    """
    Validates a list of raster image paths to ensure they are compatible for processing.

    Args:
        input_image_paths (list[str]): Paths to input raster images.
        debug_logs (bool): If True, prints debug messages.
        check_geotransform (bool): Check that all images have a valid geotransform.
        check_crs (bool): Check that all images have the same CRS.
        check_bands (bool): Check that all images have the same number of bands.
        check_nodata (bool): Check that all images have the same nodata values per band.
        check_resolution (bool): Check that all images have the same resolution.

    Returns:
        bool: True if all checks pass.

    Raises:
        ValueError: If any check fails.
    """

    if debug_logs:
        print(f"Found {len(input_image_paths)} images")

    datasets = [gdal.Open(p, gdal.GA_ReadOnly) for p in input_image_paths]

    ref_ds = datasets[0]
    ref_crs = ref_ds.GetProjectionRef()
    ref_count = ref_ds.RasterCount
    ref_gt = ref_ds.GetGeoTransform()
    ref_res = (ref_gt[1], abs(ref_gt[5]))
    ref_nodata = [ref_ds.GetRasterBand(b + 1).GetNoDataValue() for b in range(ref_count)]

    for i, ds in enumerate(datasets):
        if check_geotransform:
            gt = ds.GetGeoTransform()
            if gt is None or gt[1] == 0 or gt[5] == 0:
                raise ValueError(f"Fail: Image {i} has no valid geotransform.")
        if check_crs and ds.GetProjectionRef() != ref_crs:
            raise ValueError(f"Fail: Image {i} has different CRS.")
        if check_bands and ds.RasterCount != ref_count:
            raise ValueError(f"Fail: Image {i} has {ds.RasterCount} bands; expected {ref_count}.")
        if check_resolution:
            gt = ds.GetGeoTransform()
            res = (gt[1], abs(gt[5]))
            if res != ref_res:
                raise ValueError(f"Fail: Image {i} has resolution {res}; expected {ref_res}.")
        if check_nodata:
            for b in range(ref_count):
                nd = ds.GetRasterBand(b + 1).GetNoDataValue()
                if nd != ref_nodata[b]:
                    raise ValueError(f"Fail: Image {i}, band {b+1} has different nodata value.")

    if debug_logs:
        passed_checks = []
        if check_geotransform: passed_checks.append("geotransform")
        if check_crs: passed_checks.append("crs")
        if check_bands: passed_checks.append("bands")
        if check_nodata: passed_checks.append("nodata")
        if check_resolution: passed_checks.append("resolution")
        print(f"Input data checks passed: {', '.join(passed_checks)}")

    for ds in datasets:
        ds = None
    return True


def _resolve_nodata_value(
    image_path: str | None = None,
    custom_nodata_value: float | int | None = None,
) -> float | int | None:
    """
    Get the NoData value for a raster.

    Args:
        image_path (str): Path to the raster image.
        custom_nodata_value (float | int | None, optional): If provided, overrides the image's NoData.

    Returns:
        float | None: The chosen NoData value.
    """
    if custom_nodata_value is not None:
        return custom_nodata_value
    else:
        ds = gdal.Open(image_path, gdal.GA_ReadOnly)
        nodata = ds.GetRasterBand(1).GetNoDataValue()
        ds = None
        if nodata is None:
            raise ValueError(f"No NoData value found on rasters and custom_nodata_value was not provided. One of these is required.")
        return nodata