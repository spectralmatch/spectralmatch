import subprocess
import re
from .utils_test import create_dummy_raster


def test_cli_general_help():
    cli_function_names = [
        "align_rasters",
        "band_math",
        "compare_image_spectral_profiles",
        "compare_image_spectral_profiles_pairs",
        "compare_spatial_spectral_difference_band_average",
        "create_cloud_mask_with_omnicloudmask",
        "global_regression",
        "local_block_adjustment",
        "mask_rasters",
        "match_paths",
        "merge_rasters",
        "merge_vectors",
        "process_raster_values_to_vector_polygons",
        "search_paths",
        "voronoi_center_seamline",
        "create_paths",
    ]

    result = subprocess.run(["spectralmatch", "--help"], capture_output=True, text=True)
    output = result.stdout + result.stderr
    assert result.returncode == 0
    for name in cli_function_names:
        assert name in output, f"'{name}' not found in CLI help output"


def test_cli_command_help():
    result = subprocess.run(
        ["spectralmatch", "global_regression", "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "global_regression" in (result.stdout + result.stderr)


def test_cli_version():
    result = subprocess.run(
        ["spectralmatch", "--version"], capture_output=True, text=True
    )
    assert result.returncode == 0
    assert re.search(
        r"\b\d+\.\d+\.\d+\b", result.stdout
    ), "Version number not found in output"