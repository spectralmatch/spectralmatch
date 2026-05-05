import subprocess
import re
import sys


def _run_cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "spectralmatch", *args],
        capture_output=True,
        text=True,
    )


def test_cli_general_help():
    cli_function_names = [
        "Pif",
        "align_rasters",
        "band_math",
        "compare_image_spectral_profiles_pairs",
        "compare_spatial_spectral_difference_band_average",
        "compare_before_after_all_images",
        "compute_overviews",
        "create_cloud_mask_with_omnicloudmask",
        "global_regression",
        "local_block_adjustment",
        "mask_rasters",
        "match_paths",
        "merge_rasters",
        "merge_vectors",
        "pipeline",
        "process_raster_values_to_vector_polygons",
        "search_paths",
        "create_paths",
        "voronoi_center_seamline",
    ]

    result = _run_cli("--help")
    output = result.stdout + result.stderr
    assert result.returncode == 0
    for name in cli_function_names:
        assert name in output, f"'{name}' not found in CLI help output"


def test_cli_command_help():
    result = _run_cli("global_regression", "--help")
    assert result.returncode == 0
    assert "global_regression" in (result.stdout + result.stderr)


def test_cli_version():
    result = _run_cli("--version")
    assert result.returncode == 0
    assert re.search(
        r"\b\d+\.\d+\.\d+\b", result.stdout
    ), "Version number not found in output"
