import os
import pytest

from spectralmatch import voronoi_center_seamline
from .test_utils import create_dummy_raster


# voronoi_center_seamline
@pytest.mark.parametrize("image_prefix, fill_value", [("A", 100), ("B", 120)])
def test_voronoi_center_seamline_all_params(tmp_path, image_prefix, fill_value):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()

    # Create dummy input rasters
    input_paths = []
    for name in ["A", "B"]:
        path = input_dir / f"{name}.tif"
        create_dummy_raster(
            path,
            width=256,
            height=256,
            count=1,
            transform=(1, 0, 10 if name == "A" else 20, 0, -1, -10),
            fill_value=fill_value if name == image_prefix else fill_value + 50,
        )
        input_paths.append(str(path))

    output_mask = str(output_dir / "seamlines.gpkg")
    debug_vectors = str(output_dir / "debug_vectors.gpkg")

    voronoi_center_seamline(
        input_images=input_paths,
        output_mask=output_mask,
        image_field_name="source",
        debug_logs=True,
        debug_vectors_path=debug_vectors,
    )

    assert os.path.exists(output_mask)
    assert os.path.exists(debug_vectors)


def test_voronoi_center_seamline_minimal(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()

    paths = []
    for name in ["X", "Y"]:
        path = input_dir / f"{name}.tif"
        create_dummy_raster(path, 16, 16, count=1, fill_value=75 if name == "X" else 85)
        paths.append(str(path))

    out_path = str(output_dir / "seamlines.gpkg")

    voronoi_center_seamline(input_images=paths, output_mask=out_path)

    assert os.path.exists(out_path)
