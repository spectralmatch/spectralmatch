from importlib.metadata import version, PackageNotFoundError

from .match.global_regression import global_regression
from .match.local_block_adjustment import local_block_adjustment
from .handlers import search_paths, create_paths, match_paths
from .utils import merge_rasters, mask_rasters, merge_vectors, align_rasters
from .mask.mask import (
    create_cloud_mask_with_omnicloudmask,
    create_ndvi_raster,
    band_math,
)
from .mask.utils_mask import threshold_raster, process_raster_values_to_vector_polygons
from .statistics import (
    compare_image_spectral_profiles_pairs,
    compare_before_after_all_images,
    compare_spatial_spectral_difference_band_average,
)
from .seamline.voronoi_center_seamline import voronoi_center_seamline

__all__ = [
    # Match
    "global_regression",
    "local_block_adjustment",
    # Mask
    "band_math",
    "create_cloud_mask_with_omnicloudmask",
    "create_ndvi_raster",
    "process_raster_values_to_vector_polygons",
    "threshold_raster",
    # Seamlines
    "voronoi_center_seamline",
    # Handlers
    "search_paths",
    "create_paths",
    "match_paths",
    # Utils
    "merge_rasters",
    "mask_rasters",
    "merge_vectors",
    "align_rasters",
    # Statistics
    "compare_image_spectral_profiles_pairs",
    "compare_before_after_all_images",
    "compare_spatial_spectral_difference_band_average",
]

# Import version from pyproject.toml
try:
    __version__ = version("spectralmatch")
except PackageNotFoundError:
    __version__ = "0.0.0"