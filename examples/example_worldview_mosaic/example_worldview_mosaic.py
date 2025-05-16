# %% Worldview Mosaic
# This file demonstrates how to preprocess Worldview3 imagery into a mosaic using spectralmatch.
# Starting from two overlapping Worldview3 images in reflectance, the process includes global matching, local matching, starting from saved block maps (optional for demonstration purposes), generating seamlines, and marging images, and before vs after statistics.
# This script is set up to perform matching on all .tif files from a folder within the working directory called "Input" e.g. working_directory/Input/*.tif.

# %% Setup
import os
import importlib

from spectralmatch.match.global_regression import global_regression
from spectralmatch.match.local_block_adjustment import local_block_adjustment
from spectralmatch.handlers import merge_rasters, mask_rasters
from spectralmatch.voronoi_center_seamline import voronoi_center_seamline

working_directory = '/Users/kanoalindiwe/Downloads/Projects/spectralmatch/docs/examples/data_worldview3' # If this does not automatically find the correct CWD, manually copy the path to the data_worldview3 folder
print(working_directory)

input_folder = os.path.join(working_directory, "Input")
global_folder = os.path.join(working_directory, "GlobalMatch")
local_folder = os.path.join(working_directory, "LocalMatch")
clipped_images = os.path.join(working_directory, "ClippedImages")

window_size = 128
num_workers = 5


# %% Global matching
saved_adjustments_path = os.path.join(global_folder, "GlobalAdjustments.json")

global_regression(
    input_folder,
    (global_folder, "_global"),
    custom_mean_factor = 3, # Defualt 1; 3 often works better to 'move' the spectral mean of images closer together
    debug_logs=True,
    window_size=window_size,
    parallel_workers=num_workers,
    save_adjustments=saved_adjustments_path,
    )

# %% Local matching
input_image_paths = [os.path.join(global_folder, f) for f in os.listdir(global_folder) if f.lower().endswith(".tif")]

local_block_adjustment(
    input_image_paths,
    local_folder,
    number_of_blocks=100,
    debug_logs=True,
    window_size=window_size,
    parallel_workers=num_workers,
    save_block_maps=True,
    )

# %% Start from saved block maps (optional)
input_image_paths = [os.path.join(global_folder, f) for f in os.listdir(global_folder) if f.lower().endswith(".tif")]

old_local_folder = os.path.join(working_directory, "LocalMatch")
new_local_folder = os.path.join(working_directory, "LocalMatch_New")

saved_reference_block_path = os.path.join(old_local_folder, "BlockReferenceMean", "BlockReferenceMean.tif")
saved_local_block_paths = [os.path.join(os.path.join(old_local_folder, "BlockLocalMean"), f) for f in os.listdir(os.path.join(old_local_folder, "BlockLocalMean")) if f.lower().endswith(".tif")]

local_block_adjustment(
    input_image_paths,
    new_local_folder,
    number_of_blocks=100,
    debug_logs=True,
    window_size=window_size,
    parallel_workers=num_workers,
    load_block_maps=(saved_reference_block_path, saved_local_block_paths)
    )


# %% Generate seamlines
input_image_paths = [os.path.join(local_folder, f) for f in os.listdir(local_folder) if f.lower().endswith(".tif")]
output_vector_mask = os.path.join(working_directory, "ImageClips.gpkg")

voronoi_center_seamline(
    input_image_paths,
    output_vector_mask,
    image_field_name='image',
    debug_logs=True,
    )


# %% Clip and merge
input_image_paths = sorted([os.path.join(local_folder, f) for f in os.listdir(local_folder) if f.lower().endswith(".tif")])
output_clipped_image_paths = sorted([os.path.join(clipped_images, os.path.splitext(os.path.basename(path))[0] + "_Clipped.tif") for path in input_image_paths])

input_vector_mask_path = os.path.join(working_directory, "ImageClips.gpkg")
output_merged_image_path = os.path.join(working_directory, "MergedImage.tif")

mask_rasters(
    input_image_paths,
    output_clipped_image_paths,
    input_vector_mask_path,
    tap=True,
    debug_logs=True,
    split_mask_by_attribute="image",
    window_size=window_size,
    )

merge_rasters(
    output_clipped_image_paths,
    output_merged_image_path,
    window_size=window_size,
    debug_logs=True,
)

# %% Statistics
from spectralmatch import (
    compare_spatial_spectral_difference_individual_bands,
    compare_image_spectral_profiles_pairs,
    compare_image_spectral_profiles,
    compare_spatial_spectral_difference_average)

compare_spatial_spectral_difference_individual_bands(
    (
    '/image/a.tif',
    '/image/b.tif'),
    '/output.png'
)


compare_image_spectral_profiles_pairs(
    {
        'Image A': [
            '/image/before/a.tif',
            'image/after/a.tif'
        ],
        'Image B': [
            '/image/before/b.tif',
            '/image/after/b.tif'
        ]
    },
    '/output.png'
)


compare_image_spectral_profiles(
    {
        'Image A': 'image/a.tif',
        'Image B': '/image/b.tif'
    },
    "/output.png",
    "Digital Number Spectral Profile Comparison",
    'Band',
    'Digital Number(0-2,047)',

)


compare_spatial_spectral_difference_average(
    [
        '/image/a.tif',
        '/image/a.tif'
     ],
    '/output.png'
)