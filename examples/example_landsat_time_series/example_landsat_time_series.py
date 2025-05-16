# %% Landsat Time Series
# This notebook demonstrates how to preprocess Landsat 8-9 into a time series with spectralmatch. Starting from 5 Landsat 8-9 OLI/TIRS C2 L1 images, the process includes clipping clouds with OmniCloudMask, masking high NDVI areas as Pseudo Invariant Features (PIFs), applying global regression Relative Radiometric Normalization, fine-tuning overlap areas with local block adjustment, and before vs after statistics.

# This script is setup to perform matching on all tif files from a folder within the working directory called "Input" e.g. working_directory/Input/*.tif.


# %% Setup
import os
import importlib

from spectralmatch.match.global_regression import global_regression
from spectralmatch.match.local_block_adjustment import local_block_adjustment
from spectralmatch.handlers import merge_rasters
from spectralmatch.mask import create_cloud_mask_with_omnicloudmask, post_process_raster_cloud_mask_to_vector, create_ndvi_mask, post_process_threshold_to_vector, mask_image_with_vector

working_directory = os.path.join(os.getcwd(), "data_landsat")
print(working_directory)


# %% Create cloud masks
input_folder = os.path.join(working_directory, "Input")
output_folder = os.path.join(working_directory, 'Masks');
os.makedirs(output_folder, exist_ok=True)
input_image_paths_array = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if
                           f.lower().endswith(".tif")]

for path in input_image_paths_array:
    create_cloud_mask_with_omnicloudmask(
        path,
        5,
        3,
        8,
        os.path.join(output_folder, f"{os.path.splitext(os.path.basename(path))[0]}_CloudMask.tif"),
        # down_sample_m=10
    )

input_mask_rasters = [os.path.join(output_folder, f) for f in os.listdir(output_folder) if f.lower().endswith(".tif")]

for path in input_mask_rasters:
    post_process_raster_cloud_mask_to_vector(
        path,
        os.path.join(output_folder, f"{os.path.splitext(os.path.basename(path))[0]}_CloudMask.gpkg"),
        None,
        {1: 50},
        {0: 0, 1: 1, 2: 1, 3: 1}
    )


# %% Use cloud masks
input_folder = os.path.join(working_directory, "Input")
mask_folder = os.path.join(working_directory, "Masks")
output_folder = os.path.join(working_directory, "Masked")
input_paths = sorted([os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.lower().endswith(".tif")])
input_mask_vectors = sorted(
    [os.path.join(mask_folder, f) for f in os.listdir(mask_folder) if f.lower().endswith(".gpkg")])
output_paths = sorted(
    [os.path.join(output_folder, os.path.splitext(os.path.basename(path))[0] + "_CloudMasked.tif") for path in
     input_paths])

for input_path, vector_path, output_path in zip(input_paths, input_mask_vectors, output_paths):
    mask_image_with_vector(
        input_path,
        vector_path,
        output_path,
        {"value": 0},
        True
    )


# %% Mask trees as a non-PIF for isolated analysis
input_folder = os.path.join(working_directory, "Input")
mask_folder = os.path.join(working_directory, "Pifs")
output_folder = os.path.join(working_directory, "Pifed")
input_paths = sorted([os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.lower().endswith(".tif")])
output_raster_masks = sorted(
    [os.path.join(mask_folder, os.path.splitext(os.path.basename(path))[0] + "_CloudMasked.tif") for path in
     input_paths])
output_vectors_masks = sorted(
    [os.path.join(mask_folder, os.path.splitext(os.path.basename(path))[0] + "_CloudMasked.gpkg") for path in
     input_paths])

for input_path, raster_path, vector_path in zip(input_paths, output_raster_masks, output_vectors_masks):
    create_ndvi_mask(
        input_path,
        raster_path,
        5,
        4,
    )

for input_path, raster_path, vector_path in zip(input_paths, output_raster_masks, output_vectors_masks):
    post_process_threshold_to_vector(
        raster_path,
        vector_path,
        0.2,
        "<=",
    )


# %% Global matching
vector_mask_path = working_directory + "/Input/Masks.gpkg"
input_folder = os.path.join(working_directory, "Masked")
global_folder = os.path.join(working_directory, "GlobalMatch")
input_image_paths = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.lower().endswith(".tif")]

global_regression(
    input_image_paths,
    global_folder,
    custom_mean_factor = 3, # Defualt 1; 3 often works better to 'move' the spectral mean of images closer together
    custom_std_factor = 1,
    # vector_mask_path=vector_mask_path,
    debug_logs=False,
    window_size=(512, 512),
    parallel=True,
    )


# %% Local matching
global_folder = os.path.join(working_directory, "GlobalMatch")
input_image_paths = [os.path.join(global_folder, f) for f in os.listdir(global_folder) if f.lower().endswith(".tif")]
local_folder = os.path.join(working_directory, "LocalMatch")
global_image_paths_array = [os.path.join(global_folder, f) for f in os.listdir(global_folder) if f.lower().endswith(".tif")]

matched_local_images_paths = local_block_adjustment(
    global_image_paths_array,
    local_folder,
    number_of_blocks=100,
    debug_logs=False,
    window_size=(512, 512),
    parallel=True,
    )

merge_rasters(
    matched_local_images_paths, # Rasters are layered with the last ones on top
    os.path.join(working_directory, "MatchedLocalImages.tif"),
    window_size=(512, 512),
    )

print("Done with global and local histogram matching")


# %% Statistics
