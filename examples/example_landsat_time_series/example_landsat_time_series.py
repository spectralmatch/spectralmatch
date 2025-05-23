# %% Landsat Time Series
# This notebook demonstrates how to preprocess Landsat 8-9 into a time series with spectralmatch.
# Starting from 5 Landsat 8-9 OLI/TIRS C2 L1 images, the process includes clipping clouds with OmniCloudMask, masking high NDVI areas as Pseudo Invariant Features (PIFs), applying global regression Relative Radiometric Normalization, fine-tuning overlap areas with local block adjustment, and before vs after statistics.
# This script is set up to perform matching on all tif files from a folder within the working directory called "Input" e.g. working_directory/Input/*.tif.

# %% Setup
import os
from spectralmatch import *

# Important: If this does not automatically find the correct CWD, manually copy the path to the /data_worldview3 folder
working_directory = '/Users/kanoalindiwe/Downloads/Projects/spectralmatch/docs/examples/data_landsat'
print(working_directory)

input_folder = os.path.join(working_directory, "Input")
global_folder = os.path.join(working_directory, "GlobalMatch")
local_folder = os.path.join(working_directory, "LocalMatch")
mask_cloud_folder = os.path.join(working_directory, "MaskCloud")
mask_vegetation_folder = os.path.join(working_directory, "MaskVegetation")
masked_folder = os.path.join(working_directory, "Masked")
stats_folder = os.path.join(working_directory, "Stats")

window_size = 128
num_workers = 5

# %% Create cloud masks
input_image_paths = search_paths(input_folder, "*.tif")

for path in input_image_paths:
    create_cloud_mask_with_omnicloudmask(
        path,
        5,
        3,
        8,
        os.path.join(mask_cloud_folder, f"{os.path.splitext(os.path.basename(path))[0]}_CloudMask.tif"),
        # down_sample_m=10
    )

input_mask_rasters_paths = search_paths(mask_cloud_folder, "*.tif")

for path in input_mask_rasters_paths:
    post_process_raster_cloud_mask_to_vector(
        path,
        os.path.join(mask_cloud_folder, f"{os.path.splitext(os.path.basename(path))[0]}.gpkg"),
        None,
        {1: 50},
        {0: None, 1: 1, 2: 1, 3: 1}
    )

# %% Use cloud masks
input_image_paths = search_paths(input_folder, "*.tif")
input_mask_vectors = search_paths(mask_cloud_folder, "*.gpkg", match_to_paths=(input_image_paths, r"(.*)_CloudMask\.gpkg$"))
output_paths = create_paths(masked_folder, "$_CloudMasked.tif", input_image_paths)

for input_path, vector_path, output_path in zip(input_image_paths, input_mask_vectors, output_paths):
    mask_image_with_vector(
        input_path,
        vector_path,
        output_path,
        ("include", "value", 1),
    )

# %% Create vegetation mask for isolated analysis of vegetation
input_image_paths = search_paths(input_folder, "*.tif")
raster_mask_paths = create_paths(mask_vegetation_folder, "$_VegetationMask.tif", input_image_paths)
vector_mask_paths = create_paths(mask_vegetation_folder, "$.gpkg", input_image_paths)

for input_path, raster_path in zip(input_image_paths, raster_mask_paths):
    create_ndvi_mask(
        input_path,
        raster_path,
        5,
        4,
    )

for raster_path, vector_path in zip(raster_mask_paths, vector_mask_paths):
    post_process_threshold_to_vector(
        raster_path,
        vector_path,
        0.1,
        ">=",
    )

# %% Merge vegetation mask to create an inverted-PIF vector
# This is just a simple example of creating PIFs based on NDVI values, for a more robust methodology use other techniques to create a better mask vector file

input_vector_paths = search_paths(mask_vegetation_folder, "*.gpkg")
merged_vector_pif_path = os.path.join(working_directory, "Pifs.gpkg")

merge_vectors(
    input_vector_paths,
    merged_vector_pif_path,
    create_name_attribute=("image", ", "),
    method="intersection",
    # method="keep_all", # Create a unique mask per image
    )

# %% Global matching
vector_mask_path = os.path.join(working_directory , "Pifs.gpkg")

global_regression(
    (masked_folder, "*.tif"),
    (global_folder, "$_GlobalMatch.tif"),
    custom_mean_factor = 3, # Defualt 1; 3 often works better to 'move' the spectral mean of images closer together
    vector_mask_path=("exclude", vector_mask_path),
    # vector_mask_path=("exclude", vector_mask_path, "image"), # Use unique mask per image
    window_size=window_size,
    parallel_workers=num_workers,
    debug_logs=True,
    )

# %% Local matching
vector_mask_path = os.path.join(working_directory , "Pifs.gpkg")

local_block_adjustment(
    (global_folder, "*.tif"),
    (local_folder, "$_LocalMatch.tif"),
    number_of_blocks=100,
    vector_mask_path=("exclude", vector_mask_path),
    # vector_mask_path=("exclude", vector_mask_path, "image"), # Use unique mask per image
    window_size=window_size,
    parallel_workers=num_workers,
    debug_logs=True,
    save_block_maps=(os.path.join(local_folder, "ReferenceBlockMap", "ReferenceBlockMap.tif"), os.path.join(local_folder, "LocalBlockMap", "$_LocalBlockMap.tif")),
    )

# %% Pre-coded quick Statistics

# Compare image spectral profiles
compare_image_spectral_profiles(
    input_image_dict={
        os.path.splitext(os.path.basename(p))[0]: p
        for p in search_paths(local_folder, "*.tif")
    },
    output_figure_path=os.path.join(stats_folder,'LocalMatch_CompareImageSpectralProfiles.png'),
    title="Global to Local Match Comparison of Image Spectral Profiles",
    xlabel='Band',
    ylabel='Reflectance(0-10,000)',
)

# Compare image spectral profiles pairs
before_paths = search_paths(input_folder, "*.tif")
after_paths = search_paths(local_folder, "*.tif")

image_pairs = {
    os.path.splitext(os.path.basename(b))[0]: [b, a]
    for b, a in zip(sorted(before_paths), sorted(after_paths))
    }

compare_image_spectral_profiles_pairs(
    image_pairs,
    os.path.join(stats_folder, 'LocalMatch_CompareImageSpectralProfilesPairs.png'),
    title="Global to Local Match Comparison of Image Spectral Profiles Pairs",
    xlabel='Band',
    ylabel='Reflectance(0-10,000)',
    )

# Compare spatial spectral difference band average
input_paths = search_paths(input_folder, "*.tif")
local_paths = search_paths(local_folder, "*.tif")
before_path, after_path = next(zip(sorted(input_paths), sorted(local_paths)))

compare_spatial_spectral_difference_band_average(
    input_images=[before_path, after_path],
    output_image_path=os.path.join(stats_folder, 'LocalMatch_CompareSpatialSpectralDifferenceBandAverage.png'),
    title="Global to Local Match Comparison of Spatial Spectral Difference Band Average",
    diff_label="Reflectance Difference (0–10,000)",
    subtitle=f"Image: {os.path.splitext(os.path.basename(before_path))[0]}",
)
