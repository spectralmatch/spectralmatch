# %% Landsat Time Series
# This notebook demonstrates how to preprocess Landsat 8-9 into a time series with spectralmatch.
# Starting from 5 Landsat 8-9 OLI/TIRS C2 L1 images, the process includes clipping clouds with OmniCloudMask, masking high NDVI areas as Pseudo Invariant Features (PIFs), applying global regression Relative Radiometric Normalization, fine-tuning overlap areas with local block adjustment, and before vs after statistics.
# This script is set up to perform matching on all tif files from a folder within the working directory called "Input" e.g. working_directory/Input/*.tif.

# %% Setup
import os
from spectralmatch import *

# Important: If this does not automatically find the correct CWD, manually copy the path to the /data_worldview folder
working_directory = os.path.join(os.getcwd(), "data_landsat")
print(working_directory)

input_folder = os.path.join(working_directory, "Input")
global_folder = os.path.join(working_directory, "GlobalMatch")
local_folder = os.path.join(working_directory, "LocalMatch")
mask_cloud_folder = os.path.join(working_directory, "MaskCloud")
mask_vegetation_folder = os.path.join(working_directory, "MaskVegetation")
masked_folder = os.path.join(working_directory, "Masked")
stats_folder = os.path.join(working_directory, "Stats")

window_size = 128
num_image_workers = 3
num_window_workers = 5

# %% Create cloud masks

create_cloud_mask_with_omnicloudmask(
    input_images=(input_folder, "*.tif"),
    output_images=(mask_cloud_folder, "$_CloudMask.tif"),
    red_band_index=5,
    green_band_index=3,
    nir_band_index=8,
    debug_logs=True,
    image_parallel_workers=("thread", num_image_workers),
)

process_raster_values_to_vector_polygons(
    input_images=(mask_cloud_folder, "*.tif"),
    output_vectors=(mask_cloud_folder, "$.gpkg"),
    extraction_expression="b1==1",
    value_mapping={0: None, 1: 1, 2: 1, 3: 1},
    polygon_buffer=50,
    image_parallel_workers=("process", num_image_workers),
    window_parallel_workers=("process", num_window_workers),
    window_size=window_size,
)

merge_vectors(
    input_vectors=(mask_cloud_folder, "*.gpkg"),
    merged_vector_path=os.path.join(working_directory, "CloudMasks.gpkg"),
    method="keep",
    create_name_attribute=("image", ", "),
)

# %% Use cloud masks

mask_rasters(
    input_images=(input_folder, "*.tif"),
    output_images=(masked_folder, "$_CloudMasked.tif"),
    vector_mask=(
        "exclude",
        os.path.join(working_directory, "CloudMasks.gpkg"),
        "image",
    ),
)

# %% Create vegetation mask for isolated analysis of vegetation. This will be used to mask statistics for adjustment model not to directly clip images. This is just a simple example of creating PIFs based on NDVI values, for a more robust methodology use other techniques to create a better mask vector file.

create_ndvi_raster(
    input_images=(input_folder, "*.tif"),
    output_images=(mask_vegetation_folder, "$_Vegetation.tif"),
    nir_band_index=5,
    red_band_index=4,
)

process_raster_values_to_vector_polygons(
    input_images=(mask_vegetation_folder, "*.tif"),
    output_vectors=(mask_vegetation_folder, "$.gpkg"),
    extraction_expression="b1>=0.1",
)

merge_vectors(
    input_vectors=(mask_vegetation_folder, "*.gpkg"),
    merged_vector_path=os.path.join(working_directory, "VegetationMasks.gpkg"),
    method="keep",
    create_name_attribute=("image", ", "),
)

# %% Global matching

global_regression(
    input_images=(masked_folder, "*.tif"),
    output_images=(global_folder, "$_GlobalMatch.tif"),
    vector_mask=(
        "exclude",
        os.path.join(working_directory, "VegetationMasks.gpkg"),
        "image",
    ),  # Use unique mask per image
    window_size=window_size,
    save_as_cog=True,  # Save output as a Cloud Optimized GeoTIFF
    debug_logs=True,
)

# %% Local matching

local_block_adjustment(
    input_images=(global_folder, "*.tif"),
    output_images=(local_folder, "$_LocalMatch.tif"),
    number_of_blocks=100,
    window_size=window_size,
    vector_mask=(
        "exclude",
        os.path.join(working_directory, "VegetationMasks.gpkg"),
        "image",
    ),
    save_as_cog=True,
    debug_logs=True,
    save_block_maps=(
        os.path.join(local_folder, "BlockMaps", "ReferenceBLockMap.tif"),
        os.path.join(local_folder, "BlockMaps", "$_LocalBlockMap.tif"),
    ),
)

# %% Pre-coded quick Statistics

# Compare image spectral profiles
compare_image_spectral_profiles(
    input_image_dict={
        os.path.splitext(os.path.basename(p))[0]: p
        for p in search_paths(local_folder, "*.tif")
    },
    output_figure_path=os.path.join(
        stats_folder, "LocalMatch_CompareImageSpectralProfiles.png"
    ),
    title="Global to Local Match Comparison of Image Spectral Profiles",
    xlabel="Band",
    ylabel="Reflectance(0-10,000)",
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
    os.path.join(stats_folder, "LocalMatch_CompareImageSpectralProfilesPairs.png"),
    title="Global to Local Match Comparison of Image Spectral Profiles Pairs",
    xlabel="Band",
    ylabel="Reflectance(0-10,000)",
)

# Compare spatial spectral difference band average
input_paths = search_paths(input_folder, "*.tif")
local_paths = search_paths(local_folder, "*.tif")
before_path, after_path = next(zip(sorted(input_paths), sorted(local_paths)))

compare_spatial_spectral_difference_band_average(
    input_images=[before_path, after_path],
    output_image_path=os.path.join(
        stats_folder, "LocalMatch_CompareSpatialSpectralDifferenceBandAverage.png"
    ),
    title="Global to Local Match Comparison of Spatial Spectral Difference Band Average",
    diff_label="Reflectance Difference (0–10,000)",
    subtitle=f"Image: {os.path.splitext(os.path.basename(before_path))[0]}",
)
