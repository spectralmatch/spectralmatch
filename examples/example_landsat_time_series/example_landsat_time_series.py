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
debug_mode = False

# %% Create cloud masks

create_cloud_mask_with_omnicloudmask(
    input_images=input_folder,
    output_images=mask_cloud_folder,
    red_band_index=5,
    green_band_index=3,
    nir_band_index=7,
    debug_logs=debug_mode,
    image_parallel_workers=("thread", num_image_workers),
    omnicloud_kwargs={"patch_size": 200, "patch_overlap": 100},
)

process_raster_values_to_vector_polygons(
    input_images=mask_cloud_folder,
    output_vectors=mask_cloud_folder,
    extraction_expression="b1==1",
    value_mapping={0: None, 1: 1, 2: 1, 3: 1},
    polygon_buffer=50,
    image_parallel_workers=("process", num_image_workers),
    window_parallel_workers=("process", num_window_workers),
    window_size=window_size,
    debug_logs=debug_mode,
)

merge_vectors(
    input_vectors=mask_cloud_folder,
    merged_vector_path=os.path.join(working_directory, "CloudMasks.gpkg"),
    method="keep",
    create_name_attribute=("image", ", "),
    debug_logs=debug_mode,
)

# %% Use cloud masks

mask_rasters(
    input_images=input_folder,
    output_images=os.path.join(masked_folder, "$_CloudMasked.tif"),
    vector_mask=(
        "exclude",
        os.path.join(working_directory, "CloudMasks.gpkg"),
        "image",
    ),
    debug_logs=debug_mode,
)

# %% Create vegetation mask for isolated analysis of vegetation. This will be used to mask statistics for adjustment model not to directly clip images. This is just a simple example of creating PIFs based on NDVI values, for a more robust methodology use other techniques to create a better mask vector file.

create_ndvi_raster(
    input_images=input_folder,
    output_images=mask_vegetation_folder,
    nir_band_index=5,
    red_band_index=4,
    debug_logs=debug_mode,
)

process_raster_values_to_vector_polygons(
    input_images=mask_vegetation_folder,
    output_vectors=mask_vegetation_folder,
    extraction_expression="b1>=0.1",
    debug_logs=debug_mode,
)

merge_vectors(
    input_vectors=mask_vegetation_folder,
    merged_vector_path=os.path.join(working_directory, "VegetationMasks.gpkg"),
    method="keep",
    create_name_attribute=("image", ", "),
    debug_logs=debug_mode,
)

# %% Global matching

global_regression(
    input_images=masked_folder,
    output_images=global_folder,
    vector_mask=(
        "exclude",
        os.path.join(working_directory, "VegetationMasks.gpkg"),
        "image",
    ),  # Use unique mask per image
    window_size=window_size,
    save_as_cog=True,  # Save output as a Cloud Optimized GeoTIFF
    debug_logs=debug_mode,
)

# %% Local matching

local_block_adjustment(
    input_images=global_folder,
    output_images=local_folder,
    number_of_blocks=100,
    window_size=window_size,
    vector_mask=(
        "exclude",
        os.path.join(working_directory, "VegetationMasks.gpkg"),
        "image",
    ),
    save_as_cog=True,
    debug_logs=debug_mode,
    save_block_maps=(
        os.path.join(local_folder, "BlockMaps", "ReferenceBLockMap.tif"),
        os.path.join(local_folder, "BlockMaps", "$_LocalBlockMap.tif"),
    ),
)

# %% Pre-coded quick Statistics

# Compare image spectral profiles pairs
image_pairs = {
    os.path.splitext(os.path.basename(b))[0]: [b, a]
    for b, a in zip(search_paths(os.path.join(input_folder, "*.tif")), search_paths(os.path.join(local_folder, "*.tif")))
}

compare_image_spectral_profiles_pairs(
    image_pairs,
    os.path.join(stats_folder, "ImageSpectralProfilesPairs.png"),
    title="Comparison of Image Spectral Profile Pairs",
    xlabel="Band",
    ylabel="Reflectance",
)

# Compare spatial spectral difference band average
before_paths, after_paths = zip(*zip(search_paths(os.path.join(input_folder, "*.tif")), search_paths(os.path.join(local_folder, "*.tif"))))

for before_path, after_path in zip(before_paths, after_paths):
    compare_spatial_spectral_difference_band_average(
        input_images=[before_path, after_path],
        output_figure_path=os.path.join(
            stats_folder,
            f"PixelChange_{os.path.splitext(os.path.basename(before_path))[0]}.png"
        ),
        title="Input to Output Comparison of Pixel Change",
        diff_label="Pixel Difference",
        subtitle=f"Image: {os.path.splitext(os.path.basename(before_path))[0]}",
    )

# Compare before after all images
compare_before_after_all_images(
    input_images_1=search_paths(os.path.join(masked_folder, "*.tif")),
    input_images_2=search_paths(os.path.join(local_folder, "*.tif")),
    output_figure_path=os.path.join(stats_folder, "CompareBeforeAfterAllImages.png"),
    image_names=[os.path.splitext(os.path.basename(p))[0] for p in search_paths(os.path.join(input_folder, "*.tif"))],
    title="Comparison of Before to After of all Images",
    ylabel_1="Before",
    ylabel_2="After",
)