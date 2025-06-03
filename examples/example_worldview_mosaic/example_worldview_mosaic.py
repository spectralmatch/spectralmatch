# %% Worldview Mosaic
# This file demonstrates how to preprocess Worldview3 imagery into a mosaic using spectralmatch.
# Starting from two overlapping Worldview3 images in reflectance, the process includes global matching, local matching, starting from saved block maps (optional for demonstration purposes), generating seamlines, and marging images, and before vs after statistics.
# This script is set up to perform matching on all .tif files from a folder within the working directory called "Input" e.g. working_directory/Input/*.tif. The easiest way to process your own imagery is to move it inside that folder or change the working_directory to another folder with this structure, alternatively, you can pass in custom lists of image paths.

# %% Setup
import os
from spectralmatch import *

# Important: If this does not automatically find the correct CWD, manually copy the path to the /data_worldview folder
working_directory = os.path.join(os.getcwd(), "data_worldview")
print(working_directory)

input_folder = os.path.join(working_directory, "Input")
global_folder = os.path.join(working_directory, "GlobalMatch")
local_folder = os.path.join(working_directory, "LocalMatch")
aligned_folder = os.path.join(working_directory, "Aligned")
clipped_folder = os.path.join(working_directory, "Clipped")
stats_folder = os.path.join(working_directory, "Stats")


window_size = 128
num_image_workers = 5
num_window_workers = 5

# %% Global matching

global_regression(
    (input_folder, "*.tif"),
    (global_folder, "$_Global.tif"),
    debug_logs=True,
    window_size=window_size,
    image_parallel_workers=("process", num_image_workers),
    window_parallel_workers=("process", num_window_workers),
    save_as_cog=True,
    # specify_model_images=("include", ['Worldview_2016-09-22']), # Global matching all input images to the spectral profile of any number of specified images (regression will still be based on overlapping areas, however, only the *included* images statistics will influence the solution)
    # custom_mean_factor=3, # Default is 1; 3 often works better to 'move' the spectral mean of images closer together (applied when creating model)
    custom_std_factor=3,
    save_adjustments=os.path.join(global_folder, "GlobalAdjustments.json"), # Start from precomputed statistics for images whole and overlap stats
    # load_adjustments=os.path.join(global_folder, "GlobalAdjustments.json"), # Load Statistics
    )

# %% Local matching
reference_map_path = os.path.join(local_folder, "ReferenceBlockMap", "ReferenceBlockMap.tif")
local_maps_path = os.path.join(local_folder, "LocalBlockMap", "$_LocalBlockMap.tif")
searched_paths = search_paths(os.path.join(local_folder, "LocalBlockMap"), "*.tif")

local_block_adjustment(
    (global_folder, "*.tif"),
    (local_folder, "$_Local.tif"),
    debug_logs=True,
    window_size=window_size,
    image_parallel_workers=("process", num_image_workers),
    window_parallel_workers=("process", num_window_workers),
    save_as_cog=True,
    number_of_blocks="coefficient_of_variation", # Target number of blocks
    # override_bounds_canvas_coords = (193011.1444011169369332, 2184419.3597142999060452, 205679.2836037494416814, 2198309.8632259583100677), # Local match with a larger canvas than images bounds (perhaps to anticipate adding additional imagery so you don't have to recalculate local block maps each rematch)
    save_block_maps=(reference_map_path, local_maps_path),
    # load_block_maps=(reference_map_path, searched_paths), # Local match from saved block maps (this code just passes in local maps, but if a reference map is passed in, it will match images to the reference map without recomputing it)
    )

#%% Align rasters

align_rasters(
    (local_folder, "*.tif"),
    (aligned_folder, "$_Aligned.tif"),
    tap=True,
    resolution='lowest',
    debug_logs=True,
    window_size=window_size,
    image_parallel_workers=("process", num_image_workers),
    window_parallel_workers=("process", num_window_workers),
    )

# %% Generate voronoi center seamlines
output_vector_mask = os.path.join(working_directory, "ImageMasks.gpkg")
debug_vectors_path = os.path.join(working_directory, "DebugVectors.gpkg")

voronoi_center_seamline(
    (aligned_folder, "*.tif"),
    output_vector_mask,
    image_field_name='image',
    debug_logs=True,
    debug_vectors_path=debug_vectors_path,
    )

# %% Clip
input_mask_path = os.path.join(working_directory, "ImageMasks.gpkg")

mask_rasters(
    (aligned_folder, "*.tif"),
    (clipped_folder, "$_Clipped.tif"),
    vector_mask=("include", input_mask_path, "image"),
    debug_logs=True,
    window_size=window_size,
    image_parallel_workers=("process", num_image_workers),
    window_parallel_workers=("process", num_window_workers),
    )

# %% Merge rasters
output_merged_image_path = os.path.join(working_directory, "MergedImage.tif")

merge_rasters(
    (clipped_folder, "*.tif"),
    output_merged_image_path,
    debug_logs=True,
    window_size=window_size,
    image_parallel_workers=("process", num_image_workers),
    window_parallel_workers=("process", num_window_workers),
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
