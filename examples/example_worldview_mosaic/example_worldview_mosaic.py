# %% Worldview Mosaic
# This file demonstrates how to preprocess Worldview3 imagery into a mosaic using spectralmatch.
# Starting from two overlapping Worldview3 images in reflectance, the process includes global matching, local matching, starting from saved block maps (optional for demonstration purposes), generating seamlines, and marging images, and before vs after statistics.
# This script is set up to perform matching on all .tif files from a folder within the working directory called "Input" e.g. working_directory/Input/*.tif. The easiest way to process your own imagery is to move it inside that folder or change the working_directory to another folder with this structure, alternatively, you can pass in custom lists of image paths.

# %% Setup
import os

from fiona.env import local
from spectralmatch import *

# Important: If this does not automatically find the correct CWD, manually copy the path to the /data_worldview3 folder
working_directory = '/Users/kanoalindiwe/Downloads/Projects/spectralmatch/docs/examples/data_worldview3'
print(working_directory)

input_folder = os.path.join(working_directory, "Input")
global_folder = os.path.join(working_directory, "GlobalMatch")
local_folder = os.path.join(working_directory, "LocalMatch")
aligned_folder = os.path.join(working_directory, "Aligned")
clipped_folder = os.path.join(working_directory, "Clipped")
stats_folder = os.path.join(working_directory, "Stats")

window_size = 128
num_workers = 5

# %% Global matching
saved_adjustments_path = os.path.join(global_folder, "GlobalAdjustments.json")


global_regression(
    (input_folder, "*.tif"),
    (global_folder, "$_Global.tif"),
    custom_mean_factor = 3, # Default is 1; 3 often works better to 'move' the spectral mean of images closer together
    debug_logs=True,
    window_size=window_size,
    parallel_workers=num_workers,
    )

# %% (OPTIONAL) Global matching all input images to the spectral profile of any number of specified images (regression will still be based on overlapping areas, however, only the *included* images statistics will influence the solution)
new_global_folder = os.path.join(working_directory, "GlobalMatch_New")
saved_adjustments_path = os.path.join(new_global_folder, "GlobalAdjustments.json")


global_regression(
    (input_folder, "*.tif"),
    (new_global_folder, "$_Global.tif"),
    custom_mean_factor = 3, # Default is 1; 3 often works better to 'move' the spectral mean of images closer together
    debug_logs=True,
    window_size=window_size,
    parallel_workers=num_workers,
    specify_model_images=("include", ['worldview3_example_image_right']),
    save_adjustments=saved_adjustments_path,
    )

# %% (OPTIONAL) Global matching starting from precomputed statistics for images whole and overlap stats
new_global_folder = os.path.join(working_directory, "GlobalMatch_New")
saved_adjustments_path = os.path.join(new_global_folder, "GlobalAdjustments.json")


global_regression(
    (input_folder, "*.tif"),
    (new_global_folder, "$_Global.tif"),
    custom_mean_factor = 3, # Default is 1; 3 often works better to 'move' the spectral mean of images closer together
    debug_logs=True,
    window_size=window_size,
    parallel_workers=num_workers,
    load_adjustments=saved_adjustments_path,
    )

# %% Local matching
local_block_adjustment(
    (global_folder, "*.tif"),
    (local_folder, "$_Local.tif"),
    number_of_blocks=100,
    debug_logs=True,
    window_size=window_size,
    parallel_workers=num_workers,
    )

# %% (OPTIONAL) Local match with a larger canvas than images bounds (perhaps to anticipate adding additional imagery so you don't have to recalculate local block maps each rematch)
new_local_folder = os.path.join(working_directory, "LocalMatch_New")
reference_map_path = os.path.join(new_local_folder, "ReferenceBlockMap", "ReferenceBlockMap.tif")
local_maps_path = os.path.join(new_local_folder, "LocalBlockMap", "$_LocalBlockMap.tif")

local_block_adjustment(
    (global_folder, "*.tif"),
    (new_local_folder, "$_Local.tif"),
    number_of_blocks=(30,30),
    debug_logs=True,
    window_size=window_size,
    parallel_workers=num_workers,
    override_bounds_canvas_coords = (193011.1444011169369332, 2184419.3597142999060452, 205679.2836037494416814, 2198309.8632259583100677),
    save_block_maps=(reference_map_path, local_maps_path),
    )

# %% (OPTIONAL) Local match from saved block maps (this code just passes in local maps, but if a reference map is passed in, it will match images to the reference map without recomputing it)

old_local_folder = os.path.join(working_directory, "LocalMatch")
new_local_folder = os.path.join(working_directory, "LocalMatch_New")
saved_reference_block_path = os.path.join(old_local_folder, "ReferenceBlockMap", "ReferenceBlockMap.tif")
saved_local_block_paths = [os.path.join(os.path.join(new_local_folder, "LocalBlockMap"), f) for f in os.listdir(os.path.join(new_local_folder, "LocalBlockMap")) if f.lower().endswith(".tif")]

local_block_adjustment(
    (global_folder, "*.tif"),
    (new_local_folder, "$_Local.tif"),
    number_of_blocks=100,
    debug_logs=True,
    window_size=window_size,
    parallel_workers=num_workers,
    load_block_maps=(None, saved_local_block_paths)
    )

#%% Align rasters
input_image_paths = search_paths(local_folder, "*.tif")
output_clipped_image_paths = create_paths(aligned_folder, "$_Aligned.tif", input_image_paths)

process_rasters(
    input_image_paths,
    output_clipped_image_paths,
    tap=True,
    resolution='highest',
    debug_logs=True,
    window_size=window_size,
    )

# %% Generate seamlines
input_image_paths = search_paths(aligned_folder, "*.tif")
output_vector_mask = os.path.join(working_directory, "ImageClips.gpkg")

voronoi_center_seamline(
    input_image_paths,
    output_vector_mask,
    image_field_name='image',
    debug_logs=True,
    )

# %% Clip and merge
input_image_paths = search_paths(aligned_folder, "*.tif")
output_clipped_image_paths = create_paths(clipped_folder, "$_Clipped.tif", input_image_paths)
input_vector_mask_path = os.path.join(working_directory, "ImageClips.gpkg")
output_merged_image_path = os.path.join(working_directory, "MergedImage.tif")

process_rasters(
    input_image_paths,
    output_clipped_image_paths,
    vector_mask_path=input_vector_mask_path,
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
