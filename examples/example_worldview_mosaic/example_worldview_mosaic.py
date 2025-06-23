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
num_image_workers = 3
num_window_workers = 5
debug_mode = False

# %% Global matching

global_regression(
    input_images=input_folder, # Automatically searches for all *.tif files if passed this way
    output_images=global_folder,
    debug_logs=debug_mode,
    window_size=window_size,
    image_parallel_workers=("process", num_image_workers),
    window_parallel_workers=("process", num_window_workers),
    # specify_model_images=("include", ['Worldview_2016-09-22']), # Global matching all input images to the spectral profile of any number of specified images (regression will still be based on overlapping areas, however, only the *included* images statistics will influence the solution)
    # custom_mean_factor=3, # Default is 1; 3 often works better to 'move' the spectral mean of images closer together (applied when creating model)
    custom_std_factor=3,
    save_adjustments=os.path.join(
        global_folder, "GlobalAdjustments.json"
    ),  # Start from precomputed statistics for images whole and overlap stats
    # load_adjustments=os.path.join(global_folder, "GlobalAdjustments.json"), # Load Statistics
)

# %% Local matching
reference_map_path = os.path.join(local_folder, "ReferenceBlockMap", "ReferenceBlockMap.tif")
local_maps_path = os.path.join(local_folder, "LocalBlockMap", "$_LocalBlockMap.tif")
searched_paths = search_paths(os.path.join(local_folder, "LocalBlockMap", "*.tif"))

local_block_adjustment(
    input_images=global_folder,
    output_images=local_folder,
    debug_logs=debug_mode,
    window_size=window_size,
    image_parallel_workers=("process", num_image_workers),
    window_parallel_workers=("process", num_window_workers),
    number_of_blocks="coefficient_of_variation",  # Target number of blocks
    # override_bounds_canvas_coords = (193011.1444011169369332, 2184419.3597142999060452, 205679.2836037494416814, 2198309.8632259583100677), # Local match with a larger canvas than images bounds (perhaps to anticipate adding additional imagery so you don't have to recalculate local block maps each rematch)
    save_block_maps=(reference_map_path, local_maps_path),
    # load_block_maps=(reference_map_path, searched_paths), # Local match from saved block maps (this code just passes in local maps, but if a reference map is passed in, it will match images to the reference map without recomputing it)
)

# %% Align rasters

align_rasters(
    input_images=local_folder,
    output_images=aligned_folder,
    tap=True,
    resolution="lowest",
    debug_logs=debug_mode,
    window_size=window_size,
    image_parallel_workers=("process", num_image_workers),
    window_parallel_workers=("process", num_window_workers),
)

# %% Generate voronoi center seamlines

voronoi_center_seamline(
    input_images=aligned_folder,
    output_mask=os.path.join(working_directory, "ImageMasks.gpkg"),
    image_field_name="image",
    debug_logs=debug_mode,
    debug_vectors_path=os.path.join(working_directory, "DebugVectors.gpkg"),
)

# %% Clip

mask_rasters(
    input_images=aligned_folder,
    output_images=clipped_folder,
    vector_mask=("include", os.path.join(working_directory, "ImageMasks.gpkg"), "image"),
    debug_logs=debug_mode,
    window_size=window_size,
    image_parallel_workers=("process", num_image_workers),
    window_parallel_workers=("process", num_window_workers),
)

# %% Merge rasters

merge_rasters(
    input_images=clipped_folder,
    output_image_path=os.path.join(working_directory, "MergedImage.tif"),
    debug_logs=debug_mode,
    window_size=window_size,
    image_parallel_workers=("process", num_image_workers),
    window_parallel_workers=("process", num_window_workers),
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
    input_images_1=search_paths(os.path.join(input_folder, "*.tif")),
    input_images_2=search_paths(os.path.join(local_folder, "*.tif")),
    output_figure_path=os.path.join(stats_folder, "CompareBeforeAfterAllImages.png"),
    image_names=[os.path.splitext(os.path.basename(p))[0] for p in search_paths(os.path.join(input_folder, "*.tif"))],
    title="Comparison of Before to After of all Images",
    ylabel_1="Before",
    ylabel_2="After",
)