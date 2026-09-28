# %% Worldview Mosaic
# This file demonstrates how to preprocess Worldview3 imagery into a mosaic using spectralmatch.
# Starting from two overlapping Worldview3 images in reflectance, the process includes joint coregistration, global matching, local matching, starting from saved block maps (optional for demonstration purposes), polygonizing valid-pixel footprints, smoothing footprint polygons, generating seamlines, and merging images, and before vs after statistics.
# This script is set up to perform matching on all .tif files from a folder within the working directory called "Input" e.g. working_directory/Input/*.tif. The easiest way to process your own imagery is to move it inside that folder or change the working_directory to another folder with this structure, alternatively, you can pass in custom lists of image paths.

# %% Setup
import os
from spectralmatch import (
    align_rasters,
    compare_before_after_all_images,
    compare_image_spectral_profiles_pairs,
    compare_spatial_spectral_difference_band_average,
    create_footprints,
    postprocess_footprints,
    joint_coregistration,
    global_regression,
    local_block_adjustment,
    mask_rasters,
    markov_triangles,
    merge_rasters,
    search_paths,
    voronoi_center_seamline,
    weighted_seamline,
)

# Important: If this does not automatically find the correct CWD, manually copy the path to the /data_worldview folder
working_directory = os.path.join(os.getcwd(), "data_worldview")
print(working_directory)

input_folder = os.path.join(working_directory, "Input")
coregistration_folder = os.path.join(working_directory, "Coregistration")
global_folder = os.path.join(working_directory, "GlobalMatch")
local_folder = os.path.join(working_directory, "LocalMatch")
clipped_folder = os.path.join(working_directory, "Clipped")
stats_folder = os.path.join(working_directory, "Stats")
footprints_path = os.path.join(working_directory, "Footprints.gpkg")
metadata_csv_path = os.path.join(input_folder, "ImageMetadata.csv")
smoothed_footprints_path = os.path.join(working_directory, "SmoothedFootprints.gpkg")
seamlines_path = os.path.join(working_directory, "ImageMasks.gpkg")

window_size = 1024
image_threads = 3 # Dask: None | int
io_threads = 3
tile_threads = 3
debug_mode = True

concurrent_processing_backend = "process_pool"  # process_pool | dask
dask_scheduler = None  # None | ("file", f"/tmp/spectralmatch-dask-1.json") | ("address", "tcp://scheduler:8786")

# %% Joint coregistration
joint_coregistration(
    input_images=input_folder,
    output_images=coregistration_folder,
    tap=True,
    resolution="highest",
    debug_logs=debug_mode,
    window_size=window_size,
    image_threads=image_threads,
    io_threads=io_threads,
    tile_threads=tile_threads,
    concurrent_processing_backend=concurrent_processing_backend,
    dask_scheduler=dask_scheduler,
)

# %% Align rasters (or skip this step if images are already aligned or this is set in the joint_coregistration step)

# align_rasters(
#     input_images=input_folder,
#     output_images=coregistration_folder,
#     tap=True,
#     resolution="highest",
#     debug_logs=debug_mode,
#     window_size=window_size,
#     image_threads=image_threads,
#     io_threads=io_threads,
#     tile_threads=tile_threads,
#     concurrent_processing_backend=concurrent_processing_backend,
#     dask_scheduler=dask_scheduler,
# )

# %% Global matching
inz_output_path = os.path.join(global_folder, "INZ", "$_to_$_INZ.tif")

global_regression(
    input_images=coregistration_folder, # Automatically searches for all *.tif files if passed this way
    output_images=global_folder,
    debug_logs=debug_mode,
    window_size=window_size,
    image_threads=image_threads,
    io_threads=io_threads,
    tile_threads=tile_threads,
    concurrent_processing_backend=concurrent_processing_backend,
    dask_scheduler=dask_scheduler,
    save_as_cog=True,
    # custom_nodata_value=-9999,
    pif_method="flood_from_match_points",
    estimate_stats=True,
    # specify_model_images=("include", ['Worldview_2016-09-22']), # Global matching all input images to the spectral profile of any number of specified images (regression will still be based on overlapping areas, however, only the *included* images statistics will influence the solution)
    # custom_mean_factor=3, # Default is 1; 3 often works better to 'move' the spectral mean of images closer together (applied when creating model)
    # custom_std_factor=3,
    save_adjustments=os.path.join(
        global_folder, "GlobalAdjustments.json"
    ),  # Start from precomputed statistics for images whole and overlap stats
    pif_save_inz=inz_output_path,  # Saves one INZ raster per overlap pair; first $ = main/sensed image name, second $ = reference image name
    # load_adjustments=os.path.join(global_folder, "GlobalAdjustments.json"), # Load Statistics
)

# %% Local matching
reference_map_path = os.path.join(local_folder, "ReferenceBlockMap", "ReferenceBlockMap.tif")
local_maps_path = os.path.join(local_folder, "LocalBlockMap", "$_LocalBlockMap.tif")
# Only search for saved maps when enabling load_block_maps below.
# searched_paths = search_paths(os.path.join(local_folder, "LocalBlockMap", "*.tif"))

local_block_adjustment(
    input_images=global_folder,
    output_images=local_folder,
    debug_logs=debug_mode,
    window_size=window_size,
    image_threads=image_threads,
    io_threads=io_threads,
    tile_threads=tile_threads,
    concurrent_processing_backend=concurrent_processing_backend,
    dask_scheduler=dask_scheduler,
    save_as_cog=True,
    # custom_nodata_value=-9999,
    correction_method="offset",
    number_of_blocks=50,  # Target number of blocks
    # override_bounds_canvas_coords = (193011.1444011169369332, 2184419.3597142999060452, 205679.2836037494416814, 2198309.8632259583100677), # Local match with a larger canvas than images bounds (perhaps to anticipate adding additional imagery so you don't have to recalculate local block maps each rematch)
    save_block_maps=(reference_map_path, local_maps_path),
    # load_block_maps=(reference_map_path, searched_paths), # Local match from saved block maps (this code just passes in local maps, but if a reference map is passed in, it will match images to the reference map without recomputing it)
)

# %% Polygonize valid-pixel footprints
# Use the locally matched images so footprint identifiers match the images clipped below.
# GDAL's validity mask defines valid pixels. Apply cloud masks as nodata or a raster validity mask before this step; cloud class values alone do not mark pixels invalid.
# ImageMetadata.csv contains illustrative scores (0–100), cloud percentages and sun/off-nadir angles in degrees; replace them with measured metadata for your images.
# Its image column includes the full LocalMatch basename, e.g. Worldview_20160922_Coregistered_Global_Local; .tif is optional.
create_footprints(
    input_images=local_folder,
    output_polygons=footprints_path,
    metadata_csv=metadata_csv_path,
    metadata_image_field_name="image",  # CSV values must contain the full current basename, including processing suffixes.
    image_field_name="image",
    output_layer="footprints",
    band=1,
    eight_connected=True,
    image_threads=image_threads,
    concurrent_processing_backend=concurrent_processing_backend,
    dask_scheduler=dask_scheduler,
    debug_logs=debug_mode,
)

# %% Cut edge holes and smooth footprints
# Input polygons must use a suitable projected CRS; distances below use its linear units (metres for the example imagery).
# These are starting values to tune for your imagery. Cuts use the original outer ring to select holes, and processing cannot expand the valid area.
postprocess_footprints(
    input_polygons=footprints_path,
    output_polygons=smoothed_footprints_path,
    input_layer="footprints",
    output_layer="footprints",
    hole_edge_distance=800,  # Maximum hole-to-edge distance; 0 disables edge-hole cuts.
    hole_to_hole_distance=800,  # Maximum distance between holes; 0 disables hole-pair cuts.
    hole_relative_edge_distance=None,  # Optionally also limit distance / sqrt(hole_area / pi).
    hole_cut_width="maximum_inscribed_circle",  # Inscribed diameter; pairs use the smaller width. Also accepts a positive integer or "hole_size".
    hole_cut_method="corridor",  # "corridor" subtracts a shortest connection; "buffer" expands the selected hole to the edge.
    simplify_smoothing_radius=200,  # Erode then dilate within the cut polygon; 0 disables smoothing.
    simplify_tolerance=120,  # Maximum inward-shortcut deviation; 0 disables simplification.
    simplify_area_weight=0.5,  # Larger values favor retaining area over reducing perimeter.
    filter_area_size=None,  # Optional minimum component area in squared CRS units.
    filter_area_rank=2,  # Keep the largest component per feature; negative N keeps the smallest abs(N), 0 or None keeps all.
    image_threads=image_threads,
    concurrent_processing_backend=concurrent_processing_backend,
    dask_scheduler=dask_scheduler,
    debug_logs=debug_mode,
)

# %% Generate seamlines

# Option 1: Voronoi center seamlines from the smoothed footprints

# voronoi_center_seamline(
#     input_polygons=smoothed_footprints_path,
#     input_layer="footprints",
#     output_mask=seamlines_path,
#     image_field_name="image",
#     debug_logs=debug_mode,
#     debug_vectors_path=os.path.join(working_directory, "DebugVectors.gpkg"),
# )

# Option 2: Rank the smoothed footprints using CSV attributes, which postprocessing preserves. Adjust the example coefficients to your priorities.
# weighted_seamline(
#     input_polygons=smoothed_footprints_path,
#     output_mask=seamlines_path,
#     input_layer="footprints",
#     image_field_name="image",
#     rank_function="{quality_score} - {cloud_cover} + 0.1 * {sun_elevation} - 0.2 * {off_nadir_angle}",
#     rank_descending=True,
#     debug_logs=debug_mode,
# )

# Option 3: Optimize triangle seamlines using image texture, inter-image differences, and image level metadata.
# GSDs are positive integer CRS distances (metres here) or "native", independent of mesh_spacing.
# Omit input_polygons to generate valid-data footprints automatically.
# Single GSDs/bands use [value]; multiple values require a reducer, e.g. {"average": ["native", 4]}.
markov_triangles(
    input_images=local_folder,
    input_polygons=smoothed_footprints_path,
    input_layer="footprints",
    output_mask=seamlines_path,
    image_field_name="image",
    mesh_spacing=10,
    edge_variables_weights_gsds_bands=[
        ["laplacian_difference", 0.7, ["native"], [1]],
        ["laplacian_magnitude", 1.0, ["native"], {"largest": [1, 2, 3]}],
        # ["/path/to/costs.tif", 0.5, {"largest": [2, 8]}, [1]],
    ],
    # image_rank_function="{quality_score} - {cloud_cover} + 0.1 * {sun_elevation} - 0.2 * {off_nadir_angle}",
    # image_rank_descending=True,
    # image_quality_weight=1.0,
    # foreground_path="/path/to/protected_objects.gpkg",
    # foreground_layer="objects",
    image_threads=image_threads,
    concurrent_processing_backend=concurrent_processing_backend,
    dask_scheduler=dask_scheduler,
    debug_logs=debug_mode,
)

# %% Clip

mask_rasters(
    input_images=local_folder,
    output_images=clipped_folder,
    vector_mask=("include", seamlines_path, "image"),
    debug_logs=debug_mode,
    window_size=window_size,
    image_threads=image_threads,
    io_threads=io_threads,
    tile_threads=tile_threads,
    concurrent_processing_backend=concurrent_processing_backend,
    dask_scheduler=dask_scheduler,
)

# %% Merge rasters

merge_rasters(
    input_images=clipped_folder,
    # output_image_path=os.path.join(working_directory, "MergedImage.tif"), # Use this for single tif output and set output_tiles=False
    output_image_path=os.path.join(working_directory, "MergedImage"),
    debug_logs=debug_mode,
    window_size=window_size,
    io_threads=io_threads,
    tile_threads=tile_threads,
    build_overviews=True,
    output_tiles=True
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
