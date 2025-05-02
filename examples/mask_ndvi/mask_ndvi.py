from spectralmatch import (
    create_ndvi_mask,
    post_process_threshold_to_vector
)

create_ndvi_mask(
    "input_image_path.tif",
    "output_image_path.tif",
    4,
    3,
)
post_process_threshold_to_vector(
    "input_image_path.tif",
    'output_vector_path.gpkg',
    0.2,
    "<=",
)