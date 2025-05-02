from spectralmatch import (
    create_cloud_mask_with_omnicloudmask,
    post_process_raster_cloud_mask_to_vector,
)

from spectralmatch import write_vector

create_cloud_mask_with_omnicloudmask(
    "input_image_path.tif",
    5,
    3,
    8,
    "output_mask>path.tif",
    down_sample_m=10
)
write_vector(
    post_process_raster_cloud_mask_to_vector(
        "input_image_path.tif",
        None,
        {1: 50},
        {0: 0, 1: 1, 2: 1, 3: 1}
    ),
    "output_vector_path.tif",
)