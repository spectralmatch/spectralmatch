`mask_rasters` defaults to `include_touched_pixels=True`, matching the pipeline.
This retains pixels touched by cutline boundaries and helps prevent sparse gaps
when adjacent clipped images use different raster grids. Set
`include_touched_pixels=False` for strict pixel-center masking. Original source
NoData remains excluded.

Set `seperate_multipolygons=True` in `mask_rasters` to export disconnected
retained mask polygons as independent cropped rasters. Output bases such as
`scene_Mask.tif` become `scene_Mask_1.tif`, `scene_Mask_2.tif`, etc.; numbering
restarts for each input and sorts by polygon bounds. The default is `False`.
Overlapping mask features are dissolved, holes remain masked, and exclude masks
split the remaining raster area. Components outside the raster are omitted.

The returned flat list contains the actual numbered paths; the unnumbered base
is not written. `resume_from_outputs` checks each numbered file independently.
Use `mask_rasters_seperate_multipolygons=True` with `pipeline`.

```python
from spectralmatch import mask_rasters

pieces = mask_rasters(
    input_images=["scene.tif"],
    output_images=["scene_Mask.tif"],
    vector_mask=("include", "coverage.gpkg", "image"),
    seperate_multipolygons=True,
)
```

::: spectralmatch.utils
