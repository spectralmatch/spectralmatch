import numpy as np
import pytest
import geopandas as gpd

from osgeo import gdal, osr
from shapely.geometry import box


_DTYPE_MAP = {
    "uint8": gdal.GDT_Byte,
    "byte": gdal.GDT_Byte,
    "uint16": gdal.GDT_UInt16,
    "int16": gdal.GDT_Int16,
    "uint32": gdal.GDT_UInt32,
    "int32": gdal.GDT_Int32,
    "float32": gdal.GDT_Float32,
    "float64": gdal.GDT_Float64,
}


def create_dummy_raster(
    path,
    width=10,
    height=10,
    count=3,
    dtype="uint8",
    nodata=0,
    transform=None,
    crs="EPSG:4326",
    fill_value=100,
    band_data=None,
):
    if transform is None:
        transform = (0, 1, 0, 10, 0, -1)

    if band_data is None:
        data = np.full((count, height, width), fill_value, dtype=dtype)
    else:
        data = np.asarray(band_data)
        if data.ndim == 2:
            data = data[np.newaxis, ...]
        count, height, width = data.shape
        dtype = str(data.dtype)

    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(
        str(path),
        width,
        height,
        count,
        _DTYPE_MAP[str(dtype).lower()],
    )
    ds.SetGeoTransform(transform)

    if crs:
        srs = osr.SpatialReference()
        srs.SetFromUserInput(crs)
        ds.SetProjection(srs.ExportToWkt())

    for band_index in range(count):
        band = ds.GetRasterBand(band_index + 1)
        band.WriteArray(data[band_index])
        if nodata is not None:
            band.SetNoDataValue(nodata)

    ds.FlushCache()
    ds = None


def create_dummy_vector(path, bounds=(0, 0, 5, 5), crs="EPSG:4326"):
    gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[box(*bounds)], crs=crs)
    gdf.to_file(path)
