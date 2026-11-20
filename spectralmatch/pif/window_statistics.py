"""Batched bounded statistics with the sampling layout of expression VRTs.

GDAL's non-overview ComputeStatistics samples every floor(sqrt(block_count))
block (with its square-grid adjustment). Our expression VRTs have no overviews;
query their block size instead of assuming the source TIFF's block layout.
See gcore/gdalrasterband.cpp, GDALRasterBand::ComputeStatistics.
"""
import math
import numpy as np
from osgeo import gdal

NODATA = np.float32(-999999999)


def statistics_windows(width, height, approximate):
    prototype = gdal.Open(f'<VRTDataset rasterXSize="{width}" rasterYSize="{height}">'
                         '<VRTRasterBand dataType="Float32" band="1" subClass="VRTDerivedRasterBand">'
                         '<PixelFunctionType>expression</PixelFunctionType>'
                         '<PixelFunctionArguments expression="0"/>'
                         '</VRTRasterBand></VRTDataset>')
    bx, by = prototype.GetRasterBand(1).GetBlockSize()
    prototype = None
    nx, ny = math.ceil(width/bx), math.ceil(height/by)
    stride = max(1, int(math.sqrt(nx*ny))) if approximate else 1
    if approximate and stride == nx and nx > 1:
        stride += 1
    for index in range(0, nx*ny, stride):
        x, y = index % nx * bx, index // nx * by
        yield x, y, min(bx, width-x), min(by, height-y)


def batched_statistics(width, height, channels, reader, approximate):
    """Reader returns channels x height x width Float32 masked expressions.

    Only channels missed by approximate sampling retry the exact population.
    Blockwise pooled moments avoid a full-overlap allocation and cancellation.
    """
    counts = np.zeros(channels, dtype=np.int64)
    means = np.zeros(channels, dtype=np.float64)
    moments = np.zeros(channels, dtype=np.float64)

    def scan(indices, approx):
        for window in statistics_windows(width, height, approx):
            data = reader(window, indices)
            if data is None:
                continue
            for slot, channel in enumerate(indices):
                values = data[slot].ravel()
                values = values[(values != NODATA) & ~np.isnan(values)].astype(np.float64)
                n = values.size
                if not n:
                    continue
                mean = values.mean()
                m2 = np.sum((values-mean)**2)
                total = counts[channel]+n
                delta = mean-means[channel]
                moments[channel] += m2 + delta*delta*counts[channel]*n/total
                means[channel] += delta*n/total
                counts[channel] = total
    scan(list(range(channels)), approximate)
    missing = np.flatnonzero(counts == 0).tolist()
    if missing and approximate:
        scan(missing, False)
    if np.any(counts == 0):
        raise RuntimeError('Failed to compute statistics, no valid pixels found in sampling.')
    return means, np.sqrt(moments/counts)


def read_bands(dataset, bands, window):
    values = dataset.ReadAsArray(*window, band_list=bands, buf_type=gdal.GDT_Float32)
    if values is None:
        raise RuntimeError('Failed to read source raster block.')
    return values.reshape(len(bands), window[3], window[2])


def difference_statistics(reference, sensed, mask, bands, width, height, approximate):
    def read(window, indices):
        valid = mask.GetRasterBand(1).ReadAsArray(*window) > 0
        if not valid.any():
            return None
        selected = [i+1 for i in indices]
        ref = read_bands(reference, selected, window)
        sen = read_bands(sensed, selected, window)
        # Expression VRT inputs and output are Float32; arithmetic is Float64.
        result = (sen.astype(np.float64)-ref).astype(np.float32)
        result[:, ~valid] = NODATA
        return result
    return batched_statistics(width, height, bands, read, approximate)


def selected_statistics(reference, sensed, mask, bands, width, height, approximate):
    def read(window, indices):
        valid = mask.GetRasterBand(1).ReadAsArray(*window) > 0
        if not valid.any():
            return None
        values = np.empty((len(indices), window[3], window[2]), dtype=np.float32)
        for side, dataset in enumerate((reference, sensed)):
            slots = [j for j, channel in enumerate(indices) if channel//bands == side]
            if slots:
                selected = [indices[j] % bands + 1 for j in slots]
                values[slots] = read_bands(dataset, selected, window)
        values[:, ~valid] = NODATA
        return values
    return batched_statistics(width, height, 2*bands, read, approximate)
