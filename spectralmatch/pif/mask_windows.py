"""Sparse binary masks: JSON windows with offsets in the mask's pixel grid.

The JSON array begins with grid metadata, followed by nonempty windows. Grids
are bit-packed (base64, big bit order), or constant for rectangular seeds.
Readers and writers stream records; no full-overlap NumPy array is required.
Offsets refer to the overlap grid (or packed grid for matched_windows), whose
transform and CRS are in the header, not necessarily the original image grid.
"""
from spectralmatch.utils_compression import raster_creation_options
import base64
import json
import math

import numpy as np
from osgeo import gdal

MAX_PIXELS = 1024 * 1024


def _windows(width, height):
    """Bounded windows in global row-major order, including extremely wide grids."""
    rows = max(1, MAX_PIXELS // width)
    for y in range(0, height, rows):
        for x in range(0, width, MAX_PIXELS):
            yield x, y, min(width-x, MAX_PIXELS), min(height-y, rows)


def _record(x, y, grid):
    height, width = grid.shape
    return dict(x_offset=x, y_offset=y, width=width, height=height,
                grid=base64.b64encode(np.packbits(grid != 0).tobytes()).decode('ascii'))


def _write(path, width, height, transform, projection, records):
    header = dict(format='spectralmatch.mask_windows', version=1, width=width,
                  height=height, transform=list(transform), projection=projection,
                  grid_encoding='base64-packbits-big', offsets='mask_pixels')
    with open(path, 'w', encoding='utf-8') as stream:
        stream.write('[' + json.dumps(header))
        for record in records:
            stream.write(',\n' + json.dumps(record))
        stream.write(']\n')


def _records(path):
    """Incrementally decode a JSON array, including files reformatted by users."""
    decoder = json.JSONDecoder()
    with open(path, encoding='utf-8') as stream:
        buffer, eof = '', False

        def refill():
            nonlocal buffer, eof
            chunk = stream.read(65536)
            buffer += chunk
            eof = not chunk

        def whitespace():
            nonlocal buffer
            buffer = buffer.lstrip()
            while not buffer and not eof:
                refill()
                buffer = buffer.lstrip()

        whitespace()
        if not buffer.startswith('['):
            raise ValueError('Expected mask-window JSON array.')
        buffer = buffer[1:]
        whitespace()
        if buffer.startswith(']'):
            raise ValueError('Missing mask-window metadata.')
        while True:
            while True:
                try:
                    record, end = decoder.raw_decode(buffer)
                    break
                except json.JSONDecodeError:
                    if eof:
                        raise ValueError('Invalid or incomplete mask-window JSON.') from None
                    refill()
            buffer = buffer[end:]
            yield record
            whitespace()
            if buffer.startswith(']'):
                buffer = buffer[1:]
                whitespace()
                if buffer:
                    raise ValueError('Unexpected data after mask-window JSON.')
                return
            if not buffer.startswith(','):
                raise ValueError('Missing separator in mask-window JSON.')
            buffer = buffer[1:]
            whitespace()


def save_mask_windows(mask_path, json_path):
    """Save a binary raster as streamed, nonempty offset/grid records."""
    ds = gdal.Open(str(mask_path))
    if ds is None:
        raise RuntimeError(f'Could not open mask: {mask_path}')
    band = ds.GetRasterBand(1)

    def records():
        for x, y, w, h in _windows(ds.RasterXSize, ds.RasterYSize):
            grid = band.ReadAsArray(x, y, w, h)
            if grid.any():
                yield _record(x, y, grid)

    _write(json_path, ds.RasterXSize, ds.RasterYSize, ds.GetGeoTransform(),
           ds.GetProjectionRef(), records())
    band = ds = None
    return str(json_path)


def load_mask_windows(json_path, output_path):
    """Restore a sparse tiled TIFF, unioning overlapping window grids."""
    records = _records(json_path)
    header = next(records)
    if header.get('format') != 'spectralmatch.mask_windows' or header.get('version') != 1:
        raise ValueError('Unsupported mask-window JSON format.')
    width, height = header['width'], header['height']
    if not isinstance(width, int) or not isinstance(height, int) or min(width, height) < 1:
        raise ValueError('Invalid mask grid dimensions.')
    ds = gdal.GetDriverByName('GTiff').Create(str(output_path), width, height, 1, gdal.GDT_Byte,
        options=raster_creation_options(['TILED=YES', 'COMPRESS=DEFLATE', 'BIGTIFF=IF_SAFER', 'SPARSE_OK=YES'], "GTiff"))
    if ds is None:
        raise RuntimeError(f'Could not create mask: {output_path}')
    ds.SetGeoTransform(header['transform'])
    ds.SetProjection(header['projection'])
    band = ds.GetRasterBand(1)
    try:
        for record in records:
            x, y, w, h = (record[key] for key in ('x_offset', 'y_offset', 'width', 'height'))
            if not all(isinstance(v, int) for v in (x, y, w, h)) or min(x, y) < 0 or min(w, h) < 1 or x+w > width or y+h > height:
                raise ValueError('Invalid mask-window bounds.')
            if record.get('fill') == 1:
                # A rectangular seed may be large; write it in bounded pieces.
                for dx, dy, tw, th in _windows(w, h):
                    band.WriteArray(np.ones((th, tw), dtype=np.uint8), x+dx, y+dy)
            else:
                if w*h > MAX_PIXELS:
                    raise ValueError('Encoded mask window exceeds the bounded grid size.')
                packed = base64.b64decode(record['grid'], validate=True)
                if len(packed) != (w*h+7)//8:
                    raise ValueError('Invalid packed mask size.')
                grid = np.unpackbits(np.frombuffer(packed, dtype=np.uint8), count=w*h).reshape(h, w)
                grid |= band.ReadAsArray(x, y, w, h)
                band.WriteArray(grid, x, y)
    finally:
        band = ds = None
        records.close()
    return str(output_path)


def seed_mask(seed_points, width, height, transform, projection, radius, json_path, output_path):
    """Represent square dilation exactly as clipped constant-valued windows."""
    points = np.asarray(seed_points)
    valid = points[(points[:, 0] >= 0) & (points[:, 0] < height) &
                   (points[:, 1] >= 0) & (points[:, 1] < width)]
    if not len(valid):
        raise ValueError('Conjugate seed points fall outside the overlap raster.')
    radius = max(0, int(radius))

    def records():
        for row, col in valid:
            x, y = max(0, int(col)-radius), max(0, int(row)-radius)
            right, bottom = min(width, int(col)+radius+1), min(height, int(row)+radius+1)
            yield dict(x_offset=x, y_offset=y, width=right-x, height=bottom-y, fill=1)

    _write(json_path, width, height, transform, projection, records())
    return load_mask_windows(json_path, output_path)


def sample_mask(mask_path, max_samples, json_path, output_path, *, count=None):
    """Select the original linspace ranks in row-major order with bounded memory."""
    if isinstance(max_samples, bool) or not isinstance(max_samples, int) or max_samples < 1:
        raise ValueError('max_samples must be a positive integer.')
    ds = gdal.Open(mask_path)
    if ds is None:
        raise RuntimeError(f'Could not open mask: {mask_path}')
    band = ds.GetRasterBand(1)
    # Reuse the pipeline count, or count in bounded windows for standalone use.
    if count is None:
        count = sum(int(np.count_nonzero(band.ReadAsArray(x, y, w, h)))
                    for x, y, w, h in _windows(ds.RasterXSize, ds.RasterYSize))
    if count <= max_samples:
        band = ds = None
        return mask_path
    step = (count-1) / (max_samples-1) if max_samples > 1 else 0

    def records():
        offset = cursor = 0
        for x, y, w, h in _windows(ds.RasterXSize, ds.RasterYSize):
            grid = band.ReadAsArray(x, y, w, h)
            indices = np.flatnonzero(grid)
            end = offset + indices.size
            if indices.size and cursor < max_samples:
                upper = min(max_samples, math.ceil(end/step)+1) if step else 1
                ranks = np.arange(cursor, upper, dtype=np.int64)
                targets = (ranks * step).astype(np.int64)
                if step and ranks.size and ranks[-1] == max_samples-1:
                    targets[-1] = count-1  # np.linspace includes its exact endpoint.
                targets = targets[(targets >= offset) & (targets < end)]
                cursor += targets.size
                if targets.size:
                    grid.fill(0)
                    grid.flat[indices[targets-offset]] = 1
                    yield _record(x, y, grid)
            offset = end
            if cursor == max_samples:
                break

    _write(json_path, ds.RasterXSize, ds.RasterYSize, ds.GetGeoTransform(), ds.GetProjectionRef(), records())
    band = ds = None
    return load_mask_windows(json_path, output_path)
