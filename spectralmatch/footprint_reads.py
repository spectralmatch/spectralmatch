"""Optional exact valid-data footprints for avoiding empty raster reads.

Footprints are hints about the current raster's validity, not additional cutlines.
No buffering or simplification is applied. Boundary blocks retain the raster's own
nodata/mask rules; callers must regenerate footprints after geometric changes.
"""
from copy import deepcopy
from pathlib import Path
import xml.etree.ElementTree as ET

from osgeo import gdal
from shapely import from_wkb, union_all
from shapely.affinity import affine_transform
from shapely.geometry import box
from shapely.prepared import prep


def matching_identifier(name, identifiers):
    """Accept an exact ID or a uniquely longest ID followed by processing suffixes."""
    def stem(value):
        filename = Path(str(value).replace('\\', '/')).name.casefold()
        suffix = Path(filename).suffix
        # An ID may contain dots without being a filename (e.g. scene.v1).
        return filename[:-len(suffix)] if suffix in {'.tif', '.tiff', '.vrt', '.img', '.jp2', '.bil', '.bsq'} else filename

    name = stem(name)
    candidates = []
    for key in identifiers:
        identifier = stem(key)
        if identifier == name or name.startswith(identifier + '_'):
            candidates.append((len(identifier), key))
    if not candidates:
        return None
    length = max(item[0] for item in candidates)
    matches = {key for size, key in candidates if size == length}
    if len(matches) != 1:
        raise ValueError(f'Ambiguous input_footprints image identifier for {name}: {sorted(matches)}')
    return matches.pop()


def load_footprints(input_footprints, image_paths, *, optional_paths=()):
    """Read once, match IDs, reproject to each raster CRS, return picklable WKB.

    A file or list of files uses its first layer and the standard ``image`` ID
    (falling back to ``image_path``). Fixed-reference images may be omitted.
    Missing moving-image IDs are errors, never silently empty footprints.
    """
    if input_footprints is None:
        return {}
    import geopandas as gpd
    paths = [input_footprints] if isinstance(input_footprints, (str, Path)) else list(input_footprints)
    if not paths:
        raise ValueError('input_footprints must contain at least one vector file, or be None')
    frames = []
    for path in paths:
        frame = gpd.read_file(path)
        if frame.crs is None:
            raise ValueError(f'input_footprints has no CRS: {path}')
        field = 'image' if 'image' in frame else 'image_path'
        if field not in frame:
            raise ValueError(f'input_footprints needs an image or image_path field: {path}')
        if frame[field].isna().any():
            raise ValueError(f'input_footprints contains a missing image ID: {path}')
        if frame.geometry.isna().any() or frame.geometry.is_empty.any() or not frame.geometry.is_valid.all():
            raise ValueError(f'input_footprints contains empty or invalid geometry: {path}')
        if not frame.geom_type.isin(['Polygon', 'MultiPolygon']).all():
            raise ValueError(f'input_footprints must contain polygons: {path}')
        frame = frame.copy()
        frame['_footprint_id'] = frame[field].astype(str)
        frames.append(frame)
    identifiers = {key for frame in frames for key in frame['_footprint_id']}
    optional = {str(path) for path in optional_paths}
    projected = {}
    result = {}
    for path in image_paths:
        key = matching_identifier(path, identifiers)
        if key is None:
            if str(path) in optional:
                continue
            raise ValueError(f'No input_footprints geometry matches image: {path}')
        ds = gdal.Open(str(path))
        crs = ds.GetProjection()
        ds = None
        if not crs:
            raise ValueError(f'Raster has no CRS: {path}')
        if crs not in projected:
            projected[crs] = [frame.to_crs(crs) for frame in frames]
        geometries = [geom for frame in projected[crs]
                      for geom in frame.loc[frame['_footprint_id'] == key].geometry]
        result[str(path)] = union_all(geometries).wkb
    return result


def footprint_pairs(pairs, paths, footprints):
    """Prune empty polygon intersections; leave unspecified references unchanged."""
    if not footprints:
        return pairs
    geometries = {name: from_wkb(footprints[path]) for name, path in paths.items() if path in footprints}
    return tuple((a, b) for a, b in pairs
                 if a not in geometries or b not in geometries
                 or geometries[a].intersection(geometries[b]).area > 0)


class FootprintWindows:
    """Prepared polygon in raster pixel coordinates; no pixel reads or buffer."""
    def __init__(self, geometries, transform):
        inverse = gdal.InvGeoTransform(transform)
        converted = [affine_transform(from_wkb(g),
                    [inverse[1], inverse[2], inverse[4], inverse[5], inverse[0], inverse[3]])
                     for g in geometries if g is not None]
        self.geometries = [prep(g) for g in converted]

    def intersects(self, window, padding=0):
        x, y, width, height = window
        rectangle = box(x-padding, y-padding, x + width+padding, y + height+padding)
        return all(geom.intersects(rectangle) for geom in self.geometries)


def sparse_footprint_vrt(path, footprint, destination, *, preserve_masks=True):
    """Keep full grid/values but omit VRT source blocks outside the footprint.

    Adjacent blocks are coalesced into rectangles. No image pixels are read to
    construct this view. This preserves GDAL's statistics sampling grid and the
    original data in boundary blocks. Without nodata, retain the original source
    because GDAL band statistics do not universally honor explicit masks.
    For derived outputs whose validity is already defined by their calculated
    nodata values, preserve_masks=False avoids manufacturing explicit mask files.
    """
    if footprint is None:
        return path
    ds = gdal.Open(str(path))
    if any(ds.GetRasterBand(b).GetNoDataValue() is None for b in range(1, ds.RasterCount + 1)):
        ds = None
        return path
    width, height = ds.RasterXSize, ds.RasterYSize
    block_x, block_y = ds.GetRasterBand(1).GetBlockSize()
    # A row strip is a read unit too; avoid one XML source for each individual row.
    block_y = max(block_y, 128) if block_x == width else block_y
    index = FootprintWindows([footprint], ds.GetGeoTransform())
    rectangles, active = [], {}
    for y in range(0, height, block_y):
        runs = []
        start = None
        h = min(block_y, height - y)
        for x in range(0, width, block_x):
            if index.intersects((x, y, min(block_x, width - x), h)):
                if start is None:
                    start = x
            elif start is not None:
                runs.append((start, x - start))
                start = None
        if start is not None:
            runs.append((start, width - start))
        following = {}
        for x, w in runs:
            key = (x, w)
            if key in active:
                rectangle = active.pop(key)
                rectangle[3] += h
            else:
                rectangle = [x, y, w, h]
            following[key] = rectangle
        rectangles.extend(active.values())
        active = following
    rectangles.extend(active.values())
    # A dense image needs no extra layer of VRT indirection.
    if rectangles == [[0, 0, width, height]]:
        ds = None
        return path
    # Preserve existing overview pixels and GDAL's approximate-statistics
    # population. A multi-source VRT otherwise loses implicit overviews.
    overview_paths = []
    for level in range(ds.GetRasterBand(1).GetOverviewCount()):
        overview_path = str(destination) + f".overview_{level}.vrt"
        overview = gdal.Translate(overview_path, ds, format='VRT', overviewLevel=level)
        overview = None
        overview_paths.append(overview_path)
    view = gdal.Translate(str(destination), ds, format='VRT')
    view = ds = None
    tree = ET.parse(destination)
    root = tree.getroot()
    # Translate may flatten an input VRT with scaled/derived sources. Wrap the
    # input itself instead, so every SrcRect remains in this raster's grid.
    root.attrib.pop('subClass', None)
    for child in list(root):
        if child.tag in ('GDALWarpOptions', 'BlockXSize', 'BlockYSize', 'MaskBand'):
            root.remove(child)

    def source(band_number):
        node = ET.Element('SimpleSource')
        ET.SubElement(node, 'SourceFilename', relativeToVRT='0').text = str(Path(path).resolve())
        ET.SubElement(node, 'SourceBand').text = str(band_number)
        for tag in ('SrcRect', 'DstRect'):
            ET.SubElement(node, tag, xOff='0', yOff='0', xSize=str(width), ySize=str(height))
        return node

    for band in root.findall('VRTRasterBand'):
        band.attrib.pop('subClass', None)
        for child in list(band):
            if child.tag.endswith('Source') or child.tag.startswith('PixelFunction') or child.tag in ('SourceTransferType', 'MaskBand'):
                band.remove(child)
        band.append(source(band.get('band')))
        if preserve_masks:
            mask = ET.SubElement(ET.SubElement(band, 'MaskBand'), 'VRTRasterBand', dataType='Byte')
            mask.append(source('mask,' + band.get('band')))
    for band in root.findall('VRTRasterBand'):
        for existing in band.findall('Overview'):
            band.remove(existing)
        for overview_path in overview_paths:
            overview = ET.SubElement(band, 'Overview')
            ET.SubElement(overview, 'SourceFilename', relativeToVRT='0').text = str(Path(overview_path).resolve())
            ET.SubElement(overview, 'SourceBand').text = band.get('band')
    for band in root.iter('VRTRasterBand'):
        sources = [child for child in band if child.tag in ('SimpleSource', 'ComplexSource')]
        for source in sources:
            band.remove(source)
            for x, y, w, h in rectangles:
                part = deepcopy(source)
                for tag in ('SrcRect', 'DstRect'):
                    rect = part.find(tag)
                    rect.attrib.update(xOff=str(x), yOff=str(y), xSize=str(w), ySize=str(h))
                band.append(part)
    tree.write(destination, encoding='utf-8', xml_declaration=True)
    return str(destination)


def occupied_rectangles(index, x, y, width, height, block_width=1, block_height=1):
    """Coalesce occupied grid cells into rectangles without changing their grid."""
    active = {}
    for top in range(y, y + height, block_height):
        h = min(block_height, y + height - top)
        runs, start = [], None
        for left in range(x, x + width, block_width):
            w = min(block_width, x + width - left)
            if index.intersects((left, top, w, h)):
                if start is None:
                    start = left
            elif start is not None:
                runs.append((start, left-start))
                start = None
        if start is not None:
            runs.append((start, x+width-start))
        following = {}
        for left, w in runs:
            key = left, w
            rectangle = active.pop(key, [left, top, w, 0])
            rectangle[3] += h
            following[key] = rectangle
        yield from active.values()
        active = following
    yield from active.values()
