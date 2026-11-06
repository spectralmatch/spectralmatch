import geopandas as gpd
import pytest
from shapely.geometry import box, MultiPolygon, mapping
from spectralmatch import create_footprints
from spectralmatch.seamline.footprints import _coverage_polygons

@pytest.mark.parametrize('threshold,parts',[(1.0,1),(None,2),(0,2),(0.00001,2)])
def test_filter_before_buffer(threshold,parts):
    tiny=box(100,0,100.01,0.01)
    frame=gpd.GeoDataFrame({'image':['a']},geometry=[MultiPolygon([box(0,0,10,10),tiny])],crs=3857)
    result=_coverage_polygons(frame,'image',outline_path=None,outline_layer=None,image_rank_function=None,image_rank_descending=True,extrude_distance=5,min_polygon_area=threshold)
    geom=result.geometry.iloc[0]
    assert (len(geom.geoms) if hasattr(geom,'geoms') else 1)==parts

@pytest.mark.parametrize('bad',[-1,float('nan'),float('inf'),True,'1'])
def test_bad_threshold(tmp_path,bad):
    with pytest.raises(ValueError,match='min_polygon_area'):
        create_footprints(['a.tif'],str(tmp_path/'out.gpkg'),min_polygon_area=bad)

def test_public_default_and_none(tmp_path):
    records=[{'quality':2,'geometry':mapping(box(0,0,10,10))},{'quality':1,'geometry':mapping(box(9,0,10.01,0.01))}]
    for value,expected in [(1.0,1),(None,2)]:
        out=str(tmp_path/f'{value}.gpkg')
        create_footprints(['a.tif','b.tif'],out,metadata_records=records,footprint_source='metadata',metadata_crs='EPSG:3857',epsg=3857,image_rank_function='{quality}',extrude_distance=500,min_polygon_area=value)
        assert len(gpd.read_file(out))==expected

def test_geographic_area_is_square_metres():
    frame=gpd.GeoDataFrame({'image':['a']},geometry=[box(0,0,0.001,0.001)],crs=4326)
    result=_coverage_polygons(frame,'image',outline_path=None,outline_layer=None,image_rank_function=None,image_rank_descending=True,extrude_distance=0)
    assert len(result)==1
