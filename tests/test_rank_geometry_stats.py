"""stat: is calculated from individual geometry and never aliases a table column."""
import math
import geopandas as gpd
import pytest
from shapely.geometry import MultiPolygon, Polygon, box
from spectralmatch import create_footprints
from spectralmatch.seamline.weighted_seamline import _rank_polygons

EXPRESSION = '5 * log(max({stat:area_crs}, 1) / 1000000) / log(2) - ({var:view_zenith} + {var:solar_zenith})'


def test_stat_area_crs_uses_each_component_and_keeps_column_namespace():
    shape = MultiPolygon([box(0, 0, 4, 5), box(10, 0, 11, 2)])
    frame = gpd.GeoDataFrame({'image':['a'], 'area_crs':[999]}, geometry=[shape], crs=3857)
    result = _rank_polygons(frame, 'image', '{stat: area_crs} + {var:area_crs}', True)
    assert result.weighted_score.tolist() == [1019, 1001]
    assert result.area_crs.tolist() == [999, 999]
    assert all(g.geom_type == 'Polygon' for g in result.geometry)
    assert frame.geometry.iloc[0].equals(shape)


def test_area_crs_is_in_native_units_and_subtracts_holes():
    polygon = Polygon(box(0, 0, 10, 10).exterior.coords, [box(2, 2, 4, 4).exterior.coords])
    frame = gpd.GeoDataFrame({'image':['a']}, geometry=[polygon], crs=2263)
    result = _rank_polygons(frame, 'image', '{stat:area_crs}', True)
    assert result.weighted_score.iloc[0] == 96  # Square US survey feet; no m² conversion.
    assert 'area_crs' not in result.columns


def test_doubling_component_area_offsets_five_degrees():
    frame = gpd.GeoDataFrame({'image':['a','b'], 'view_zenith':[10,15], 'solar_zenith':[20,20]},
                            geometry=[box(0,0,1000,1000),box(2000,0,4000,1000)],crs=3857)
    result = _rank_polygons(frame, 'image', EXPRESSION, True)
    assert result.weighted_score.tolist() == pytest.approx([-30, -30])


def test_small_island_does_not_borrow_large_component_score(tmp_path):
    # a has a large remote component and a small high-quality island inside b.
    a = MultiPolygon([box(0,0,10000,10000),box(20000,0,21000,1000)])
    b = box(19000,-1000,22000,2000)
    output = str(tmp_path / 'coverage.gpkg')
    create_footprints(['a.tif','b.tif'],output,footprint_source='metadata',metadata_crs='EPSG:3857',
                      metadata_records=[{'view_zenith':0,'solar_zenith':0,'geometry':a.__geo_interface__},
                                        {'view_zenith':1,'solar_zenith':0,'geometry':b.__geo_interface__}],
                      image_rank_function=EXPRESSION,epsg=3857)
    result=gpd.read_file(output).set_index('image')
    assert result.loc['a'].geometry.equals(box(0,0,10000,10000))
    assert result.loc['b'].geometry.equals(b)


def test_area_stat_is_calculated_after_outline_clip_and_before_buffer(tmp_path):
    goal = str(tmp_path/'goal.gpkg')
    gpd.GeoDataFrame(geometry=[box(0,0,2,3)],crs=3857).to_file(goal)
    out = str(tmp_path/'coverage.gpkg')
    create_footprints(['a.tif'],out,footprint_source='metadata',metadata_crs='EPSG:3857',
                      metadata_records=[{'geometry':box(0,0,100,100).__geo_interface__}],
                      outline_path=goal,image_rank_function='{stat:area_crs}',extrude_distance=10,epsg=3857)
    result=gpd.read_file(out)
    assert result.weighted_score.iloc[0] == 6
    assert result.geometry.iloc[0].area > 6


def test_unknown_stat_is_rejected_and_var_stays_an_arbitrary_column():
    frame=gpd.GeoDataFrame({'image':['a'],'other':[7]},geometry=[box(0,0,2,3)],crs=3857)
    with pytest.raises(ValueError,match='Unknown geometry statistic'):
        _rank_polygons(frame,'image','{stat:other}',True)
    assert _rank_polygons(frame,'image','{var:other}',True).weighted_score.iloc[0] == 7


def test_image_level_quality_rejects_multipart_component_stat():
    frame=gpd.GeoDataFrame({'image':['a']},geometry=[MultiPolygon([box(0,0,1,1),box(2,0,3,1)])],crs=3857)
    with pytest.raises(ValueError,match='individual polygon components'):
        _rank_polygons(frame,'image','{stat:area_crs}',True,split_components=False)
