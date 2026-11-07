"""Saved vectors supply geometry, original image identities and attributes."""
import geopandas as gpd
import pytest
from shapely.geometry import Polygon, box
from spectralmatch import create_footprints


def vector(tmp_path, name, geometries, crs=3857, *, image=None, field='image', **attributes):
    path = str(tmp_path / f'{name}.gpkg')
    image = image or name
    data = {field: [image] * len(geometries), 'image_path': [f'/missing/{image}.tif'] * len(geometries)}
    data.update({key: [value] * len(geometries) for key, value in attributes.items()})
    gpd.GeoDataFrame(data, geometry=geometries, crs=crs).to_file(path, layer='saved')
    return path


def test_saved_geometry_preserves_holes_islands_identity_and_attributes(tmp_path, monkeypatch):
    shape = Polygon(box(0, 0, 10, 10).exterior.coords, [box(2, 2, 3, 3).exterior.coords])
    island = box(20, 0, 22, 2)
    source = vector(tmp_path, 'unrelated_vector_name', [shape, island], image='original_raster',
                    field='raster_id', quality=7, solar_zenith=20)
    monkeypatch.setattr('spectralmatch.seamline.footprints._footprint_from_image',
                        lambda *args: pytest.fail('vector_bounds read raster pixels'))
    output = str(tmp_path / 'result.gpkg')
    create_footprints([source], output, footprint_source='vector_bounds', vector_layer='saved',
                      image_field_name='raster_id', epsg=3857)
    result = gpd.read_file(output)
    assert result.geometry.iloc[0].equals(shape.union(island))
    assert result.quality.iloc[0] == 7
    assert result.solar_zenith.iloc[0] == 20
    assert result.raster_id.iloc[0] == 'original_raster'
    assert result.image_path.iloc[0] == '/missing/original_raster.tif'


def test_vector_bounds_reprojects_and_applies_rank_buffer_and_area_filter(tmp_path):
    a = vector(tmp_path, 'a', [box(0, 0, 10, 10)], quality=2)
    b = vector(tmp_path, 'b', [box(5, 0, 20, 10), box(40, 0, 40.01, 0.01)], quality=1)
    output = str(tmp_path / 'ranked.gpkg')
    create_footprints([a, b], output, footprint_source='vector_bounds', vector_layer='saved',
                      epsg=3857, image_rank_function='{var:quality}', extrude_distance=2)
    result = gpd.read_file(output).set_index('image')
    assert result.loc['b'].geometry.equals(box(8, 0, 20, 10))
    assert result.loc['a'].geometry.intersection(result.loc['b'].geometry).area > 0
    geo = gpd.read_file(a).to_crs(4326)
    geo_path = str(tmp_path / 'geographic.gpkg'); geo.to_file(geo_path)
    projected = str(tmp_path / 'projected.gpkg')
    create_footprints(geo_path, projected, footprint_source='vector_bounds', epsg=3857)
    assert gpd.read_file(projected).geometry.iloc[0].symmetric_difference(box(0, 0, 10, 10)).area < 1e-6


def test_explicit_metadata_overrides_saved_attributes_and_csv_uses_original_id(tmp_path):
    source = vector(tmp_path, 'vector_name', [box(0, 0, 10, 10)], image='actual_image', quality=1, note='saved')
    output = str(tmp_path / 'override.gpkg')
    create_footprints([source], output, footprint_source='vector_bounds', metadata_records=[{'quality': 3}])
    result = gpd.read_file(output)
    assert result.quality.iloc[0] == 3 and result.note.iloc[0] == 'saved'
    csv = tmp_path / 'attributes.csv'; csv.write_text('image,quality\nactual_image.tif,4\n')
    create_footprints([source], str(tmp_path / 'csv.gpkg'), footprint_source='vector_bounds', metadata_csv=str(csv))
    assert gpd.read_file(tmp_path / 'csv.gpkg').quality.iloc[0] == 4


def test_vector_output_alias_is_rejected(tmp_path):
    source = vector(tmp_path, 'source', [box(0, 0, 10, 10)])
    with pytest.raises(ValueError, match='differ from the input vector'):
        create_footprints([source], source, footprint_source='vector_bounds')
    assert len(gpd.read_file(source)) == 1


@pytest.mark.parametrize('problem', ['missing_id', 'missing_path', 'multiple_ids', 'multiple_paths'])
def test_invalid_vector_identity_is_rejected(tmp_path, problem):
    source = vector(tmp_path, 'source', [box(0, 0, 10, 10), box(20, 0, 30, 10)])
    frame = gpd.read_file(source)
    if problem == 'missing_id': frame = frame.drop(columns='image')
    if problem == 'missing_path': frame = frame.drop(columns='image_path')
    if problem == 'multiple_ids': frame.loc[1, 'image'] = 'other'
    if problem == 'multiple_paths': frame.loc[1, 'image_path'] = '/other.tif'
    frame.to_file(source, layer='saved')
    with pytest.raises(ValueError, match='image|image_path'):
        create_footprints([source], str(tmp_path / 'out.gpkg'), footprint_source='vector_bounds')


def test_duplicate_image_ids_in_different_vectors_are_rejected(tmp_path):
    a = vector(tmp_path, 'a', [box(0, 0, 10, 10)], image='same')
    b = vector(tmp_path, 'b', [box(10, 0, 20, 10)], image='same')
    with pytest.raises(ValueError, match='identifiers must be unique'):
        create_footprints([a, b], str(tmp_path / 'out.gpkg'), footprint_source='vector_bounds')
