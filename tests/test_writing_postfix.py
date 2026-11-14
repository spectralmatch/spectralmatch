"""Native reuse always inspects final filenames before temporary writes begin."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import inspect

import geopandas as gpd
import pytest
from shapely.geometry import box, mapping

from spectralmatch import create_footprints, align_rasters, merge_rasters
from spectralmatch.seamline import footprints
from spectralmatch.utils_output import output_context, writing_outputs
from .utils_test import create_dummy_raster


def footprint_options():
    return dict(footprint_source='metadata', metadata_records=[{'geometry': mapping(box(0, 0, 10, 10))}],
                metadata_crs='EPSG:32604', validity_writing_postfix='_writing')


def test_completed_footprint_is_reused_at_original_path(tmp_path, monkeypatch):
    output = str(tmp_path / 'footprint.gpkg')
    assert create_footprints(['original.tif'], output, **footprint_options()) == output
    previous = Path(output).read_bytes()
    stale = tmp_path / 'footprint_writing.gpkg'
    stale.write_bytes(b'interrupted attempt')
    def unexpected(*args, **kwargs):
        pytest.fail('A completed vector must be reused before polygonization')
    monkeypatch.setattr(footprints, '_footprint_from_metadata', unexpected)
    assert create_footprints(['original.tif'], output, resume_from_outputs='validate', **footprint_options()) == output
    assert Path(output).read_bytes() == previous
    assert stale.read_bytes() == b'interrupted attempt'
    assert gpd.read_file(output).image_path.tolist() == ['original.tif']


def test_incomplete_final_rebuilds_and_removes_old_marker(tmp_path):
    output = tmp_path / 'footprint.gpkg'
    output.write_bytes(b'partial')
    marker = Path(str(output) + '.incomplete')
    marker.write_text('')
    assert create_footprints(['original.tif'], str(output), resume_from_outputs='validate', **footprint_options()) == str(output)
    assert len(gpd.read_file(output)) == 1
    assert not marker.exists()
    assert not (tmp_path / 'footprint_writing.gpkg').exists()


def test_failed_rebuild_preserves_existing_final(tmp_path, monkeypatch):
    output = tmp_path / 'footprint.gpkg'
    create_footprints(['original.tif'], str(output), **footprint_options())
    previous = output.read_bytes()
    def fail(self, *args, **kwargs):
        raise RuntimeError('interrupted polygon write')
    monkeypatch.setattr(footprints._PolygonWriter, 'write', fail)
    with pytest.raises(RuntimeError, match='interrupted polygon write'):
        create_footprints(['original.tif'], str(output), **footprint_options())
    assert output.read_bytes() == previous


@pytest.mark.parametrize('postfix', [None, '_writing', '_partial'])
def test_native_raster_writers_preserve_returns_and_resume(tmp_path, postfix, monkeypatch):
    source, aligned, merged = [str(tmp_path / n) for n in ('source.tif','aligned.tif','merged.tif')]
    create_dummy_raster(source, width=32, height=32, count=1, fill_value=100)
    with ThreadPoolExecutor(2) as executor:
        assert align_rasters([source], [aligned], executor=executor, validity_writing_postfix=postfix) == [aligned]
    assert merge_rasters([aligned], merged, validity_writing_postfix=postfix) == merged
    original = Path(merged).read_bytes()
    # Reuse needs no input pixels, even when the input is no longer available.
    Path(aligned).unlink()
    assert merge_rasters([aligned], merged, resume_from_outputs='validate', validity_writing_postfix=postfix) == merged
    assert Path(merged).read_bytes() == original
    assert not list(tmp_path.glob('*_writing.*'))
    assert not list(tmp_path.glob('*_partial.*'))


def test_writer_failure_preserves_final_and_retry_ignores_partial(tmp_path):
    final = str(tmp_path / 'a.txt')
    Path(final).write_text('old')
    with output_context('_writing'):
        with pytest.raises(RuntimeError):
            with writing_outputs(final) as (work,):
                Path(work).write_text('partial')
                raise RuntimeError('failure')
        assert Path(final).read_text() == 'old'
        with writing_outputs(final) as (work,):
            assert not Path(work).exists()
            Path(work).write_text('new')
    assert Path(final).read_text() == 'new'
    assert not (tmp_path / 'a_writing.txt').exists()


def test_public_parameter_is_only_optional_suffix():
    params = inspect.signature(create_footprints).parameters
    assert params['validity_writing_postfix'].default is None
    assert 'output_policy' not in params


@pytest.mark.parametrize('method', ['global_regression', 'local_block_adjustment'])
def test_matching_workers_receive_suffix_and_publish_original_names(tmp_path, method):
    from spectralmatch import Match
    from .test_match import _make_two_test_rasters
    images, outputs, _, _ = _make_two_test_rasters(tmp_path, [('A', 100), ('B', 120)])
    options = {'pif_method': 'entire'} if method == 'global_regression' else {'number_of_blocks': (2, 2)}
    with ThreadPoolExecutor(2) as executor:
        getattr(Match, method)(images, outputs, executor=executor, tile_threads=1,
            validity_writing_postfix='_writing', **options)
    assert all(Path(path).is_file() for path in outputs)
    assert not list(tmp_path.rglob('*_writing.tif'))


@pytest.mark.parametrize('suffix', [True, 3, '../writing', '/writing', '\\writing'])
def test_rejects_non_suffix_values(tmp_path, suffix):
    with pytest.raises(ValueError, match='validity_writing_postfix'):
        create_footprints(['unused.tif'], str(tmp_path / 'unused.gpkg'), validity_writing_postfix=suffix)
