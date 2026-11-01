With `global_regression(pif_method="flood_from_match_points")`,
`pif_stats_scope="overlap"` is the default: difference mean/std come from the
whole valid overlap. Set `pif_stats_scope="matched_windows"` to warp only
`pif_window_size`-pixel windows around accepted distributed tie points (default
257, an odd side length). Difference statistics are pooled across those windows;
overlapping source pixels are counted once. Seed regions retain the existing
`pif_region_radius` behavior and stop at window boundaries. The global solver
and output correction are unchanged.

Both modes use joint coregistration's distributed ORB search, map-coordinate
RANSAC and per-window inlier selection. The `tie_grid_spacing`,
`tie_search_radius`, `tie_orb_max_features`, `tie_max_matches_per_window`,
`tie_maximum_displacement` and `tie_ransac_reprojection_threshold` controls have
the same meaning as in joint coregistration.

`estimate_stats=True` requests approximate GDAL mean/std for both the difference
population and the selected PIF pixels; `False` requests exact GDAL statistics.
Only selected-PIF statistics retry with exact GDAL statistics if approximate
sampling finds no valid pixels; other statistics calculations keep their requested mode.
The matched-window mode also avoids the preliminary whole-image/overlap
statistics pass. Optional `pif_save_inz` diagnostics use packed window
coordinates in this mode, rather than geographic mosaic coordinates.

For direct `Pif.flood_from_match_points` calls, the corresponding names are
`stats_scope`, `window_size` and `estimate_stats`.

Set `skip_overlap_when_not_enough_matches=True` on `global_regression` or
`Pif.flood_from_match_points` to warn and omit pairs with fewer than three
usable tie points or fewer than the required PIF samples. The default is
`False`, which preserves the existing errors. Skipped pairs contribute neither
pair statistics nor model constraints and do not appear in saved overlap
statistics. Their images remain available to other overlaps; images without
any usable pair keep identity corrections. If no usable pairs remain, the
existing no-valid-PIF-overlaps error still stops the calculation. Malformed tie
caches, missing cache pairs, I/O errors and unrelated statistics errors are
not suppressed. Both overlap and matched-window statistics modes support this
option; the `entire` method is unaffected.

::: spectralmatch.pif.Pif
