Global matching now defaults to `tie_sampling_method="overlap_spread"`.
Search centers are placed only inside the intersection of the two images'
valid-data masks on their common matching grid. The first center is the center
of the occupied bounds, snapped to a valid pixel when necessary. The region
with the most valid area is then bisected along its longest CRS dimension,
and the next center is placed in the largest remaining region. Holes and
separate islands are respected; ties in placement use a stable order.

```python
global_regression(
    input_images, output_images,
    tie_sampling_method="overlap_spread",
    tie_max_samples="1/100crs",
    tie_min_samples="1/500crs",
    pif_region_radius=30,
)
```

`tie_max_samples` is the goal for **accepted matched-point pairs**, after
validity checks, RANSAC, per-window selection, and duplicate removal.
`tie_min_samples` is the required minimum. Each accepts a positive integer or
`"N/Dcrs"`: resolve the latter as `ceil(valid_overlap_area * N / D**2)`.
Thus `"5/3crs"` means five pairs per 3 x 3 CRS-unit square. Defaults are
`"1/100crs"` and `"1/500crs"`. PIF's internal alignment always requires at least
three pairs, so density-derived counts are raised to that floor; an explicit
integer maximum below the required minimum is an error. Cached pairs are
checked against the same valid overlap and quotas and capped at the goal.

Search continues toward the goal. `tie_max_attempts=None` allows up to
`max(100, 10 * resolved_goal)` search centers; a positive integer overrides
that limit. Exhausting valid centers or that budget succeeds if the minimum
was reached and otherwise raises an insufficient-pairs error. Reaching the
minimum alone does not stop the search. With `overlap_spread`, pairs with no
shared valid pixels are always skipped, even when
`skip_overlap_when_not_enough_matches=False`. This is distinct from an overlap
with valid pixels but too few accepted matches; that still errors unless the
skip option is enabled. Skipped pairs are omitted from saved overlap statistics,
and images without a usable neighbor retain identity corrections. A model still
requires at least one usable pair overall. I/O and unrelated errors still raise.

These pair quotas are separate from `pif_max_samples` / `pif_min_samples`,
which still count **stable PIF pixels** after neighborhood and spectral
filtering. More matched pairs do not guarantee enough stable pixels.

Use `tie_sampling_method="grid"` for the previous rectangular-overlap grid.
`tie_grid_spacing` only applies to that method; the new pair quotas and
attempt budget apply to `overlap_spread`. Both use the existing feature
matcher, search radius, RANSAC, and per-window match selection settings.
Full-resolution validity is scanned in tiles into a temporary disk-backed
mask; no full-resolution multispectral overlap is loaded into RAM for
placement. The scan also occurs for cached pairs and matched-window PIF mode
because density quotas require the actual shared valid area.

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
Difference and selected-PIF statistics retry with exact GDAL statistics only if
approximate sampling misses all valid pixels; unrelated failures still raise.
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
