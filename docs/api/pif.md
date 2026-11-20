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
    tie_max_samples="500crs",
    tie_min_samples="1000crs",
    pif_region_radius=30,
)
```

`tie_max_samples` is the goal for **accepted matched-point pairs**, after
validity checks, RANSAC, per-window selection, and duplicate removal.
`tie_min_samples` is the required minimum. Each accepts a positive integer,
`"Ncnt"` for a fixed count, or `"Dcrs"` for a density. The density resolves to
`ceil(valid_overlap_area / D**2)` across the whole valid overlap; it does not
require a match in each D × D square. Defaults are `"500crs"` and `"1000crs"`.
Fraction-form strings are rejected. PIF's internal alignment always requires at least
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
with valid pixels but too few accepted matches; that errors when the
skip option is disabled. Skipped pairs are omitted from saved overlap statistics,
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

`skip_overlap_when_not_enough_matches=True` is the default on `global_regression` and
`Pif.flood_from_match_points`: warn and omit pairs with fewer than three
unique usable tie points (after overlap-grid conversion and duplicate removal)
or fewer than the required PIF samples. Set `False` to raise on these shortages. Skipped pairs contribute neither
pair statistics nor model constraints and do not appear in saved overlap
statistics. Their images remain available to other overlaps; images without
any usable pair keep identity corrections. If no usable pairs remain, the
existing no-valid-PIF-overlaps error still stops the calculation. Malformed tie
caches, missing cache pairs, I/O errors and unrelated statistics errors are
not suppressed. Both overlap and matched-window statistics modes support this
option; the `entire` method is unaffected.

::: spectralmatch.pif.Pif

The live accepted-tie bar reports accepted pairs / resolved maximum for each overlap.
Rejected searches do not advance it. A successful overlap may finish below the
maximum when it meets the minimum and exhausts the search budget.


### Sparse PIF masks and memory use

Seed masks are stored as pixel-offset rectangles in temporary `seed_windows.json`.
Sampling scans bounded row-major windows and keeps the same evenly spaced PIF
pixel ranks as before; it writes `sampled_pif_windows.json` when subsampling is
needed. Raster masks remain sparse tiled TIFFs for bounded statistics reads. Neither seed
creation nor sampling allocates a NumPy array for the entire overlap. This does
not change `stats_scope`: overlap statistics still describe the whole overlap.

`from spectralmatch.pif import save_mask_windows, load_mask_windows` provides
`save_mask_windows(mask_path, json_path)` and
`load_mask_windows(json_path, output_path)` for saving/restoring binary masks.
JSON is an array: grid metadata first, then nonempty windows with `x_offset`,
`y_offset`, `width`, `height`, and a bit-packed base64 `grid` (big bit order).
Constant seed rectangles use `fill: 1`. Overlapping windows are unioned.
Offsets use the saved mask grid; the header includes its transform and CRS.
For `matched_windows`, that is the packed diagnostic grid, not source-image
coordinates. Readers/writers stream records rather than loading all grids.
Temporary JSON files follow the pair's existing temporary-directory lifetime;
these helpers do not enable automatic resume or change output reuse policy.

Masked VRTs in global and local statistics now honor `tile_threads`, including
an active thread-balancing budget, instead of unconditionally using all CPUs.


In `global_regression`, PIF mode skips preliminary ordinary overlap statistics:
the PIF solver computes the overlap statistics it actually uses. Traditional
whole-image statistics are retained only when required by a requested saved
adjustment file in `overlap` mode. `entire` mode still computes both ordinary
statistics passes. These shortcuts do not change `pif_stats_scope`, PIF
selection, or the fitted correction.

`overlap_spread` builds a compact block-count index during its shared-valid-pixel
scan. Region counts and exact nearest-center searches use this index without
repeated full-region scans. Actual pixel validity, valid-area quotas,
subdivision priorities, and row/column tie-breaking are unchanged.


PIF difference normalization still uses the valid whole-overlap population
(or the explicitly selected `matched_windows` population). Difference and final
PIF statistics read bands together in bounded windows. Approximate mode keeps
the expression-VRT block sampling layout used by GDAL; channels with no sampled
valid pixels retry exact statistics. Blockwise pooled moments can differ in
floating-point rounding from GDAL's accumulation, without changing the sampled
population. Read errors are never treated as empty samples.

After normalization, INZ and vegetation eligibility are evaluated only in seed
regions. The score is rounded to Float32 before thresholding, as before. The
pipeline writes the final binary PIF mask directly and counts selected pixels
while writing; it no longer materializes intermediate INZ/stability TIFFs.
Requesting `pif_save_inz` still writes the full score grid, including outside seed
regions. Zero in this binary mask means excluded, not zero-valued image data:
valid zero-valued image samples remain eligible. Capped PIF selection retains
the same evenly spaced row-major pixel ranks.
