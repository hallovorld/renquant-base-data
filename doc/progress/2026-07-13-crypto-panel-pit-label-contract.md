# Crypto panel PIT label contract fix (D-C3)

Date: 2026-07-13
PR: feat(crypto): alpha158 panel builder (D-C3) — #44

## Problem

Forward-return labels used calendar-day ffill to cover observation gaps, but
the ffill extended past the last real observation into purely synthetic
territory. A label whose terminal date had no real close was computed from a
forward-filled price, which could appear as zero-return or unchanged — an
artifact, not a measurement.

## Fix

`compute_forward_returns()` now requires a real observation at the terminal
date for each label. A boolean mask `real_obs = c.reindex(cal).notna()` marks
dates with actual closes, and `terminal_has_real = real_obs.shift(-n)` gates
each horizon's output — any label where the terminal date lacks a real
observation is set to NaN. BTC-excess labels additionally require the BTC leg's
terminal date to have a real observation.

## Manifest provenance additions

- `observation_end`: per-pair last real observation date
- `label_available_at`: per-pair, per-horizon last date with valid labels
- `input_bar_watermarks`: per-pair last observation date from input bars
- `calendar_identity_digest`: SHA-256 of the UTC daily calendar parameters
- `terminal_obs_required: true` in `label_contract`

## Bar-close PIT timing (codex r5)

Bar index = bar OPEN timestamp. Close[D] is known at D+1 00:00 UTC, not at D.
`_available_after` columns were off by one: `date + N` should be `date + N + 1`
because the terminal bar at D+N closes at D+N+1.

Fix:
- `compute_forward_returns()`: `_available_after = date + (N+1)` days
- `build_crypto_panel()`: adds `feature_available_after = date + 1 day` column
- Manifest `label_contract`: added `bar_timestamp_convention`, `bar_close_offset_days`,
  `availability_rule`
- `_load_ohlcv()`: retains `bar_close_utc` column if present in source data

## Tests

27 tests pass (up from 24 — 3 new bar-close PIT regression tests):
- `test_label_available_after_uses_bar_close_offset`: proves fwd_5d at Jan 1
  has available_after = Jan 7 (D+5+1), not Jan 6
- `test_btc_excess_available_after_bar_close_offset`: proves BTC-excess
  available_after uses the same D+N+1 convention
- `test_feature_available_after_in_panel`: proves feature_available_after =
  date + 1 day for every row in the built panel
- Prior tests updated for the +1 offset and new columns
