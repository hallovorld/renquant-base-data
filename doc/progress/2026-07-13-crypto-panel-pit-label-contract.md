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

## Tests

20 tests pass (up from 18 — 2 new terminal-gap regression tests):
- `test_no_labels_after_last_real_observation`: 10-day series, proves labels
  are NaN when terminal date has no real observation, valid when it does
- `test_btc_excess_terminal_gap`: proves BTC-excess labels are NaN when
  either leg's terminal is missing
- Existing gap-calendar and btc-calendar tests updated to reflect the
  stricter terminal-obs requirement
