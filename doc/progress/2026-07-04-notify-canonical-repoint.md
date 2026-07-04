# 2026-07-04 — Re-point ntfy sender to the canonical (campaign B6, audit XC-4)

`watchlist_screen.notify` local urllib transport deleted; now a thin seam over
`renquant_common.notify.send` (same signature, same default topic).
renquant-common floor bumped to 0.10 (lockstep).

Behavior preserved: this repo already honored `RENQUANT_NO_NOTIFY` — that
semantics (plus never-raise and the 5 s timeout) now comes from the canonical.
Only nominal delta: suppression accepts the truthy set 1/true/yes/on (was
`== "1"` — a superset, existing usage unaffected) and the suppressed/failure
log lines are emitted by the `renquant_common.notify` logger.

Suite green (239 passed) including `test_watchlist_screen`'s existing
`RENQUANT_NO_NOTIFY=1` suppression test, unchanged.
