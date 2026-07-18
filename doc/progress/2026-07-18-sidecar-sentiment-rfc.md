# Progress: rawlabel-sidecar sentiment contract reconciliation RFC

Date: 2026-07-18

## What

doc/design/2026-07-18-rawlabel-sidecar-sentiment-reconciliation.md —
RFC resolving the weekly PatchTST retrain deadlock: the sidecar builder
excludes the 3 sentiment columns by design while the served sidecar
carries them, so the refresh guard rejects every rebuild (identical
failures 07-11 and 07-18, verified). Recommendation: migrate the served
sidecar to the builder's 176-col contract (option b), gated on a full
cross-repo consumer sweep (AC-1); migration itself is a supervised
ask-first landing step, never the scheduled job. Guard fail-closed
direction unchanged.
