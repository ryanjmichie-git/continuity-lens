# Product brief

## Candidate user and job

The initial candidate is a generative-video creator or editor who reviews many short clips and needs to decide which transitions deserve manual attention or regeneration.

Candidate job to be done:

> When I assemble or generate a sequence of clips, help me find transitions that violate expected motion or appearance so I can spend review time and generation credits where they matter most.

This is a hypothesis until the discovery gate in [discovery.md](discovery.md) is satisfied.

Problem interviews are still pending, so the current positioning is a **research probe**, not a
validated creator-QA product.

## Product wedge

Continuity Lens is not a general video-quality score. Its wedge is boundary triage:

1. Accept the end of Clip A and beginning of Clip B.
2. Predict target latent tokens from the context.
3. Compare predicted and observed target representations.
4. Show the learned signal beside inexpensive alternatives and the exact boundary frames.
5. Let the reviewer decide whether to accept, edit, or regenerate.

## User value and risks

Potential value:

- Reduce repetitive frame scrubbing.
- Surface subtle within-shot temporal jumps.
- Make a research signal inspectable rather than hiding it behind a single score.

Principal risks:

- A high score may mean distribution shift rather than a visible defect.
- A creator may over-trust a benchmark-calibrated number.
- Stylized editing and intentional cuts can look discontinuous by design.
- Inference cost may exceed the value of cheap filtering.

The UI therefore presents evidence and limitations, not an automated rejection.

## Evidence-driven product decision

The frozen DAVIS result does not justify V-JEPA in the operational path: the cheap-only ensemble
reached 0.975 AUPRC versus 0.778 for the predictor, and the hybrid slightly regressed. The local app
keeps the hybrid visible for research inspection, but explicitly says it is not the recommended
production model. A real product pilot should start with the inexpensive baseline and revisit a
learned signal only on creator-authored failures that the baseline misses.

## Metrics

Research metric: held-out ΔAUPRC between masked-future prediction error and the cheap-only ensemble, with a grouped confidence interval.

Product metrics for a later pilot:

- Review time per accepted minute of footage.
- Recall of creator-identified defects at a fixed review budget.
- Regenerations avoided or correctly prioritized.
- Reviewer override and false-alert rates.

No A/B-test claim is made in this 24-hour prototype. A suitable next test would randomize boundary review with and without model-assisted ordering while holding the clips and review budget fixed.
