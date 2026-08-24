# Evaluation card

## Claim under test

Masked-future V-JEPA prediction error adds held-out discrimination for within-source temporal discontinuities beyond a cheap visual ensemble.

## Data and split

- Primary corpus: DAVIS 2017 480p real-video sequences.
- Development: official train split, 60 source groups.
- Held out: official val split, 30 source groups.
- Synthetic geometry and owned/generated clips are excluded from headline metrics.

Each source contributes three deterministic anchors. Every anchor produces one continuous case, a temporal skip, a reordered target, and a secondary cross-video splice. Grouping is by context source; inference excludes the secondary lane.

## Signals

- Predictor: training-aligned mean L1 error against layer-normalized EMA target tokens.
- Encoder: cosine distance at the boundary.
- Cheap: HSV histogram distance, SSIM distance, and flow discontinuity.
- Hybrid: fixed logistic model on cheap features plus predictor error.

Candidate future horizons are 2, 4, and 8 frames within a 16-frame input. Selection, calibration, and thresholding use development groups only.

## Metrics and uncertainty

- Primary: ΔAUPRC, predictor minus cheap-only.
- Secondary: hybrid ΔAUPRC, AUROC, balanced accuracy, corruption recall, latency, and VRAM.
- Interval: 5,000 paired bootstrap resamples grouped by source video.

Reporting policy:

- Interval above zero: evidence the predictor adds value.
- Interval below zero: cheap signals outperform.
- Interval crossing zero: inconclusive at this sample size.

There is no arbitrary minimum effect and no pre-committed winner.

## Frozen result

- Selected development horizon: 2 frames.
- Held-out primary sample: 30 source groups, 270 transitions, 0.667 prevalence.
- Predictor AUPRC: 0.778; cheap AUPRC: 0.975; hybrid AUPRC: 0.969.
- Predictor-minus-cheap point ΔAUPRC: −0.197.
- Grouped 95% bootstrap interval: [−0.229, −0.143] across 5,000 resamples.
- Hybrid-minus-cheap interval: [−0.0122, −0.0002].
- Reporting outcome: **cheap signals outperform on this benchmark**.
- Frozen protocol hash: `24a0e6d55dc4e295c69a2e0377930b60d5257275ba3a88ce2574c07b344da16c`.

The held-out output was generated once and is protected against overwrite. Exact predictions,
bootstrap draws, figures, and JSON metrics are in `artifacts/test/`.

## Known limitations

- A contiguous future-only mask differs from V-JEPA pretraining masks.
- DAVIS is a real-video segmentation corpus, not a creator-authored continuity benchmark.
- Programmatic skips and block reorders cover only a subset of real failures.
- Cross-video splices share targets across context groups and are descriptive only.
- AUPRC depends on the constructed class prevalence.
- The calibrated risk is not transferable without new data.
- The benchmark result should not be generalized from constructed DAVIS corruptions to
  creator-authored generative-video failures.
