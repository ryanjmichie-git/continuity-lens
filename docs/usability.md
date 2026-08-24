# Expert usability walkthrough

## Status and evidence boundary

**AI-assisted expert heuristic review complete; human usability sessions were not conducted.**
Archetype responses below are simulated perspective checks, not participant statements, observed
completion, or measured human timing.

Evidence labels:

- **EMPIRICAL — system behavior:** all six generated pairs were executed through the real model.
- **EXPERT REVIEW — simulated use:** one known-good and one known-bad workflow were visually
  exercised in the local browser, then interpreted from three archetype perspectives.
- **ABSENT — human usability:** no participant success or comprehension claim is made.

See [evidence-register.md](evidence-register.md).

## System walkthrough

Command: `uv run continuity-lens walkthrough`

| Generated condition | Expected label | Cheap score | Experimental hybrid score | Frozen threshold flag |
|---|---:|---:|---:|---:|
| Continuous, variant 0 | Good | 12.0/100 | 5.0/100 | No |
| Object disappearance | Bad | 15.6/100 | 3.8/100 | No |
| Teleportation, variant 0 | Bad | 12.4/100 | 8.8/100 | No |
| Velocity change | Bad | 12.3/100 | 8.6/100 | No |
| Continuous, variant 1 | Good | 12.4/100 | 6.4/100 | No |
| Teleportation, variant 1 | Bad | 12.2/100 | 10.6/100 | No |

The frozen hybrid threshold detected **0 of 4 known-bad app examples** and produced no false alert
on the two continuous examples. This is out-of-distribution system evidence, not a statistical
estimate. Mean measured system time was 14 ms decoding, 17 ms cheap features, 187 ms model
inference, and 252 ms total warm processing; cold model load was 7.7 seconds. Exact rows are in
`artifacts/usability/`.

## Five-step expert walkthrough

1. **Load good and bad transitions:** the labeled dropdown loads all six self-generated pairs. The
   continuous and teleportation workflows were visually verified end to end.
2. **Identify the boundary:** a teal divider separates the last four displayed context frames from
   the two unique target frames used by the frozen horizon.
3. **Explain risk and a component:** the cheap and hybrid values are DAVIS-corruption benchmark
   scores, not physical probabilities. Prediction error is latent L1 disagreement; flow
   discontinuity measures boundary motion change relative to neighboring frame pairs.
4. **Choose an action:** the score does not justify an action. A reviewer may accept the visible
   continuous example and edit or regenerate a visibly teleported object, but that decision comes
   from the displayed evidence—the known-bad teleportation score remains low.
5. **Explain error:** generated geometry is out of domain, global token averages can miss small
   objects, median flow ignores sparse motion, and the future-only mask differs from pretraining.

## A1 — Generative-video creator perspective (simulated)

| Capture field | Modeled answer |
|---|---|
| Task completion | Expected to complete using the labeled example selector; not observed from a human. |
| Time | First-use hypothesis: 2–4 minutes. This is not measured timing. |
| Boundary | “The teal divider is where Clip A ends and the two analyzed frames from Clip B begin.” |
| Risk interpretation | “The low score does not mean the teleport is valid; it says this benchmark calibration did not flag it.” |
| Decision | Accept the visibly continuous pair; regenerate or edit visible teleportation despite the low score. |
| Why wrong | The model and calibration did not learn this stylized, sparse-object failure. |
| Main confusion | A 0–100 number initially looks actionable even when labeled experimental. |
| Requested evidence | Region-level heatmap, examples from generated footage, privacy statement, and similar false negatives. |

## A2 — Professional editor perspective (simulated)

| Capture field | Modeled answer |
|---|---|
| Task completion | Expected to complete; arbitrary uploads still require compatible clip lengths. |
| Time | First-use hypothesis: 3–5 minutes. This is not measured timing. |
| Boundary | “Four frames before the divider and two analyzed frames after it define the review point.” |
| Risk interpretation | “SSIM and histogram describe appearance change; none determines whether a cut was intentional.” |
| Decision | Use the strip to inspect the edit; do not accept or reject from the score. |
| Why wrong | Intentional cuts, fast camera motion, lighting changes, and eyeline edits are absent from calibration. |
| Main confusion | No timecodes, audio, or intentional-cut control; raw signal units require explanation. |
| Requested evidence | Source timecodes, frame stepping, intentional-cut labels, and false-alert examples by edit type. |

## A3 — Creative technologist perspective (simulated)

| Capture field | Modeled answer |
|---|---|
| Task completion | Expected to complete; the CSV and JSON artifacts support deeper inspection. |
| Time | First-use hypothesis: 2–3 minutes for the UI, longer for provenance review; not measured. |
| Boundary | “The frozen 14-context/2-target window is summarized by the 4+2 visible strip.” |
| Risk interpretation | “Cheap and hybrid scores share development calibration; component values expose why they disagree.” |
| Decision | Keep cheap features as the control and reject deployment of the V-JEPA feature from current evidence. |
| Why wrong | Domain shift, global aggregation, mask shift, and constructed prevalence limit transfer. |
| Main confusion | The single target tubelet limits the usefulness of a curve; calibration distributions are not in-app. |
| Requested evidence | Prediction export, score distributions, config hashes, per-case errors, equal-latency comparisons, and localized tokens. |

## Bounded iteration applied

The expert review changed only the product surface; the frozen benchmark was not rerun:

- Removed accept/edit/regenerate recommendations.
- Replaced unlabeled example thumbnails with a labeled dropdown.
- Corrected “eight-frame strip” to four context plus the frozen two-frame target.
- Led with the stronger cheap-only score and labeled the hybrid experimental.
- Added direction and plain-language interpretation for every component signal.
- Separated decode, cheap-feature, model, cold-start, request-total, and warm-pipeline timing.
- Added the held-out no-go decision above the workflow.

## Completion statement

The five questions are answerable and the misleading action language is removed. Human acceptance
criteria remain **not evaluated**. The completed artifact supports an expert-reviewed research-demo
claim, not a human-validated usability claim.
