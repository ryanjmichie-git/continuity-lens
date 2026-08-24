# Problem-discovery proxy

## Status and evidence boundary

**Archetype proxy complete; human discovery was not conducted.** The material below is a
structured assumption set, not interview evidence, quotations, observed behavior, or validated
demand. It may guide product choices and future recruiting, but it cannot support claims about
frequency, willingness to pay, or market size.

Evidence label: **HYPOTHESIS — archetype-based**. See [evidence-register.md](evidence-register.md).

## Method

Three plausible users were constructed from common generative-video, editorial, and ML-prototyping
workflows. Each archetype answers the same discovery guide. Answers intentionally avoid invented
identities, quotations, precise frequencies, and cost figures.

## A1 — Generative-video creator

| Discovery question | Archetype hypothesis |
|---|---|
| Most recent continuity failure | A subject’s identity, hand geometry, or motion trajectory changes between generated clips. |
| How it is found and handled | The creator notices it while comparing candidates or assembling a sequence, then regenerates, trims around it, or hides it with a cut. |
| Likely cost | Additional generations plus manual review; magnitude is unknown without real interviews or account data. |
| Highest-priority failures | Subject identity, geometry, and motion continuity. |
| Existing workaround | Frame scrubbing, side-by-side playback, trying another seed, and manual masking or editing. |
| Evidence required for trust | Exact boundary frames, a reason for the flag, representative false positives, and assurance that uploaded media stays local. |
| Frequency/severity hypothesis | Likely recurrent in multi-shot generative work, but not validated. Severe when a hero subject or expensive shot must be regenerated. |

## A2 — Professional video editor

| Discovery question | Archetype hypothesis |
|---|---|
| Most recent continuity failure | Motion direction, eyeline, lighting, or object position creates an unintended jump across an edit. |
| How it is found and handled | The editor catches it during playback or frame stepping, then trims, retimes, reframes, or chooses another take. |
| Likely cost | Editorial attention and schedule risk; precise cost is unknown. |
| Highest-priority failures | Edit continuity, motion, lighting, and intentional-versus-accidental cuts. |
| Existing workaround | Timeline playback, frame stepping, scopes, markers, and peer review. |
| Evidence required for trust | Timecode-level localization, visible before/after frames, controllable false alerts, and an intentional-cut override. |
| Frequency/severity hypothesis | Depends strongly on project volume and genre; false alarms on intentional edits could erase any benefit. |

## A3 — Creative technologist or ML prototyper

| Discovery question | Archetype hypothesis |
|---|---|
| Most recent continuity failure | A generation pipeline produces geometry, identity, or velocity drift across a batch of clips. |
| How it is found and handled | The technologist samples outputs, compares seeds, runs basic visual checks, and rejects or reruns failures. |
| Likely cost | Compute budget plus engineering and review time; no amount is asserted. |
| Highest-priority failures | Geometry, identity, motion, and systematic pipeline regressions. |
| Existing workaround | Contact sheets, embedding similarity, pixel or flow checks, and manually curated regression sets. |
| Evidence required for trust | Reproducible configuration, calibration data, baseline comparisons, latency, downloadable predictions, and failure examples. |
| Frequency/severity hypothesis | More likely to matter at batch scale, but the incremental value over inexpensive checks is the key unknown. |

## Cross-archetype synthesis

Shared hypotheses:

- Boundary localization and visible evidence matter more than an unexplained scalar.
- Intentional cuts and high camera motion are likely false-positive modes.
- A learned signal must add value beyond inexpensive checks to justify latency and complexity.
- Local processing, reproducibility, and failure examples are important trust mechanisms.

Archetype-specific needs conflict: creators may want simple regeneration guidance, editors need
intentional-cut controls and timecodes, while technical users want raw signals and exports. A single
interface should not claim to validate all three jobs.

## Positioning decision

Because no human behavior was observed, the creator-QA demand gate is **not satisfied**. Continuity
Lens remains positioned as a **research probe and inspectable evaluation case study**. The next
validating step, if time becomes available, is three recent-behavior interviews using the original
guide—not asking whether participants like this prototype.

## Product decisions allowed by this proxy

- Provide generated examples so the research workflow is self-contained.
- Show the exact boundary and component evidence.
- Keep the score explicitly benchmark-calibrated and non-causal.
- Lead with the held-out no-go decision rather than a deployment recommendation.
- Preserve downloadable results and provenance for technical review.

These are design hypotheses and methodological safeguards, not evidence of usability or demand.
