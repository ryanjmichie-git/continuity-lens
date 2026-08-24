# Two-minute walkthrough

## 0:00–0:20 — Frame the decision

“Continuity Lens asks whether V-JEPA’s masked-future prediction error detects implausible video
transitions better than inexpensive visual signals. I designed the benchmark to answer whether the
world-model signal earns its product cost—not to force a win.”

## 0:20–0:50 — Show the workflow

Launch `uv run continuity-lens app`, select a generated teleportation example, and run analysis.
Point out the two clips, boundary strip, experimental hybrid risk, component scores, tubelet error,
and measured latency. State that the risk is benchmark-calibrated triage evidence, not physical
truth.

## 0:50–1:25 — Explain methodological safeguards

Show the evaluation card and frozen spec. Mention 60 development and 30 held-out DAVIS groups,
2/4/8-frame development selection, one held-out pass, source-grouped bootstrap, exact V-JEPA
source/checkpoint pins, no repeated-frame padding, and real-video-first evaluation.

## 1:25–1:50 — Report the result

Show the AUPRC chart. Predictor error reached 0.778; cheap-only reached 0.975. The grouped 95%
ΔAUPRC interval was [−0.229, −0.143]. The hybrid also slightly regressed. High-motion natural
boundaries were a predictor failure mode, while flow captured the constructed skips well.

## 1:50–2:00 — Make the product call

“I would not ship V-JEPA for this task from this evidence. I would deploy the cheap baseline as a
research control, collect creator-authored failures, and test incremental recall at a fixed review
budget. The useful artifact is the decision system: research signal, baselines, freeze, evidence,
and an explicit no-go when the economics do not work.”
