# Local release-candidate checklist

- [x] `uv run --isolated --locked continuity-lens check` passes in a fresh environment.
- [x] `uv run continuity-lens check` passes.
- [x] `uv run continuity-lens doctor --model` records a real checkpoint smoke.
- [x] DAVIS manifest lists 60 development and 30 held-out sources.
- [x] Frozen held-out results exist under protocol `24a0e6d5…`.
- [x] Discovery and usability proxies are labeled as archetype hypotheses and expert simulation;
  they contain no invented participants, quotations, or human-validation claim.
- [x] Six self-generated qualitative demo pairs are reproducible and excluded from headline metrics.
- [x] README claims match the evidence status.
- [x] Candidate-file scan excludes datasets, checkpoints, recordings, private notes, and raw or
  uploaded media; the only tracked video is the owned, captioned demo.
- [x] Third-party notices and citations are present.
- [x] Render the two-minute captioned walkthrough using [demo-script.md](demo-script.md).
- [ ] Human has reviewed the repository before any remote is created or pushed.
