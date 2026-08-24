# Continuity Lens working agreement

## Product question

Does masked-future latent prediction identify implausible video transitions better than inexpensive pixel, similarity, and optical-flow signals?

## Commands

- Setup: `uv sync --locked`
- Environment: `uv run continuity-lens doctor`
- Data: `uv run continuity-lens data prepare`
- Generated demos: `uv run continuity-lens data demos`
- Synthetic diagnostics: `uv run continuity-lens diagnostics`
- Real-system walkthrough: `uv run continuity-lens walkthrough`
- Development benchmark: `uv run continuity-lens benchmark --split dev`
- Frozen test: `uv run continuity-lens benchmark --split test --frozen`
- App: `uv run continuity-lens app`
- Captioned walkthrough: `uv run python scripts/render_demo.py`
- Quality gate: `uv run continuity-lens check`

## Non-negotiable evaluation contract

- DAVIS train is development data; DAVIS val is held out until the protocol is frozen.
- Never tune features, horizons, coefficients, or thresholds against held-out results.
- Never pad videos by repeating frames.
- Never describe masked-future prediction as causal simulation or physical truth.
- If a frozen test has already run, do not overwrite it. Version the protocol and rerun the full test.

## Repository boundaries

- Do not commit checkpoints, datasets, interview recordings, private notes, or uploaded videos.
- Do not create a public remote, publish, or push without explicit human approval.
- Preserve anonymization and media-license evidence.

## Definition of done

Code, tests, docs, benchmark artifacts, limitations, and a local demo are reviewable. Claims must match the evidence, including negative or inconclusive evidence.
