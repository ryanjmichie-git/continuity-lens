# Benchmark provenance

## Frozen evaluation identity

The development protocol was frozen at `2026-08-24T00:15:46Z` with:

- Protocol hash: `24a0e6d55dc4e295c69a2e0377930b60d5257275ba3a88ce2574c07b344da16c`
- Test-time package hash: `a256b40ac9e7fd014c9dacb77a47e1c12f1acd058674fa8b220a2450a3115a71`
- DAVIS manifest hash: `0543e897653244c9be77fc75a0d3ad4996a13e6753d8e7dc5d0f792aa6a5568a`
- Checkpoint SHA-256: `5346856ec9df69487fe72a25bf2632aaa8112df33fb67708e3f7374edc1f7012`

The one-time held-out run completed at `2026-08-24T00:17:42Z`. Its predictions, grouped
bootstrap draws, metrics, and figures are committed as immutable result artifacts.

## Post-test changes

After the held-out result existed, only product, auxiliary-analysis, and demo surfaces changed:

- `app.py`: separates the cheap and experimental hybrid scores, explains every signal, exposes the
  exact boundary, and reports cold and warm latency without making an automated decision.
- `cli.py`: adds generated-demo, diagnostic, and system-walkthrough commands.
- `synthetic.py`: creates six H.264 qualitative demo pairs.
- `diagnostics.py`: scores the controlled auxiliary suite and writes explicitly non-headline
  summaries and a figure.
- `walkthrough.py`: records real system outputs and latency for the six owned examples; it does not
  claim human usability evidence.
- `scripts/render_demo.py`: renders the deterministic captioned two-minute walkthrough from owned
  repository artifacts.

No feature definition, case generator, mask, model loader, calibrator, metric, bootstrap, or frozen
test path changed. Tests and documentation were expanded, and no second held-out run was made.

Because protocol v1 conservatively hashes the entire package rather than only scoring modules, the
current package hash intentionally differs from the test-time hash. The freeze guard refuses a
new test run under the old protocol. Re-evaluation requires a new protocol version and a complete
development freeze; deleting or overwriting the existing held-out artifacts is prohibited.
