from pathlib import Path

from continuity_lens.diagnostics import run_synthetic_diagnostics
from continuity_lens.synthetic import generate_diagnostics


def test_synthetic_diagnostics_preserve_the_claim_boundary(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    artifacts_root = tmp_path / "artifacts"
    generate_diagnostics(data_root / "diagnostics", horizon=4)
    result = run_synthetic_diagnostics(
        data_root=data_root,
        artifacts_root=artifacts_root,
        mock_model=True,
    )
    assert result["summary"]["case_count"] == 12
    assert "cannot support claims" in result["summary"]["claim_boundary"]
    assert (artifacts_root / "diagnostic" / "predictions.csv").exists()
