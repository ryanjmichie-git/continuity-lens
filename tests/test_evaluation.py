from __future__ import annotations

from pathlib import Path

import pytest

from continuity_lens.evaluation import (
    FrozenEvaluationError,
    run_development,
    run_frozen_test,
)


def test_freeze_then_single_held_out_pass(tiny_davis: Path, tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    development = run_development(
        data_root=tiny_davis,
        artifacts_root=artifacts,
        mock_model=True,
    )
    assert development["frozen_spec"]["expected_test_groups"] == 2
    result = run_frozen_test(
        data_root=tiny_davis,
        artifacts_root=artifacts,
        mock_model=True,
        bootstrap_resamples=100,
    )
    assert result["metrics"]["bootstrap"]["resamples"] == 100
    assert result["metrics"]["reporting_outcome"] in {
        "predictor_adds_evidence",
        "cheap_signals_outperform",
        "inconclusive_at_this_sample_size",
    }
    with pytest.raises(FrozenEvaluationError, match="will not be overwritten"):
        run_frozen_test(
            data_root=tiny_davis,
            artifacts_root=artifacts,
            mock_model=True,
            bootstrap_resamples=100,
        )
