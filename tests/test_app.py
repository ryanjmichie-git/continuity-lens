from pathlib import Path

import gradio as gr

from continuity_lens.app import create_app
from continuity_lens.synthetic import generate_demo_clips
from continuity_lens.walkthrough import run_demo_walkthrough


def test_gradio_app_builds_without_loading_checkpoint(tmp_path: Path) -> None:
    demo = create_app(artifacts_root=tmp_path, mock_model=True)
    assert isinstance(demo, gr.Blocks)


def test_system_walkthrough_is_explicitly_not_human_evidence(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    artifacts_root = tmp_path / "artifacts"
    generate_demo_clips(data_root / "demo", horizon=4)
    result = run_demo_walkthrough(
        data_root=data_root,
        artifacts_root=artifacts_root,
        mock_model=True,
    )
    assert result["summary"]["case_count"] == 6
    assert "not human usability evidence" in result["summary"]["evidence_level"]
    assert (artifacts_root / "usability" / "system_walkthrough.csv").exists()
