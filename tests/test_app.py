from pathlib import Path

import gradio as gr

from continuity_lens.app import create_app


def test_gradio_app_builds_without_loading_checkpoint(tmp_path: Path) -> None:
    demo = create_app(artifacts_root=tmp_path, mock_model=True)
    assert isinstance(demo, gr.Blocks)
