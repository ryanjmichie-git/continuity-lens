from __future__ import annotations

import json
import platform
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

import torch
import typer

from continuity_lens import __version__
from continuity_lens.app import create_app
from continuity_lens.config import (
    DAVIS_URL,
    DEFAULT_HORIZON,
    VJEPA_CHECKPOINT_URL,
    VJEPA_COMMIT,
    VJEPA_REPOSITORY,
    default_artifacts_dir,
    default_data_dir,
)
from continuity_lens.dataset import prepare_davis
from continuity_lens.diagnostics import run_synthetic_diagnostics
from continuity_lens.evaluation import run_development, run_frozen_test
from continuity_lens.synthetic import generate_demo_clips, generate_diagnostics
from continuity_lens.vjepa import load_vjepa
from continuity_lens.walkthrough import run_demo_walkthrough

app = typer.Typer(no_args_is_help=True, help="Research video continuity with V-JEPA.")
data_app = typer.Typer(no_args_is_help=True, help="Prepare licensed local datasets.")
app.add_typer(data_app, name="data")


def _head_status(url: str) -> str:
    request = urllib.request.Request(
        url,
        method="HEAD",
        headers={"User-Agent": "continuity-lens/0.1"},
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:  # noqa: S310
            length = response.headers.get("Content-Length", "unknown")
            return f"reachable ({response.status}, {length} bytes)"
    except (urllib.error.URLError, TimeoutError) as exc:
        return f"unreachable ({exc})"


@app.command()
def doctor(
    model: bool = typer.Option(False, "--model", help="Also download and load the real model."),
) -> None:
    """Report environment compatibility without downloading the model by default."""
    cuda = torch.cuda.is_available()
    report = {
        "continuity_lens": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": cuda,
        "gpu": torch.cuda.get_device_name(0) if cuda else None,
        "gpu_arches": torch.cuda.get_arch_list() if cuda else [],
        "vjepa_source": f"{VJEPA_REPOSITORY}@{VJEPA_COMMIT}",
        "checkpoint": _head_status(VJEPA_CHECKPOINT_URL),
        "davis": _head_status(DAVIS_URL),
    }
    typer.echo(json.dumps(report, indent=2))
    if cuda and "sm_120" not in report["gpu_arches"]:
        typer.secho("Warning: this PyTorch build does not advertise sm_120 kernels.", fg="yellow")
    if model:
        bundle = load_vjepa(horizon=DEFAULT_HORIZON)
        typer.secho(
            f"Loaded real model on {bundle.device}; "
            f"checkpoint {bundle.manifest['checkpoint_sha256']}",
            fg="green",
        )


@data_app.command("prepare")
def data_prepare(
    root: Path = typer.Option(default_data_dir(), help="Local data root; ignored by Git."),
    download: bool = typer.Option(True, "--download/--no-download"),
) -> None:
    """Download/validate DAVIS and write a provenance manifest."""
    manifest = prepare_davis(root, download=download)
    typer.echo(json.dumps(manifest, indent=2))


@data_app.command("diagnostics")
def data_diagnostics(
    root: Path = typer.Option(Path("data/diagnostics"), help="Diagnostic output root."),
    horizon: int = typer.Option(DEFAULT_HORIZON),
) -> None:
    records = generate_diagnostics(root, horizon=horizon)
    typer.echo(f"Generated {len(records)} controlled diagnostic cases under {root.resolve()}")


@data_app.command("demos")
def data_demos(
    root: Path = typer.Option(Path("data/demo"), help="Generated qualitative clip root."),
    artifacts_root: Path = typer.Option(default_artifacts_dir()),
    horizon: int | None = typer.Option(
        None,
        help="Target frames; defaults to the frozen horizon when available.",
    ),
) -> None:
    """Generate six self-owned clip pairs for the local product demonstration."""
    frozen_path = artifacts_root.resolve() / "dev" / "frozen_spec.json"
    if horizon is None and frozen_path.exists():
        horizon = int(json.loads(frozen_path.read_text(encoding="utf-8"))["selected_horizon"])
    selected_horizon = horizon if horizon is not None else DEFAULT_HORIZON
    pairs = generate_demo_clips(root, horizon=selected_horizon)
    typer.echo(
        f"Generated {len(pairs)} qualitative pairs at horizon={selected_horizon} "
        f"under {root.resolve()}"
    )


@app.command()
def benchmark(
    split: str = typer.Option(..., help="Development (`dev`) or held-out (`test`)."),
    frozen: bool = typer.Option(False, help="Required for the held-out test split."),
    data_root: Path = typer.Option(default_data_dir()),
    artifacts_root: Path = typer.Option(default_artifacts_dir()),
    mock_model: bool = typer.Option(
        False,
        help="Deterministic CI model; never for reported results.",
    ),
    force: bool = typer.Option(False, help="Replace a development freeze before any held-out run."),
    bootstrap_resamples: int = typer.Option(5_000, min=100),
) -> None:
    """Run development selection or the immutable held-out benchmark."""
    if split == "dev":
        if frozen:
            raise typer.BadParameter("--frozen is only valid with --split test")
        result = run_development(
            data_root=data_root,
            artifacts_root=artifacts_root,
            mock_model=mock_model,
            force=force,
        )
    elif split == "test":
        if not frozen:
            raise typer.BadParameter("Held-out evaluation requires --frozen")
        result = run_frozen_test(
            data_root=data_root,
            artifacts_root=artifacts_root,
            mock_model=mock_model,
            bootstrap_resamples=bootstrap_resamples,
        )
    else:
        raise typer.BadParameter("split must be `dev` or `test`")
    typer.echo(json.dumps(result, indent=2))


@app.command("app")
def launch_app(
    artifacts_root: Path = typer.Option(default_artifacts_dir()),
    mock_model: bool = typer.Option(False, help="Use the deterministic CI scorer."),
    share: bool = typer.Option(False, help="Create a temporary Gradio share link."),
) -> None:
    """Launch the local two-clip continuity review experience."""
    create_app(artifacts_root=artifacts_root, mock_model=mock_model).launch(share=share)


@app.command()
def walkthrough(
    data_root: Path = typer.Option(default_data_dir()),
    artifacts_root: Path = typer.Option(default_artifacts_dir()),
    mock_model: bool = typer.Option(False, help="Use the deterministic CI scorer."),
) -> None:
    """Measure all six generated examples; this is not human usability evidence."""
    result = run_demo_walkthrough(
        data_root=data_root,
        artifacts_root=artifacts_root,
        mock_model=mock_model,
    )
    typer.echo(json.dumps(result["summary"], indent=2))


@app.command()
def diagnostics(
    data_root: Path = typer.Option(default_data_dir()),
    artifacts_root: Path = typer.Option(default_artifacts_dir()),
    mock_model: bool = typer.Option(False, help="Use the deterministic CI scorer."),
) -> None:
    """Score the controlled synthetic lane; never use it for headline claims."""
    result = run_synthetic_diagnostics(
        data_root=data_root,
        artifacts_root=artifacts_root,
        mock_model=mock_model,
    )
    typer.echo(json.dumps(result["summary"], indent=2))


@app.command()
def check() -> None:
    """Run the compact local quality gate."""
    commands = ([sys.executable, "-m", "ruff", "check", "."], [sys.executable, "-m", "pytest"])
    for command in commands:
        typer.echo(f"$ {' '.join(command)}")
        result = subprocess.run(command, check=False)
        if result.returncode:
            raise typer.Exit(result.returncode)


if __name__ == "__main__":
    app()
