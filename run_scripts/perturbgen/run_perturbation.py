"""Run PerturbGen inference from its native perturbation YAML config."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from run_scripts.perturbgen.config import PerturbGenConfig
from run_scripts.perturbgen.config import load_perturbgen_config
from run_scripts.perturbgen.utils import run_command


def build_perturbation_command(
    config: PerturbGenConfig,
    *,
    perturbation_config: str | Path | None = None,
    python_executable: str | None = None,
) -> list[str]:
    """Build the existing PerturbGen perturbation-validation command."""
    settings = config.perturbation
    requested_path = (
        Path(perturbation_config).expanduser().resolve()
        if perturbation_config is not None
        else settings.native_config_path
    )
    if requested_path is None:
        raise ValueError(
            "Set perturbation.native_config_path or pass --perturbation-config."
        )
    return [
        python_executable or sys.executable,
        str(config.perturbation_script_path),
        "--config",
        str(requested_path),
    ]


def _parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("config.yaml"),
    )
    parser.add_argument("--perturbation-config", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run configured PerturbGen perturbation inference."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    arguments = _parse_arguments()
    config = load_perturbgen_config(arguments.config)
    run_command(
        build_perturbation_command(
            config,
            perturbation_config=arguments.perturbation_config,
        ),
        perturbgen_directory=config.project.perturbgen_directory,
        dry_run=arguments.dry_run,
    )


if __name__ == "__main__":
    main()
