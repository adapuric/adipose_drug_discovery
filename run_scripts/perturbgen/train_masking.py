"""Train the configured PerturbGen masking model."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from run_scripts.perturbgen.config import PerturbGenConfig
from run_scripts.perturbgen.config import load_perturbgen_config
from run_scripts.perturbgen.utils import bool_text
from run_scripts.perturbgen.utils import model_data_arguments
from run_scripts.perturbgen.utils import run_command
from run_scripts.perturbgen.utils import shared_model_arguments


def build_masking_command(
    config: PerturbGenConfig,
    *,
    python_executable: str | None = None,
) -> list[str]:
    """Build the PerturbGen masking-model training command."""
    model = config.model
    masking = config.masking
    command = [
        python_executable or sys.executable,
        "-m",
        "perturbgen",
        "train-mask",
        "--train_mode",
        "masking",
        "--split",
        "False",
        "--splitting_mode",
        "stratified",
        "--split_obs",
        *model.conditioning_obs_cols,
        "--output_dir",
        str(config.masking_output_directory),
        *model_data_arguments(config),
        "--batch_size",
        str(masking.batch_size),
        "--epochs",
        str(masking.epochs),
        "--cellgen_lr",
        str(masking.learning_rate),
        "--cellgen_wd",
        str(masking.weight_decay),
        "--n_workers",
        str(masking.data_loader_workers),
        *shared_model_arguments(model),
        "--mask_scheduler",
        masking.mask_scheduler,
        "--num_node",
        str(model.num_nodes),
        "--use_weighted_sampler",
        bool_text(masking.use_weighted_sampler),
        "--ckpt_every_n_epochs",
        str(masking.checkpoint_interval_epochs),
    ]
    resume_checkpoint = masking.resume_checkpoint_path
    if resume_checkpoint is not None:
        command.extend(["--ckpt_masking_path", str(resume_checkpoint)])
    return command


def _parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("config.yaml"),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run configured PerturbGen masking-model training."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    arguments = _parse_arguments()
    config = load_perturbgen_config(arguments.config)
    run_command(
        build_masking_command(config),
        perturbgen_directory=config.project.perturbgen_directory,
        dry_run=arguments.dry_run,
    )


if __name__ == "__main__":
    main()
