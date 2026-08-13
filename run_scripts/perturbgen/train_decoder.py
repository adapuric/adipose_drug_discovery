"""Train the configured PerturbGen count decoder."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from run_scripts.perturbgen.config import PerturbGenConfig
from run_scripts.perturbgen.config import load_perturbgen_config
from run_scripts.perturbgen.utils import bool_text
from run_scripts.perturbgen.utils import checkpoint_path
from run_scripts.perturbgen.utils import model_data_arguments
from run_scripts.perturbgen.utils import run_command
from run_scripts.perturbgen.utils import shared_model_arguments


def build_decoder_command(
    config: PerturbGenConfig,
    *,
    masking_checkpoint: str | Path | None = None,
    python_executable: str | None = None,
) -> list[str]:
    """Build the PerturbGen count-decoder training command."""
    model = config.model
    decoder = config.decoder
    checkpoint = checkpoint_path(
        decoder.masking_checkpoint_path,
        override=masking_checkpoint,
    )
    return [
        python_executable or sys.executable,
        "-m",
        "perturbgen",
        "train-decoder",
        "--train_mode",
        "count",
        "--split",
        "False",
        "--splitting_mode",
        "stratified",
        "--split_obs",
        *model.conditioning_obs_cols,
        "--output_dir",
        str(config.decoder_output_directory),
        "--ckpt_masking_path",
        str(checkpoint),
        *model_data_arguments(config),
        "--batch_size",
        str(decoder.batch_size),
        "--epochs",
        str(decoder.epochs),
        "--count_lr",
        str(decoder.count_learning_rate),
        "--count_wd",
        str(decoder.count_weight_decay),
        "--cellgen_lr",
        str(decoder.masking_learning_rate),
        "--cellgen_wd",
        str(decoder.masking_weight_decay),
        "--mlm_prob",
        str(decoder.masking_probability),
        "--count_dropout",
        str(decoder.count_dropout),
        "--n_workers",
        str(decoder.data_loader_workers),
        "--loss_mode",
        decoder.loss_mode,
        *shared_model_arguments(model),
        "--mask_scheduler",
        decoder.mask_scheduler,
        "--use_positional_encoding",
        bool_text(decoder.use_positional_encoding),
        "--layer_norm",
        bool_text(decoder.use_layer_normalization),
        "--num_node",
        str(model.num_nodes),
        "--ckpt_every_n_epochs",
        str(decoder.checkpoint_interval_epochs),
    ]


def _parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("config.yaml"),
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run configured PerturbGen count-decoder training."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    arguments = _parse_arguments()
    config = load_perturbgen_config(arguments.config)
    run_command(
        build_decoder_command(
            config,
            masking_checkpoint=arguments.checkpoint,
        ),
        perturbgen_directory=config.project.perturbgen_directory,
        dry_run=arguments.dry_run,
    )


if __name__ == "__main__":
    main()
