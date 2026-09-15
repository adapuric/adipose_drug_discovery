"""Shared command and process utilities for PerturbGen stages."""

from __future__ import annotations

import logging
import math
import re
import shlex
import subprocess
from collections.abc import Sequence
from pathlib import Path

from run_scripts.perturbgen.config import ModelConfig
from run_scripts.perturbgen.config import PerturbGenConfig


logger = logging.getLogger(__name__)


def run_command(
    command: Sequence[str],
    *,
    perturbgen_directory: str | Path,
    dry_run: bool = False,
) -> None:
    """Print or execute a PerturbGen command in its project directory."""
    command_text = shlex.join(command)
    if dry_run:
        print(command_text)
        return

    workdir = Path(perturbgen_directory).expanduser().resolve()
    if not workdir.is_dir():
        raise FileNotFoundError(
            f"PerturbGen working directory not found: {workdir}"
        )
    logger.info("Running: %s", command_text)
    subprocess.run(list(command), cwd=workdir, check=True)


def run_training_and_report_checkpoints(
    command: Sequence[str],
    *,
    perturbgen_directory: str | Path,
    checkpoint_directory: str | Path,
    stage_name: str,
    dry_run: bool = False,
    limit: int = 5,
) -> None:
    """Run training and print the new checkpoints with the lowest losses."""
    checkpoint_dir = Path(checkpoint_directory).expanduser().resolve()
    existing_checkpoints = set(checkpoint_dir.glob("*.ckpt"))
    run_command(
        command,
        perturbgen_directory=perturbgen_directory,
        dry_run=dry_run,
    )
    if dry_run:
        return

    new_checkpoints = set(checkpoint_dir.glob("*.ckpt")) - existing_checkpoints
    ranked_checkpoints: list[tuple[float, int, Path]] = []
    filename_pattern = re.compile(
        r"-epoch_(?P<epoch>\d+)-loss_"
        r"(?P<loss>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
        r"(?:-v\d+)?\.ckpt$"
    )
    for checkpoint in new_checkpoints:
        match = filename_pattern.search(checkpoint.name)
        if match is None:
            continue
        loss = float(match.group("loss"))
        if not math.isfinite(loss):
            continue
        ranked_checkpoints.append((loss, int(match.group("epoch")), checkpoint))

    if not ranked_checkpoints:
        raise RuntimeError(
            f"No loss-labelled {stage_name} checkpoints were created in "
            f"{checkpoint_dir}."
        )

    ranked_checkpoints.sort(key=lambda result: (result[0], result[1]))
    selected = ranked_checkpoints[:limit]
    print(f"\nBest {len(selected)} {stage_name} checkpoints by loss:")
    print("rank\tepoch\tloss\tcheckpoint")
    for rank, (loss, epoch, checkpoint) in enumerate(selected, start=1):
        print(f"{rank}\t{epoch:02d}\t{loss:.12g}\t{checkpoint}")


def model_data_arguments(config: PerturbGenConfig) -> list[str]:
    """Return shared tokenized-data arguments for model stages."""
    return [
        "--src_dataset",
        str(config.source_token_dataset_path),
        "--tgt_dataset_folder",
        str(config.target_token_dataset_directory),
        "--src_adata",
        str(config.source_pairing_h5ad_path),
        "--tgt_adata_folder",
        str(config.target_pairing_h5ad_directory),
        "--mapping_dict_path",
        str(config.token_to_gene_mapping_path),
    ]


def shared_model_arguments(
    model: ModelConfig,
    *,
    metadata_cols: Sequence[str] | None = None,
) -> list[str]:
    """Return model arguments shared by training and embedding stages."""
    resolved_metadata_cols = (
        model.retained_obs_cols if metadata_cols is None else metadata_cols
    )
    arguments = [
        "--pred_tps",
        *model.predicted_time_points,
        "--var_list",
        *resolved_metadata_cols,
        "--encoder",
        model.encoder,
        "--encoder_path",
        str(model.encoder_checkpoint_path),
        "--seed",
        str(model.random_seed),
        "--context_mode",
        bool_text(model.context_mode),
        "--pos_encoding_mode",
        model.positional_encoding_mode,
        "--num_layers",
        str(model.transformer_layers),
        "--d_ff",
        str(model.feedforward_dimension),
        "--d_model",
        str(model.model_dimension),
    ]
    if model.conditioning_obs_cols:
        arguments.extend(["--cond_list", *model.conditioning_obs_cols])
    return arguments


def checkpoint_path(
    configured_checkpoint: Path | None,
    *,
    override: str | Path | None,
) -> Path:
    """Resolve a required masking checkpoint from an override or config."""
    if override is not None:
        return Path(override).expanduser().resolve()
    if configured_checkpoint is None:
        raise ValueError(
            "Set masking_checkpoint in the config or pass --checkpoint."
        )
    return configured_checkpoint


def bool_text(value: bool) -> str:
    """Return the string Boolean representation expected by PerturbGen."""
    return "True" if value else "False"
