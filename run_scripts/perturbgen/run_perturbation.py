"""Run and summarize donor-aware PerturbGen perturbation inference."""

from __future__ import annotations

import argparse
import logging
import pickle
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import yaml

from run_scripts.perturbgen.config import PerturbGenConfig
from run_scripts.perturbgen.config import load_perturbgen_config
from run_scripts.perturbgen.summarize_perturbation import (
    summarize_perturbation_result,
)
from run_scripts.perturbgen.utils import run_command


logger = logging.getLogger(__name__)

_VALID_MODES = {"delete", "mask", "overexpress", "pad"}
_VALID_SEQUENCES = {"src", "tgt"}
_SAFE_GENE = re.compile(r"^[A-Za-z0-9_.-]+$")


def build_native_perturbation_config(
    config: PerturbGenConfig,
    *,
    decoder_checkpoint: str | Path,
    gene: str,
) -> dict[str, object]:
    """Build the native YAML mapping consumed by ``Perturb/val.py``."""
    settings = config.perturbation
    _validate_settings(config, gene=gene)
    output_directory = perturbation_gene_directory(config, gene=gene)
    predicted_time_points = [
        _native_time_point(value)
        for value in config.model.predicted_time_points
    ]
    data: dict[str, object] = {
        "src_dataset_file": str(config.source_token_dataset_path),
        "tgt_dataset_folder": str(config.target_token_dataset_directory),
        "src_adata": str(config.source_pairing_h5ad_path),
        "tgt_adata_folder": str(config.target_pairing_h5ad_directory),
    }
    if config.model.conditioning_obs_cols:
        data["cond_list"] = list(config.model.conditioning_obs_cols)

    return {
        "data": data,
        "trainer": {
            "use_count_decoder": True,
            "n_samples": settings.n_samples,
            "d_model": config.model.model_dimension,
            "generate": False,
            "use_positional_encoding": (config.decoder.use_positional_encoding),
            "layer_norm": config.decoder.use_layer_normalization,
            "dropout": config.decoder.count_dropout,
            "num_heads": 8,
            "num_layers": config.model.transformer_layers,
            "loss_mode": config.decoder.loss_mode,
            "d_ff": config.model.feedforward_dimension,
            "mask_scheduler": config.decoder.mask_scheduler,
            "output_dir": str(output_directory),
            "mapping_dict_path": str(config.token_to_gene_mapping_path),
            "tokenid_to_rowid_path": str(config.token_to_row_mapping_path),
            "encoder": config.model.encoder,
            "encoder_path": str(config.model.encoder_checkpoint_path),
            "context_mode": settings.context_mode,
            "pos_encoding_mode": config.model.positional_encoding_mode,
            "exclude_src": False,
            "perturbation_mode": settings.mode,
            "genes_to_perturb": [gene],
            "validation_mode": "inference",
            "perturbation_sequence": [settings.sequence],
            "var_list": list(config.model.retained_obs_cols),
            "pred_tps": predicted_time_points,
        },
        "datamodule": {
            "batch_size": settings.batch_size,
            "num_workers": settings.data_loader_workers,
            "shuffle": False,
            "split": False,
            "var_list": list(config.model.retained_obs_cols),
            "pred_tps": predicted_time_points,
        },
        "model": {
            "precision": settings.precision,
            "ckpt_masking_path": str(
                Path(decoder_checkpoint).expanduser().resolve()
            ),
        },
    }


def perturbation_gene_directory(
    config: PerturbGenConfig,
    *,
    gene: str,
) -> Path:
    """Return an isolated output directory for one gene and edit."""
    settings = config.perturbation
    return (
        config.perturbation_output_directory
        / f"{settings.mode}_{settings.sequence}"
        / gene
    )


def resolve_decoder_checkpoint(
    config: PerturbGenConfig,
    *,
    override: str | Path | None,
) -> Path:
    """Resolve the required decoder checkpoint."""
    requested = override or config.perturbation.decoder_checkpoint_path
    if requested is None:
        raise ValueError(
            "Set perturbation.decoder_checkpoint_path or pass --checkpoint."
        )
    return Path(requested).expanduser().resolve()


def resolve_genes(
    config: PerturbGenConfig,
    *,
    overrides: Sequence[str] | None,
) -> tuple[str, ...]:
    """Resolve separate single-gene perturbations from CLI or configuration."""
    genes = tuple(overrides or config.perturbation.genes_to_perturb)
    if not genes:
        raise ValueError(
            "Set perturbation.genes_to_perturb or pass at least one --gene."
        )
    if len(set(genes)) != len(genes):
        raise ValueError("Perturbation genes must not contain duplicates.")
    for gene in genes:
        if not _SAFE_GENE.fullmatch(gene):
            raise ValueError(f"Invalid perturbation gene identifier: {gene!r}.")
    return genes


def write_native_perturbation_config(
    native_config: Mapping[str, object],
    *,
    path: str | Path,
) -> Path:
    """Persist the exact native config used for one inference run."""
    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_config = dict(native_config)
    if output_path.exists():
        with output_path.open(encoding="utf-8") as handle:
            existing_config = yaml.safe_load(handle)
        if existing_config != resolved_config:
            raise FileExistsError(
                "Refusing to mix a new perturbation config with an existing "
                f"gene output: {output_path.parent}. Move that directory or "
                "use a new run.run_name before changing the checkpoint or "
                "inference settings."
            )
        return output_path
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(resolved_config, handle, sort_keys=False)
    return output_path


def validate_runtime_inputs(
    config: PerturbGenConfig,
    *,
    decoder_checkpoint: Path,
    genes: Sequence[str],
) -> None:
    """Fail early when required artifacts or gene tokens are absent."""
    required_files = {
        "decoder checkpoint": decoder_checkpoint,
        "source pairing H5AD": config.source_pairing_h5ad_path,
        "token-to-gene mapping": config.token_to_gene_mapping_path,
        "token-to-row mapping": config.token_to_row_mapping_path,
        "encoder checkpoint": config.model.encoder_checkpoint_path,
        "PerturbGen perturbation script": config.perturbation_script_path,
    }
    required_directories = {
        "source token dataset": config.source_token_dataset_path,
        "target token datasets": config.target_token_dataset_directory,
        "target pairing H5ADs": config.target_pairing_h5ad_directory,
    }
    for label, path in required_files.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {label}: {path}")
    for label, path in required_directories.items():
        if not path.is_dir():
            raise FileNotFoundError(f"Missing {label}: {path}")

    with config.token_to_gene_mapping_path.open("rb") as handle:
        row_to_gene = pickle.load(handle)
    if not isinstance(row_to_gene, Mapping):
        raise ValueError("Token-to-gene mapping is not a mapping.")
    gene_to_row = {str(gene): row for row, gene in row_to_gene.items()}
    if missing_genes := sorted(set(genes).difference(gene_to_row)):
        raise ValueError(
            "Perturbation genes are absent from the tokenized vocabulary: "
            f"{missing_genes}."
        )

    if config.perturbation.sequence == "src":
        with config.token_to_row_mapping_path.open("rb") as handle:
            token_to_row = pickle.load(handle)
        if not isinstance(token_to_row, Mapping):
            raise ValueError("Token-to-row mapping is not a mapping.")
        source_rows = set(token_to_row.values())
        if missing_source_genes := sorted(
            gene for gene in genes if gene_to_row[gene] not in source_rows
        ):
            raise ValueError(
                "Source perturbation genes have no source token: "
                f"{missing_source_genes}."
            )


def find_perturbation_result(
    config: PerturbGenConfig,
    *,
    gene: str,
) -> Path:
    """Find the newest non-empty native output for one configured edit."""
    settings = config.perturbation
    output_directory = perturbation_gene_directory(config, gene=gene)
    suffix = f"_g{gene}_s{settings.sequence}_t{settings.mode}.h5ad"
    if candidates := sorted(
        (
            path
            for path in output_directory.glob(f"*{suffix}")
            if path.stat().st_size > 0
        ),
        key=lambda path: path.stat().st_mtime,
    ):
        return candidates[-1]
    else:
        raise RuntimeError(
            "PerturbGen returned without a non-empty result for "
            f"{gene}. The gene may be absent from all selected source nuclei."
        )


def _validate_settings(config: PerturbGenConfig, *, gene: str) -> None:
    """Validate native options that PerturbGen otherwise accepts opaquely."""
    settings = config.perturbation
    if settings.mode not in _VALID_MODES:
        raise ValueError(
            f"perturbation.mode must be one of {sorted(_VALID_MODES)}."
        )
    if settings.sequence not in _VALID_SEQUENCES:
        raise ValueError(
            f"perturbation.sequence must be one of {sorted(_VALID_SEQUENCES)}."
        )
    if not _SAFE_GENE.fullmatch(gene):
        raise ValueError(f"Invalid perturbation gene identifier: {gene!r}.")

    positive_values = {
        "n_samples": settings.n_samples,
        "batch_size": settings.batch_size,
        "precision": settings.precision,
    }
    if invalid := sorted(
        name for name, value in positive_values.items() if value <= 0
    ):
        raise ValueError(f"Perturbation values must be positive: {invalid}.")
    if settings.data_loader_workers < 0:
        raise ValueError(
            "perturbation.data_loader_workers must not be negative."
        )


def _native_time_point(value: str) -> int:
    """Convert configured CLI time-point text to PerturbGen's native integer."""
    try:
        return int(value)
    except ValueError as error:
        raise ValueError(
            "PerturbGen native perturbation time points must be integers; "
            f"got {value!r}."
        ) from error


def _parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("config.yaml"),
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--gene",
        action="append",
        help="Gene identifier to run separately; repeat for multiple genes.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run configured perturbations and write donor-aware summaries."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    arguments = _parse_arguments()
    config = load_perturbgen_config(arguments.config)
    checkpoint = resolve_decoder_checkpoint(
        config,
        override=arguments.checkpoint,
    )
    genes = resolve_genes(config, overrides=arguments.gene)
    if not arguments.dry_run:
        validate_runtime_inputs(
            config,
            decoder_checkpoint=checkpoint,
            genes=genes,
        )

    for gene in genes:
        output_directory = perturbation_gene_directory(config, gene=gene)
        native_path = output_directory / "native_config.yaml"
        native_config = build_native_perturbation_config(
            config,
            decoder_checkpoint=checkpoint,
            gene=gene,
        )
        if not arguments.dry_run:
            write_native_perturbation_config(native_config, path=native_path)
        command = [
            sys.executable,
            str(config.perturbation_script_path),
            "--config",
            str(native_path),
        ]
        run_command(
            command,
            perturbgen_directory=config.project.perturbgen_directory,
            dry_run=arguments.dry_run,
        )
        if arguments.dry_run:
            continue

        result_path = find_perturbation_result(config, gene=gene)
        logger.info("PerturbGen result: %s", result_path)
        summary_paths = summarize_perturbation_result(
            result_path,
            output_directory=output_directory / "summary",
            donor_col=config.prepare.donor_col,
            state_col=config.prepare.state_alias_col,
        )
        for name, path in summary_paths.items():
            logger.info("%s: %s", name, path)


if __name__ == "__main__":
    main()
