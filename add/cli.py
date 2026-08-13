"""Command-line entry point for adipose drug-discovery analyses."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from add.data import AnalysisConfig
from add.data import load_config
from add.workflows import cache_perturbation_signatures
from add.workflows import create_pseudobulk
from add.workflows import estimate_rescue_vectors
from add.workflows import rank_drug_candidates


def build_parser() -> argparse.ArgumentParser:
    """Build the `add` command-line parser."""
    parser = argparse.ArgumentParser(
        prog="add",
        description="Donor-aware adipose rescue and drug-response baselines.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/baselines.yaml"),
        help="YAML configuration (default: config/baselines.yaml).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    pseudobulk_parser = _add_file_stage_parser(
        subparsers,
        command="pseudobulk",
        help_text="Sum raw counts into donor-level adipose profiles.",
        input_option="--input",
    )
    pseudobulk_parser.add_argument("--chunk-size", type=int, default=4096)
    pseudobulk_parser.set_defaults(handler=_pseudobulk_command)

    rescue_parser = _add_file_stage_parser(
        subparsers,
        command="rescue",
        help_text="Estimate paired baseline-to-weightloss rescue vectors.",
        input_option="--pseudobulk",
    )
    rescue_parser.set_defaults(handler=_rescue_command)

    perturb_parser = subparsers.add_parser(
        "perturb",
        help="Process and cache an external perturbation source.",
    )
    perturb_parser.add_argument(
        "--source",
        choices=("tahoe", "lincs"),
        required=True,
    )
    perturb_parser.add_argument("--input", type=Path, help="Tahoe H5AD input.")
    perturb_parser.add_argument("--matrix", type=Path, help="LINCS matrix.")
    perturb_parser.add_argument("--metadata", type=Path, help="LINCS metadata.")
    perturb_parser.add_argument(
        "--gene-metadata",
        type=Path,
        help="LINCS measured-landmark annotation.",
    )
    perturb_parser.add_argument("--output-prefix", type=Path)
    perturb_parser.set_defaults(handler=_perturb_command)

    baseline_parser = subparsers.add_parser(
        "baseline",
        help="Run one of the four drug-response baselines.",
    )
    baseline_parser.add_argument(
        "--model",
        choices=("perturbed-mean", "pca-ridge", "mean-drug", "cmap"),
        required=True,
    )
    baseline_parser.add_argument("--signatures", type=Path)
    baseline_parser.add_argument("--rescue", type=Path)
    baseline_parser.add_argument("--pseudobulk", type=Path)
    baseline_parser.add_argument("--output-dir", type=Path)
    baseline_parser.add_argument(
        "--workers",
        type=int,
        help="Worker processes (default: physical cores minus one).",
    )
    baseline_parser.set_defaults(handler=_baseline_command)
    return parser


def _add_file_stage_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    *,
    command: str,
    help_text: str,
    input_option: str,
) -> argparse.ArgumentParser:
    """Add a stage parser with one file input and an output directory."""
    parser = subparsers.add_parser(command, help=help_text)
    parser.add_argument(input_option, type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser


def main() -> None:
    """Parse arguments and execute the requested analysis stage."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    arguments = build_parser().parse_args()
    config = load_config(arguments.config)
    arguments.handler(arguments, config)


def _pseudobulk_command(
    arguments: argparse.Namespace,
    config: AnalysisConfig,
) -> None:
    """Forward pseudobulk CLI arguments to its workflow."""
    create_pseudobulk(
        config,
        input_path=arguments.input,
        output_dir=arguments.output_dir,
        chunk_size=arguments.chunk_size,
    )


def _rescue_command(
    arguments: argparse.Namespace,
    config: AnalysisConfig,
) -> None:
    """Forward rescue CLI arguments to its workflow."""
    estimate_rescue_vectors(
        config,
        pseudobulk_path=arguments.pseudobulk,
        output_dir=arguments.output_dir,
    )


def _perturb_command(
    arguments: argparse.Namespace,
    config: AnalysisConfig,
) -> None:
    """Forward perturbation CLI arguments to its workflow."""
    cache_perturbation_signatures(
        config,
        source=arguments.source,
        input_path=arguments.input,
        matrix_path=arguments.matrix,
        metadata_path=arguments.metadata,
        gene_metadata_path=arguments.gene_metadata,
        output_prefix=arguments.output_prefix,
    )


def _baseline_command(
    arguments: argparse.Namespace,
    config: AnalysisConfig,
) -> None:
    """Forward baseline CLI arguments to its workflow."""
    rank_drug_candidates(
        config,
        model=arguments.model,
        signature_prefix=arguments.signatures,
        rescue_path=arguments.rescue,
        pseudobulk_path=arguments.pseudobulk,
        output_dir=arguments.output_dir,
        workers=arguments.workers,
    )


if __name__ == "__main__":
    main()
