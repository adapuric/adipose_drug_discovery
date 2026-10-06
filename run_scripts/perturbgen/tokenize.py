"""Tokenize and pair the configured H5AD with PerturbGen."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from add.utils import get_physical_cores
from run_scripts.perturbgen.config import PerturbGenConfig
from run_scripts.perturbgen.config import load_perturbgen_config
from run_scripts.perturbgen.utils import run_command


def build_tokenize_command(
    config: PerturbGenConfig,
    *,
    python_executable: str | None = None,
) -> list[str]:
    """Build the paired PerturbGen tokenization command."""
    tokenize = config.tokenize
    command = [
        python_executable or sys.executable,
        "-m",
        "perturbgen",
        "tokenise",
        "--h5ad_path",
        str(config.tokenizer_input_h5ad_path),
        "--dataset",
        config.run.run_name,
        "--gene_filtering_mode",
        tokenize.gene_filtering_mode,
        "--hvg_mode",
        tokenize.hvg_mode,
        "--n_hvg",
        str(tokenize.highly_variable_gene_count),
        "--var_list",
        *tokenize.retained_obs_cols,
        "--pairing_mode",
        tokenize.pairing_mode,
        "--time_obs",
        tokenize.time_obs_col,
        "--main_pairing_obs",
        tokenize.main_pairing_obs,
        "--nproc",
        str(get_physical_cores()),
        "--reference_time",
        tokenize.reference_time,
        "--time_point_order",
        *tokenize.time_point_order,
        "--gene_median_path",
        str(tokenize.gene_median_dictionary_path),
        "--token_dict_path",
        str(tokenize.token_dictionary_path),
        "--gene_mapping_path",
        str(tokenize.ensembl_mapping_path),
    ]
    if tokenize.optional_pairing_obs:
        command.extend(["--opt_pairing_obs", *tokenize.optional_pairing_obs])
    return command


def _parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "configs" / "config.yaml",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run configured PerturbGen tokenization."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    arguments = _parse_arguments()
    config = load_perturbgen_config(arguments.config)
    input_path = config.tokenizer_input_h5ad_path
    if not arguments.dry_run and not input_path.is_file():
        raise FileNotFoundError(
            f"Tokenizer input H5AD not found: {input_path}. Run the prepare "
            "stage first or set tokenize.input_h5ad_path to an existing file."
        )

    run_command(
        build_tokenize_command(config),
        perturbgen_directory=config.project.perturbgen_directory,
        dry_run=arguments.dry_run,
    )


if __name__ == "__main__":
    main()
