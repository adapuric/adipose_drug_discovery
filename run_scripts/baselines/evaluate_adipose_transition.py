"""Evaluate paired surgery-transition predictions with donor holdouts."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import anndata as ad

from add.validation.adipose_benchmark import benchmark_settings
from add.validation.adipose_benchmark import paired_donor_profiles
from add.validation.adipose_benchmark import write_adipose_benchmark
from add.validation.artifacts import configured_path
from add.validation.artifacts import read_report_config


def _parse_arguments() -> argparse.Namespace:
    """Parse the independent donor-benchmark configuration and output path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("config/adipose_benchmark.yaml")
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Validate inputs or publish a complete donor-benchmark report."""
    args = _parse_arguments()
    logging.basicConfig(level=logging.INFO)
    values, root = read_report_config(
        args.config,
        allowed_keys=("pseudobulk_path", "output_path", "parameters"),
    )
    parameters = values.get("parameters", {})
    if not isinstance(parameters, dict):
        raise TypeError("parameters must be a mapping")
    settings = benchmark_settings(parameters)
    source = configured_path(values, "pseudobulk_path", root=root)
    destination = args.output_dir or configured_path(
        values, "output_path", root=root
    )
    if args.dry_run:
        baseline, _, exclusions = paired_donor_profiles(
            ad.read_h5ad(source), settings=settings
        )
        logging.info(
            "Validated %d donor/state pairs; %d exclusions; output %s",
            len(baseline),
            len(exclusions),
            destination,
        )
    else:
        write_adipose_benchmark(source, destination, settings=settings)


if __name__ == "__main__":
    main()
