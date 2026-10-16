"""Write a PerturbGen report manifest from completed perturbation outputs."""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
from collections.abc import Sequence
from pathlib import Path

from run_scripts.perturbgen.config import PerturbGenConfig
from run_scripts.perturbgen.config import load_perturbgen_config
from run_scripts.perturbgen.run_perturbation import apply_perturbation_overrides
from run_scripts.perturbgen.run_perturbation import find_perturbation_result


logger = logging.getLogger(__name__)


def build_report_manifest(
    config: PerturbGenConfig,
    *,
    results_root: Path | None = None,
    source: str = "summary",
) -> list[dict[str, object]]:
    """List every completed single-gene edit of one PerturbGen run.

    Each entry names the model input H5AD and the condition whose sequence
    was edited, so the report can form the naive co-expression contrast.
    Edits without the requested output are logged and left out.

    Args:
      config: PerturbGen workflow configuration of the run.
      results_root: Directory replacing `run.results_root_directory`, for
        results copied to another cluster.
      source: Effect input to list ("summary" or "native").

    Returns:
      Report manifest entries in mode, sequence and gene order.
    """
    if source not in {"summary", "native"}:
        raise ValueError('source must be "summary" or "native"')
    if results_root is not None:
        config = dataclasses.replace(
            config,
            run=dataclasses.replace(
                config.run, results_root_directory=results_root
            ),
        )
    perturbation_directory = config.perturbation_output_directory
    if not perturbation_directory.is_dir():
        raise FileNotFoundError(
            f"No perturbation outputs for {config.run.run_name}: "
            f"{perturbation_directory}"
        )

    path_key = "summary_path" if source == "summary" else "result_path"
    entries: list[dict[str, object]] = []
    unavailable: list[str] = []
    for edit_directory in sorted(perturbation_directory.iterdir()):
        if not edit_directory.is_dir():
            continue
        mode, _, sequence = edit_directory.name.rpartition("_")
        edit_config = apply_perturbation_overrides(
            config, mode=mode, sequence=sequence
        )
        condition = _edited_condition(edit_config)
        for gene_directory in sorted(edit_directory.iterdir()):
            if not gene_directory.is_dir():
                continue
            gene = gene_directory.name
            try:
                effect_path = (
                    find_perturbation_result(edit_config, gene=gene)
                    if source == "native"
                    else gene_directory / "summary/donor_state_effects.csv.gz"
                )
            except RuntimeError:
                effect_path = None
            if effect_path is None or not effect_path.is_file():
                unavailable.append(f"{edit_directory.name}/{gene}")
                continue
            entries.append(
                {
                    "run_id": config.run.run_name,
                    "edit_id": f"{mode}_{sequence}_{gene}",
                    "edited_genes": [gene],
                    "edit_mode": mode,
                    "edit_sequence": sequence,
                    path_key: str(effect_path),
                    "coexpression_h5ad_path": str(
                        edit_config.tokenizer_input_h5ad_path
                    ),
                    "edited_condition": condition,
                    "gene_namespace": "ensembl_gene_id",
                }
            )
    if unavailable:
        logger.warning(
            "%s: %d edits lack %s output and are not listed: %s",
            config.run.run_name,
            len(unavailable),
            source,
            unavailable,
        )
    return entries


def _edited_condition(config: PerturbGenConfig) -> str:
    """Return the condition label of the sequence the edit changed."""
    settings = config.perturbation
    if settings.sequence not in {"src", "tgt"} or settings.mode not in {
        "delete",
        "mask",
        "overexpress",
        "pad",
    }:
        raise ValueError(
            "Perturbation directories must be named <mode>_<sequence>; got "
            f"{settings.mode}_{settings.sequence}"
        )
    if settings.sequence == "src":
        return config.tokenize.reference_time
    order = config.tokenize.time_point_order
    targets = {
        order[int(point)] for point in config.model.predicted_time_points
    }
    if len(targets) != 1:
        raise ValueError(
            "Target edits span several conditions; list them separately: "
            f"{sorted(targets)}"
        )
    return targets.pop()


def _write_manifest(
    config_paths: Sequence[Path],
    *,
    results_root: Path | None,
    source: str,
    output: Path,
) -> None:
    """Combine the runs' edits into one manifest without overwriting."""
    if output.exists():
        raise FileExistsError(f"Manifest already exists: {output}")
    entries = [
        entry
        for config_path in config_paths
        for entry in build_report_manifest(
            load_perturbgen_config(config_path),
            results_root=results_root,
            source=source,
        )
    ]
    if not entries:
        raise ValueError("No completed edits were found to list")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(entries, indent=2) + "\n")
    logger.info("Wrote %d edits to %s", len(entries), output)


def _parse_arguments() -> argparse.Namespace:
    """Parse run configurations and the manifest destination."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        action="append",
        required=True,
        help="PerturbGen workflow YAML; repeat for several runs.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        help="Replacement for run.results_root_directory (copied results).",
    )
    parser.add_argument(
        "--source",
        choices=("summary", "native"),
        default="summary",
        help="List donor_state_effects summaries or native result H5ADs.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Write the report manifest for the requested PerturbGen runs."""
    logging.basicConfig(level=logging.INFO)
    args = _parse_arguments()
    _write_manifest(
        args.config,
        results_root=args.results_root,
        source=args.source,
        output=args.output,
    )


if __name__ == "__main__":
    main()
