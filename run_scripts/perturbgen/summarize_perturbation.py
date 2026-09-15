"""Summarize one PerturbGen result at the donor and target-state level."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, cast

import anndata as ad  # type: ignore[import]
import numpy as np
import pandas as pd


def summarize_perturbation_result(
    result_path: str | Path,
    *,
    output_directory: str | Path,
    donor_col: str,
    state_col: str,
    top_genes_per_state: int = 25,
) -> dict[str, Path]:
    """Write descriptive donor-level perturbation effects and QC tables.

    PerturbGen stores perturbed predictions in X, unperturbed predictions in
    layers["pred_counts"], and observed target counts in layers["true_counts"].
    Nuclei are averaged within donor and target state before donors are
    summarized.
    """
    input_path = Path(result_path).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"PerturbGen result not found: {input_path}")

    adata = ad.read_h5ad(input_path)
    _validate_result(adata, donor_col=donor_col, state_col=state_col)

    genes = adata.var_names.astype(str).to_numpy()
    obs = cast(pd.DataFrame, adata.obs)
    frames: list[pd.DataFrame] = []

    grouping = obs.groupby(
        [donor_col, state_col],
        observed=True,
        dropna=False,
        sort=True,
    )
    for group_key, positions in grouping.indices.items():
        donor, state = cast(tuple[object, object], group_key)
        perturbed = _mean_rows(adata.X, positions)  # type: ignore
        predicted = _mean_rows(adata.layers["pred_counts"], positions)  # type: ignore
        observed = _mean_rows(adata.layers["true_counts"], positions)  # type: ignore
        frames.append(
            pd.DataFrame(
                {
                    donor_col: donor,
                    state_col: state,
                    "gene": genes,
                    "n_nuclei": len(positions),
                    "mean_perturbed_count": perturbed,
                    "mean_unperturbed_count": predicted,
                    "mean_observed_target_count": observed,
                    "model_perturbation_delta": perturbed - predicted,
                    "model_log2_fold_change": np.log2(
                        (perturbed + 1.0) / (predicted + 1.0)
                    ),
                }
            )
        )

    donor_effects = pd.concat(frames, ignore_index=True)
    state_summary = (
        donor_effects.groupby(
            [state_col, "gene"],
            observed=True,
            dropna=False,
            sort=True,
        )
        .agg(
            n_donors=(donor_col, "nunique"),
            mean_model_perturbation_delta=(
                "model_perturbation_delta",
                "mean",
            ),
            sd_model_perturbation_delta=(
                "model_perturbation_delta",
                "std",
            ),
            mean_model_log2_fold_change=(
                "model_log2_fold_change",
                "mean",
            ),
            sd_model_log2_fold_change=(
                "model_log2_fold_change",
                "std",
            ),
            mean_perturbed_count=("mean_perturbed_count", "mean"),
            mean_unperturbed_count=("mean_unperturbed_count", "mean"),
            mean_observed_target_count=(
                "mean_observed_target_count",
                "mean",
            ),
        )
        .reset_index()
    )
    state_summary["absolute_mean_log2_fold_change"] = state_summary[
        "mean_model_log2_fold_change"
    ].abs()

    top_effects = (
        state_summary.sort_values(
            [state_col, "absolute_mean_log2_fold_change", "gene"],
            ascending=[True, False, True],
        )
        .groupby(state_col, observed=True, dropna=False, sort=False)
        .head(top_genes_per_state)
        .drop(columns="absolute_mean_log2_fold_change")
    )
    state_summary = state_summary.drop(columns="absolute_mean_log2_fold_change")
    group_qc = donor_effects[
        [donor_col, state_col, "n_nuclei"]
    ].drop_duplicates()

    destination = Path(output_directory).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    paths = {
        "donor_state_effects": destination / "donor_state_effects.csv.gz",
        "state_summary": destination / "state_summary.csv.gz",
        "top_effects": destination / "top_effects.csv",
        "group_qc": destination / "group_qc.csv",
    }
    donor_effects.to_csv(paths["donor_state_effects"], index=False)
    state_summary.to_csv(paths["state_summary"], index=False)
    top_effects.to_csv(paths["top_effects"], index=False)
    group_qc.to_csv(paths["group_qc"], index=False)
    return paths


def _validate_result(
    adata: ad.AnnData,
    *,
    donor_col: str,
    state_col: str,
) -> None:
    """Validate fields required for a donor-aware descriptive summary."""
    if adata.n_obs == 0 or adata.n_vars == 0:
        raise ValueError("PerturbGen result has no observations or genes.")

    if adata.X is None:
        raise ValueError("PerturbGen result X has no perturbed counts.")

    if missing_layers := {
        layer
        for layer in ("pred_counts", "true_counts")
        if layer not in adata.layers
    }:
        raise ValueError(
            "PerturbGen result is missing required layers: "
            f"{sorted(missing_layers)}."
        )

    if missing_obs := {
        column for column in (donor_col, state_col) if column not in adata.obs
    }:
        raise ValueError(
            "PerturbGen result is missing required obs columns: "
            f"{sorted(missing_obs)}."
        )

    obs = cast(pd.DataFrame, adata.obs)
    if columns := [
        column
        for column in (donor_col, state_col)
        if bool(obs[column].isna().any())
    ]:
        raise ValueError(
            f"PerturbGen result has missing donor/state values in: {columns}."
        )

    if not adata.var_names.is_unique:
        raise ValueError("PerturbGen result gene identifiers are not unique.")


def _mean_rows(
    matrix: Any,
    positions: np.ndarray[Any, np.dtype[np.integer]],
) -> np.ndarray:
    """Return column means without densifying the complete matrix."""
    return np.asarray(matrix[positions].mean(axis=0)).reshape(-1)


def _parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--donor-col", required=True)
    parser.add_argument("--state-col", required=True)
    parser.add_argument("--top-genes-per-state", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    """Summarize one PerturbGen result."""
    arguments = _parse_arguments()
    paths = summarize_perturbation_result(
        arguments.input,
        output_directory=arguments.output_dir,
        donor_col=arguments.donor_col,
        state_col=arguments.state_col,
        top_genes_per_state=arguments.top_genes_per_state,
    )
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
