"""Read model-sensitivity exports without importing native model
infrastructure.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

from add.validation.adipose_benchmark import change_prediction_metrics


def classify_perturbgen_run(
    manifest: Mapping[str, object],
    *,
    test_donors: Sequence[str],
) -> str:
    """Classify training exposure and access to observed target information."""
    required = (
        "native_revision",
        "checkpoint_id",
        "config_id",
        "preprocessing_policy",
    )
    if any(not manifest.get(key) for key in required):
        return "unverified_provenance"
    stages = manifest.get("training_donors_by_stage")
    information = manifest.get("target_information")
    if not isinstance(stages, dict) or not isinstance(information, dict):
        return "unverified_provenance"
    if set(stages) != {
        "masking",
        "decoder",
        "checkpoint_selection",
        "preprocessing",
    }:
        return "unverified_provenance"
    if set(information) != {"expression", "tokens_or_order", "size_factors"}:
        return "unverified_provenance"
    if not all(isinstance(value, bool) for value in information.values()):
        return "unverified_provenance"
    exposed: set[str] = set()
    for donors in stages.values():
        if not isinstance(donors, list) or not all(
            isinstance(donor, str) and donor for donor in donors
        ):
            return "unverified_provenance"
        exposed.update(donors)
    if not stages["masking"] or not stages["decoder"]:
        return "unverified_provenance"
    heldout = not exposed.intersection(test_donors)
    if manifest["preprocessing_policy"] == "transductive_hvg":
        heldout = False
    prefix = "heldout" if heldout else "in_sample"
    suffix = "target_informed" if any(information.values()) else "target_blind"
    return f"{prefix}_{suffix}"


def no_downstream_effect(
    unedited: pd.DataFrame,
    *,
    edited_genes: Sequence[str],
) -> pd.DataFrame:
    """Return exact downstream zero with missing predictions kept missing."""
    if not unedited.columns.is_unique or not unedited.index.is_unique:
        raise ValueError("Unedited profiles require unique sample and gene IDs")
    downstream = unedited.loc[:, ~unedited.columns.isin(edited_genes)]
    return downstream.where(downstream.isna(), 0.0)


def normalized_edit_effects(
    edited: pd.DataFrame,
    unedited: pd.DataFrame,
    *,
    target_sum: float = 1_000_000,
) -> pd.DataFrame:
    """Use an unedited size factor for both profiles to avoid closure
    artifacts.
    """
    if not edited.index.equals(unedited.index) or set(edited.columns) != set(
        unedited.columns
    ):
        raise ValueError(
            "Edited and unedited profiles must have identical support"
        )
    edited = edited.loc[:, unedited.columns]
    if not np.isfinite(target_sum) or target_sum <= 0:
        raise ValueError("target_sum must be positive and finite")
    if (edited < 0).any().any() or (unedited < 0).any().any():
        raise ValueError("Model count predictions must be nonnegative")
    complete = unedited.notna().all(axis=1) & edited.notna().all(axis=1)
    totals = unedited.sum(axis=1).where(complete)
    factors = target_sum / totals.where(totals > 0)
    return pd.DataFrame(
        np.log1p(edited.mul(factors, axis=0).to_numpy())
        - np.log1p(unedited.mul(factors, axis=0).to_numpy()),
        index=unedited.index,
        columns=unedited.columns,
    )


def read_native_edit_effects(
    result_path: Path,
    *,
    run_id: str,
    edit_id: str,
    edited_genes: Sequence[str],
    donor_col: str = "Donor",
    state_col: str = "cell_states_adipocytes",
    chunk_size: int = 4096,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate existing edited/unedited count predictions in bounded
    chunks.
    """
    if chunk_size < 1 or not edited_genes:
        raise ValueError(
            "Positive chunk size and explicit edited gene IDs required"
        )
    with h5py.File(result_path, "r") as handle:
        obs = cast(
            pd.DataFrame, ad.io.read_elem(cast(h5py.Group, handle["obs"]))
        )
        var = cast(
            pd.DataFrame, ad.io.read_elem(cast(h5py.Group, handle["var"]))
        )
        if not {donor_col, state_col} <= set(obs):
            raise ValueError(
                "Native output lacks declared donor/state metadata"
            )
        if (
            obs[[donor_col, state_col]].isna().any().any()
            or not obs.index.is_unique
        ):
            raise ValueError(
                "Native output requires nonmissing groups and unique row IDs"
            )
        if not var.index.is_unique:
            raise ValueError("Native output gene IDs must be unique")
        if not set(edited_genes) <= set(var.index.astype(str)):
            raise ValueError(
                "Edited gene IDs must use the native output namespace"
            )
        if "X" not in handle or "layers/pred_counts" not in handle:
            raise ValueError(
                "Expected edited X and unedited layers/pred_counts"
            )
        group_index = pd.MultiIndex.from_frame(
            obs[[donor_col, state_col]].astype(str)
        )
        codes, groups = pd.factorize(group_index, sort=True)
        summed: list[np.ndarray] = []
        for key in ("X", "layers/pred_counts"):
            node = handle[key]
            if isinstance(node, h5py.Group):
                matrix = ad.io.sparse_dataset(node)
            elif isinstance(node, h5py.Dataset):
                matrix = node
            else:
                raise TypeError(f"Unsupported matrix encoding: {key}")
            total = np.zeros((len(groups), len(var)), dtype=float)
            if matrix.shape != (len(obs), len(var)):
                raise ValueError(
                    "Native count matrix and metadata are misaligned"
                )
            for start in range(0, len(obs), chunk_size):
                stop = min(start + chunk_size, len(obs))
                block = matrix[start:stop]
                values = (
                    sp.csr_matrix(block).toarray()
                    if sp.issparse(block)
                    else np.asarray(block)
                )
                if not np.isfinite(values).all() or (values < 0).any():
                    raise ValueError(
                        "Native predictions must be finite nonnegative counts"
                    )
                np.add.at(total, codes[start:stop], values)
            summed.append(total)
    genes = pd.Index(var.index.astype(str), name="gene_id")
    index = pd.MultiIndex.from_tuples(list(groups), names=["donor_id", "state"])
    edited = pd.DataFrame(summed[0], index=index, columns=genes)
    unedited = pd.DataFrame(summed[1], index=index, columns=genes)
    normalized = normalized_edit_effects(edited, unedited)
    cell_counts = np.bincount(codes, minlength=len(groups))
    tables: list[pd.DataFrame] = []
    qc_rows: list[dict[str, object]] = []
    for position, (donor, state) in enumerate(index):
        ids = sorted(obs.index[codes == position].astype(str))
        support_hash = hashlib.sha256("\n".join(ids).encode()).hexdigest()
        qc_rows.append(
            {
                "run_id": run_id,
                "edit_id": edit_id,
                "donor_id": donor,
                "state": state,
                "n_nuclei": int(cell_counts[position]),
                "support_hash": support_hash,
            }
        )
        for scale, effects in (
            ("shared_unedited_log1p_cpm", normalized),
            ("raw_pseudobulk_count", edited - unedited),
        ):
            base = pd.DataFrame(
                {
                    "run_id": run_id,
                    "edit_id": edit_id,
                    "donor_id": donor,
                    "state": state,
                    "gene_id": genes,
                    "delta": effects.iloc[position].to_numpy(),
                    "edited_prediction_count": summed[0][position],
                    "unedited_prediction_count": summed[1][position],
                    "is_edited_gene": genes.isin(edited_genes),
                    "scale": scale,
                    "n_nuclei": cell_counts[position],
                    "support_hash": support_hash,
                    "model": f"perturbgen:{run_id}",
                    "truth_kind": "model_sensitivity",
                }
            )
            base["status"] = np.where(
                np.isfinite(base["delta"]), "ok", "zero_unedited_library"
            )
            tables.append(base)
            null = base.loc[~base["is_edited_gene"]].copy()
            null["delta"] = np.where(null["delta"].notna(), 0.0, np.nan)
            null["edited_prediction_count"] = null["unedited_prediction_count"]
            null["model"] = "no-downstream-effect"
            tables.append(null)
    return pd.concat(tables, ignore_index=True), pd.DataFrame(qc_rows)


def read_legacy_edit_effects(
    summary_path: Path,
    *,
    run_id: str,
    edit_id: str,
    edited_genes: Sequence[str],
    donor_col: str,
    state_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read historical descriptive means without changing their scale
    semantics.
    """
    source = pd.read_csv(
        summary_path, dtype={donor_col: str, state_col: str, "gene": str}
    )
    required = {
        donor_col,
        state_col,
        "gene",
        "n_nuclei",
        "model_log2_fold_change",
    }
    if (
        not required <= set(source)
        or source.duplicated([donor_col, state_col, "gene"]).any()
    ):
        raise ValueError(
            "Legacy summary requires unique donor/state/gene effects"
        )
    result = source.rename(
        columns={
            donor_col: "donor_id",
            state_col: "state",
            "gene": "gene_id",
            "model_log2_fold_change": "delta",
        }
    )
    result["run_id"] = run_id
    result["edit_id"] = edit_id
    result["model"] = f"perturbgen:{run_id}"
    result["scale"] = "log2_ratio_of_nucleus_means_plus1"
    result["truth_kind"] = "model_sensitivity"
    result["is_edited_gene"] = result["gene_id"].isin(edited_genes)
    result["support_hash"] = pd.NA
    result["status"] = np.where(
        np.isfinite(result["delta"]), "ok", "non_finite_effect"
    )
    null = result.loc[~result["is_edited_gene"]].copy()
    null["delta"] = null["delta"].where(null["delta"].isna(), 0.0)
    null["model"] = "no-downstream-effect"
    if "mean_unperturbed_count" in null:
        null["mean_perturbed_count"] = null["mean_unperturbed_count"]
    if "model_perturbation_delta" in null:
        null["model_perturbation_delta"] = null["delta"]
    qc = result[
        ["run_id", "edit_id", "donor_id", "state", "n_nuclei", "support_hash"]
    ].drop_duplicates()
    return pd.concat([result, null], ignore_index=True), qc


def evaluate_transition_export(
    predictions: pd.DataFrame,
    reference_predictions: pd.DataFrame,
    *,
    manifest: Mapping[str, object],
) -> pd.DataFrame:
    """Admit only donor-held-out target-blind exports on exact benchmark
    support.
    """
    keys = ["fold_id", "donor_id", "state", "gene_id"]
    required = {
        *keys,
        "predicted_delta",
        "baseline_expression",
        "observed_expression",
        "scale",
    }
    if (
        predictions.empty
        or not required <= set(predictions)
        or predictions.duplicated(keys).any()
    ):
        raise ValueError(
            "Transition export requires unique fold/donor/state/gene rows"
        )
    if manifest.get("prediction_kind") != "unedited_transition":
        raise ValueError(
            "Surgery comparisons require prediction_kind=unedited_transition"
        )
    if (
        not manifest.get("fold_id")
        or not predictions["fold_id"].eq(str(manifest["fold_id"])).all()
    ):
        raise ValueError(
            "Export must identify exactly its checkpoint's outer fold"
        )
    donors = predictions["donor_id"].astype(str).unique().tolist()
    classification = classify_perturbgen_run(manifest, test_donors=donors)
    if classification != "heldout_target_blind":
        raise ValueError(
            "Direct comparison requires heldout_target_blind; "
            f"received {classification}"
        )
    if manifest.get(
        "population_match_verified"
    ) is not True or not manifest.get("profile_identity"):
        raise ValueError(
            "Verified population matching and profile identity are required"
        )
    reference = reference_predictions.loc[
        reference_predictions["model"].eq("adipose-no-change")
    ]
    predictions = predictions.loc[:, sorted(required)].copy()
    selected = reference.merge(
        predictions[keys[:3]].drop_duplicates(),
        on=keys[:3],
        validate="many_to_one",
    )
    aligned = selected.merge(
        predictions,
        on=keys,
        how="outer",
        suffixes=("_reference", ""),
        indicator=True,
        validate="one_to_one",
    )
    if not aligned["_merge"].eq("both").all():
        raise ValueError(
            "Native export and benchmark gene/population support differ"
        )
    for column in ("baseline_expression", "observed_expression"):
        if not np.allclose(
            aligned[column], aligned[f"{column}_reference"], equal_nan=False
        ):
            raise ValueError(f"Native export uses incompatible {column}")
    if not aligned["scale"].eq(aligned["scale_reference"]).all():
        raise ValueError("Native export and benchmark scales differ")
    rows: list[dict[str, object]] = []
    for (fold, donor, state), group in aligned.groupby(keys[:3], sort=True):
        observed = (
            group["observed_expression"].to_numpy()
            - group["baseline_expression"].to_numpy()
        )
        minimum = int(group["n_genes"].iloc[0])
        values = change_prediction_metrics(
            group["predicted_delta"].to_numpy(),
            observed,
            pd.Index(group["gene_id"]),
            minimum_shared_genes=minimum,
        )
        rows.append(
            {
                "task": "surgery_transition",
                "model": f"perturbgen:{manifest['run_id']}",
                "fold_id": fold,
                "donor_id": donor,
                "state": state,
                "n_genes": len(group),
                "scale": str(group["scale"].iloc[0]),
                "status": values["error_status"],
                "comparison_class": classification,
            }
            | values
        )
    return pd.DataFrame(rows)
