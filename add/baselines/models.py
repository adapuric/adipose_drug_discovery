"""PCA-ridge baseline model fitting, prediction, and evaluation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import anndata as ad  # type: ignore[import]
import numpy as np
import pandas as pd
import scipy.sparse as sp  # type: ignore[import]
from sklearn.decomposition import PCA  # type: ignore[import]
from sklearn.linear_model import Ridge  # type: ignore[import]
from sklearn.preprocessing import OneHotEncoder  # type: ignore[import]

from add.baselines.evaluation import split_signature_contexts
from add.baselines.scoring import _canonical_score_row
from add.baselines.scoring import _mimic_fields
from add.baselines.scoring import _rank_scores
from add.baselines.scoring import _validated_rescue_vectors
from add.baselines.signatures import _collapse_metadata
from add.baselines.signatures import _context_columns
from add.baselines.signatures import _context_keys
from add.baselines.signatures import _context_label
from add.baselines.signatures import _dense_matrix
from add.baselines.signatures import _dense_row
from add.baselines.signatures import _format_context_key
from add.baselines.signatures import _gene_positions
from add.baselines.signatures import _metadata_annotation_from_frame
from add.baselines.signatures import _require_finite
from add.baselines.signatures import _required_string_values
from add.baselines.signatures import _root_mean_squared_error
from add.baselines.signatures import _signature_ids
from add.baselines.signatures import _signature_positions
from add.baselines.signatures import _subset_signatures
from add.baselines.signatures import _with_signature_ids
from add.perturb import PerturbSignatures


@dataclass(frozen=True)
class PcaRidgeModel:
    """Fitted context PCA, drug encoding, and multi-output ridge model."""

    genes: tuple[str, ...]
    pca: PCA
    drug_encoder: OneHotEncoder
    ridge: Ridge
    drug_col: str
    context_cols: tuple[str, ...]
    training_drugs: tuple[str, ...]
    training_contexts: tuple[str, ...]
    drug_metadata: pd.DataFrame
    random_seed: int


def fit_pca_ridge(
    signatures: PerturbSignatures,
    *,
    drug_col: str,
    context_col: str | Sequence[str],
    n_components: int = 30,
    ridge_alpha: float = 10.0,
    model_genes: Sequence[str] | None = None,
    max_model_genes: int | None = None,
    target_col: str | None = "target",
    mechanism_col: str | None = "mechanism",
    random_seed: int = 0,
) -> PcaRidgeModel:
    """Fit PCA(control state) plus one-hot drug identity to measured deltas."""
    if signatures.control is None:
        raise ValueError("PCA + ridge requires matched control expression.")
    context_cols = _context_columns(context_col)
    required_columns = [drug_col, *context_cols]
    missing_columns = [
        column for column in required_columns if column not in signatures.meta
    ]
    if missing_columns:
        raise KeyError(
            f"Perturbation metadata lacks model columns: {missing_columns}"
        )
    if n_components < 1:
        raise ValueError("n_components must be at least 1.")
    if ridge_alpha < 0.0 or not np.isfinite(ridge_alpha):
        raise ValueError("ridge_alpha must be finite and non-negative.")

    gene_positions = _select_model_gene_positions(
        signatures,
        model_genes=model_genes,
        max_model_genes=max_model_genes,
    )
    genes = tuple(str(signatures.genes[index]) for index in gene_positions)
    control = _dense_matrix(signatures.control[:, gene_positions])
    delta = _dense_matrix(signatures.delta[:, gene_positions])
    _require_finite(control, name="control expression")
    _require_finite(delta, name="perturbation deltas")

    context_values = _context_keys(signatures.meta, context_cols)
    unique_control = _mean_matrix_by_labels(control, context_values)
    resolved_components = min(
        n_components,
        unique_control.shape[0],
        unique_control.shape[1],
    )
    pca = PCA(
        n_components=resolved_components,
        svd_solver="full",
        random_state=random_seed,
    )
    pca.fit(unique_control)
    control_components = pca.transform(control)

    drug_values = _required_string_values(signatures.meta, drug_col)
    drug_encoder = OneHotEncoder(
        handle_unknown="error",
        sparse_output=False,
        dtype=np.float64,
    )
    encoded_drugs = np.asarray(
        drug_encoder.fit_transform(drug_values.reshape(-1, 1)),
        dtype=float,
    )
    predictors = np.concatenate(
        [control_components, encoded_drugs],
        axis=1,
    )
    ridge = Ridge(alpha=ridge_alpha)
    ridge.fit(predictors, delta)

    drug_metadata = _summarize_drug_metadata(
        signatures.meta,
        drug_col=drug_col,
        context_col=context_col,
        target_col=target_col,
        mechanism_col=mechanism_col,
    )
    return PcaRidgeModel(
        genes=genes,
        pca=pca,
        drug_encoder=drug_encoder,
        ridge=ridge,
        drug_col=drug_col,
        context_cols=context_cols,
        training_drugs=tuple(sorted(set(drug_values))),
        training_contexts=tuple(
            _format_context_key(key, context_cols)
            for key in sorted(set(context_values))
        ),
        drug_metadata=drug_metadata,
        random_seed=random_seed,
    )


def build_adipose_starting_expression(
    pseudobulk: ad.AnnData,
    *,
    donor_col: str,
    condition_col: str,
    state_col: str,
    baseline_label: str,
    paired_donors: Mapping[str, Sequence[str]] | None = None,
    count_layer: str | None = None,
    counts_per_million: float = 1_000_000.0,
) -> pd.DataFrame:
    """Return equal-donor mean baseline adipose log1p-CPM by state.

    Each pseudobulk donor is library-normalized before averaging. Nuclei-rich
    donors therefore do not receive additional biological weight.
    """
    required = [donor_col, condition_col, state_col]
    missing = [column for column in required if column not in pseudobulk.obs]
    if missing:
        raise KeyError(f"Pseudobulk metadata lacks columns: {missing}")
    if counts_per_million <= 0.0 or not np.isfinite(counts_per_million):
        raise ValueError("counts_per_million must be finite and positive.")
    if count_layer is not None and count_layer not in pseudobulk.layers:
        raise KeyError(f"Pseudobulk count layer {count_layer!r} is absent.")

    baseline_mask = (
        pseudobulk.obs[condition_col].astype(str).to_numpy() == baseline_label
    )
    baseline_positions = np.flatnonzero(baseline_mask)
    if baseline_positions.size == 0:
        raise ValueError(
            f"No pseudobulk rows have condition {baseline_label!r}."
        )

    baseline_obs = pd.DataFrame(pseudobulk.obs).iloc[baseline_positions].copy()
    duplicated = baseline_obs.duplicated([state_col, donor_col], keep=False)
    if duplicated.any():
        duplicate_pairs = baseline_obs.loc[
            duplicated,
            [state_col, donor_col],
        ].drop_duplicates()
        raise ValueError(
            "Baseline pseudobulk must have one row per donor and state; found "
            f"duplicates: {duplicate_pairs.to_dict(orient='records')}"
        )

    if count_layer is None:
        matrix = pseudobulk.X
    else:
        matrix = pseudobulk.layers[count_layer]
    if matrix is None:
        raise ValueError("Pseudobulk contains no count matrix.")
    selected_counts = matrix[baseline_positions]
    baseline_matrix = (
        sp.csr_matrix(selected_counts)
        if sp.issparse(selected_counts)
        else np.asarray(selected_counts)
    )
    baseline_counts = _dense_matrix(baseline_matrix)
    _require_finite(baseline_counts, name="adipose pseudobulk counts")
    if (baseline_counts < 0.0).any():
        raise ValueError("Adipose pseudobulk counts must be non-negative.")
    library_sizes = baseline_counts.sum(axis=1)
    if (library_sizes <= 0.0).any():
        raise ValueError(
            "Adipose baseline pseudobulk contains an empty library."
        )
    log_cpm = np.log1p(
        baseline_counts / library_sizes[:, np.newaxis] * counts_per_million
    )

    states = baseline_obs[state_col].astype(str).to_numpy()
    donors = baseline_obs[donor_col].astype(str).to_numpy()
    state_profiles: list[np.ndarray] = []
    retained_states: list[str] = []
    for state in sorted(set(states)):
        state_mask = states == state
        if paired_donors is not None:
            if state not in paired_donors:
                continue
            allowed = {str(donor) for donor in paired_donors[state]}
            state_mask &= np.fromiter(
                (donor in allowed for donor in donors),
                dtype=bool,
                count=len(donors),
            )
        if not state_mask.any():
            continue
        state_profiles.append(np.mean(log_cpm[state_mask], axis=0))
        retained_states.append(state)
    if not state_profiles:
        raise ValueError(
            "No baseline adipose states remain after donor selection."
        )

    return pd.DataFrame(
        np.vstack(state_profiles),
        index=pd.Index(retained_states, name="state"),
        columns=pseudobulk.var_names.astype(str),
    )


def predict_pca_ridge(
    model: PcaRidgeModel,
    starting_expression: pd.DataFrame,
    *,
    drug_ids: Sequence[str],
) -> PerturbSignatures:
    """Predict drug deltas for every supplied adipose starting state."""
    if not starting_expression.index.is_unique:
        raise ValueError("Starting-expression state labels must be unique.")
    if not starting_expression.columns.is_unique:
        raise ValueError("Starting-expression gene labels must be unique.")
    missing_genes = [
        gene for gene in model.genes if gene not in starting_expression.columns
    ]
    if missing_genes:
        raise ValueError(
            f"Starting expression lacks model genes: {missing_genes[:10]}"
        )
    requested_drugs = [str(drug) for drug in drug_ids]
    if not requested_drugs:
        raise ValueError("At least one drug must be requested.")
    if len(requested_drugs) != len(set(requested_drugs)):
        raise ValueError("Requested drug IDs must be unique.")
    unknown = sorted(set(requested_drugs).difference(model.training_drugs))
    if unknown:
        raise ValueError(
            f"PCA + ridge cannot encode drugs absent from training: {unknown}"
        )

    states = starting_expression.index.astype(str).tolist()
    state_matrix = starting_expression.loc[:, list(model.genes)].to_numpy(
        dtype=float,
    )
    _require_finite(state_matrix, name="adipose starting expression")

    prediction_rows: list[np.ndarray] = []
    control_rows: list[np.ndarray] = []
    metadata_rows: list[dict[str, object]] = []
    for state, starting_row in zip(states, state_matrix, strict=True):
        repeated_start = np.repeat(
            starting_row[np.newaxis, :],
            len(requested_drugs),
            axis=0,
        )
        predicted = _predict_pca_ridge_matrix(
            model,
            repeated_start,
            requested_drugs,
        )
        prediction_rows.extend(predicted)
        control_rows.extend(repeated_start)
        for drug in requested_drugs:
            record: dict[str, object] = {
                "drug": drug,
                "state": state,
                "context": f"adipose:{state}",
                "source": "tahoe",
                "n_external_contexts": int(
                    model.drug_metadata.loc[drug, "n_external_contexts"]
                ),
            }
            for column, value in model.drug_metadata.loc[drug].items():
                if column not in record:
                    record[column] = value
            metadata_rows.append(record)

    meta = _with_signature_ids(
        pd.DataFrame(metadata_rows),
        prefix="pca_ridge",
    )
    return PerturbSignatures(
        delta=np.vstack(prediction_rows),
        genes=list(model.genes),
        meta=meta,
        control=np.vstack(control_rows),
        provenance={
            "baseline": "pca-ridge",
            "random_seed": model.random_seed,
            "input": "equal-donor baseline adipose log1p-CPM",
        },
    )


def evaluate_pca_ridge(
    signatures: PerturbSignatures,
    *,
    drug_col: str,
    context_col: str | Sequence[str],
    n_components: int = 30,
    ridge_alpha: float = 10.0,
    model_genes: Sequence[str] | None = None,
    max_model_genes: int | None = None,
    target_col: str | None = "target",
    mechanism_col: str | None = "mechanism",
    test_fraction: float = 0.2,
    random_seed: int = 0,
    minimum_shared_genes: int = 3,
) -> pd.DataFrame:
    """Evaluate PCA + ridge on contexts absent from model fitting."""
    if signatures.control is None:
        raise ValueError("PCA + ridge evaluation requires control expression.")
    train_ids, test_ids = split_signature_contexts(
        signatures,
        context_col=context_col,
        test_fraction=test_fraction,
        random_seed=random_seed,
    )
    training = _subset_signatures(signatures, train_ids)
    model = fit_pca_ridge(
        training,
        drug_col=drug_col,
        context_col=context_col,
        n_components=n_components,
        ridge_alpha=ridge_alpha,
        model_genes=model_genes,
        max_model_genes=max_model_genes,
        target_col=target_col,
        mechanism_col=mechanism_col,
        random_seed=random_seed,
    )

    test_positions = _signature_positions(signatures, test_ids)
    model_gene_positions = _gene_positions(signatures.genes, model.genes)
    test_control = _dense_matrix(
        signatures.control[test_positions][:, model_gene_positions]
    )
    observed_delta = _dense_matrix(
        signatures.delta[test_positions][:, model_gene_positions]
    )
    test_meta = signatures.meta.iloc[test_positions]
    context_cols = _context_columns(context_col)
    test_drugs = _required_string_values(test_meta, drug_col).tolist()
    signature_ids = _signature_ids(signatures)

    rows: list[dict[str, object]] = []
    for row_index, position in enumerate(test_positions):
        drug = test_drugs[row_index]
        base = {
            "signature_id": signature_ids[position],
            "drug": drug,
            "context": _context_label(
                test_meta.iloc[row_index],
                context_cols,
            ),
            "random_seed": random_seed,
            "baseline": "pca-ridge",
        }
        if drug not in model.training_drugs:
            rows.append(
                {
                    **base,
                    "prediction_pearson": np.nan,
                    "prediction_spearman": np.nan,
                    "prediction_rmse": np.nan,
                    "n_shared_genes": len(model.genes),
                    "score_status": "unseen_drug",
                }
            )
            continue

        predicted = _predict_pca_ridge_matrix(
            model,
            test_control[row_index : row_index + 1],
            [drug],
        )[0]
        observed = observed_delta[row_index]
        score = _mimic_fields(
            predicted,
            model.genes,
            observed,
            model.genes,
            minimum_shared_genes=minimum_shared_genes,
        )
        rows.append(
            {
                **base,
                "prediction_pearson": score["score_mimic"],
                "prediction_spearman": score["score_spearman"],
                "prediction_rmse": _root_mean_squared_error(
                    predicted,
                    observed,
                ),
                "n_shared_genes": score["n_shared_genes"],
                "score_status": score["score_status"],
            }
        )
    return pd.DataFrame(rows).sort_values(
        "signature_id",
        kind="stable",
        ignore_index=True,
    )


def score_pca_ridge(
    predicted_signatures: PerturbSignatures,
    rescue_vectors: Mapping[str, pd.Series],
    *,
    minimum_shared_genes: int = 3,
) -> pd.DataFrame:
    """Rank context-aware adipose PCA + ridge predictions by mimicry."""
    if "state" not in predicted_signatures.meta:
        raise KeyError("Predicted signature metadata lacks 'state'.")
    rescue_by_state = dict(_validated_rescue_vectors(rescue_vectors))
    rows: list[dict[str, object]] = []
    signature_ids = _signature_ids(predicted_signatures)
    for position, (_, metadata) in enumerate(
        predicted_signatures.meta.iterrows()
    ):
        state = str(metadata["state"])
        if state not in rescue_by_state:
            raise KeyError(
                f"No rescue vector was supplied for state {state!r}."
            )
        rescue = rescue_by_state[state]
        score = _mimic_fields(
            _dense_row(predicted_signatures.delta, position),
            predicted_signatures.genes,
            rescue.to_numpy(dtype=float),
            [str(gene) for gene in rescue.index],
            minimum_shared_genes=minimum_shared_genes,
        )
        rows.append(
            _canonical_score_row(
                state=state,
                score=score,
                source=str(metadata.get("source", "tahoe")),
                baseline="pca-ridge",
                score_name="pearson_mimic",
                drug=str(metadata["drug"]),
                signature_id=signature_ids[position],
                context=str(metadata["context"]),
                n_external_contexts=int(str(metadata["n_external_contexts"])),
                target=metadata.get("target", pd.NA),
                mechanism=metadata.get("mechanism", pd.NA),
            )
        )
    return _rank_scores(pd.DataFrame(rows), score_col="score")


def _select_model_gene_positions(
    signatures: PerturbSignatures,
    *,
    model_genes: Sequence[str] | None,
    max_model_genes: int | None,
) -> np.ndarray:
    """Choose top-variance genes within the caller's allowed gene set."""
    if model_genes is not None:
        selected = [str(gene) for gene in model_genes]
        if not selected or len(selected) != len(set(selected)):
            raise ValueError(
                "model_genes must contain unique gene identifiers."
            )
        candidate_positions = _gene_positions(signatures.genes, selected)
    else:
        candidate_positions = np.arange(len(signatures.genes), dtype=int)
    if max_model_genes is not None and max_model_genes < 1:
        raise ValueError("max_model_genes must be at least 1 when supplied.")
    if max_model_genes is None or max_model_genes >= len(candidate_positions):
        return candidate_positions

    delta = _dense_matrix(signatures.delta[:, candidate_positions])
    _require_finite(delta, name="training perturbation deltas")
    variance = np.var(delta, axis=0)
    gene_names = np.asarray(signatures.genes, dtype=str)[candidate_positions]
    order = np.lexsort((gene_names, -variance))
    return candidate_positions[order[:max_model_genes]]


def _predict_pca_ridge_matrix(
    model: PcaRidgeModel,
    starting_expression: np.ndarray,
    drug_ids: Sequence[str],
) -> np.ndarray:
    """Apply a fitted model to aligned starting-expression rows."""
    unknown = sorted(set(drug_ids).difference(model.training_drugs))
    if unknown:
        raise ValueError(f"Unknown drug IDs for PCA + ridge: {unknown}")
    components = model.pca.transform(starting_expression)
    encoded = np.asarray(
        model.drug_encoder.transform(
            np.asarray(drug_ids, dtype=object).reshape(-1, 1)
        ),
        dtype=float,
    )
    predictors = np.concatenate([components, encoded], axis=1)
    return np.asarray(model.ridge.predict(predictors), dtype=float)


def _summarize_drug_metadata(
    metadata: pd.DataFrame,
    *,
    drug_col: str,
    context_col: str | Sequence[str],
    target_col: str | None,
    mechanism_col: str | None,
) -> pd.DataFrame:
    """Return consensus annotations and context support per training drug."""
    context_cols = _context_columns(context_col)
    rows: list[dict[str, object]] = []
    for drug, group in metadata.groupby(
        drug_col,
        observed=True,
        sort=True,
        dropna=False,
    ):
        record: dict[str, object] = {
            "drug": str(drug),
            "n_external_contexts": int(
                group.loc[:, list(context_cols)].drop_duplicates().shape[0]
            ),
            "target": _metadata_annotation_from_frame(group, target_col),
            "mechanism": _metadata_annotation_from_frame(
                group,
                mechanism_col,
            ),
        }
        for column in group.columns:
            if (
                column == drug_col
                or column in context_cols
                or column in {target_col, mechanism_col}
            ):
                continue
            record[column] = _collapse_metadata(
                pd.Series(group[column], index=group.index)
            )
        rows.append(record)
    result = pd.DataFrame(rows).set_index("drug", drop=False)
    return result


def _mean_matrix_by_labels(
    matrix: np.ndarray,
    labels: Sequence[tuple[str, ...]],
) -> np.ndarray:
    """Average matrix rows once per sorted label."""
    return np.vstack(
        [
            np.mean(
                matrix[
                    np.fromiter(
                        (observed == label for observed in labels),
                        dtype=bool,
                        count=len(labels),
                    )
                ],
                axis=0,
            )
            for label in sorted(set(labels))
        ]
    )
