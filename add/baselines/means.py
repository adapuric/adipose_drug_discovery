"""Perturbed-mean and per-drug mean baseline methods."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from add.baselines.evaluation import _context_score_summary
from add.baselines.evaluation import split_signature_contexts
from add.baselines.scoring import _canonical_score_row
from add.baselines.scoring import _mimic_fields
from add.baselines.scoring import _rank_scores
from add.baselines.scoring import _score_all_signatures
from add.baselines.scoring import _validated_rescue_vectors
from add.baselines.signatures import _average_drug_contexts
from add.baselines.signatures import _average_signature_groups
from add.baselines.signatures import _context_columns
from add.baselines.signatures import _context_keys
from add.baselines.signatures import _context_label
from add.baselines.signatures import _dense_matrix
from add.baselines.signatures import _dense_row
from add.baselines.signatures import _require_finite
from add.baselines.signatures import _root_mean_squared_error
from add.baselines.signatures import _signature_ids
from add.baselines.signatures import _signature_positions
from add.perturb import PerturbSignatures


def perturbed_mean_signature(
    signatures: PerturbSignatures,
    *,
    training_signature_ids: Sequence[str] | None = None,
) -> np.ndarray:
    """Return the gene-wise mean over training perturbation signatures.

    The caller can supply the training IDs from a held-out split. Test rows are
    never consulted when that argument is present.
    """
    positions = _signature_positions(signatures, training_signature_ids)
    training_delta = _dense_matrix(signatures.delta[positions])
    _require_finite(training_delta, name="training perturbation deltas")
    return np.mean(training_delta, axis=0)


def evaluate_perturbed_mean(
    signatures: PerturbSignatures,
    *,
    context_col: str | Sequence[str],
    drug_col: str,
    test_fraction: float = 0.2,
    random_seed: int = 0,
    minimum_shared_genes: int = 3,
) -> pd.DataFrame:
    """Evaluate the training-only mean against held-out context signatures."""
    train_ids, test_ids = split_signature_contexts(
        signatures,
        context_col=context_col,
        test_fraction=test_fraction,
        random_seed=random_seed,
    )

    prediction = perturbed_mean_signature(
        signatures,
        training_signature_ids=train_ids,
    )

    test_positions = _signature_positions(signatures, test_ids)
    context_cols = _context_columns(context_col)
    signature_ids = _signature_ids(signatures)
    rows: list[dict[str, object]] = []
    for position in test_positions:
        observed = _dense_row(signatures.delta, position)
        score = _mimic_fields(
            prediction,
            signatures.genes,
            observed,
            signatures.genes,
            minimum_shared_genes=minimum_shared_genes,
        )
        rows.append(
            {
                "signature_id": signature_ids[position],
                "drug": str(signatures.meta.iloc[position][drug_col]),
                "context": _context_label(
                    signatures.meta.iloc[position],
                    context_cols,
                ),
                "prediction_pearson": score["score_mimic"],
                "prediction_spearman": score["score_spearman"],
                "prediction_rmse": _root_mean_squared_error(
                    prediction,
                    observed,
                ),
                "n_shared_genes": score["n_shared_genes"],
                "score_status": score["score_status"],
                "n_training_signatures": len(train_ids),
                "random_seed": random_seed,
                "baseline": "perturbed-mean",
            }
        )

    return pd.DataFrame(rows).sort_values(
        "signature_id",
        kind="stable",
        ignore_index=True,
    )


def score_perturbed_mean(
    signatures: PerturbSignatures,
    rescue_vectors: Mapping[str, pd.Series],
    *,
    training_signature_ids: Sequence[str] | None = None,
    context_col: str | Sequence[str] | None = None,
    minimum_shared_genes: int = 3,
    source: str = "external",
) -> pd.DataFrame:
    """Score one generic training mean per adipose state.

    The output deliberately contains no drug rows or drug ordering because the
    prediction is identical for every perturbation identity.
    """
    mean_delta = perturbed_mean_signature(
        signatures,
        training_signature_ids=training_signature_ids,
    )
    training_positions = _signature_positions(
        signatures,
        training_signature_ids,
    )
    n_training = len(training_positions)
    n_contexts: object = pd.NA
    if context_col is not None:
        context_cols = _context_columns(context_col)
        context_keys = _context_keys(
            signatures.meta.iloc[training_positions],
            context_cols,
        )
        n_contexts = len(set(context_keys))

    rows: list[dict[str, object]] = []
    for state, rescue in _validated_rescue_vectors(rescue_vectors):
        score = _mimic_fields(
            mean_delta,
            signatures.genes,
            rescue.to_numpy(dtype=float),
            [str(gene) for gene in rescue.index],
            minimum_shared_genes=minimum_shared_genes,
        )
        rows.append(
            _canonical_score_row(
                state=state,
                score=score,
                source=source,
                baseline="perturbed-mean",
                score_name="pearson_mimic",
                drug=pd.NA,
                signature_id="PERTURBED_MEAN",
                context=pd.NA,
                n_external_contexts=n_contexts,
                n_training_signatures=n_training,
            )
        )

    result = pd.DataFrame(rows)
    result["rank"] = pd.array(
        [1 if status == "ok" else pd.NA for status in result["score_status"]],
        dtype="Int64",
    )

    return result


def mean_drug_signatures(
    signatures: PerturbSignatures,
    *,
    drug_col: str,
    context_col: str | Sequence[str],
) -> PerturbSignatures:
    """Average each drug equally across its observed external contexts."""
    context_cols = _context_columns(context_col)
    context_means = _average_signature_groups(
        signatures,
        group_cols=(drug_col, *context_cols),
        id_prefix="drug_context",
    )
    return _average_drug_contexts(
        context_means,
        drug_col=drug_col,
        context_cols=context_cols,
    )


def score_mean_drug(
    signatures: PerturbSignatures,
    rescue_vectors: Mapping[str, pd.Series],
    *,
    drug_col: str,
    context_col: str | Sequence[str],
    minimum_shared_genes: int = 3,
    source: str = "tahoe",
    target_col: str | None = "target",
    mechanism_col: str | None = "mechanism",
    workers: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return mean-drug rankings and the corresponding context scores."""
    context_cols = _context_columns(context_col)
    context_means = _average_signature_groups(
        signatures,
        group_cols=(drug_col, *context_cols),
        id_prefix="drug_context",
    )
    drug_means = _average_drug_contexts(
        context_means,
        drug_col=drug_col,
        context_cols=context_cols,
    )
    context_scores = _score_all_signatures(
        context_means,
        rescue_vectors,
        drug_col=drug_col,
        context_col=context_cols,
        minimum_shared_genes=minimum_shared_genes,
        source=source,
        baseline="mean-drug-context",
        target_col=target_col,
        mechanism_col=mechanism_col,
        workers=workers,
    )
    ranked = _score_all_signatures(
        drug_means,
        rescue_vectors,
        drug_col=drug_col,
        context_col=None,
        minimum_shared_genes=minimum_shared_genes,
        source=source,
        baseline="mean-drug",
        target_col=target_col,
        mechanism_col=mechanism_col,
        workers=workers,
    )
    ranked = _rank_scores(ranked, score_col="score")

    context_summary = _context_score_summary(context_scores)
    ranked = ranked.merge(
        context_summary,
        on=["state", "drug"],
        how="left",
        validate="one_to_one",
    )
    return ranked, context_scores
