"""Canonical score construction shared by baseline methods."""

from __future__ import annotations

import multiprocessing as mp
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from add.baselines.signatures import _context_columns
from add.baselines.signatures import _context_label
from add.baselines.signatures import _dense_row
from add.baselines.signatures import _metadata_annotation
from add.baselines.signatures import _signature_ids
from add.perturb import PerturbSignatures
from add.scoring import score_mimicry
from add.scoring import weighted_cmap_connectivity


_PARALLEL_SIGNATURES: PerturbSignatures | None = None


def _score_all_signatures(
    signatures: PerturbSignatures,
    rescue_vectors: Mapping[str, pd.Series],
    *,
    drug_col: str,
    context_col: str | Sequence[str] | None,
    minimum_shared_genes: int,
    source: str,
    baseline: str,
    target_col: str | None,
    mechanism_col: str | None,
    workers: int = 1,
) -> pd.DataFrame:
    """Score every signature against every state using shared alignment."""
    if workers < 1:
        raise ValueError("workers must be at least 1")
    context_cols = (
        None if context_col is None else _context_columns(context_col)
    )
    rescues = _validated_rescue_vectors(rescue_vectors)
    resolved_workers = min(workers, len(rescues))
    tasks = [
        (
            state,
            rescue,
            drug_col,
            context_cols,
            minimum_shared_genes,
            source,
            baseline,
            target_col,
            mechanism_col,
        )
        for state, rescue in rescues
    ]
    if resolved_workers == 1:
        frames = [_score_signature_state(signatures, *task) for task in tasks]
    else:
        global _PARALLEL_SIGNATURES
        _PARALLEL_SIGNATURES = signatures
        try:
            context = mp.get_context("fork")
            with context.Pool(processes=resolved_workers) as pool:
                frames = pool.starmap(_score_signature_state_worker, tasks)
        finally:
            _PARALLEL_SIGNATURES = None
    return pd.concat(frames, axis=0, ignore_index=True)


def _score_signature_state_worker(
    state: str,
    rescue: pd.Series,
    drug_col: str,
    context_cols: tuple[str, ...] | None,
    minimum_shared_genes: int,
    source: str,
    baseline: str,
    target_col: str | None,
    mechanism_col: str | None,
) -> pd.DataFrame:
    """Score one rescue state in a forked worker."""
    if _PARALLEL_SIGNATURES is None:
        raise RuntimeError("parallel signature matrix is unavailable")
    return _score_signature_state(
        _PARALLEL_SIGNATURES,
        state,
        rescue,
        drug_col,
        context_cols,
        minimum_shared_genes,
        source,
        baseline,
        target_col,
        mechanism_col,
    )


def _score_signature_state(
    signatures: PerturbSignatures,
    state: str,
    rescue: pd.Series,
    drug_col: str,
    context_cols: tuple[str, ...] | None,
    minimum_shared_genes: int,
    source: str,
    baseline: str,
    target_col: str | None,
    mechanism_col: str | None,
) -> pd.DataFrame:
    """Score every signature against one prevalidated rescue state."""
    rows: list[dict[str, object]] = []
    signature_ids = _signature_ids(signatures)
    rescue_values = rescue.to_numpy(dtype=float)
    rescue_genes = tuple(str(gene) for gene in rescue.index)
    for position, (_, metadata) in enumerate(signatures.meta.iterrows()):
        score = _mimic_fields(
            _dense_row(signatures.delta, position),
            signatures.genes,
            rescue_values,
            rescue_genes,
            minimum_shared_genes=minimum_shared_genes,
        )
        context: object = pd.NA
        if context_cols is not None:
            context = _context_label(metadata, context_cols)
        rows.append(
            _canonical_score_row(
                state=state,
                score=score,
                source=source,
                baseline=baseline,
                score_name="pearson_mimic",
                drug=str(metadata[drug_col]),
                signature_id=signature_ids[position],
                context=context,
                n_external_contexts=int(
                    str(metadata.get("n_external_contexts", 1))
                ),
                target=_metadata_annotation(metadata, target_col),
                mechanism=_metadata_annotation(metadata, mechanism_col),
            )
        )
    return pd.DataFrame(rows)


def _rank_scores(results: pd.DataFrame, *, score_col: str) -> pd.DataFrame:
    """Assign descending within-state ranks while preserving missing scores."""
    ranked = results.copy()
    ranked["rank"] = pd.Series(pd.NA, index=ranked.index, dtype="Int64")
    scorable = ranked["score_status"].eq("ok") & np.isfinite(ranked[score_col])
    ranked.loc[scorable, "rank"] = (
        ranked.loc[scorable]
        .groupby("state", observed=True)[score_col]
        .rank(method="min", ascending=False)
        .astype("Int64")
    )
    return ranked.sort_values(
        ["state", "rank", "drug"],
        kind="stable",
        na_position="last",
        ignore_index=True,
    )


def _canonical_score_row(
    *,
    state: str,
    score: Mapping[str, object],
    source: str,
    baseline: str,
    score_name: str,
    drug: object,
    signature_id: object,
    context: object,
    n_external_contexts: object,
    target: object = pd.NA,
    mechanism: object = pd.NA,
    n_training_signatures: int | None = None,
    primary_score: object | None = None,
    connectivity_score: object = np.nan,
    score_status: str | None = None,
) -> dict[str, object]:
    """Build one stable score row shared by all four baselines."""
    resolved_primary = score["score_mimic"]
    if primary_score is not None:
        resolved_primary = primary_score
    row: dict[str, object] = {
        "signature_id": signature_id,
        "drug": drug,
        "target": target,
        "mechanism": mechanism,
        "context": context,
        "state": state,
        "score": resolved_primary,
        "score_name": score_name,
        "score_mimic": score["score_mimic"],
        "score_spearman": score["score_spearman"],
        "score_connectivity": connectivity_score,
        "n_shared_genes": score["n_shared_genes"],
        "n_external_contexts": n_external_contexts,
        "source": source,
        "baseline": baseline,
        "score_status": score["score_status"]
        if score_status is None
        else score_status,
    }
    if n_training_signatures is not None:
        row["n_training_signatures"] = n_training_signatures
    return row


def _mimic_fields(
    candidate_delta: np.ndarray,
    candidate_genes: Sequence[str],
    rescue_delta: np.ndarray,
    rescue_genes: Sequence[str],
    *,
    minimum_shared_genes: int,
) -> dict[str, object]:
    """Adapt the shared mimic scorer to stable baseline column names."""
    result = score_mimicry(
        candidate_delta,
        candidate_genes,
        rescue_delta,
        rescue_genes,
        minimum_shared_genes=minimum_shared_genes,
    )
    return {
        "score_mimic": result.pearson,
        "score_spearman": result.spearman,
        "n_shared_genes": result.n_shared,
        "score_status": result.status,
    }


def _connectivity_fields(
    candidate_delta: np.ndarray,
    candidate_genes: Sequence[str],
    rescue_delta: np.ndarray,
    rescue_genes: Sequence[str],
    *,
    query_genes_per_direction: int,
    minimum_query_genes: int,
) -> dict[str, object]:
    """Adapt shared weighted connectivity to canonical output fields."""
    result = weighted_cmap_connectivity(
        candidate_delta,
        candidate_genes,
        rescue_delta,
        rescue_genes,
        maximum_query_genes=query_genes_per_direction,
        minimum_query_genes=minimum_query_genes,
    )
    return {
        "score_connectivity": result.score_connectivity,
        "score_status": result.status,
    }


def _validated_rescue_vectors(
    rescue_vectors: Mapping[str, pd.Series],
) -> list[tuple[str, pd.Series]]:
    """Return sorted rescue vectors with unique gene identifiers."""
    if not rescue_vectors:
        raise ValueError("At least one rescue vector is required.")
    validated: list[tuple[str, pd.Series]] = []
    for state in sorted(rescue_vectors):
        rescue = rescue_vectors[state]
        if not rescue.index.is_unique:
            raise ValueError(
                f"Rescue genes must be unique for state {state!r}."
            )
        validated.append((str(state), rescue))
    return validated
