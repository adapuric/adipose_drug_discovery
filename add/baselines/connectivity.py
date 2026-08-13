"""Direct CMap connectivity baseline scoring."""

from __future__ import annotations

import multiprocessing as mp
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from add.baselines.scoring import _canonical_score_row
from add.baselines.scoring import _connectivity_fields
from add.baselines.scoring import _mimic_fields
from add.baselines.scoring import _rank_scores
from add.baselines.scoring import _validated_rescue_vectors
from add.baselines.signatures import _collapse_metadata
from add.baselines.signatures import _context_columns
from add.baselines.signatures import _context_label
from add.baselines.signatures import _dense_row
from add.baselines.signatures import _metadata_annotation
from add.baselines.signatures import _signature_ids
from add.perturb import PerturbSignatures


_PARALLEL_SIGNATURES: PerturbSignatures | None = None


def score_cmap(
    signatures: PerturbSignatures,
    rescue_vectors: Mapping[str, pd.Series],
    *,
    drug_col: str,
    context_col: str | Sequence[str],
    query_genes_per_direction: int = 150,
    minimum_query_genes: int = 10,
    minimum_shared_genes: int = 3,
    source: str = "lincs",
    target_col: str | None = "target",
    mechanism_col: str | None = "mechanism",
    workers: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score measured LINCS contexts and rank drugs by median connectivity.

    Connectivity is positive when rescue-up genes rank high and rescue-down
    genes rank low in the measured drug signature. Drug-level ranking uses the
    median of direct context scores; individual contexts remain in the second
    returned table.
    """
    if workers < 1:
        raise ValueError("workers must be at least 1")
    context_cols = _context_columns(context_col)
    rescues = _validated_rescue_vectors(rescue_vectors)
    resolved_workers = min(workers, len(rescues))
    tasks = [
        (
            state,
            rescue,
            drug_col,
            context_cols,
            query_genes_per_direction,
            minimum_query_genes,
            minimum_shared_genes,
            source,
            target_col,
            mechanism_col,
        )
        for state, rescue in rescues
    ]
    if resolved_workers == 1:
        frames = [_score_cmap_state(signatures, *task) for task in tasks]
    else:
        global _PARALLEL_SIGNATURES
        _PARALLEL_SIGNATURES = signatures
        try:
            context = mp.get_context("fork")
            with context.Pool(processes=resolved_workers) as pool:
                frames = pool.starmap(_score_cmap_state_worker, tasks)
        finally:
            _PARALLEL_SIGNATURES = None
    context_scores = pd.concat(frames, axis=0, ignore_index=True)
    ranked = _median_cmap_drug_scores(context_scores)
    return ranked, context_scores


def _score_cmap_state_worker(
    state: str,
    rescue: pd.Series,
    drug_col: str,
    context_cols: tuple[str, ...],
    query_genes_per_direction: int,
    minimum_query_genes: int,
    minimum_shared_genes: int,
    source: str,
    target_col: str | None,
    mechanism_col: str | None,
) -> pd.DataFrame:
    """Score one CMap rescue state in a forked worker."""
    if _PARALLEL_SIGNATURES is None:
        raise RuntimeError("parallel signature matrix is unavailable")

    return _score_cmap_state(
        _PARALLEL_SIGNATURES,
        state,
        rescue,
        drug_col,
        context_cols,
        query_genes_per_direction,
        minimum_query_genes,
        minimum_shared_genes,
        source,
        target_col,
        mechanism_col,
    )


def _score_cmap_state(
    signatures: PerturbSignatures,
    state: str,
    rescue: pd.Series,
    drug_col: str,
    context_cols: tuple[str, ...],
    query_genes_per_direction: int,
    minimum_query_genes: int,
    minimum_shared_genes: int,
    source: str,
    target_col: str | None,
    mechanism_col: str | None,
) -> pd.DataFrame:
    """Score every measured signature against one rescue state."""
    rows: list[dict[str, object]] = []
    signature_ids = _signature_ids(signatures)
    rescue_values = rescue.to_numpy(dtype=float)
    rescue_genes = tuple(str(gene) for gene in rescue.index)
    for position, (_, metadata) in enumerate(signatures.meta.iterrows()):
        candidate = _dense_row(signatures.delta, position)
        mimic = _mimic_fields(
            candidate,
            signatures.genes,
            rescue_values,
            rescue_genes,
            minimum_shared_genes=minimum_shared_genes,
        )
        connectivity = _connectivity_fields(
            candidate,
            signatures.genes,
            rescue_values,
            rescue_genes,
            query_genes_per_direction=query_genes_per_direction,
            minimum_query_genes=minimum_query_genes,
        )
        status = str(connectivity["score_status"])
        rows.append(
            _canonical_score_row(
                state=state,
                score=mimic,
                source=source,
                baseline="cmap-context",
                score_name="weighted_cmap_connectivity",
                drug=str(metadata[drug_col]),
                signature_id=signature_ids[position],
                context=_context_label(metadata, context_cols),
                n_external_contexts=1,
                target=_metadata_annotation(metadata, target_col),
                mechanism=_metadata_annotation(metadata, mechanism_col),
                primary_score=connectivity["score_connectivity"],
                connectivity_score=connectivity["score_connectivity"],
                score_status=status,
            )
        )
    return pd.DataFrame(rows)


def _median_cmap_drug_scores(context_scores: pd.DataFrame) -> pd.DataFrame:
    """Aggregate direct LINCS context scores with a documented median."""
    rows: list[dict[str, object]] = []
    for keys, group in context_scores.groupby(
        ["state", "drug"],
        observed=True,
        sort=True,
        dropna=False,
    ):
        key_values = list(keys if isinstance(keys, tuple) else (keys,))
        if len(key_values) != 2:
            raise RuntimeError("CMap aggregation expected state and drug keys.")
        state = str(key_values[0])
        drug = str(key_values[1])
        finite = group.loc[np.isfinite(group["score"])]
        status = "no_scorable_contexts" if finite.empty else "ok"
        score = np.nan if finite.empty else float(finite["score"].median())

        rows.append(
            {
                "signature_id": pd.NA,
                "drug": drug,
                "target": _collapse_metadata(
                    pd.Series(group["target"], index=group.index)
                ),
                "mechanism": _collapse_metadata(
                    pd.Series(group["mechanism"], index=group.index)
                ),
                "state": state,
                "score": score,
                "score_name": "median_weighted_cmap_connectivity",
                "score_mimic": np.nan
                if finite.empty
                else float(finite["score_mimic"].median()),
                "score_spearman": np.nan
                if finite.empty
                else float(finite["score_spearman"].median()),
                "score_connectivity": score,
                "n_shared_genes": int(
                    str(pd.Series(group["n_shared_genes"]).max())
                ),
                "n_external_contexts": int(
                    pd.Series(group["context"]).nunique()
                ),
                "source": _collapse_metadata(
                    pd.Series(group["source"], index=group.index)
                ),
                "baseline": "cmap",
                "score_status": status,
                "context_score_max": np.nan
                if finite.empty
                else float(finite["score"].max()),
                "context_fraction_positive": np.nan
                if finite.empty
                else float((finite["score"] > 0).mean()),
            }
        )

    return _rank_scores(pd.DataFrame(rows), score_col="score")
