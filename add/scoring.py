"""Gene-aligned mimicry and rank-connectivity scoring."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats  # type: ignore[import]

from add.perturb import PerturbSignatures


@dataclass(frozen=True)
class MimicScore:
    """Pearson and Spearman mimicry diagnostics for one candidate.

    Positive values mean that the candidate follows the beneficial
    baseline-to-weightloss rescue direction. `n_shared` counts aligned genes
    with finite values in both vectors.

    Attributes:
      score_mimic: Pearson correlation in the rescue direction.
      score_spearman: Spearman correlation in the rescue direction.
      n_shared: Number of finite, explicitly aligned genes.
      status: "ok" or the reason a correlation was not estimable.
    """

    score_mimic: float
    score_spearman: float
    n_shared: int
    status: str

    @property
    def pearson(self) -> float:
        """Return the primary Pearson mimicry score."""
        return self.score_mimic

    @property
    def spearman(self) -> float:
        """Return the companion rank correlation."""
        return self.score_spearman


@dataclass(frozen=True)
class CMapScore:
    """Weighted bidirectional rank-connectivity result.

    Attributes:
      score_connectivity: Weighted connectivity in [-1, 1]. Positive values
        place rescue-up genes near the candidate top and rescue-down genes near
        its bottom.
      n_shared: Number of finite genes in both source vectors.
      n_up: Rescue-up query genes used after truncation.
      n_down: Rescue-down query genes used after truncation.
      enrichment_up: Weighted enrichment of rescue-up genes.
      enrichment_down: Weighted enrichment of rescue-down genes.
      status: "ok" or the reason connectivity was not estimable.
    """

    score_connectivity: float
    n_shared: int
    n_up: int
    n_down: int
    enrichment_up: float
    enrichment_down: float
    status: str


def score_mimicry(
    delta_candidate: Sequence[float] | np.ndarray,
    candidate_genes: Sequence[str],
    delta_rescue: Sequence[float] | np.ndarray,
    rescue_genes: Sequence[str],
    *,
    minimum_shared_genes: int = 3,
) -> MimicScore:
    """Score a candidate against rescue after explicit gene-name alignment.

    Args:
      delta_candidate: Candidate response values.
      candidate_genes: Gene identifiers aligned to the candidate values.
      delta_rescue: Baseline-to-weightloss rescue values.
      rescue_genes: Gene identifiers aligned to the rescue values.
      minimum_shared_genes: Minimum finite aligned genes required.

    Returns:
      Pearson and Spearman mimicry scores with an estimability status.
    """
    if minimum_shared_genes < 2:
        raise ValueError("minimum_shared_genes must be at least 2")
    candidate, rescue, _ = _aligned_finite_vectors(
        delta_candidate,
        candidate_genes,
        delta_rescue,
        rescue_genes,
    )
    n_shared = candidate.size
    if n_shared < minimum_shared_genes:
        return _unscorable_mimic(n_shared, "insufficient_shared_genes")
    if _is_constant(candidate):
        return _unscorable_mimic(n_shared, "constant_candidate")
    if _is_constant(rescue):
        return _unscorable_mimic(n_shared, "constant_rescue")

    pearson = float(np.corrcoef(candidate, rescue)[0, 1])
    spearman_result: Any = stats.spearmanr(candidate, rescue)
    spearman = float(spearman_result.statistic)
    if not np.isfinite(pearson) or not np.isfinite(spearman):
        return _unscorable_mimic(n_shared, "non_finite_correlation")
    return MimicScore(
        score_mimic=pearson,
        score_spearman=spearman,
        n_shared=n_shared,
        status="ok",
    )


def score_signatures(
    signatures: PerturbSignatures,
    delta_rescue: Sequence[float] | np.ndarray,
    rescue_genes: Sequence[str],
    *,
    state: str | None = None,
    minimum_shared_genes: int = 3,
) -> pd.DataFrame:
    """Score every aligned perturbation row against one adipose rescue vector.

    Args:
      signatures: Candidate response matrix and row metadata.
      delta_rescue: Rescue values for one adipose state.
      rescue_genes: Gene identifiers aligned to `delta_rescue`.
      state: Optional adipocyte-state label added to every output row.
      minimum_shared_genes: Minimum finite aligned genes required.

    Returns:
      Candidate metadata plus mimicry scores, diagnostics, and descending rank.
    """
    output_columns = {
        "score_mimic",
        "score_spearman",
        "n_shared",
        "score_status",
        "rank",
    }
    if state is not None:
        output_columns.add("state")
    if collisions := output_columns.intersection(signatures.meta.columns):
        raise ValueError(
            f"signature metadata collides with score columns: {collisions}"
        )

    rows = [
        score_mimicry(
            signature,
            signatures.genes,
            delta_rescue,
            rescue_genes,
            minimum_shared_genes=minimum_shared_genes,
        )
        for signature in signatures.delta
    ]
    result = signatures.meta.copy()
    result["score_mimic"] = [row.score_mimic for row in rows]
    result["score_spearman"] = [row.score_spearman for row in rows]
    result["n_shared"] = [row.n_shared for row in rows]
    result["score_status"] = [row.status for row in rows]
    if state is not None:
        result["state"] = state

    result["rank"] = pd.Series(pd.NA, index=result.index, dtype="Int64")
    estimable = result["score_status"].eq("ok")
    result.loc[estimable, "rank"] = (
        result.loc[estimable, "score_mimic"]
        .rank(method="min", ascending=False)
        .astype(int)
    )
    return result


def weighted_cmap_connectivity(
    delta_candidate: Sequence[float] | np.ndarray,
    candidate_genes: Sequence[str],
    delta_rescue: Sequence[float] | np.ndarray,
    rescue_genes: Sequence[str],
    *,
    maximum_query_genes: int = 150,
    minimum_query_genes: int = 10,
) -> CMapScore:
    """Calculate weighted CMap-style bidirectional rank connectivity.

    Candidate genes are ranked from most upregulated to most downregulated.
    Rescue-positive and rescue-negative genes form separate weighted queries.
    Mimicry therefore has positive sign: rescue-up genes enriched at the top
    and rescue-down genes enriched at the bottom yield positive connectivity.

    Args:
      delta_candidate: Measured candidate response values.
      candidate_genes: Gene identifiers aligned to candidate values.
      delta_rescue: Signed rescue values, such as moderated t statistics.
      rescue_genes: Gene identifiers aligned to rescue values.
      maximum_query_genes: Maximum strongest genes retained per rescue arm.
      minimum_query_genes: Minimum genes required in each signed query arm.

    Returns:
      Weighted connectivity and query-support diagnostics.
    """
    if maximum_query_genes < 1:
        raise ValueError("maximum_query_genes must be at least 1")
    if minimum_query_genes < 1:
        raise ValueError("minimum_query_genes must be at least 1")
    if minimum_query_genes > maximum_query_genes:
        raise ValueError(
            "minimum_query_genes cannot exceed maximum_query_genes"
        )

    candidate, rescue, shared_genes = _aligned_finite_vectors(
        delta_candidate,
        candidate_genes,
        delta_rescue,
        rescue_genes,
    )
    n_shared = candidate.size
    up_indices = _strongest_query_indices(
        rescue,
        shared_genes,
        positive=True,
        maximum=maximum_query_genes,
    )
    down_indices = _strongest_query_indices(
        rescue,
        shared_genes,
        positive=False,
        maximum=maximum_query_genes,
    )
    if (
        up_indices.size < minimum_query_genes
        or down_indices.size < minimum_query_genes
    ):
        return CMapScore(
            score_connectivity=np.nan,
            n_shared=n_shared,
            n_up=int(up_indices.size),
            n_down=int(down_indices.size),
            enrichment_up=np.nan,
            enrichment_down=np.nan,
            status="insufficient_query_genes",
        )

    ranked_indices = np.lexsort((np.asarray(shared_genes), -candidate))
    ranked_genes = np.asarray(shared_genes)[ranked_indices]
    up_weights = {
        shared_genes[index]: abs(float(rescue[index])) for index in up_indices
    }
    down_weights = {
        shared_genes[index]: abs(float(rescue[index])) for index in down_indices
    }
    enrichment_up = _weighted_enrichment(ranked_genes, up_weights)
    enrichment_down = _weighted_enrichment(ranked_genes, down_weights)

    # The two arms must point to opposite ends of the ranked candidate to
    # support a bidirectional connectivity call. Concordant arm signs are set
    # to the CMap null rather than rewarded as one-sided enrichment.
    if enrichment_up * enrichment_down < 0.0:
        connectivity = 0.5 * (enrichment_up - enrichment_down)
    else:
        connectivity = 0.0
    return CMapScore(
        score_connectivity=float(np.clip(connectivity, -1.0, 1.0)),
        n_shared=n_shared,
        n_up=int(up_indices.size),
        n_down=int(down_indices.size),
        enrichment_up=enrichment_up,
        enrichment_down=enrichment_down,
        status="ok",
    )


def score_cmap_signatures(
    signatures: PerturbSignatures,
    delta_rescue: Sequence[float] | np.ndarray,
    rescue_genes: Sequence[str],
    *,
    state: str | None = None,
    maximum_query_genes: int = 150,
    minimum_query_genes: int = 10,
) -> pd.DataFrame:
    """Score each measured signature with weighted rank connectivity.

    Args:
      signatures: Measured candidate signatures and metadata.
      delta_rescue: Signed rescue values for one adipose state.
      rescue_genes: Gene identifiers aligned to `delta_rescue`.
      state: Optional adipocyte-state label added to every output row.
      maximum_query_genes: Maximum strongest genes per rescue arm.
      minimum_query_genes: Minimum genes required per rescue arm.

    Returns:
      Metadata plus connectivity, arm support, diagnostics, and rank.
    """
    output_columns = {
        "score_connectivity",
        "n_shared",
        "n_query_up",
        "n_query_down",
        "enrichment_up",
        "enrichment_down",
        "score_status",
        "rank",
    }
    if state is not None:
        output_columns.add("state")
    if collisions := output_columns.intersection(signatures.meta.columns):
        raise ValueError(
            f"signature metadata collides with score columns: {collisions}"
        )

    scores = [
        weighted_cmap_connectivity(
            signature,
            signatures.genes,
            delta_rescue,
            rescue_genes,
            maximum_query_genes=maximum_query_genes,
            minimum_query_genes=minimum_query_genes,
        )
        for signature in signatures.delta
    ]
    result = signatures.meta.copy()
    result["score_connectivity"] = [
        score.score_connectivity for score in scores
    ]
    result["n_shared"] = [score.n_shared for score in scores]
    result["n_query_up"] = [score.n_up for score in scores]
    result["n_query_down"] = [score.n_down for score in scores]
    result["enrichment_up"] = [score.enrichment_up for score in scores]
    result["enrichment_down"] = [score.enrichment_down for score in scores]
    result["score_status"] = [score.status for score in scores]
    if state is not None:
        result["state"] = state

    result["rank"] = pd.Series(pd.NA, index=result.index, dtype="Int64")
    estimable = result["score_status"].eq("ok")
    result.loc[estimable, "rank"] = (
        result.loc[estimable, "score_connectivity"]
        .rank(method="min", ascending=False)
        .astype(int)
    )
    return result


def _aligned_finite_vectors(
    candidate_values: Sequence[float] | np.ndarray,
    candidate_genes: Sequence[str],
    rescue_values: Sequence[float] | np.ndarray,
    rescue_genes: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return finite vectors aligned by unique gene identifiers."""
    candidate = _validated_vector(
        candidate_values,
        candidate_genes,
        vector_name="candidate",
    )
    rescue = _validated_vector(
        rescue_values,
        rescue_genes,
        vector_name="rescue",
    )
    candidate_gene_list = [str(gene) for gene in candidate_genes]
    rescue_gene_list = [str(gene) for gene in rescue_genes]
    candidate_lookup = {
        gene: index for index, gene in enumerate(candidate_gene_list)
    }
    shared_genes = [
        gene for gene in rescue_gene_list if gene in candidate_lookup
    ]
    candidate_aligned = np.asarray(
        [candidate[candidate_lookup[gene]] for gene in shared_genes],
        dtype=np.float64,
    )
    rescue_lookup = {gene: index for index, gene in enumerate(rescue_gene_list)}
    rescue_aligned = np.asarray(
        [rescue[rescue_lookup[gene]] for gene in shared_genes],
        dtype=np.float64,
    )
    finite = np.isfinite(candidate_aligned) & np.isfinite(rescue_aligned)
    finite_genes = [
        gene for gene, keep in zip(shared_genes, finite, strict=True) if keep
    ]
    return candidate_aligned[finite], rescue_aligned[finite], finite_genes


def _validated_vector(
    values: Sequence[float] | np.ndarray,
    genes: Sequence[str],
    *,
    vector_name: str,
) -> np.ndarray:
    """Validate one numeric vector and its unique gene identifiers."""
    try:
        vector = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{vector_name} values must be numeric") from error
    if vector.ndim != 1:
        raise ValueError(f"{vector_name} values must be one-dimensional")
    gene_list = [str(gene) for gene in genes]
    if len(vector) != len(gene_list):
        raise ValueError(
            f"{vector_name} values and genes differ in length: "
            f"{len(vector)} versus {len(gene_list)}"
        )
    if any(not gene.strip() for gene in gene_list):
        raise ValueError(f"{vector_name} genes contains a blank identifier")
    if len(gene_list) != len(set(gene_list)):
        raise ValueError(f"{vector_name} genes must be unique")
    return vector


def _is_constant(values: np.ndarray) -> bool:
    """Return whether all finite values are exactly equal."""
    return values.size == 0 or np.ptp(values) == 0.0


def _unscorable_mimic(n_shared: int, status: str) -> MimicScore:
    """Return a missing correlation with an explicit diagnostic status."""
    return MimicScore(
        score_mimic=np.nan,
        score_spearman=np.nan,
        n_shared=n_shared,
        status=status,
    )


def _strongest_query_indices(
    rescue: np.ndarray,
    genes: Sequence[str],
    *,
    positive: bool,
    maximum: int,
) -> np.ndarray:
    """Return strongest signed rescue indices with deterministic tie breaks."""
    eligible = np.flatnonzero(rescue > 0.0 if positive else rescue < 0.0)
    if eligible.size == 0:
        return eligible
    gene_values = np.asarray(genes)
    order = np.lexsort((gene_values[eligible], -np.abs(rescue[eligible])))
    return eligible[order[:maximum]]


def _weighted_enrichment(
    ranked_genes: np.ndarray,
    query_weights: dict[str, float],
) -> float:
    """Return a weighted running-sum enrichment score for ranked genes."""
    hit_mask = np.asarray([gene in query_weights for gene in ranked_genes])
    n_hits = int(hit_mask.sum())
    n_misses = len(ranked_genes) - n_hits
    if n_hits == 0 or n_misses == 0:
        return np.nan

    hit_weights = np.asarray(
        [query_weights.get(str(gene), 0.0) for gene in ranked_genes],
        dtype=np.float64,
    )
    total_weight = float(hit_weights.sum())
    if total_weight <= 0.0 or not np.isfinite(total_weight):
        return np.nan
    steps = np.where(
        hit_mask,
        hit_weights / total_weight,
        -1.0 / n_misses,
    )
    running = np.cumsum(steps)
    maximum = float(np.max(running))
    minimum = float(np.min(running))
    return maximum if abs(maximum) >= abs(minimum) else minimum
