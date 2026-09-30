"""Tests for shared rescue-mimicry scoring."""

from __future__ import annotations

import numpy as np
import pandas as pd

from add.perturb import PerturbSignatures
from add.scoring import score_cmap_signatures
from add.scoring import score_mimicry
from add.scoring import weighted_cmap_connectivity


def test_cmap_rejects_unsupported_candidates() -> None:
    """Insufficient signatures cannot acquire connectivity scores or ranks."""
    n_genes = 20
    genes = [f"g{i:03}" for i in range(n_genes)]
    rescue = np.concatenate([np.ones(n_genes // 2), -np.ones(n_genes // 2)])
    candidate = rescue.copy()
    signatures = PerturbSignatures(
        delta=candidate[None, :],
        genes=genes,
        meta=pd.DataFrame({"drug": ["candidate"]}),
    )

    scores = score_cmap_signatures(
        signatures,
        rescue,
        genes,
        minimum_shared_genes=50,
    )

    assert scores.loc[0, "score_status"] == "insufficient_shared_genes"
    assert np.isnan(scores.loc[0, "score_connectivity"])
    assert pd.isna(scores.loc[0, "rank"])


def test_cmap_shared_support_counts_only_aligned_finite_values() -> None:
    """Missing values cannot satisfy the enrichment gene-support threshold."""
    genes = [f"g{i}" for i in range(6)]
    rescue = np.array([3.0, 2.0, 1.0, -1.0, -2.0, -3.0])
    candidate = rescue.copy()
    candidate[0] = np.nan

    result = weighted_cmap_connectivity(
        candidate[::-1],
        genes[::-1],
        rescue,
        genes,
        minimum_shared_genes=6,
        minimum_query_genes=1,
    )

    assert result.n_shared == 5
    assert result.status == "insufficient_shared_genes"


def test_mimicry_aligns_shuffled_gene_order() -> None:
    """Correlation follows gene identifiers rather than input column order."""
    result = score_mimicry(
        [3.0, 1.0, 2.0],
        ["C", "A", "B"],
        [1.0, 2.0, 3.0],
        ["A", "B", "C"],
    )

    assert result.status == "ok"
    assert result.n_shared == 3
    assert result.score_mimic == 1.0
    assert result.score_spearman == 1.0


def test_positive_score_means_candidate_mimics_rescue() -> None:
    """Same-direction candidates score positive and reversed ones negative."""
    genes = ["A", "B", "C", "D"]
    rescue = np.array([-2.0, -0.5, 1.0, 3.0])

    mimic = score_mimicry(rescue, genes, rescue, genes)
    reverse = score_mimicry(-rescue, genes, rescue, genes)

    assert mimic.score_mimic > 0.99
    assert mimic.score_spearman > 0.99
    assert reverse.score_mimic < -0.99
    assert reverse.score_spearman < -0.99


def test_mimicry_reports_finite_shared_gene_support() -> None:
    """Missing values reduce usable gene support rather than becoming zero."""
    result = score_mimicry(
        [1.0, np.nan, 3.0],
        ["A", "B", "C"],
        [1.0, 2.0, 3.0],
        ["A", "B", "C"],
        minimum_shared_genes=3,
    )

    assert result.n_shared == 2
    assert result.status == "insufficient_shared_genes"
    assert np.isnan(result.score_mimic)
