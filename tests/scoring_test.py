"""Tests for shared rescue-mimicry scoring."""

from __future__ import annotations

import numpy as np
import pandas as pd

from add.perturb import PerturbSignatures
from add.scoring import score_mimicry
from add.scoring import score_signatures
from add.scoring import weighted_cmap_connectivity


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


def test_signature_table_uses_canonical_state_column() -> None:
    """Table scoring labels rows with the canonical adipocyte-state field."""
    signatures = PerturbSignatures(
        delta=np.array([[1.0, 2.0, 3.0]]),
        genes=["A", "B", "C"],
        meta=pd.DataFrame({"drug": ["candidate"]}),
    )

    result = score_signatures(
        signatures,
        [1.0, 2.0, 3.0],
        ["A", "B", "C"],
        state="AD_ALL",
    )

    assert result["state"].tolist() == ["AD_ALL"]
    assert "adipocyte_state" not in result


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


def test_weighted_cmap_connectivity_uses_mimicry_sign() -> None:
    """Rescue-up at the top and rescue-down at the bottom scores positive."""
    genes = ["UP1", "UP2", "UP3", "DOWN1", "DOWN2", "DOWN3"]
    rescue = np.array([3.0, 2.0, 1.0, -1.0, -2.0, -3.0])

    mimic = weighted_cmap_connectivity(
        rescue,
        genes,
        rescue,
        genes,
        maximum_query_genes=3,
        minimum_query_genes=2,
    )
    reverse = weighted_cmap_connectivity(
        -rescue,
        genes,
        rescue,
        genes,
        maximum_query_genes=3,
        minimum_query_genes=2,
    )

    assert mimic.status == "ok"
    assert mimic.enrichment_up > 0.0
    assert mimic.enrichment_down < 0.0
    assert mimic.score_connectivity > 0.9
    assert reverse.score_connectivity < -0.9
