"""Tests for donor-aware simple perturbation baselines."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from add.baselines import build_adipose_starting_expression
from add.baselines import evaluate_pca_ridge
from add.baselines import fit_pca_ridge
from add.baselines import mean_drug_signatures
from add.baselines import predict_pca_ridge
from add.baselines import score_cmap
from add.baselines import score_mean_drug
from add.baselines import split_signature_contexts
from add.baselines import train_mean_signature
from add.perturb import PerturbSignatures


def test_train_mean_uses_only_training_signatures() -> None:
    """A held-out outlier cannot leak into the generic training mean."""
    signatures = _make_signatures(
        delta=np.array(
            [
                [1.0, 2.0, 3.0],
                [3.0, 4.0, 5.0],
                [100.0, 100.0, 100.0],
            ]
        ),
        contexts=["c1", "c2", "heldout"],
        drugs=["a", "b", "outlier"],
    )

    mean_delta = train_mean_signature(
        signatures,
        training_signature_ids=["s0", "s1"],
    )
    np.testing.assert_allclose(mean_delta, [2.0, 3.0, 4.0])


def test_pca_ridge_predictions_use_adipose_starting_expression() -> None:
    """Known context effects are recovered for the same drug in two states."""
    signatures = _context_dependent_signatures()
    signatures.meta["signature_id"] = signatures.meta.index.astype(str)
    model = fit_pca_ridge(
        signatures,
        drug_col="drug",
        context_col="context_id",
        n_components=2,
        ridge_alpha=1e-8,
        model_genes=["g1", "g2", "g3"],
        max_model_genes=2,
        random_seed=5,
    )
    adipose_states = pd.DataFrame(
        [[1.5, 3.5, 0.5], [3.5, 1.5, 2.5]],
        index=["state_a", "state_b"],
        columns=["g1", "g2", "g3"],
    )

    predicted = predict_pca_ridge(
        model,
        adipose_states,
        drug_ids=["drug_a"],
    )

    assert list(predicted.genes) == ["g1", "g2"]
    assert predicted.meta["state"].tolist() == ["state_a", "state_b"]
    # The fixture defines delta = 0.5 * control + [1, 0] for drug_a.
    np.testing.assert_allclose(
        predicted.delta, [[1.75, 1.75], [2.75, 0.75]], atol=1e-7
    )


def test_evaluation_holds_out_entire_cell_lines() -> None:
    """Plate matching cannot allow the same cell line on both split sides."""
    signatures = _context_dependent_signatures()
    signatures.meta["line"] = ["a"] * 4 + ["b"] * 4
    train, test = split_signature_contexts(
        signatures,
        context_col="line",
        test_fraction=0.5,
        random_seed=3,
    )

    evaluation = evaluate_pca_ridge(
        signatures,
        context_col="context_id",
        split_group_cols="line",
        drug_col="drug",
        n_components=1,
        test_fraction=0.5,
        random_seed=3,
    )

    assert set(signatures.meta.loc[list(train), "line"]).isdisjoint(
        signatures.meta.loc[list(test), "line"],
    )
    assert set(evaluation["signature_id"]) == set(test)
    assert evaluation["train_mean_rmse"].notna().all()


def test_adipose_starting_expression_weights_donors_equally() -> None:
    """Baseline state expression averages donor log-CPM profiles, not counts."""
    pseudobulk = ad.AnnData(
        X=np.array(
            [
                [90.0, 10.0],
                [10.0, 90.0],
                [40.0, 60.0],
            ]
        ),
        obs=pd.DataFrame(
            {
                "Donor": ["d1", "d2", "d1"],
                "condition": ["baseline", "baseline", "weightloss"],
                "state": ["AD_ALL", "AD_ALL", "AD_ALL"],
            },
            index=["p1", "p2", "p3"],
        ),
        var=pd.DataFrame(index=["g1", "g2"]),
    )

    result = build_adipose_starting_expression(
        pseudobulk,
        donor_col="Donor",
        condition_col="condition",
        state_col="state",
        baseline_label="baseline",
    )
    expected = np.mean(
        np.log1p(
            np.array(
                [
                    [900_000.0, 100_000.0],
                    [100_000.0, 900_000.0],
                ]
            )
        ),
        axis=0,
    )

    np.testing.assert_allclose(result.loc["AD_ALL"], expected)
    assert not np.allclose(
        result.loc["AD_ALL"],
        np.log1p([500_000.0, 500_000.0]),
    )


def test_mean_drug_weights_contexts_equally_and_retains_context_scores() -> (
    None
):
    """Replicate-rich contexts do not outweigh other drug contexts."""
    signatures = _make_signatures(
        delta=np.array(
            [
                [0.0, 0.0, 0.0],
                [2.0, 4.0, 6.0],
                [9.0, 7.0, 5.0],
            ]
        ),
        contexts=["c1", "c1", "c2"],
        drugs=["drug_a", "drug_a", "drug_a"],
    )

    averaged = mean_drug_signatures(
        signatures,
        drug_col="drug",
        context_col="context_id",
    )
    ranked, context_scores = score_mean_drug(
        signatures,
        {
            "AD_ALL": pd.Series(
                [5.0, 4.5, 4.0],
                index=signatures.genes,
            )
        },
        drug_col="drug",
        context_col="context_id",
        source="fixture",
    )

    np.testing.assert_allclose(averaged.delta[0], [5.0, 4.5, 4.0])
    assert averaged.meta.loc[averaged.meta.index[0], "n_external_contexts"] == 2
    assert len(context_scores) == 2
    assert ranked.loc[0, "rank"] == 1


def test_parallel_mean_drug_matches_serial_output() -> None:
    """Forked state scoring preserves every mean-drug result and row order."""
    signatures, rescues = _parallel_scoring_fixture()

    serial = score_mean_drug(
        signatures,
        rescues,
        drug_col="drug",
        context_col="context_id",
        workers=1,
    )
    parallel = score_mean_drug(
        signatures,
        rescues,
        drug_col="drug",
        context_col="context_id",
        workers=2,
    )

    for serial_table, parallel_table in zip(serial, parallel, strict=True):
        pd.testing.assert_frame_equal(serial_table, parallel_table)


def test_cmap_connectivity_is_positive_for_rescue_mimicry() -> None:
    """Direct LINCS matching ranks a rescue mimic above its sign inverse."""
    rescue_values = np.array(
        [6.0, 5.0, 4.0, 3.0, 2.0, 1.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0]
    )
    genes = [f"g{index}" for index in range(len(rescue_values))]
    signatures = _make_signatures(
        delta=np.vstack([rescue_values, -rescue_values]),
        contexts=["line_a", "line_b"],
        drugs=["mimic", "inverse"],
        genes=genes,
    )

    ranked, context_scores = score_cmap(
        signatures,
        {"AD_ALL": pd.Series(rescue_values, index=genes)},
        drug_col="drug",
        context_col="context_id",
        query_genes_per_direction=3,
        minimum_query_genes=2,
        source="lincs-fixture",
    )
    scores = context_scores.set_index("drug")["score"]

    assert scores["mimic"] == pytest.approx(1.0)
    assert scores["inverse"] == pytest.approx(-1.0)
    assert ranked.loc[0, "drug"] == "mimic"
    assert ranked.loc[0, "rank"] == 1


def test_cmap_aggregation_preserves_ineligible_support() -> None:
    """Unscorable contexts cannot influence medians or become zero scores."""
    rescue = np.r_[np.arange(10, 0, -1), -np.arange(1, 11)]
    signatures = _make_signatures(
        delta=np.vstack([rescue, np.ones(20), np.ones(20)]),
        contexts=["c1", "c2", "c1"],
        drugs=["mixed", "mixed", "constant"],
    )

    ranked, contexts = score_cmap(
        signatures,
        {"AD_ALL": pd.Series(rescue, index=signatures.genes)},
        drug_col="drug",
        context_col="context_id",
        minimum_shared_genes=20,
    )

    ranked = ranked.set_index("drug")
    assert ranked.loc["mixed", "score"] > 0.9
    assert ranked.loc["mixed", "n_signatures_eligible"] == 1
    assert ranked.loc["mixed", "n_signatures_total"] == 2
    assert pd.isna(ranked.loc["constant", "rank"])
    assert np.isnan(ranked.loc["constant", "score"])
    assert contexts["score_status"].eq("constant_candidate").sum() == 2


def test_parallel_cmap_matches_serial_output() -> None:
    """Forked state scoring preserves every CMap result and row order."""
    signatures, rescues = _parallel_scoring_fixture()

    serial = score_cmap(
        signatures,
        rescues,
        drug_col="drug",
        context_col="context_id",
        query_genes_per_direction=3,
        minimum_query_genes=2,
        workers=1,
    )
    parallel = score_cmap(
        signatures,
        rescues,
        drug_col="drug",
        context_col="context_id",
        query_genes_per_direction=3,
        minimum_query_genes=2,
        workers=2,
    )

    for serial_table, parallel_table in zip(serial, parallel, strict=True):
        pd.testing.assert_frame_equal(serial_table, parallel_table)


def _make_signatures(
    *,
    delta: np.ndarray,
    contexts: list[str],
    drugs: list[str],
    genes: list[str] | None = None,
    control: np.ndarray | None = None,
) -> PerturbSignatures:
    """Construct a small aligned perturbation-signature fixture."""
    resolved_genes = genes or [
        f"g{index + 1}" for index in range(delta.shape[1])
    ]
    metadata = pd.DataFrame(
        {
            "drug": drugs,
            "context_id": contexts,
            "target": ["target"] * len(drugs),
            "mechanism": ["mechanism"] * len(drugs),
            "source": ["fixture"] * len(drugs),
        },
        index=pd.Index(
            [f"s{index}" for index in range(len(drugs))],
            name="signature_id",
        ),
    )
    return PerturbSignatures(
        delta=delta,
        genes=resolved_genes,
        meta=metadata,
        control=control,
        provenance={"fixture": True},
    )


def _parallel_scoring_fixture() -> tuple[
    PerturbSignatures,
    dict[str, pd.Series],
]:
    """Return two rescue states and signatures suitable for pool tests."""
    genes = [f"g{index}" for index in range(12)]
    base = np.arange(1.0, 13.0)
    signatures = _make_signatures(
        delta=np.vstack([base, base[::-1], -base, -base[::-1]]),
        contexts=["c1", "c2", "c1", "c2"],
        drugs=["drug_a", "drug_a", "drug_b", "drug_b"],
        genes=genes,
    )
    rescues = {
        "AD1": pd.Series(
            np.array([6, 5, 4, 3, 2, 1, -1, -2, -3, -4, -5, -6]),
            index=genes,
            dtype=float,
        ),
        "AD_ALL": pd.Series(
            np.array([-4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 7, 8]),
            index=genes,
            dtype=float,
        ),
    }
    return signatures, rescues


def _context_dependent_signatures() -> PerturbSignatures:
    """Return two drugs measured in each of four control contexts."""
    context_profiles = {
        "c1": np.array([1.0, 4.0, 0.0]),
        "c2": np.array([2.0, 3.0, 1.0]),
        "c3": np.array([3.0, 2.0, 2.0]),
        "c4": np.array([4.0, 1.0, 3.0]),
    }
    drug_effects = {
        "drug_a": np.array([1.0, 0.0, 0.0]),
        "drug_b": np.array([0.0, 1.0, 0.0]),
    }
    controls: list[np.ndarray] = []
    deltas: list[np.ndarray] = []
    contexts: list[str] = []
    drugs: list[str] = []
    for context, control in context_profiles.items():
        for drug, drug_effect in drug_effects.items():
            controls.append(control)
            deltas.append(0.5 * control + drug_effect)
            contexts.append(context)
            drugs.append(drug)
    return _make_signatures(
        delta=np.vstack(deltas),
        contexts=contexts,
        drugs=drugs,
        control=np.vstack(controls),
    )
