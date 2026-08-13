"""Tests for donor-aware simple perturbation baselines."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from add.baselines import build_adipose_starting_expression
from add.baselines import evaluate_pca_ridge
from add.baselines import evaluate_perturbed_mean
from add.baselines import fit_pca_ridge
from add.baselines import mean_drug_signatures
from add.baselines import perturbed_mean_signature
from add.baselines import predict_pca_ridge
from add.baselines import score_cmap
from add.baselines import score_mean_drug
from add.baselines import score_perturbed_mean
from add.baselines import split_signature_contexts
from add.perturb import PerturbSignatures


def test_context_split_is_deterministic_and_keeps_contexts_disjoint() -> None:
    """A fixed seed gives one split with no context on both sides."""
    signatures = _make_signatures(
        delta=np.arange(18, dtype=float).reshape(6, 3),
        contexts=["c1", "c1", "c2", "c2", "c3", "c4"],
        drugs=["a", "b", "a", "b", "a", "b"],
    )

    first = split_signature_contexts(
        signatures,
        context_col="context_id",
        test_fraction=0.4,
        random_seed=17,
    )
    second = split_signature_contexts(
        signatures,
        context_col="context_id",
        test_fraction=0.4,
        random_seed=17,
    )

    assert first == second
    context_by_id = signatures.meta["context_id"].to_dict()
    train_contexts = {context_by_id[identifier] for identifier in first[0]}
    test_contexts = {context_by_id[identifier] for identifier in first[1]}
    assert train_contexts.isdisjoint(test_contexts)


def test_perturbed_mean_uses_only_training_signatures() -> None:
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

    mean_delta = perturbed_mean_signature(
        signatures,
        training_signature_ids=["s0", "s1"],
    )
    generic_scores = score_perturbed_mean(
        signatures,
        {"AD_ALL": pd.Series([2.0, 3.0, 4.0], index=signatures.genes)},
        training_signature_ids=["s0", "s1"],
        source="fixture",
    )

    np.testing.assert_allclose(mean_delta, [2.0, 3.0, 4.0])
    assert len(generic_scores) == 1
    assert pd.isna(generic_scores.loc[0, "drug"])
    assert generic_scores.loc[0, "signature_id"] == "PERTURBED_MEAN"


def test_perturbed_mean_evaluation_is_deterministic() -> None:
    """Held-out mean-prediction metrics repeat exactly for a fixed seed."""
    signatures = _make_signatures(
        delta=np.array(
            [
                [1.0, 2.0, 4.0],
                [2.0, 3.0, 6.0],
                [3.0, 5.0, 7.0],
                [4.0, 7.0, 8.0],
                [5.0, 8.0, 10.0],
                [6.0, 9.0, 12.0],
            ]
        ),
        contexts=["c1", "c1", "c2", "c2", "c3", "c3"],
        drugs=["a", "b", "a", "b", "a", "b"],
    )

    first = evaluate_perturbed_mean(
        signatures,
        context_col="context_id",
        drug_col="drug",
        test_fraction=0.34,
        random_seed=23,
    )
    second = evaluate_perturbed_mean(
        signatures,
        context_col="context_id",
        drug_col="drug",
        test_fraction=0.34,
        random_seed=23,
    )

    pd.testing.assert_frame_equal(first, second)


def test_pca_ridge_predictions_use_adipose_starting_expression() -> None:
    """The same drug has different predictions in two adipose states."""
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

    assert model.genes == ("g1", "g2")
    assert predicted.meta["state"].tolist() == ["state_a", "state_b"]
    assert predicted.meta["signature_id"].is_unique
    assert predicted.meta["signature_id"].tolist() == list(predicted.meta.index)
    assert not np.allclose(predicted.delta[0], predicted.delta[1])
    with pytest.raises(ValueError, match="absent from training"):
        predict_pca_ridge(
            model,
            adipose_states,
            drug_ids=["unknown_drug"],
        )


def test_pca_ridge_evaluation_repeats_for_a_grouped_split() -> None:
    """Grouped PCA-ridge evaluation is deterministic for a fixed seed."""
    signatures = _context_dependent_signatures()

    first = evaluate_pca_ridge(
        signatures,
        drug_col="drug",
        context_col="context_id",
        n_components=2,
        ridge_alpha=0.1,
        test_fraction=0.25,
        random_seed=11,
    )
    second = evaluate_pca_ridge(
        signatures,
        drug_col="drug",
        context_col="context_id",
        n_components=2,
        ridge_alpha=0.1,
        test_fraction=0.25,
        random_seed=11,
    )

    pd.testing.assert_frame_equal(first, second)


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

    assert scores["mimic"] > 0.0
    assert scores["inverse"] < 0.0
    assert ranked.loc[0, "drug"] == "mimic"
    assert ranked.loc[0, "rank"] == 1


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


def test_baseline_workers_must_be_positive() -> None:
    """Invalid pool sizes fail before starting worker processes."""
    signatures, rescues = _parallel_scoring_fixture()

    with pytest.raises(ValueError, match="workers must be at least 1"):
        score_cmap(
            signatures,
            rescues,
            drug_col="drug",
            context_col="context_id",
            workers=0,
        )


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
