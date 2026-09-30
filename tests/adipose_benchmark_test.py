"""Scientific acceptance checks for paired donor transition predictors."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd

from add.baselines.adipose import fit_adipose_mean_shift
from add.baselines.adipose import fit_adipose_ridge
from add.baselines.adipose import predict_adipose_mean_shift
from add.baselines.adipose import predict_adipose_ridge
from add.validation.adipose_benchmark import BenchmarkSettings
from add.validation.adipose_benchmark import donor_metric_summary
from add.validation.adipose_benchmark import evaluate_adipose_transition


def test_mean_shift_averages_independent_donors() -> None:
    """Every donor has equal weight in the predicted paired change."""
    changes = pd.DataFrame(
        [[1.0, 3.0], [5.0, -1.0]], index=["a", "b"], columns=["g1", "g2"]
    )
    model = fit_adipose_mean_shift(changes)
    baseline = pd.DataFrame(
        [[10.0, 20.0]], index=["heldout"], columns=["g1", "g2"]
    )

    result = predict_adipose_mean_shift(model, baseline)

    np.testing.assert_allclose(result, [[3.0, 1.0]])


def test_small_ridge_recovers_a_linear_change() -> None:
    """Weakly regularized ridge recovers a known low-rank donor response."""
    x = pd.DataFrame(
        [[i, 2 * i, -i] for i in range(8)],
        columns=["a", "b", "c"],
        index=[f"d{i}" for i in range(8)],
    )
    changes = x * 0.4 + 2
    model = fit_adipose_ridge(x, changes, alpha=1e-10, n_components=1)
    test = pd.DataFrame([[8.0, 16.0, -8.0]], columns=x.columns, index=["new"])

    result = predict_adipose_ridge(model, test)

    np.testing.assert_allclose(result, test * 0.4 + 2, atol=1e-6)


def test_heldout_target_cannot_change_its_predictions() -> None:
    """Held-out targets cannot affect their model predictions."""
    rng = np.random.default_rng(9)
    counts = rng.integers(20, 300, size=(16, 6)).astype(float)
    obs = pd.DataFrame(
        [
            {
                "Donor": f"d{donor}",
                "condition": condition,
                "cell_state_t2d": state,
                "n_cells": 30,
            }
            for donor in range(4)
            for state in ("AD1", "AD2")
            for condition in ("baseline", "weightloss")
        ],
        index=[str(i) for i in range(16)],
    )
    pseudobulk = ad.AnnData(
        X=counts, obs=obs, var=pd.DataFrame(index=[f"g{i}" for i in range(6)])
    )
    pseudobulk.uns["pseudobulk"] = {"aggregation": "sum_raw_counts"}
    settings = BenchmarkSettings(minimum_shared_genes=3, bootstrap_draws=20)
    first = evaluate_adipose_transition(pseudobulk, settings=settings)
    changed = pseudobulk.copy()
    changed.X[1] *= np.arange(1, 7)
    second = evaluate_adipose_transition(changed, settings=settings)

    columns = ["model", "state", "gene_id", "predicted_delta"]
    before = first["predictions.parquet"].query("donor_id == 'd0'")[columns]
    after = second["predictions.parquet"].query("donor_id == 'd0'")[columns]
    pd.testing.assert_frame_equal(before, after)
    folds = first["folds.csv"]
    assert not folds.duplicated(["fold_id", "donor_id"]).any()
    assert (
        folds.groupby("fold_id")["role"]
        .apply(lambda x: x.eq("test").sum())
        .eq(1)
        .all()
    )
    null = first["metrics.csv"].query("model == 'adipose-no-change'")
    assert null["delta_rmse"].notna().all()
    assert null["correlation_status"].eq("constant_candidate").all()

    # Insufficient ridge support must remain missing, never become mean shift.
    three_donors = pseudobulk[pseudobulk.obs["Donor"].ne("d3")].copy()
    limited = evaluate_adipose_transition(three_donors, settings=settings)
    ridge = limited["metrics.csv"].query("model == 'adipose-ridge'")
    assert ridge["status"].eq("insufficient_training_donors").all()
    assert ridge["delta_rmse"].isna().all()
    means = limited["metrics.csv"].query("model == 'adipose-mean-shift'")
    assert means["status"].eq("ok").all()


def test_bootstrap_keeps_donor_clusters_and_unsupported_intervals() -> None:
    """Scaled state values share resamples; constants have no interval."""
    metrics = pd.DataFrame(
        {
            "donor_id": ["a", "b", "c"] * 3,
            "state": ["AD1"] * 3 + ["AD2"] * 3 + ["AD3"] * 3,
            "value": [1.0, 2.0, 5.0, 2.0, 4.0, 10.0, 0.0, 0.0, 0.0],
        }
    )
    first = donor_metric_summary(
        metrics,
        value_columns=("value",),
        group_columns=("state",),
        random_seed=7,
        bootstrap_draws=100,
    ).set_index("state")
    second = donor_metric_summary(
        metrics,
        value_columns=("value",),
        group_columns=("state",),
        random_seed=7,
        bootstrap_draws=100,
    ).set_index("state")

    pd.testing.assert_frame_equal(first, second)
    np.testing.assert_allclose(
        first.loc["AD2", ["mean", "lower", "upper"]].astype(float),
        first.loc["AD1", ["mean", "lower", "upper"]].astype(float) * 2,
    )
    assert first.loc["AD3", "interval_status"] == "degenerate_donor_values"
    assert pd.isna(first.loc["AD3", "lower"])
