"""Scientifically decisive tests for donor-level pseudobulk counts."""

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from add.pseudobulk import adipose_group_support
from add.pseudobulk import aggregate_counts
from add.pseudobulk import build_adipose_pseudobulk


def _cell_level_adipose() -> ad.AnnData:
    counts = np.array(
        [
            [1, 0, 2],
            [0, 3, 1],
            [4, 0, 0],
            [2, 2, 0],
            [1, 1, 5],
            [7, 0, 1],
            [0, 6, 2],
            [1, 2, 3],
        ],
        dtype=np.int32,
    )
    obs = pd.DataFrame(
        {
            "Donor": ["D1", "D1", "D1", "D1", "D1", "D2", "D2", "D2"],
            "condition": [
                "baseline",
                "baseline",
                "baseline",
                "weightloss",
                "weightloss",
                "baseline",
                "weightloss",
                "weightloss",
            ],
            "cell_state_t2d": [
                "AD1",
                "AD1",
                "Unassigned",
                "AD1",
                "AD1",
                "AD2",
                "AD2",
                "AD2",
            ],
            "sex": ["F", "F", "F", "F", "F", "M", "M", "M"],
            "sample": ["S1", "S1", "S2", "S3", "S3", "S4", "S5", "S5"],
            "BMI": [45.0, 45.0, 45.0, 45.0, 45.0, np.nan, np.nan, np.nan],
        },
        index=[f"nucleus_{index}" for index in range(len(counts))],
    )
    adata = ad.AnnData(
        X=np.full(counts.shape, 999.0),
        obs=obs,
        var=pd.DataFrame(index=["G1", "G2", "G3"]),
    )
    adata.layers["counts"] = sparse.csr_matrix(counts)
    return adata


def _one_profile(
    pseudobulk: ad.AnnData,
    *,
    donor: str,
    condition: str,
    state: str,
) -> tuple[np.ndarray, pd.Series]:
    mask = (
        (pseudobulk.obs["Donor"] == donor)
        & (pseudobulk.obs["condition"] == condition)
        & (pseudobulk.obs["cell_state_t2d"] == state)
    )
    assert int(mask.sum()) == 1
    position = int(np.flatnonzero(mask.to_numpy())[0])
    vector = np.asarray(pseudobulk.X[position].toarray()).ravel()
    return vector, pseudobulk.obs.iloc[position]


def test_pseudobulk_exactly_sums_configured_raw_counts_in_chunks() -> None:
    """Chunked state and pooled outputs equal direct raw-count sums."""
    adata = _cell_level_adipose()

    pseudobulk = build_adipose_pseudobulk(
        adata,
        count_layer="counts",
        min_cells=2,
        metadata_cols=("sex", "sample", "not_available"),
        chunk_size=2,
    )

    state_vector, state_obs = _one_profile(
        pseudobulk,
        donor="D1",
        condition="baseline",
        state="AD1",
    )
    pooled_vector, pooled_obs = _one_profile(
        pseudobulk,
        donor="D1",
        condition="baseline",
        state="AD_ALL",
    )

    np.testing.assert_array_equal(state_vector, np.array([1, 3, 3]))
    # AD_ALL includes the otherwise unassigned third adipocyte nucleus.
    np.testing.assert_array_equal(pooled_vector, np.array([5, 3, 3]))
    assert state_obs["n_cells"] == 2
    assert pooled_obs["n_cells"] == 3
    assert state_obs["sample"] == "S1"
    assert pd.isna(pooled_obs["sample"])
    assert pooled_obs["sex"] == "F"
    assert "Unassigned" not in set(pseudobulk.obs["cell_state_t2d"])
    assert sparse.isspmatrix_csr(pseudobulk.X)


def test_pseudobulk_has_one_row_per_biological_group_not_per_nucleus() -> None:
    """Nuclei contribute counts but never become replicate output rows."""
    pseudobulk = build_adipose_pseudobulk(
        _cell_level_adipose(),
        count_layer="counts",
        min_cells=2,
        chunk_size=3,
    )

    group_columns = ["Donor", "condition", "cell_state_t2d"]
    assert not pseudobulk.obs.duplicated(group_columns).any()
    assert pseudobulk.n_obs == 6
    assert pseudobulk.n_obs < 8  # eight nuclei entered the aggregation

    # D2 baseline has one nucleus, so it fails min_cells independently in both
    # the state-level and pooled estimators.
    d2_baseline = (pseudobulk.obs["Donor"] == "D2") & (
        pseudobulk.obs["condition"] == "baseline"
    )
    assert not d2_baseline.any()


def test_aggregate_counts_never_falls_back_to_x() -> None:
    """A missing configured layer fails even when X contains expression."""
    adata = _cell_level_adipose()
    del adata.layers["counts"]

    with pytest.raises(KeyError, match="No expression fallback"):
        aggregate_counts(
            adata,
            group_cols=("Donor", "condition"),
            count_layer="counts",
        )


def test_group_support_retains_below_threshold_candidates() -> None:
    """The support audit exposes every applied min-cells decision."""
    adata = _cell_level_adipose()
    support = adipose_group_support(adata.obs, min_cells=2)
    pseudobulk = build_adipose_pseudobulk(
        adata,
        count_layer="counts",
        min_cells=2,
    )

    assert len(support) == 8
    assert support["retained"].dtype == bool
    assert not support.duplicated(
        ["Donor", "condition", "cell_state_t2d"]
    ).any()
    assert "Unassigned" not in set(support["cell_state_t2d"])

    d2_baseline = support.loc[
        (support["Donor"] == "D2") & (support["condition"] == "baseline")
    ]
    assert len(d2_baseline) == 2
    assert d2_baseline["n_cells"].eq(1).all()
    assert not d2_baseline["retained"].any()

    retained_groups = support.loc[
        support["retained"],
        ["Donor", "condition", "cell_state_t2d"],
    ].reset_index(drop=True)
    output_groups = pseudobulk.obs.loc[
        :,
        ["Donor", "condition", "cell_state_t2d"],
    ].reset_index(drop=True)
    pd.testing.assert_frame_equal(retained_groups, output_groups)


def test_backed_sparse_count_layer_produces_persistable_profiles(
    tmp_path,
) -> None:
    """The CLI's backed-H5AD path works without substituting expression X."""
    input_path = tmp_path / "cells.h5ad"
    output_path = tmp_path / "pseudobulk.h5ad"
    _cell_level_adipose().write_h5ad(input_path)

    backed = ad.read_h5ad(input_path, backed="r")
    try:
        pseudobulk = build_adipose_pseudobulk(
            backed,
            count_layer="counts",
            min_cells=2,
            metadata_cols=("sample", "BMI"),
            chunk_size=2,
        )
    finally:
        backed.file.close()

    pseudobulk.write_h5ad(output_path)
    restored = ad.read_h5ad(output_path)
    vector, _ = _one_profile(
        restored,
        donor="D1",
        condition="baseline",
        state="AD_ALL",
    )
    np.testing.assert_array_equal(vector, np.array([5, 3, 3]))
    assert pd.api.types.is_float_dtype(restored.obs["BMI"])
    d2_weightloss = restored.obs.loc[
        (restored.obs["Donor"] == "D2")
        & (restored.obs["condition"] == "weightloss")
    ]
    assert d2_weightloss["BMI"].isna().all()
