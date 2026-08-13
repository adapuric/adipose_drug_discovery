"""Tests for matched donor selection and paired rescue direction."""

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from add.rescue import estimate_rescue
from add.rescue import rescue_vector
from add.rescue import select_matched_pairs


def test_rescue_vector_uses_finite_tested_moderated_t_values() -> None:
    """The default scoring vector omits explicitly untested genes."""
    results = pd.DataFrame(
        {
            "gene": ["G1", "G2", "G3", "G1"],
            "state": ["AD1", "AD1", "AD1", "AD2"],
            "tested": [True, False, True, True],
            "moderated_t": [2.5, np.nan, np.inf, -1.0],
            "logFC": [1.0, np.nan, 0.5, -0.2],
        }
    )

    vector = rescue_vector(results, state="AD1")

    pd.testing.assert_series_equal(
        vector,
        pd.Series(
            [2.5],
            index=pd.Index(["G1"], name="gene"),
            name="moderated_t",
        ),
    )


def test_pair_selection_excludes_unmatched_and_duplicate_donors() -> None:
    """Only unambiguous matched donors enter the paired design."""
    obs = pd.DataFrame(
        {
            "Donor": [
                "D1",
                "D1",
                "D2",
                "D3",
                "D4",
                "D4",
                "D4",
                "D5",
                "D5",
                "D5",
                "D6",
                "D6",
            ],
            "condition": [
                "baseline",
                "weightloss",
                "baseline",
                "weightloss",
                "baseline",
                "baseline",
                "weightloss",
                "weightloss",
                "Lean",
                "baseline",
                "baseline",
                "weightloss",
            ],
            "cell_state_t2d": [
                "AD1",
                "AD1",
                "AD1",
                "AD1",
                "AD1",
                "AD1",
                "AD1",
                "AD1",
                "AD1",
                "AD1",
                "AD2",
                "AD2",
            ],
            "n_cells": [100, 90, 80, 85, 70, 30, 95, 75, 75, 88, 60, 62],
        },
        index=[f"pb_{index}" for index in range(12)],
    )
    pseudobulk = ad.AnnData(
        X=sparse.csr_matrix(np.full((12, 2), 100, dtype=np.int64)),
        obs=obs,
        var=pd.DataFrame(index=["G1", "G2"]),
    )

    paired = select_matched_pairs(pseudobulk, state="AD1")

    # D2 and D3 are unmatched; D4 has duplicate baseline profiles. D5 has an
    # irrelevant Lean row but exactly one row in each paired condition.
    assert paired.obs["Donor"].tolist() == ["D1", "D1", "D5", "D5"]
    assert paired.obs["condition"].tolist() == [
        "baseline",
        "weightloss",
        "baseline",
        "weightloss",
    ]
    assert paired.n_obs == 4
    assert paired.uns["paired_rescue"]["n_pairs"] == 2
    assert not paired.obs.duplicated(["Donor", "condition"]).any()


def test_native_limma_keeps_all_genes_and_uses_weightloss_minus_baseline() -> (
    None
):
    """Native rescue preserves genes and the beneficial direction."""
    donors = [f"D{index}" for index in range(1, 6)]
    genes = [f"stable_{index}" for index in range(40)]
    genes.extend(["rescue_up", "rescue_down", "filtered_out"])

    rows: list[np.ndarray] = []
    obs_rows: list[dict[str, object]] = []
    obs_names: list[str] = []
    for donor_index, donor in enumerate(donors):
        stable = np.array(
            [100 + donor_index * 3 + gene_index % 5 for gene_index in range(40)]
        )
        baseline = np.concatenate(
            [stable, [30 + donor_index, 180 + donor_index, 0]]
        )
        weightloss = np.concatenate(
            [stable, [180 + donor_index, 30 + donor_index, 0]]
        )
        for condition, counts in (
            ("baseline", baseline),
            ("weightloss", weightloss),
        ):
            rows.append(counts.astype(np.int64))
            obs_rows.append(
                {
                    "Donor": donor,
                    "condition": condition,
                    "cell_state_t2d": "AD_ALL",
                    "n_cells": 50,
                }
            )
            obs_names.append(f"{donor}_{condition}")

    # An unmatched donor must not become an additional biological replicate.
    rows.append(np.full(len(genes), 100, dtype=np.int64))
    obs_rows.append(
        {
            "Donor": "unmatched",
            "condition": "baseline",
            "cell_state_t2d": "AD_ALL",
            "n_cells": 500,
        }
    )
    obs_names.append("unmatched_baseline")

    pseudobulk = ad.AnnData(
        X=sparse.csr_matrix(np.vstack(rows)),
        obs=pd.DataFrame(obs_rows, index=obs_names),
        var=pd.DataFrame(index=genes),
    )

    result = estimate_rescue(
        pseudobulk,
        state="AD_ALL",
    )

    assert result["gene"].tolist() == genes
    assert result["state"].eq("AD_ALL").all()
    assert result["n_pairs"].eq(5).all()
    assert {
        "logFC",
        "moderated_t",
        "p_value",
        "adjusted_p_value",
        "tested",
        "delta_rescue",
    }.issubset(result.columns)

    up = result.set_index("gene").loc["rescue_up"]
    down = result.set_index("gene").loc["rescue_down"]
    filtered = result.set_index("gene").loc["filtered_out"]
    stable = result.loc[result["gene"].str.startswith("stable_")]
    assert stable["tested"].all()
    np.testing.assert_array_equal(stable["moderated_t"], 0.0)
    np.testing.assert_array_equal(stable["p_value"], 1.0)
    assert bool(up["tested"])
    assert up["logFC"] > 0
    assert up["moderated_t"] > 0
    assert down["logFC"] < 0
    assert down["moderated_t"] < 0
    assert not bool(filtered["tested"])
    assert pd.isna(filtered["logFC"])
    assert pd.isna(filtered["moderated_t"])
    np.testing.assert_allclose(
        result["delta_rescue"],
        result["moderated_t"],
        equal_nan=True,
    )


def test_native_limma_matches_paired_edge_r_reference() -> None:
    """Native statistics match an independently generated R reference."""
    random = np.random.default_rng(20260812)
    n_genes = 120
    n_pairs = 8
    base_means = np.exp(random.uniform(np.log(1), np.log(1000), size=n_genes))
    donor_effects = random.lognormal(0, 0.18, size=(n_pairs, n_genes))
    log2_effect = np.zeros(n_genes)
    log2_effect[:12] = random.uniform(0.4, 1.5, size=12)
    log2_effect[12:24] = -random.uniform(0.4, 1.5, size=12)

    rows: list[np.ndarray] = []
    obs_rows: list[dict[str, str]] = []
    obs_names: list[str] = []
    for donor_index in range(n_pairs):
        for condition, condition_indicator in (
            ("baseline", 0),
            ("weightloss", 1),
        ):
            means = (
                base_means
                * donor_effects[donor_index]
                * 2 ** (log2_effect * condition_indicator)
            )
            size = 1 / 0.15
            probabilities = size / (size + means)
            rows.append(random.negative_binomial(size, probabilities))
            donor = f"D{donor_index + 1}"
            obs_rows.append(
                {
                    "Donor": donor,
                    "condition": condition,
                    "cell_state_t2d": "AD_ALL",
                }
            )
            obs_names.append(f"{donor}_{condition}")

    genes = [f"g{index}" for index in range(n_genes)]
    pseudobulk = ad.AnnData(
        X=sparse.csr_matrix(np.vstack(rows)),
        obs=pd.DataFrame(obs_rows, index=obs_names),
        var=pd.DataFrame(index=genes),
    )

    result = estimate_rescue(pseudobulk, state="AD_ALL")

    filtered_genes = [
        "g0",
        "g2",
        "g5",
        "g9",
        "g19",
        "g21",
        "g28",
        "g29",
        "g31",
        "g32",
        "g38",
        "g40",
        "g45",
        "g46",
        "g50",
        "g51",
        "g60",
        "g69",
        "g86",
        "g98",
        "g100",
        "g102",
        "g105",
        "g112",
        "g113",
        "g114",
        "g117",
        "g118",
    ]
    assert result.loc[~result["tested"], "gene"].tolist() == filtered_genes

    expected = pd.DataFrame(
        {
            "logFC": [
                -1.55981876318583,
                -0.26541440024679702,
                -1.0804304967928999,
                0.228832619248346,
                0.490195032905774,
                -0.0357155229685559,
                -0.27002681418276803,
                0.18759792180206,
            ],
            "moderated_t": [
                -5.50007991315755,
                -0.796864639962077,
                -2.6416700625964902,
                0.74613835956148,
                1.7511455591518399,
                -0.117817021441458,
                -0.878527210152802,
                0.638386089651272,
            ],
            "p_value": [
                1.67677865776235e-6,
                0.429681465231745,
                0.0112799914955252,
                0.459433880654996,
                0.0866817732102362,
                0.906732868872371,
                0.384288289888743,
                0.526432908523669,
            ],
            "adjusted_p_value": [
                0.0001542636365141,
                0.741883609214415,
                0.148251316798331,
                0.741883609214415,
                0.443040174185652,
                0.958843953290324,
                0.72840670870209,
                0.756747306002775,
            ],
        },
        index=["g12", "g17", "g23", "g24", "g30", "g75", "g90", "g119"],
    )
    observed = result.set_index("gene").loc[expected.index, expected.columns]
    np.testing.assert_allclose(
        observed.to_numpy(),
        expected.to_numpy(),
        rtol=1e-8,
        atol=1e-10,
    )
