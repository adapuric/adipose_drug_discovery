"""Tests for preparing adipocytes for PerturbGen."""

import dataclasses
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from run_scripts.perturbgen.config import load_perturbgen_config
from run_scripts.perturbgen.prepare_adipose import prepare_adipose_adata


@pytest.mark.parametrize(
    ("subset_to_highly_variable_genes", "expected_genes", "gene_positions"),
    [
        (False, ["ENSG_A", "ENSG_B"], [0, 1]),
        (True, ["ENSG_A"], [0]),
    ],
)
def test_preparation_keeps_only_required_paired_raw_data(
    tmp_path: Path,
    subset_to_highly_variable_genes: bool,
    expected_genes: list[str],
    gene_positions: list[int],
) -> None:
    """Preparation writes paired raw counts with configured gene selection."""
    input_path = tmp_path / "adipose.h5ad"
    annotation_path = tmp_path / "genes.tsv"
    obs = pd.DataFrame(
        {
            "Donor": ["D1", "D1", "D2", "D3"],
            "condition": [
                "baseline",
                "weightloss",
                "baseline",
                "weightloss",
            ],
            "cell_state_t2d": ["AD1", "AD1", "AD2", "AD2"],
        },
        index=["D1_b", "D1_w", "D2_b", "D3_w"],
    )
    raw_counts = np.array(
        [
            [2, 5, 0],
            [7, 11, 1],
            [13, 17, 0],
            [19, 23, 0],
        ],
        dtype=np.int64,
    )
    adata = ad.AnnData(
        X=sparse.csr_matrix(np.full(raw_counts.shape, 0.25)),
        obs=obs,
        var=pd.DataFrame(
            {"highly_variable": [True, False, True]},
            index=["GENE_A", "GENE_B", "UNMAPPED"],
        ),
        layers={
            "raw": sparse.csr_matrix(raw_counts),
            "normalized": sparse.csr_matrix(np.full(raw_counts.shape, 0.5)),
        },
    )
    adata.write_h5ad(input_path)
    pd.DataFrame(
        {
            "Gene name": ["GENE_A", "GENE_B"],
            "Gene stable ID": ["ENSG_A", "ENSG_B"],
        }
    ).to_csv(annotation_path, sep="\t", index=False)
    repository = Path(__file__).parents[1]
    config = load_perturbgen_config(
        repository / "run_scripts" / "perturbgen" / "config.yaml"
    )
    config = dataclasses.replace(
        config,
        run=dataclasses.replace(
            config.run,
            run_name="test_run",
            results_root_directory=tmp_path,
        ),
        prepare=dataclasses.replace(
            config.prepare,
            source_h5ad_path=input_path,
            gene_annotation_path=annotation_path,
            subset_to_highly_variable_genes=(subset_to_highly_variable_genes),
        ),
    )

    result_path = prepare_adipose_adata(config)

    prepared = ad.read_h5ad(result_path)
    assert prepared.obs["Donor"].tolist() == ["D1", "D1"]
    assert prepared.obs["condition"].tolist() == ["obese", "weightloss"]
    assert prepared.obs["cell_states_adipocytes"].tolist() == ["AD1", "AD1"]
    assert "donor_id" not in prepared.obs
    assert prepared.var_names.tolist() == expected_genes
    assert prepared.var.columns.tolist() == ["ensembl_id"]
    assert "counts" not in prepared.layers
    assert "raw" not in prepared.layers
    assert "normalized" not in prepared.layers
    assert "perturbgen_preparation" not in prepared.uns
    np.testing.assert_array_equal(
        prepared.X.toarray(),
        raw_counts[:2, gene_positions],
    )
