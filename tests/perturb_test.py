"""Tests for external perturbation signature adapters."""

from __future__ import annotations

from pathlib import Path

import anndata as ad  # type: ignore[import]
import h5py
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from add.perturb import PerturbSignatures
from add.perturb import load_lincs_signatures
from add.perturb import load_perturb_signatures
from add.perturb import load_tahoe_signatures
from add.perturb import save_perturb_signatures


def test_signature_container_rejects_misaligned_metadata() -> None:
    """A signature matrix cannot silently detach rows from metadata."""
    with pytest.raises(ValueError, match="rows do not match metadata"):
        PerturbSignatures(
            delta=np.ones((2, 3)),
            genes=["A", "B", "C"],
            meta=pd.DataFrame({"drug": ["one"]}),
        )


def test_signature_cache_round_trips_control_and_provenance(
    tmp_path: Path,
) -> None:
    """The cache preserves row order, genes, controls, and source provenance."""
    signatures = PerturbSignatures(
        delta=np.array([[1.0, -1.0], [2.0, -2.0]]),
        genes=["G1", "G2"],
        meta=pd.DataFrame(
            {"drug": ["drug_b", "drug_a"]},
            index=pd.Index(["sig_b", "sig_a"], name="signature_id"),
        ),
        control=np.array([[3.0, 4.0], [5.0, 6.0]]),
        provenance={"source": "fixture", "seed": 7},
    )

    paths = save_perturb_signatures(signatures, tmp_path / "cache")
    restored = load_perturb_signatures(paths["matrix"])

    np.testing.assert_array_equal(restored.delta, signatures.delta)
    np.testing.assert_array_equal(restored.control, signatures.control)
    assert tuple(restored.genes) == ("G1", "G2")
    assert restored.meta.index.tolist() == ["sig_b", "sig_a"]
    assert restored.meta["drug"].tolist() == ["drug_b", "drug_a"]
    assert restored.provenance == {"source": "fixture", "seed": 7}


def test_tahoe_subtracts_vehicle_from_the_same_context(tmp_path: Path) -> None:
    """Tahoe deltas use plate/cell-line vehicles rather than a global DMSO."""
    obs = pd.DataFrame(
        {
            "cell_line_id": ["line_a", "line_a", "line_b", "line_b"],
            "plate": ["plate_1", "plate_1", "plate_2", "plate_2"],
            "drug": ["DMSO_TF", "drug_x", "DMSO_TF", "drug_x"],
            "sample": ["vehicle_a", "treated_a", "vehicle_b", "treated_b"],
            "dose": [0.0, 1.0, 0.0, 1.0],
            "targets": [pd.NA, "TARGET_A", pd.NA, "TARGET_B"],
            "moa-fine": [pd.NA, "MOA_A", pd.NA, "MOA_B"],
        },
        index=["a_vehicle", "a_drug", "b_vehicle", "b_drug"],
    )
    counts = np.array(
        [
            [90, 10],
            [10, 90],
            [10, 90],
            [90, 10],
        ],
        dtype=np.int64,
    )
    adata = ad.AnnData(
        X=sp.csr_matrix(np.zeros_like(counts)),
        obs=obs,
        var=pd.DataFrame(index=["G1", "G2"]),
    )
    adata.layers["counts"] = sp.csr_matrix(counts)
    path = tmp_path / "tahoe.h5ad"
    adata.write_h5ad(path)

    signatures = load_tahoe_signatures(
        path,
        dose_col="dose",
        chunk_size=2,
    )

    by_line = signatures.meta.reset_index(drop=True).set_index("cell_line_id")
    row_a = int(np.flatnonzero(signatures.meta["cell_line_id"].eq("line_a"))[0])
    row_b = int(np.flatnonzero(signatures.meta["cell_line_id"].eq("line_b"))[0])
    expected_control_a = np.log1p(np.array([900_000.0, 100_000.0]))
    expected_control_b = np.log1p(np.array([100_000.0, 900_000.0]))
    expected_treated_a = expected_control_b
    expected_treated_b = expected_control_a

    np.testing.assert_allclose(signatures.control[row_a], expected_control_a)
    np.testing.assert_allclose(signatures.control[row_b], expected_control_b)
    np.testing.assert_allclose(
        signatures.delta[row_a],
        expected_treated_a - expected_control_a,
    )
    np.testing.assert_allclose(
        signatures.delta[row_b],
        expected_treated_b - expected_control_b,
    )
    assert by_line.loc["line_a", "targets"] == "TARGET_A"
    assert by_line.loc["line_b", "moa-fine"] == "MOA_B"
    assert by_line["n_control_cells"].tolist() == [1, 1]


def test_lincs_joins_metadata_by_id_and_keeps_landmarks(tmp_path: Path) -> None:
    """LINCS rows align by signature ID and explicit landmark symbols."""
    matrix = pd.DataFrame(
        {
            "signature_id": ["sig_b", "sig_a"],
            "G3": [30.0, 3.0],
            "G1": [10.0, 1.0],
            "G2": [20.0, 2.0],
        }
    )
    metadata = pd.DataFrame(
        {
            "signature_id": ["sig_a", "sig_b", "unused"],
            "drug": ["drug_a", "drug_b", "drug_unused"],
            "cell_line": ["A", "B", "C"],
        }
    )
    gene_metadata = pd.DataFrame(
        {
            "gene": ["G1", "G2", "G3"],
            "is_landmark": [True, False, True],
        }
    )
    matrix_path = tmp_path / "matrix.csv"
    metadata_path = tmp_path / "metadata.tsv"
    genes_path = tmp_path / "genes.parquet"
    matrix.to_csv(matrix_path, index=False)
    metadata.to_csv(metadata_path, sep="\t", index=False)
    gene_metadata.to_parquet(genes_path, index=False)

    signatures = load_lincs_signatures(
        matrix_path,
        metadata_path,
        gene_metadata_path=genes_path,
    )

    assert tuple(signatures.genes) == ("G3", "G1")
    assert signatures.meta.index.tolist() == ["sig_b", "sig_a"]
    assert signatures.meta["drug"].tolist() == ["drug_b", "drug_a"]
    np.testing.assert_array_equal(
        signatures.delta,
        np.array([[30.0, 10.0], [3.0, 1.0]]),
    )
    assert signatures.provenance["representation"] == "measured_landmark_genes"


def test_lincs_loads_gctx_by_ids_and_filters_compounds(
    tmp_path: Path,
) -> None:
    """Official GCTX axes map through IDs before landmark/type filtering."""
    matrix_path = tmp_path / "matrix.gctx"
    with h5py.File(matrix_path, "w") as handle:
        handle.create_dataset(
            "0/DATA/0/matrix",
            data=np.array(
                [
                    [30.0, 10.0, 20.0],
                    [3.0, 1.0, 2.0],
                    [300.0, 100.0, 200.0],
                ]
            ),
        )
        handle.create_dataset(
            "0/META/COL/id",
            data=np.asarray([b"sig_b", b"sig_a", b"control"]),
        )
        handle.create_dataset(
            "0/META/ROW/id",
            data=np.asarray([b"103", b"101", b"102"]),
        )

    metadata = pd.DataFrame(
        {
            "sig_id": ["sig_a", "control", "sig_b"],
            "pert_iname": ["drug_a", "DMSO", "drug_b"],
            "pert_type": ["trt_cp", "ctl_vehicle", "trt_cp"],
            "cell_id": ["A", "A", "B"],
        }
    )
    genes = pd.DataFrame(
        {
            "pr_gene_id": ["101", "102", "103"],
            "pr_gene_symbol": ["G1", "G2", "G3"],
            "pr_is_lm": [1, 0, 1],
        }
    )
    metadata_path = tmp_path / "sig_info.txt"
    genes_path = tmp_path / "gene_info.txt"
    metadata.to_csv(metadata_path, sep="\t", index=False)
    genes.to_csv(genes_path, sep="\t", index=False)

    signatures = load_lincs_signatures(
        matrix_path,
        metadata_path,
        signature_id_col="sig_id",
        gene_metadata_path=genes_path,
        gene_id_col="pr_gene_id",
        gene_col="pr_gene_symbol",
        landmark_col="pr_is_lm",
        perturbation_type_col="pert_type",
        compound_type="trt_cp",
        chunk_size=2,
    )

    assert tuple(signatures.genes) == ("G3", "G1")
    assert signatures.meta.index.tolist() == ["sig_b", "sig_a"]
    assert signatures.meta["pert_iname"].tolist() == ["drug_b", "drug_a"]
    np.testing.assert_array_equal(
        signatures.delta,
        np.array([[30.0, 10.0], [3.0, 1.0]]),
    )
    assert signatures.provenance["n_matrix_signatures"] == 3
    assert signatures.provenance["n_retained_signatures"] == 2
    assert signatures.provenance["n_retained_genes"] == 2
