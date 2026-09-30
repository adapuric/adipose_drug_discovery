"""Worked native-count example for downstream-effect aggregation."""

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from add.validation.perturbgen_results import read_native_edit_effects


def test_native_effects_sum_counts_without_treating_cells_as_donors(
    tmp_path,
) -> None:
    """Native predictions are summed by donor before shared-factor scaling."""
    unedited = np.array([[10.0, 3.0, 5.0], [20.0, 6.0, 10.0], [10.0, 3.0, 5.0]])
    edited = unedited.copy()
    edited[:, 0] = 0.0
    native = ad.AnnData(
        X=sp.csr_matrix(edited),
        obs=pd.DataFrame(
            {"Donor": ["d1", "d1", "d2"], "state": ["AD1"] * 3},
            index=["a", "b", "c"],
        ),
        var=pd.DataFrame(index=["target", "g1", "g2"]),
    )
    native.layers["pred_counts"] = sp.csr_matrix(unedited)
    path = tmp_path / "native.h5ad"
    native.write_h5ad(path)

    effects, qc = read_native_edit_effects(
        path,
        run_id="run",
        edit_id="target",
        edited_genes=["target"],
        state_col="state",
        chunk_size=1,
    )

    observed = effects.query(
        "model == 'perturbgen:run' "
        "and scale == 'shared_unedited_log1p_cpm' and not is_edited_gene"
    )
    np.testing.assert_array_equal(observed["delta"], 0.0)
    assert qc.set_index("donor_id").loc["d1", "n_nuclei"] == 2
    assert set(effects["truth_kind"]) == {"model_sensitivity"}
