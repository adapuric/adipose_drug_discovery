"""Tests for PerturbGen inference wiring and donor-aware summaries."""

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from run_scripts.perturbgen.run_perturbation import (
    write_native_perturbation_config,
)
from run_scripts.perturbgen.summarize_perturbation import (
    summarize_perturbation_result,
)


def test_summary_averages_nuclei_within_donor_and_state(
    tmp_path: Path,
) -> None:
    """Cell rows are reduced to donor-state effects before donor summaries."""
    result_path = tmp_path / "result.h5ad"
    adata = ad.AnnData(
        X=sparse.csr_matrix(
            np.array(
                [
                    [3.0, 2.0],
                    [5.0, 6.0],
                    [8.0, 4.0],
                    [7.0, 9.0],
                ]
            )
        ),
        obs=pd.DataFrame(
            {
                "Donor": ["D1", "D1", "D2", "D2"],
                "cell_states_adipocytes": ["AD1", "AD1", "AD1", "AD2"],
            },
            index=["c1", "c2", "c3", "c4"],
        ),
        var=pd.DataFrame(index=["ENSG1", "ENSG2"]),
        layers={
            "pred_counts": sparse.csr_matrix(
                np.array(
                    [
                        [1.0, 1.0],
                        [3.0, 2.0],
                        [4.0, 2.0],
                        [8.0, 8.0],
                    ]
                )
            ),
            "true_counts": np.array(
                [
                    [2.0, 1.0],
                    [4.0, 3.0],
                    [5.0, 2.0],
                    [9.0, 8.0],
                ]
            ),
        },
    )
    adata.write_h5ad(result_path)

    paths = summarize_perturbation_result(
        result_path,
        output_directory=tmp_path / "summary",
        donor_col="Donor",
        state_col="cell_states_adipocytes",
    )

    donor_effects = pd.read_csv(paths["donor_state_effects"])
    d1_ad1 = donor_effects.loc[
        (donor_effects["Donor"] == "D1")
        & (donor_effects["cell_states_adipocytes"] == "AD1")
    ].set_index("gene")
    assert d1_ad1.loc["ENSG1", "n_nuclei"] == 2
    assert d1_ad1.loc["ENSG1", "model_perturbation_delta"] == 2.0
    assert d1_ad1.loc["ENSG2", "model_perturbation_delta"] == 2.5

    state_summary = pd.read_csv(paths["state_summary"])
    ad1_ensg1 = state_summary.loc[
        (state_summary["cell_states_adipocytes"] == "AD1")
        & (state_summary["gene"] == "ENSG1")
    ].iloc[0]
    assert ad1_ensg1["n_donors"] == 2
    assert ad1_ensg1["mean_model_perturbation_delta"] == 3.0


def test_native_config_cannot_relabel_an_existing_gene_run(
    tmp_path: Path,
) -> None:
    """A changed checkpoint cannot overwrite an existing run manifest."""
    config_path = tmp_path / "native_config.yaml"
    write_native_perturbation_config(
        {"model": {"ckpt_masking_path": "decoder_a.ckpt"}},
        path=config_path,
    )

    with pytest.raises(FileExistsError, match="Refusing to mix"):
        write_native_perturbation_config(
            {"model": {"ckpt_masking_path": "decoder_b.ckpt"}},
            path=config_path,
        )
