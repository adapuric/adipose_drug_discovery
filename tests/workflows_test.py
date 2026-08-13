"""Tests for workflow integration helpers."""

import numpy as np
import pandas as pd

from add.perturb import PerturbSignatures
from add.workflows import _top_candidate_series


def test_top_candidate_selects_the_requested_adipose_state() -> None:
    """Candidate plotting must not mix predictions for the same drug."""
    signatures = PerturbSignatures(
        delta=np.array([[1.0, 0.0], [0.0, 2.0]]),
        genes=["G1", "G2"],
        meta=pd.DataFrame(
            {
                "drug": ["drug_a", "drug_a"],
                "state": ["AD1", "AD_ALL"],
            },
            index=["sig_1", "sig_2"],
        ),
    )
    rankings = pd.DataFrame(
        {
            "drug": ["drug_a", "drug_a"],
            "state": ["AD1", "AD_ALL"],
            "score": [0.5, 0.8],
        }
    )

    selected = _top_candidate_series(
        rankings,
        state="AD_ALL",
        signatures=signatures,
        context_scores=None,
    )

    assert selected is not None
    label, delta = selected
    assert label == "drug_a"
    pd.testing.assert_series_equal(
        delta,
        pd.Series(
            [0.0, 2.0],
            index=pd.Index(["G1", "G2"], name="gene"),
        ),
    )
