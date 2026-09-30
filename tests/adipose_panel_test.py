"""Array-scale and replication contracts for the small adipose panel."""

import numpy as np
import pandas as pd

from add.validation.adipose_panel import adipose_panel_signatures
from add.validation.adipose_panel import map_array_expression


def test_probe_mapping_excludes_ambiguity_and_does_not_log_twice() -> None:
    """Ambiguous probes cannot multiply effects or change the log scale."""
    expression = pd.DataFrame(
        {"control": [1.0, 3.0, 100.0], "treated": [3.0, 5.0, 100.0]},
        index=["p1", "p2", "ambiguous"],
    )
    annotation = pd.DataFrame(
        {
            "probe_id": ["p1", "p2", "ambiguous", "ambiguous"],
            "gene_id": ["A", "A", "A", "B"],
        }
    )

    result, audit = map_array_expression(
        expression, annotation, input_scale="log2_reference_ratio"
    )

    np.testing.assert_allclose(result.loc["A"], [2.0, 4.0])
    assert result.index.tolist() == ["A"]
    assert (
        audit.loc[audit["probe_id"].eq("ambiguous"), "status"].iloc[0]
        == "unmapped_or_ambiguous"
    )


def test_technical_replicates_do_not_inflate_donor_support() -> None:
    """Technical duplicates cannot inflate donor support or weighting."""
    samples = pd.DataFrame(
        {
            "study_id": ["study"] * 4,
            "sample_id": ["c1", "t1", "c2", "t2"],
            "compound": ["vehicle", "drug", "vehicle", "drug"],
            "unit_id": ["d1", "d1", "d2", "d2"],
            "unit_type": ["donor"] * 4,
            "condition": ["vehicle", "treated", "vehicle", "treated"],
        }
    )
    expression = pd.DataFrame(
        [[1.0, 3.0, 2.0, 8.0]],
        index=pd.Index(["gene"], name="gene_id"),
        columns=samples["sample_id"],
    )
    first = adipose_panel_signatures(expression, samples, pairing="paired")
    duplicate = samples.iloc[[1]].assign(sample_id="t1_repeat")
    expanded = pd.concat([samples, duplicate], ignore_index=True)
    expression["t1_repeat"] = expression["t1"]

    second = adipose_panel_signatures(expression, expanded, pairing="paired")

    pd.testing.assert_frame_equal(
        first["study_signatures"], second["study_signatures"]
    )
    assert first["study_signatures"].loc[0, "delta"] == 4.0
    assert first["study_signatures"].loc[0, "n_units_treated"] == 2
