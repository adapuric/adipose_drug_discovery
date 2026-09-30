"""Observable scientific encodings for perturbation-report figures."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from add.visualization.perturbation import plot_edit_effect_heatmap


def test_effect_heatmap_keeps_missing_distinct_from_zero() -> None:
    """Missing combinations remain masked on the signed effect scale."""
    effects = pd.DataFrame(
        {
            "edit_id": ["A", "A", "B"],
            "state": ["state"] * 3,
            "donor_id": ["d1"] * 3,
            "gene_id": ["g1", "g2", "g1"],
            "delta": [0.0, 1.0, -1.0],
            "scale": ["shared_unedited_log1p_cpm"] * 3,
        }
    )

    figure, axis = plot_edit_effect_heatmap(effects)

    values = axis.images[0].get_array()
    assert values[0, 0] == 0.0
    assert np.ma.getmaskarray(values)[1, 1]
    plt.close(figure)
