"""Input-table validation and labeling for visualizations."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


def _require_columns(
    table: pd.DataFrame,
    columns: Sequence[str],
    *,
    table_name: str,
) -> None:
    """Raise when a plotting input omits required scientific fields."""
    if missing := [column for column in columns if column not in table]:
        raise ValueError(f"{table_name} is missing columns: {missing}")


def _state_column(table: pd.DataFrame) -> str:
    """Return the supported adipocyte-state column name."""
    for column in ("state", "adipocyte_state"):
        if column in table:
            return column
    raise ValueError("Table must contain 'state' or 'adipocyte_state'")


def _drug_labels(table: pd.DataFrame) -> list[str]:
    """Return the most informative available drug labels."""
    return next(
        (
            table[column].fillna("unknown").astype(str).tolist()
            for column in ("drug_name", "drug", "drug_id")
            if column in table
        ),
        ["generic perturbation"] * len(table),
    )


def _ordered_values(
    values: pd.Series,
    *,
    requested: Sequence[str] | None,
) -> list[str]:
    """Resolve explicit ordering while retaining unlisted observed values."""
    observed = values.dropna().astype(str).drop_duplicates().tolist()
    if requested is None:
        return sorted(observed)
    requested_values = list(dict.fromkeys(str(value) for value in requested))
    if missing := [
        value for value in requested_values if value not in observed
    ]:
        raise ValueError(
            f"Requested states are absent from the data: {missing}"
        )
    return requested_values + [
        value for value in observed if value not in requested_values
    ]


def _score_label(score_col: str) -> str:
    """Return an interpretable axis label for a score column."""
    labels = {
        "score": "Mimicry score (positive follows rescue)",
        "score_mimic": "Pearson mimicry (positive follows rescue)",
        "score_spearman": "Spearman mimicry (positive follows rescue)",
        "score_connectivity": "Weighted connectivity (positive follows rescue)",
    }
    return labels.get(score_col, score_col.replace("_", " "))
