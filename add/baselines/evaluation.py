"""Held-out context splitting and variability summaries."""

from __future__ import annotations

from collections.abc import Sequence
from math import ceil

import numpy as np
import pandas as pd

from add.baselines.signatures import _context_columns
from add.baselines.signatures import _context_keys
from add.baselines.signatures import _signature_ids
from add.perturb import PerturbSignatures


def split_signature_contexts(
    signatures: PerturbSignatures,
    *,
    context_col: str | Sequence[str],
    test_fraction: float,
    random_seed: int,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Split signature IDs while keeping each context on only one side."""
    context_cols = _context_columns(context_col)
    if missing := [
        column for column in context_cols if column not in signatures.meta
    ]:
        raise KeyError(
            f"Perturbation metadata lacks context columns: {missing}"
        )
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be strictly between 0 and 1.")

    context_values = _context_keys(signatures.meta, context_cols)
    unique_contexts = sorted(set(context_values))
    if len(unique_contexts) < 2:
        raise ValueError(
            "At least two contexts are required for a held-out split."
        )

    n_test = min(
        len(unique_contexts) - 1,
        max(1, ceil(len(unique_contexts) * test_fraction)),
    )
    generator = np.random.default_rng(random_seed)
    shuffled = generator.permutation(len(unique_contexts))
    test_contexts = {
        unique_contexts[index] for index in shuffled[:n_test].tolist()
    }
    signature_ids = _signature_ids(signatures)

    train_ids = tuple(
        signature_id
        for signature_id, context in zip(
            signature_ids,
            context_values,
            strict=True,
        )
        if context not in test_contexts
    )
    test_ids = tuple(
        signature_id
        for signature_id, context in zip(
            signature_ids,
            context_values,
            strict=True,
        )
        if context in test_contexts
    )

    return train_ids, test_ids


def _context_score_summary(context_scores: pd.DataFrame) -> pd.DataFrame:
    """Summarize context variability without replacing context-level rows."""
    rows: list[dict[str, object]] = []
    for keys, group in context_scores.groupby(
        ["state", "drug"],
        observed=True,
        sort=True,
        dropna=False,
    ):
        key_values = list(keys if isinstance(keys, tuple) else (keys,))
        if len(key_values) != 2:
            raise RuntimeError("Context summary expected state and drug keys.")
        state = str(key_values[0])
        drug = str(key_values[1])
        finite = group.loc[np.isfinite(group["score_mimic"]), "score_mimic"]
        rows.append(
            {
                "state": state,
                "drug": drug,
                "context_score_median": np.nan
                if finite.empty
                else float(finite.median()),
                "context_score_sd": np.nan
                if finite.empty
                else float(finite.std(ddof=0)),
                "context_fraction_positive": np.nan
                if finite.empty
                else float((finite > 0).mean()),
            }
        )
    return pd.DataFrame(rows)
