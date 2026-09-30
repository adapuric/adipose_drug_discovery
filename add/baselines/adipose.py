"""Small donor-level predictors of the paired surgery expression change."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class AdiposeMeanShift:
    """Equal-donor change learned on a named evaluation gene universe."""

    shift: pd.Series
    training_donors: tuple[str, ...]


@dataclass(frozen=True)
class AdiposeRidge:
    """Training-only scaling, low-rank baseline features, and change
    regression.
    """

    input_genes: tuple[str, ...]
    output_genes: tuple[str, ...]
    training_donors: tuple[str, ...]
    scaler: StandardScaler
    pca: PCA
    ridge: Ridge


def fit_adipose_mean_shift(
    changes: pd.DataFrame,
    *,
    minimum_donors: int = 2,
) -> AdiposeMeanShift:
    """Fit a state-specific shift with one equally weighted row per donor."""
    _validate_profiles(changes)
    if minimum_donors < 2 or len(changes) < minimum_donors:
        raise ValueError("insufficient_training_donors for mean shift")
    return AdiposeMeanShift(
        shift=changes.mean(axis=0),
        training_donors=tuple(changes.index.astype(str)),
    )


def predict_adipose_mean_shift(
    model: AdiposeMeanShift,
    baseline: pd.DataFrame,
) -> pd.DataFrame:
    """Return predicted changes without inspecting held-out target values."""
    _validate_profiles(baseline)
    selected = baseline.loc[:, model.shift.index]
    return pd.DataFrame(
        np.tile(model.shift.to_numpy(), (len(selected), 1)),
        index=selected.index,
        columns=selected.columns,
    )


def fit_adipose_ridge(
    baseline: pd.DataFrame,
    changes: pd.DataFrame,
    *,
    minimum_donors: int = 3,
    maximum_input_genes: int = 2000,
    n_components: int = 5,
    alpha: float = 10.0,
    random_seed: int = 42,
) -> AdiposeRidge:
    """Fit a small linear change model using only training donor profiles."""
    _validate_profiles(baseline)
    _validate_profiles(changes)
    if set(baseline.index) != set(changes.index):
        raise ValueError("baseline and changes must identify the same donors")
    if minimum_donors < 3 or len(baseline) < minimum_donors:
        raise ValueError("insufficient_training_donors for ridge")
    if maximum_input_genes < 1 or n_components < 1:
        raise ValueError("feature and component limits must be positive")
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be finite and nonnegative")
    variance = baseline.var(axis=0, ddof=0)
    eligible = variance.loc[variance > 0]
    if eligible.empty:
        raise ValueError("no_variable_input_features")
    order = np.lexsort((eligible.index.astype(str), -eligible.to_numpy()))
    genes = tuple(eligible.index[order[:maximum_input_genes]].astype(str))
    scaler = StandardScaler()
    scaled = scaler.fit_transform(baseline.loc[:, list(genes)])
    pca = PCA(
        n_components=min(n_components, len(baseline) - 1, len(genes)),
        svd_solver="full",
        whiten=False,
        random_state=random_seed,
    )
    predictors = pca.fit_transform(scaled)
    ridge = Ridge(alpha=alpha, fit_intercept=True)
    ridge.fit(predictors, changes.loc[baseline.index].to_numpy(dtype=float))
    return AdiposeRidge(
        input_genes=genes,
        output_genes=tuple(changes.columns.astype(str)),
        training_donors=tuple(baseline.index.astype(str)),
        scaler=scaler,
        pca=pca,
        ridge=ridge,
    )


def predict_adipose_ridge(
    model: AdiposeRidge,
    baseline: pd.DataFrame,
) -> pd.DataFrame:
    """Predict paired changes on the fitted output genes from baseline only."""
    _validate_profiles(baseline)
    selected = baseline.loc[:, list(model.input_genes)]
    components = model.pca.transform(model.scaler.transform(selected))
    return pd.DataFrame(
        model.ridge.predict(components),
        index=baseline.index,
        columns=list(model.output_genes),
    )


def _validate_profiles(profiles: pd.DataFrame) -> None:
    """Require finite, uniquely named donor rows and gene columns."""
    if (
        profiles.empty
        or not profiles.index.is_unique
        or not profiles.columns.is_unique
    ):
        raise ValueError("profiles require nonempty unique donor and gene IDs")
    if profiles.index.hasnans or profiles.columns.hasnans:
        raise ValueError("profile identifiers must not be missing")
    if not np.isfinite(profiles.to_numpy(dtype=float)).all():
        raise ValueError("profiles contain non-finite expression")
