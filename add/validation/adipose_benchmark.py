"""Donor-held-out evaluation of simple paired surgery-transition predictors."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import fields
from dataclasses import replace
from pathlib import Path
from typing import cast

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from add.baselines.adipose import fit_adipose_mean_shift
from add.baselines.adipose import fit_adipose_ridge
from add.baselines.adipose import predict_adipose_mean_shift
from add.baselines.adipose import predict_adipose_ridge
from add.scoring import score_mimicry
from add.validation.artifacts import input_identity
from add.validation.artifacts import prepare_report_directory
from add.validation.artifacts import publish_report


@dataclass(frozen=True, kw_only=True)
class BenchmarkSettings:
    """Scientific parameters shared by profile, fitting, and reporting
    stages.
    """

    donor_col: str = "Donor"
    condition_col: str = "condition"
    state_col: str = "cell_state_t2d"
    support_col: str = "n_cells"
    baseline_label: str = "baseline"
    weightloss_label: str = "weightloss"
    count_layer: str | None = None
    gene_namespace: str = "gene_symbol"
    preprocessing_policy: str = "full_raw_gene_universe"
    min_cells: int = 20
    normalization_target: float = 1_000_000.0
    minimum_gene_cpm: float = 1.0
    minimum_gene_donors: int = 2
    minimum_shared_genes: int = 50
    mean_shift_minimum_donors: int = 2
    ridge_minimum_donors: int = 3
    ridge_maximum_genes: int = 2000
    ridge_components: int = 5
    ridge_alpha: float = 10.0
    random_seed: int = 42
    bootstrap_draws: int = 2000
    confidence_level: float = 0.95

    def __post_init__(self) -> None:
        """Reject invalid thresholds before constructing folds or outputs."""
        for field in fields(self):
            value = getattr(self, field.name)
            default = field.default
            if isinstance(default, int):
                if not isinstance(value, int) or isinstance(value, bool):
                    raise TypeError(f"{field.name} must be an integer")
                if value < (0 if field.name == "random_seed" else 1):
                    raise ValueError(f"{field.name} is below its valid range")
            elif isinstance(default, float):
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not np.isfinite(value)
                    or value < 0
                ):
                    raise ValueError(
                        f"{field.name} must be finite and nonnegative"
                    )
            elif isinstance(default, str) and (
                not isinstance(value, str) or not value
            ):
                raise ValueError(f"{field.name} must be a nonempty string")
        if self.normalization_target <= 0 or not 0 < self.confidence_level < 1:
            raise ValueError(
                "normalization target and confidence level are invalid"
            )
        if (
            self.minimum_shared_genes < 2
            or self.mean_shift_minimum_donors < 2
            or self.ridge_minimum_donors < 3
        ):
            raise ValueError("insufficient minimum support thresholds")
        if self.baseline_label == self.weightloss_label:
            raise ValueError("baseline and weightloss labels must differ")
        if self.count_layer is not None and not isinstance(
            self.count_layer, str
        ):
            raise TypeError("count_layer must be a string or null")
        if self.preprocessing_policy not in {
            "full_raw_gene_universe",
            "fixed_external_vocabulary",
            "transductive_hvg",
        }:
            raise ValueError("Unknown preprocessing_policy")


def benchmark_settings(values: Mapping[str, object]) -> BenchmarkSettings:
    """Validate explicitly supplied settings against the scientific schema."""
    unknown = set(values) - {field.name for field in fields(BenchmarkSettings)}
    if unknown:
        raise ValueError(f"Unknown benchmark parameters: {sorted(unknown)}")
    return replace(BenchmarkSettings(), **values)


def paired_donor_profiles(
    pseudobulk: ad.AnnData,
    *,
    settings: BenchmarkSettings,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return aligned baseline/target log-CPM profiles and explicit
    exclusions.
    """
    if (
        pseudobulk.uns.get("pseudobulk", {}).get("aggregation")
        != "sum_raw_counts"
    ):
        raise ValueError(
            "Input must declare pseudobulk aggregation=sum_raw_counts"
        )
    if not pseudobulk.var_names.is_unique or pseudobulk.var_names.hasnans:
        raise ValueError("Pseudobulk gene IDs must be unique and nonmissing")
    obs = cast(pd.DataFrame, pseudobulk.obs)
    keys = [settings.donor_col, settings.condition_col, settings.state_col]
    required = [*keys, settings.support_col]
    if missing := set(required) - set(obs):
        raise ValueError(f"Pseudobulk metadata lacks {sorted(missing)}")
    if obs[keys].isna().any().any() or obs.duplicated(keys).any():
        raise ValueError("Pseudobulk keys must be unique and nonmissing")
    metadata = obs[required].copy()
    metadata[keys] = metadata[keys].astype(str)
    if metadata.duplicated(keys).any():
        raise ValueError(
            "String conversion caused a pseudobulk identifier collision"
        )
    counts = (
        pseudobulk.X
        if settings.count_layer is None
        else pseudobulk.layers[settings.count_layer]
    )
    counts = (
        sp.csr_matrix(counts).toarray()
        if sp.issparse(counts)
        else np.asarray(counts)
    )
    if (
        not np.isfinite(counts).all()
        or (counts < 0).any()
        or not np.allclose(counts, np.round(counts))
    ):
        raise ValueError(
            "Pseudobulk input must contain finite nonnegative raw counts"
        )
    sizes = counts.sum(axis=1)
    transformed = np.full(counts.shape, np.nan, dtype=float)
    nonempty = sizes > 0
    transformed[nonempty] = np.log1p(
        counts[nonempty] / sizes[nonempty, None] * settings.normalization_target
    )
    metadata["_position"] = np.arange(len(metadata))
    baseline_rows: list[np.ndarray] = []
    target_rows: list[np.ndarray] = []
    retained: list[tuple[str, str]] = []
    exclusions: list[dict[str, object]] = []
    surgery = metadata.loc[
        metadata[settings.condition_col].isin(
            [settings.baseline_label, settings.weightloss_label]
        )
    ]
    for (donor, state), group in surgery.groupby(
        [settings.donor_col, settings.state_col], sort=True, observed=True
    ):
        by_condition = group.set_index(settings.condition_col)
        reason = "ok"
        if not {settings.baseline_label, settings.weightloss_label} <= set(
            by_condition.index
        ):
            reason = "missing_condition_pair"
        elif (
            not pd.to_numeric(group[settings.support_col], errors="raise")
            .ge(settings.min_cells)
            .all()
        ):
            reason = "insufficient_cells"
        elif (sizes[group["_position"].to_numpy(dtype=int)] <= 0).any():
            reason = "zero_library"
        if reason != "ok":
            exclusions.append(
                {"donor_id": donor, "state": state, "status": reason}
            )
            continue
        retained.append((str(donor), str(state)))
        baseline_rows.append(
            transformed[
                int(
                    by_condition.loc[
                        [settings.baseline_label], "_position"
                    ].to_numpy(dtype=int)[0]
                )
            ]
        )
        target_rows.append(
            transformed[
                int(
                    by_condition.loc[
                        [settings.weightloss_label], "_position"
                    ].to_numpy(dtype=int)[0]
                )
            ]
        )
    if not retained:
        raise ValueError("No supported paired donor/state profiles")
    index = pd.MultiIndex.from_tuples(retained, names=["donor_id", "state"])
    genes = pd.Index(pseudobulk.var_names.astype(str), name="gene_id")
    return (
        pd.DataFrame(np.vstack(baseline_rows), index=index, columns=genes),
        pd.DataFrame(np.vstack(target_rows), index=index, columns=genes),
        pd.DataFrame(exclusions, columns=["donor_id", "state", "status"]),
    )


def donor_folds(donors: pd.Index) -> pd.DataFrame:
    """Assign every donor exactly once per outer leave-one-donor-out fold."""
    names = sorted(set(donors.astype(str)))
    if len(names) < 2:
        raise ValueError(
            "At least two paired donors are required for holdout evaluation"
        )
    return pd.DataFrame(
        [
            {
                "fold_id": f"donor_{i:04d}",
                "donor_id": donor,
                "role": "test" if donor == heldout else "train",
            }
            for i, heldout in enumerate(names)
            for donor in names
        ]
    )


def evaluate_adipose_transition(
    pseudobulk: ad.AnnData,
    *,
    settings: BenchmarkSettings | None = None,
) -> dict[str, pd.DataFrame]:
    """Evaluate three transition baselines with training-only gene selection."""
    resolved = settings or BenchmarkSettings()
    baseline, target, exclusions = paired_donor_profiles(
        pseudobulk, settings=resolved
    )
    folds = donor_folds(baseline.index.get_level_values("donor_id"))
    predictions: list[pd.DataFrame] = []
    metrics: list[dict[str, object]] = []
    genes: list[dict[str, object]] = []
    fits: list[dict[str, object]] = []
    for fold_id, split in folds.groupby("fold_id", sort=True):
        heldout = str(split.loc[split["role"].eq("test"), "donor_id"].iloc[0])
        for state in sorted(set(baseline.index.get_level_values("state"))):
            x = cast(pd.DataFrame, baseline.xs(state, level="state"))
            y = cast(pd.DataFrame, target.xs(state, level="state"))
            if heldout not in x.index:
                continue
            training = x.index[x.index != heldout]
            present = (
                (
                    np.expm1(x.loc[training])
                    * 1_000_000
                    / resolved.normalization_target
                )
                >= resolved.minimum_gene_cpm
            ) | (
                (
                    np.expm1(y.loc[training])
                    * 1_000_000
                    / resolved.normalization_target
                )
                >= resolved.minimum_gene_cpm
            )
            selected = x.columns[
                present.sum(axis=0) >= resolved.minimum_gene_donors
            ]
            for ordinal, gene in enumerate(selected):
                genes.append(
                    {
                        "fold_id": fold_id,
                        "state": state,
                        "gene_id": gene,
                        "role": "evaluation",
                        "ordinal": ordinal,
                    }
                )
            x_train = x.reindex(index=training, columns=selected)
            changes = y.reindex(index=training, columns=selected) - x_train
            x_test = x.reindex(index=[heldout], columns=selected)
            observed = (
                y.loc[[heldout], selected].iloc[0]
                - x.loc[[heldout], selected].iloc[0]
            ).to_numpy()
            for model in (
                "adipose-no-change",
                "adipose-mean-shift",
                "adipose-ridge",
            ):
                base = {
                    "task": "surgery_transition",
                    "model": model,
                    "fold_id": fold_id,
                    "donor_id": heldout,
                    "state": state,
                    "n_training_donors": len(training),
                    "n_genes": len(selected),
                    "scale": "log1p_cpm",
                    "comparison_class": (
                        "donor_heldout_transductive_preprocessing"
                        if resolved.preprocessing_policy == "transductive_hvg"
                        else "heldout_target_blind"
                    ),
                }
                status = "ok"
                prediction = np.full(len(selected), np.nan)
                if len(selected) < resolved.minimum_shared_genes:
                    status = "insufficient_shared_genes"
                elif model == "adipose-no-change":
                    prediction = np.zeros(len(selected))
                elif model == "adipose-mean-shift":
                    if len(training) < resolved.mean_shift_minimum_donors:
                        status = "insufficient_training_donors"
                    else:
                        fitted_mean = fit_adipose_mean_shift(
                            changes,
                            minimum_donors=resolved.mean_shift_minimum_donors,
                        )
                        prediction = (
                            predict_adipose_mean_shift(fitted_mean, x_test)
                            .iloc[0]
                            .to_numpy()
                        )
                elif len(training) < resolved.ridge_minimum_donors:
                    status = "insufficient_training_donors"
                elif not (x_train.var(axis=0, ddof=0) > 0).any():
                    status = "no_variable_input_features"
                else:
                    fitted_ridge = fit_adipose_ridge(
                        x_train,
                        changes,
                        minimum_donors=resolved.ridge_minimum_donors,
                        maximum_input_genes=resolved.ridge_maximum_genes,
                        n_components=resolved.ridge_components,
                        alpha=resolved.ridge_alpha,
                        random_seed=resolved.random_seed,
                    )
                    prediction = (
                        predict_adipose_ridge(fitted_ridge, x_test)
                        .iloc[0]
                        .to_numpy()
                    )
                    fits.append(
                        base
                        | {
                            "effective_components": (
                                fitted_ridge.pca.n_components_
                            ),
                            "alpha": resolved.ridge_alpha,
                        }
                    )
                    for ordinal, gene in enumerate(fitted_ridge.input_genes):
                        genes.append(
                            {
                                "fold_id": fold_id,
                                "state": state,
                                "gene_id": gene,
                                "role": "ridge_input",
                                "ordinal": ordinal,
                            }
                        )
                metric = change_prediction_metrics(
                    prediction,
                    observed,
                    selected,
                    minimum_shared_genes=resolved.minimum_shared_genes,
                )
                metrics.append(base | metric | {"status": status})
                predictions.append(
                    pd.DataFrame(
                        base
                        | {
                            "gene_id": selected,
                            "baseline_expression": x_test.iloc[0].to_numpy(),
                            "observed_expression": y.loc[
                                [heldout], selected.tolist()
                            ]
                            .iloc[0]
                            .to_numpy(),
                            "predicted_expression": x_test.iloc[0].to_numpy()
                            + prediction,
                            "observed_delta": observed,
                            "predicted_delta": prediction,
                            "status": status,
                        }
                    )
                )
    metric_table = pd.DataFrame(metrics)
    comparisons = paired_model_comparisons(metric_table)
    return {
        "folds.csv": folds,
        "fold_genes.parquet": pd.DataFrame(genes),
        "predictions.parquet": pd.concat(predictions, ignore_index=True),
        "metrics.csv": metric_table,
        "fits.csv": pd.DataFrame(fits),
        "paired_model_comparisons.csv": comparisons,
        "paired_comparison_summary.csv": donor_metric_summary(
            comparisons,
            value_columns=("rmse_improvement", "mae_improvement"),
            group_columns=("model", "comparator", "state"),
            random_seed=resolved.random_seed,
            bootstrap_draws=resolved.bootstrap_draws,
            confidence_level=resolved.confidence_level,
        ),
        "summary.csv": donor_metric_summary(
            metric_table,
            random_seed=resolved.random_seed,
            bootstrap_draws=resolved.bootstrap_draws,
            confidence_level=resolved.confidence_level,
        ),
        "exclusions.csv": exclusions,
    }


def change_prediction_metrics(
    predicted: np.ndarray,
    observed: np.ndarray,
    genes: pd.Index,
    *,
    minimum_shared_genes: int = 50,
) -> dict[str, object]:
    """Measure change-vector error while keeping constant correlation
    undefined.
    """
    finite = np.isfinite(predicted) & np.isfinite(observed)
    score = score_mimicry(
        predicted,
        genes.tolist(),
        observed,
        genes.tolist(),
        minimum_shared_genes=minimum_shared_genes,
    )
    supported = finite.sum() >= minimum_shared_genes
    error = predicted[finite] - observed[finite]
    return {
        "delta_rmse": float(np.sqrt(np.mean(error**2)))
        if supported
        else np.nan,
        "delta_mae": float(np.mean(np.abs(error))) if supported else np.nan,
        "expression_rmse": float(np.sqrt(np.mean(error**2)))
        if supported
        else np.nan,
        "delta_pearson": score.pearson,
        "delta_spearman": score.spearman,
        "error_status": "ok" if supported else "insufficient_shared_genes",
        "correlation_status": score.status,
        "n_finite_genes": int(finite.sum()),
    }


def paired_model_comparisons(metrics: pd.DataFrame) -> pd.DataFrame:
    """Compute improvements only on matched donor/state/fold/gene support."""
    keys = ["task", "fold_id", "donor_id", "state", "n_genes", "scale"]
    rows: list[pd.DataFrame] = []
    for reference in ("adipose-no-change", "adipose-mean-shift"):
        comparison = metrics.merge(
            metrics.loc[metrics["model"].eq(reference)],
            on=keys,
            suffixes=("", "_reference"),
            validate="many_to_one",
        )
        comparison = comparison.loc[~comparison["model"].eq(reference)]
        comparison["comparator"] = reference
        comparison["rmse_improvement"] = (
            comparison["delta_rmse_reference"] - comparison["delta_rmse"]
        )
        comparison["mae_improvement"] = (
            comparison["delta_mae_reference"] - comparison["delta_mae"]
        )
        comparison["comparison_status"] = np.where(
            comparison["status"].eq("ok")
            & comparison["status_reference"].eq("ok"),
            "ok",
            "non_estimable_model",
        )
        rows.append(
            comparison[
                [
                    *keys,
                    "model",
                    "comparator",
                    "rmse_improvement",
                    "mae_improvement",
                    "comparison_status",
                ]
            ]
        )
    return pd.concat(rows, ignore_index=True)


def donor_metric_summary(
    metrics: pd.DataFrame,
    *,
    random_seed: int = 42,
    bootstrap_draws: int = 2000,
    confidence_level: float = 0.95,
    value_columns: tuple[str, ...] = (
        "delta_rmse",
        "delta_mae",
        "delta_pearson",
        "delta_spearman",
    ),
    group_columns: tuple[str, ...] = ("model", "state"),
) -> pd.DataFrame:
    """Summarize fixed predictions using shared donor-cluster bootstrap
    draws.
    """
    if bootstrap_draws < 1 or not 0 < confidence_level < 1:
        raise ValueError("Invalid bootstrap draws or confidence level")
    donors = sorted(metrics["donor_id"].astype(str).unique())
    draws = np.random.default_rng(random_seed).integers(
        0, len(donors), size=(bootstrap_draws, len(donors))
    )
    rows: list[dict[str, object]] = []
    for keys, group in metrics.groupby(
        list(group_columns), observed=True, sort=True
    ):
        if group["donor_id"].duplicated().any():
            raise ValueError(
                "Summary requires one metric per independent donor and group"
            )
        labels: dict[str, object] = dict(
            zip(
                group_columns,
                keys if isinstance(keys, tuple) else (keys,),
                strict=True,
            )
        )
        for column in value_columns:
            values = (
                group.set_index("donor_id")[column]
                .reindex(donors)
                .to_numpy(dtype=float)
            )
            usable = np.isfinite(values)
            samples = values[draws]
            counts = np.isfinite(samples).sum(axis=1)
            estimates = np.divide(
                np.nansum(samples, axis=1),
                counts,
                out=np.full(bootstrap_draws, np.nan),
                where=counts > 0,
            )
            lower = upper = np.nan
            interval_status = "insufficient_donors_for_interval"
            if usable.sum() >= 3:
                interval_status = "degenerate_donor_values"
                if np.ptp(values[usable]) > 0:
                    tail = (1 - confidence_level) / 2
                    lower, upper = np.nanquantile(estimates, [tail, 1 - tail])
                    interval_status = "conditional_donor_bootstrap"
            rows.append(
                labels
                | {
                    "metric": column,
                    "mean": float(values[usable].mean())
                    if usable.any()
                    else np.nan,
                    "n_donors": int(usable.sum()),
                    "n_donors_total": len(group),
                    "lower": lower,
                    "upper": upper,
                    "interval_status": interval_status,
                    "confidence_level": confidence_level,
                }
            )
    return pd.DataFrame(rows)


def write_adipose_benchmark(
    input_path: Path,
    output_dir: Path,
    *,
    settings: BenchmarkSettings,
) -> Path:
    """Read donor pseudobulk, evaluate all folds, and publish report
    artifacts.
    """
    pseudobulk = ad.read_h5ad(input_path)
    tables = evaluate_adipose_transition(pseudobulk, settings=settings)
    destination = prepare_report_directory(output_dir)
    publish_report(
        destination,
        tables=tables,
        provenance={
            "task": "surgery_transition",
            "input": input_identity(input_path),
            "parameters": asdict(settings),
            "profile_source": str(input_path.resolve()),
            "interpretation": (
                "conditional donor-held-out prediction; no gene-edit truth"
            ),
        },
    )
    return destination
