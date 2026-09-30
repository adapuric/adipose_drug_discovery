"""Publish independent adipose validation and model-sensitivity reports."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from add.data import load_rescue_table
from add.perturb import PerturbSignatures
from add.perturb import load_perturb_signatures
from add.rescue import rescue_vector
from add.validation.adipose_benchmark import donor_metric_summary
from add.validation.adipose_benchmark import paired_model_comparisons
from add.validation.adipose_panel import adipose_panel_signatures
from add.validation.adipose_panel import compare_adipose_panel
from add.validation.adipose_panel import map_array_expression
from add.validation.adipose_panel import read_geo_table
from add.validation.artifacts import configured_path
from add.validation.artifacts import input_identity
from add.validation.artifacts import prepare_report_directory
from add.validation.artifacts import publish_report
from add.validation.artifacts import read_report_config
from add.validation.perturbgen_results import classify_perturbgen_run
from add.validation.perturbgen_results import evaluate_transition_export
from add.validation.perturbgen_results import read_legacy_edit_effects
from add.validation.perturbgen_results import read_native_edit_effects
from add.visualization.perturbation import plot_donor_edit_effects
from add.visualization.perturbation import plot_downstream_magnitude
from add.visualization.perturbation import plot_edit_effect_heatmap
from add.visualization.perturbation import plot_support_qc
from add.visualization.perturbation import plot_transition_delta_scatter
from add.visualization.perturbation import plot_transition_errors
from add.visualization.style import set_matplotlib_publication_parameters


logger = logging.getLogger(__name__)


def validate_adipose_panel(
    config_path: Path,
    *,
    output_dir: Path | None = None,
    dry_run: bool = False,
) -> Path:
    """Read the frozen two-study panel and publish separately labeled
    contrasts.
    """
    config, root = read_report_config(
        config_path,
        allowed_keys=(
            "sample_manifest",
            "compound_mapping",
            "studies",
            "external",
            "rescue_path",
            "minimum_shared_genes",
            "output_path",
        ),
    )
    sample_path = configured_path(config, "sample_manifest", root=root)
    samples = pd.read_csv(sample_path, dtype=str)
    studies = config.get("studies")
    if not isinstance(studies, list) or not studies:
        raise ValueError("Panel requires an explicit list of studies")
    destination = output_dir or configured_path(
        config, "output_path", root=root
    )
    frames, qc, inputs = _prepare_panel_studies(
        studies, samples=samples, root=root
    )
    inputs.update(
        sample_manifest=input_identity(sample_path),
        config=input_identity(config_path),
    )
    if dry_run:
        logger.info(
            "Panel validated; study statuses: %s; destination: %s",
            qc,
            destination,
        )
        return destination
    tables = {
        f"{name}.parquet": pd.concat(parts, ignore_index=True)
        if parts
        else pd.DataFrame()
        for name, parts in frames.items()
    }
    external_config = config.get("external", {})
    if not isinstance(external_config, dict):
        raise ValueError("external must map source names to cache prefixes")
    external: dict[str, PerturbSignatures] = {}
    for name, prefix in external_config.items():
        prefix_path = configured_path({"prefix": prefix}, "prefix", root=root)
        external[str(name)] = load_perturb_signatures(prefix_path)
        inputs[f"external:{name}"] = external[str(name)].provenance
    rescues: dict[str, pd.Series] = {}
    if config.get("rescue_path") is not None:
        rescue_path = configured_path(config, "rescue_path", root=root)
        rescue_table = load_rescue_table(rescue_path)
        rescues = {
            str(state): rescue_vector(rescue_table, state=str(state))
            for state in rescue_table["state"].unique()
        }
        inputs["rescue"] = input_identity(rescue_path)
    mapping_path = configured_path(config, "compound_mapping", root=root)
    compound_mapping = pd.read_csv(mapping_path, dtype=str)
    inputs["compound_mapping"] = input_identity(mapping_path)
    panel = tables["study_signatures.parquet"]
    if not panel.empty:
        external_agreement, rescue_alignment = compare_adipose_panel(
            panel,
            external=external,
            compound_mapping=compound_mapping,
            rescue_vectors=rescues,
            minimum_shared_genes=_integer(config, "minimum_shared_genes", 50),
        )
    else:
        external_agreement, rescue_alignment = pd.DataFrame(), pd.DataFrame()
    tables.update(
        {
            "external_agreement.csv": external_agreement,
            "rescue_alignment.csv": rescue_alignment,
            "sample_manifest.csv": samples,
            "qc.csv": pd.DataFrame(qc),
        }
    )
    if not external_agreement.empty and "pearson" in external_agreement:
        eligible = external_agreement.loc[external_agreement["status"].eq("ok")]
        tables["external_agreement_summary.csv"] = (
            eligible.groupby(["study_id", "compound", "source"])[
                ["pearson", "spearman", "directional_agreement"]
            ]
            .median()
            .reset_index()
        )
    destination = prepare_report_directory(destination)
    _panel_figures(
        destination,
        panel,
        external_agreement,
        unit_measurements=tables["unit_measurements.parquet"],
        unit_signatures=tables["unit_signatures.parquet"],
    )
    publish_report(
        destination,
        tables=tables,
        provenance={
            "task": "adipose_compound_validation",
            "inputs": inputs,
            "study_statuses": qc,
            "external_comparison_status": "configured"
            if external
            else "not_configured",
            "rescue_comparison_status": "configured"
            if rescues
            else "not_configured",
            "interpretation": (
                "small external tissue/culture panel; not causal validation"
            ),
        },
    )
    return destination


def perturbgen_report(
    config_path: Path,
    *,
    output_dir: Path | None = None,
    dry_run: bool = False,
) -> Path:
    """Validate existing model outputs, construct edit nulls, and draw
    reports.
    """
    config, root = read_report_config(
        config_path,
        allowed_keys=(
            "run_manifest",
            "output_path",
            "donor_col",
            "state_col",
            "readout_genes",
            "states",
            "top_n",
            "random_seed",
            "bootstrap_draws",
            "confidence_level",
            "benchmark_path",
            "gene_mapping_path",
            "png",
            "figsize",
            "point_size",
            "color_limit",
        ),
    )
    if not isinstance(config.get("png", False), bool):
        raise TypeError("png must be boolean")
    manifest_path = configured_path(config, "run_manifest", root=root)
    runs = json.loads(manifest_path.read_text())
    if not isinstance(runs, list) or not runs:
        raise ValueError("Run manifest must be a nonempty JSON list")
    destination = output_dir or configured_path(
        config, "output_path", root=root
    )
    effects, qc, provenance_rows, native_transitions, identities = (
        _read_perturbgen_runs(runs, config=config, root=root)
    )
    for optional_path in ("benchmark_path", "gene_mapping_path"):
        if config.get(optional_path) is not None:
            resolved_path = configured_path(config, optional_path, root=root)
            if not resolved_path.exists():
                raise FileNotFoundError(
                    f"Configured {optional_path} is missing: {resolved_path}"
                )
    if dry_run:
        logger.info(
            "Validated %d run/edit outputs: %s; destination %s. Planned: "
            "edit heatmap, donor effects, downstream magnitude, support QC; "
            "transition figures when benchmark_path is configured.",
            len(runs),
            provenance_rows,
            destination,
        )
        return destination
    combined = pd.concat(effects, ignore_index=True)
    if config.get("gene_mapping_path") is not None:
        gene_mapping_path = configured_path(
            config, "gene_mapping_path", root=root
        )
        gene_mapping = pd.read_csv(gene_mapping_path, dtype=str)
        if (
            not {"gene_id", "symbol"} <= set(gene_mapping)
            or gene_mapping["gene_id"].duplicated().any()
        ):
            raise ValueError(
                "Gene display mapping requires unique gene_id and symbol"
            )
        combined = combined.merge(
            gene_mapping[["gene_id", "symbol"]],
            on="gene_id",
            how="left",
            validate="many_to_one",
        )
        identities.append(input_identity(gene_mapping_path))
    support = pd.concat(qc, ignore_index=True)
    tables: dict[str, pd.DataFrame] = {
        "edit_effects.parquet": combined,
        "support_qc.csv": support,
        "run_classification.csv": pd.DataFrame(provenance_rows),
    }
    destination = prepare_report_directory(destination)
    figure_manifest = _perturbation_figures(
        destination, combined, support, config=config
    )
    transition_tables, transition_manifest = _transition_figures(
        destination,
        config=config,
        root=root,
        native_transitions=native_transitions,
    )
    tables.update(transition_tables)
    figure_manifest.extend(transition_manifest)
    (destination / "figure_manifest.json").write_text(
        json.dumps(figure_manifest, indent=2) + "\n"
    )
    publish_report(
        destination,
        tables=tables,
        provenance={
            "task": "model_sensitivity",
            "config": input_identity(config_path),
            "run_manifest": input_identity(manifest_path),
            "inputs": identities,
            "runs": runs,
            "interpretation": (
                "predicted edit sensitivity; no experimental edited outcomes"
            ),
        },
    )
    return destination


def _read_perturbgen_runs(
    runs: list[object],
    *,
    config: Mapping[str, object],
    root: Path,
) -> tuple[
    list[pd.DataFrame],
    list[pd.DataFrame],
    list[dict[str, object]],
    list[tuple[pd.DataFrame, Mapping[str, object]]],
    list[dict[str, object]],
]:
    """Read outputs with provenance classes and input identities."""
    effects: list[pd.DataFrame] = []
    qc: list[pd.DataFrame] = []
    provenance_rows: list[dict[str, object]] = []
    native_transitions: list[tuple[pd.DataFrame, Mapping[str, object]]] = []
    identities: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    for run in runs:
        if not isinstance(run, dict):
            raise TypeError("Each run must be a mapping")
        run_id, edit_id = str(run["run_id"]), str(run["edit_id"])
        if (run_id, edit_id) in seen:
            raise ValueError("Duplicate run/edit identifiers in manifest")
        seen.add((run_id, edit_id))
        genes = run.get("edited_genes")
        if (
            not isinstance(genes, list)
            or not genes
            or not all(isinstance(gene, str) for gene in genes)
        ):
            raise ValueError("edited_genes must be explicit canonical IDs")
        arguments = {
            "run_id": run_id,
            "edit_id": edit_id,
            "edited_genes": genes,
            "donor_col": str(config.get("donor_col", "Donor")),
            "state_col": str(config.get("state_col", "cell_states_adipocytes")),
        }
        if run.get("result_path") is not None:
            source = configured_path(run, "result_path", root=root)
            effect, support = read_native_edit_effects(source, **arguments)
        else:
            source = configured_path(run, "summary_path", root=root)
            effect, support = read_legacy_edit_effects(source, **arguments)
        donors = effect["donor_id"].astype(str).unique().tolist()
        classification = classify_perturbgen_run(run, test_donors=donors)
        provenance_rows.append(
            {
                "run_id": run_id,
                "edit_id": edit_id,
                "comparison_class": classification,
            }
        )
        effects.append(effect)
        qc.append(support)
        identities.append(input_identity(source))
        if run.get("transition_predictions_path") is not None:
            native_path = configured_path(
                run, "transition_predictions_path", root=root
            )
            native_transitions.append((pd.read_parquet(native_path), run))
            identities.append(input_identity(native_path))
    return effects, qc, provenance_rows, native_transitions, identities


def _transition_figures(
    destination: Path,
    *,
    config: Mapping[str, object],
    root: Path,
    native_transitions: list[tuple[pd.DataFrame, Mapping[str, object]]],
) -> tuple[dict[str, pd.DataFrame], list[dict[str, object]]]:
    """Admit compatible exports and report matched surgery predictions."""
    tables: dict[str, pd.DataFrame] = {}
    figure_manifest: list[dict[str, object]] = []
    if config.get("benchmark_path") is not None:
        benchmark = configured_path(config, "benchmark_path", root=root)
        metrics = pd.read_csv(
            benchmark / "metrics.csv", dtype={"donor_id": str}
        )
        reference = pd.read_parquet(benchmark / "predictions.parquet")
        provenance = json.loads((benchmark / "provenance.json").read_text())
        if native_transitions and (
            provenance["parameters"]["preprocessing_policy"]
            == "transductive_hvg"
        ):
            raise ValueError(
                "Strict native comparisons require training-only or fixed "
                "external preprocessing in the reference benchmark"
            )
        admitted: list[pd.DataFrame] = [metrics]
        for predictions, run in native_transitions:
            if run.get("profile_identity") != provenance["input"]["sha256"]:
                raise ValueError(
                    "Native transition export uses a different pseudobulk input"
                )
            admitted.append(
                evaluate_transition_export(predictions, reference, manifest=run)
            )
        metrics = pd.concat(admitted, ignore_index=True)
        tables["transition_metrics.csv"] = metrics
        tables["paired_model_comparisons.csv"] = paired_model_comparisons(
            metrics
        )
        tables["paired_comparison_summary.csv"] = donor_metric_summary(
            tables["paired_model_comparisons.csv"],
            value_columns=("rmse_improvement", "mae_improvement"),
            group_columns=("model", "comparator", "state"),
            random_seed=_integer(config, "random_seed", 42, minimum=0),
            bootstrap_draws=_integer(config, "bootstrap_draws", 2000),
            confidence_level=_number(config, "confidence_level", 0.95),
        )
        figure, _ = plot_transition_errors(metrics)
        _save_figure(
            figure,
            destination / "transition_model_comparison.pdf",
            png=bool(config.get("png", False)),
        )
        figure_manifest.append(
            {
                "figure": "transition_model_comparison.pdf",
                "data": "transition_metrics.csv",
                "meaning": "donor-held-out surgery prediction",
            }
        )
        eligible = reference.loc[
            reference["status"].eq("ok")
            & reference["model"].eq("adipose-ridge")
        ]
        if not eligible.empty:
            state = sorted(eligible["state"].unique())[0]
            selected = eligible.loc[eligible["state"].eq(state)]
            tables["transition_scatter.parquet"] = selected
            figure, _ = plot_transition_delta_scatter(selected)
            _save_figure(
                figure,
                destination / "transition_delta_scatter.pdf",
                png=bool(config.get("png", False)),
            )
            figure_manifest.append(
                {
                    "figure": "transition_delta_scatter.pdf",
                    "data": "transition_scatter.parquet",
                    "selection": f"adipose-ridge; first sorted state {state}",
                }
            )
    elif native_transitions:
        raise ValueError(
            "Native transition exports require a configured reference benchmark"
        )
    else:
        figure_manifest.append(
            {
                "figure": "transition_model_comparison.pdf",
                "status": "skipped",
                "reason": "benchmark_not_configured",
            }
        )
    return tables, figure_manifest


def _prepare_panel_studies(
    studies: list[object],
    *,
    samples: pd.DataFrame,
    root: Path,
) -> tuple[
    dict[str, list[pd.DataFrame]], list[dict[str, object]], dict[str, object]
]:
    """Transform verified array studies and retain explicit blocked reasons."""
    frames: dict[str, list[pd.DataFrame]] = {
        name: []
        for name in (
            "study_signatures",
            "unit_signatures",
            "unit_measurements",
            "gene_mapping_audit",
        )
    }
    inputs: dict[str, object] = {}
    qc: list[dict[str, object]] = []
    for study in studies:
        if not isinstance(study, dict):
            raise TypeError("Each study must be a mapping")
        allowed = {
            "study_id",
            "matrix_path",
            "platform_path",
            "probe_col",
            "gene_col",
            "input_scale",
            "pairing",
            "blocked_reason",
            "source_url",
            "processing_evidence",
        }
        if set(study) - allowed:
            raise ValueError(
                f"Unknown study fields: {sorted(set(study) - allowed)}"
            )
        study_id = str(study["study_id"])
        expression_path = configured_path(study, "matrix_path", root=root)
        platform_path = configured_path(study, "platform_path", root=root)
        selected_samples = samples.loc[samples["study_id"].eq(study_id)]
        if selected_samples.empty:
            raise ValueError(f"No curated samples for {study_id}")
        expression = read_geo_table(
            expression_path, table_kind="series_matrix"
        ).set_index("ID_REF")
        annotation = read_geo_table(platform_path, table_kind="platform")
        inputs[study_id] = {
            "expression": input_identity(expression_path),
            "platform": input_identity(platform_path),
            "protocol": study,
        }
        if not set(selected_samples["sample_id"]) <= set(expression.columns):
            raise ValueError(f"Manifest and matrix disagree for {study_id}")
        if study.get("blocked_reason"):
            qc.append(
                {
                    "study_id": study_id,
                    "status": "blocked",
                    "reason": study["blocked_reason"],
                    "n_samples": len(selected_samples),
                }
            )
            continue
        mapping = annotation[
            [str(study["probe_col"]), str(study["gene_col"])]
        ].copy()
        mapping.columns = ["probe_id", "gene_id"]
        mapping["gene_id"] = (
            mapping["gene_id"].astype("string").str.split(r"\s*(?:///|;|\|)\s*")
        )
        mapping = mapping.explode("gene_id").dropna()
        mapping = mapping.loc[mapping["gene_id"].astype(str).str.strip().ne("")]
        transformed, audit = map_array_expression(
            expression, mapping, input_scale=str(study["input_scale"])
        )
        audit["study_id"] = study_id
        frames["gene_mapping_audit"].append(audit)
        result = adipose_panel_signatures(
            transformed, selected_samples, pairing=str(study["pairing"])
        )
        for name, table in result.items():
            frames[name].append(table)
        qc.append(
            {
                "study_id": study_id,
                "status": "ok",
                "reason": "",
                "n_samples": len(selected_samples),
                "n_genes": len(transformed),
                "pairing": study["pairing"],
                "scale": study["input_scale"],
            }
        )
    return frames, qc, inputs


def _perturbation_figures(
    destination: Path,
    effects: pd.DataFrame,
    qc: pd.DataFrame,
    *,
    config: Mapping[str, object],
) -> list[dict[str, object]]:
    """Keep run/scale reports distinct and write the plotted donor tables."""
    set_matplotlib_publication_parameters()
    manifest: list[dict[str, object]] = []
    top_n = _integer(config, "top_n", 20)
    requested = config.get("readout_genes", [])
    states = config.get("states", [])
    if not isinstance(requested, list) or not isinstance(states, list):
        raise ValueError("readout_genes and states must be lists")
    for ordinal, (run_id, run) in enumerate(
        effects.groupby("run_id", sort=True)
    ):
        directory = destination / f"run_{ordinal:03d}"
        directory.mkdir()
        model = run.loc[
            ~run["model"].eq("no-downstream-effect") & ~run["is_edited_gene"]
        ]
        preferred = (
            "shared_unedited_log1p_cpm"
            if model["scale"].eq("shared_unedited_log1p_cpm").any()
            else "log2_ratio_of_nucleus_means_plus1"
        )
        model = model.loc[model["scale"].eq(preferred)]
        if states:
            model = model.loc[model["state"].isin(states)]
        if requested:
            genes = requested
        else:
            genes = (
                model.assign(absolute=model["delta"].abs())
                .groupby("gene_id")["absolute"]
                .mean()
                .sort_values(ascending=False, kind="stable")
                .head(top_n)
                .index.tolist()
            )
        selected = model.loc[model["gene_id"].isin(genes)].copy()
        selected.to_csv(directory / "selected_effects.csv", index=False)
        if not selected.empty:
            summary = donor_metric_summary(
                selected,
                value_columns=("delta",),
                group_columns=("edit_id", "state", "gene_id"),
                random_seed=_integer(config, "random_seed", 42, minimum=0),
                bootstrap_draws=_integer(config, "bootstrap_draws", 2000),
                confidence_level=_number(config, "confidence_level", 0.95),
            )
            summary.to_csv(directory / "donor_effect_summary.csv", index=False)
            size = config.get("figsize")
            figsize = None
            if size is not None:
                if not isinstance(size, list) or len(size) != 2:
                    raise ValueError(
                        "figsize must contain width and height in inches"
                    )
                figsize = (
                    _number({"width": size[0]}, "width", 8),
                    _number({"height": size[1]}, "height", 5),
                )
            color_limit = (
                None
                if config.get("color_limit") is None
                else _number(config, "color_limit", 1)
            )
            heatmap, _ = plot_edit_effect_heatmap(
                selected, figsize=figsize, color_limit=color_limit
            )
            donor_plot, _ = plot_donor_edit_effects(
                selected,
                figsize=figsize,
                point_size=_number(config, "point_size", 18),
                intervals=summary,
            )
            for name, figure in (
                ("edit_effect_heatmap", heatmap),
                ("donor_edit_effects", donor_plot),
            ):
                _save_figure(
                    figure,
                    directory / f"{name}.pdf",
                    png=bool(config.get("png", False)),
                )
                manifest.append(
                    {
                        "figure": f"{directory.name}/{name}.pdf",
                        "run_id": run_id,
                        "data": f"{directory.name}/selected_effects.csv",
                        "scale": preferred,
                        "selection": "prespecified"
                        if requested
                        else "exploratory mean absolute effect",
                        "interval_data": (
                            f"{directory.name}/donor_effect_summary.csv"
                        ),
                    }
                )
        downstream = run.loc[
            ~run["is_edited_gene"] & run["scale"].eq(preferred)
        ]
        if states:
            downstream = downstream.loc[downstream["state"].isin(states)]
        magnitudes = (
            downstream.assign(absolute=downstream["delta"].abs())
            .groupby(["edit_id", "state", "model", "donor_id"], observed=True)[
                "absolute"
            ]
            .mean()
            .rename("mean_absolute_delta")
            .reset_index()
        )
        magnitudes["scale"] = preferred
        magnitudes.to_csv(directory / "downstream_magnitude.csv", index=False)
        if not magnitudes.empty:
            figure, _ = plot_downstream_magnitude(magnitudes)
            _save_figure(
                figure,
                directory / "downstream_effect_magnitude.pdf",
                png=bool(config.get("png", False)),
            )
            manifest.append(
                {
                    "figure": (
                        f"{directory.name}/downstream_effect_magnitude.pdf"
                    ),
                    "data": f"{directory.name}/downstream_magnitude.csv",
                    "scale": preferred,
                    "meaning": "model sensitivity, not edit accuracy",
                }
            )
        support = qc.loc[qc["run_id"].eq(run_id)]
        support.to_csv(directory / "support_qc.csv", index=False)
        figure, _ = plot_support_qc(support)
        _save_figure(
            figure,
            directory / "support_qc.pdf",
            png=bool(config.get("png", False)),
        )
        manifest.append(
            {
                "figure": f"{directory.name}/support_qc.pdf",
                "data": f"{directory.name}/support_qc.csv",
            }
        )
    return manifest


def _panel_figures(
    destination: Path,
    panel: pd.DataFrame,
    agreement: pd.DataFrame,
    *,
    unit_measurements: pd.DataFrame,
    unit_signatures: pd.DataFrame,
) -> None:
    """Draw compact panel support and exact-compound agreement summaries."""
    set_matplotlib_publication_parameters()
    if panel.empty:
        return
    if not agreement.empty and "pearson" in agreement:
        summary = (
            agreement.groupby(["study_id", "compound", "source"])["pearson"]
            .median()
            .unstack("source")
        )
        figure, axis = plt.subplots(figsize=(6, max(3, len(summary) * 0.5)))
        artist = axis.imshow(
            np.ma.masked_invalid(summary.to_numpy()),
            vmin=-1,
            vmax=1,
            cmap="RdBu_r",
            aspect="auto",
        )
        axis.set_xticks(np.arange(len(summary.columns)), summary.columns)
        axis.set_yticks(
            np.arange(len(summary)), [" | ".join(row) for row in summary.index]
        )
        axis.set_title(
            "Exact-compound agreement: median across measured contexts"
        )
        figure.colorbar(
            artist, ax=axis, label="Pearson correlation on shared genes"
        )
        _save_figure(figure, destination / "external_agreement.pdf", png=False)
        summary.reset_index().to_csv(
            destination / "external_agreement_plot.csv", index=False
        )
    for number, ((study, compound), contrast) in enumerate(
        panel.groupby(["study_id", "compound"], sort=True)
    ):
        study, compound = str(study), str(compound)
        genes = (
            contrast.assign(absolute=contrast["delta"].abs())
            .sort_values(
                ["absolute", "gene_id"],
                ascending=[False, True],
                kind="stable",
            )
            .head(8)["gene_id"]
            .tolist()
        )
        paired = contrast["pairing"].iloc[0] == "paired"
        if paired:
            selected = unit_signatures.loc[
                unit_signatures["study_id"].eq(study)
                & unit_signatures["compound"].eq(compound)
                & unit_signatures["gene_id"].isin(genes)
            ].copy()
            selected["condition"] = "treated - vehicle"
            selected["value"] = selected["delta"]
        else:
            selected = unit_measurements.loc[
                unit_measurements["study_id"].eq(study)
                & unit_measurements["compound"].isin([compound, "vehicle"])
                & unit_measurements["gene_id"].isin(genes)
            ].copy()
            selected["value"] = selected["expression"]
        figure, axis = plt.subplots(figsize=(7, 4))
        for condition_number, (condition, group) in enumerate(
            selected.groupby("condition", sort=True)
        ):
            positions = group["gene_id"].map(
                {gene: i for i, gene in enumerate(genes)}
            )
            axis.scatter(
                positions + (condition_number - 0.5) * 0.15,
                group["value"],
                s=18,
                alpha=0.65,
                label=condition,
            )
        axis.set_xticks(np.arange(len(genes)), genes, rotation=50, ha="right")
        unit = str(contrast["unit_type"].iloc[0])
        axis.set_title(
            f"{study} | {compound}\nExploratory top effects; points = {unit}"
        )
        axis.set_ylabel(
            "Paired log2 response"
            if paired
            else "Normalized log2 sample/reference ratio"
        )
        axis.legend()
        filename = f"panel_units_{number:02d}"
        selected.to_csv(destination / f"{filename}.csv", index=False)
        _save_figure(figure, destination / f"{filename}.pdf", png=False)


def _integer(
    values: Mapping[str, object],
    key: str,
    default: int,
    *,
    minimum: int = 1,
) -> int:
    """Validate a numeric report option without silently truncating it."""
    value = values.get(key, default)
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{key} must be an integer >= {minimum}")
    return value


def _number(values: Mapping[str, object], key: str, default: float) -> float:
    """Read a finite positive report parameter in its declared units."""
    value = values.get(key, default)
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not np.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{key} must be finite and positive")
    return float(value)


def _save_figure(figure: Figure, path: Path, *, png: bool) -> None:
    """Export at the workflow boundary and release the rendered figure."""
    figure.savefig(path, bbox_inches="tight")
    if png:
        figure.savefig(path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(figure)
