"""Publish independent adipose validation and model-sensitivity reports."""

from __future__ import annotations

import dataclasses
import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from add.baselines import mean_drug_signatures
from add.baselines.edits import coexpression_edit_effects
from add.baselines.edits import edit_effect_agreement
from add.baselines.edits import generic_edit_effects
from add.data import file_identity
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
from add.validation.rescue_programs import decompose_program_mimicry
from add.validation.rescue_programs import donor_program_scores
from add.validation.rescue_programs import paired_program_changes
from add.validation.rescue_programs import program_change_correlations
from add.validation.rescue_programs import program_genes
from add.validation.rescue_programs import program_ranking_agreement
from add.validation.rescue_programs import read_gene_sets
from add.validation.rescue_programs import summarize_program_changes
from add.validation.rescue_programs import summarize_rescue_programs
from add.visualization.perturbation import plot_donor_edit_effects
from add.visualization.perturbation import plot_downstream_magnitude
from add.visualization.perturbation import plot_edit_effect_heatmap
from add.visualization.perturbation import plot_support_qc
from add.visualization.perturbation import plot_transition_delta_scatter
from add.visualization.perturbation import plot_transition_errors
from add.visualization.programs import plot_donor_program_scores
from add.visualization.programs import plot_program_matrix
from add.visualization.programs import plot_score_contributions
from add.visualization.style import set_matplotlib_publication_parameters


logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True, kw_only=True)
class _NaiveBaselineSettings:
    """Validated thresholds shared by the naive edit predictors."""

    condition_col: str
    minimum_nuclei: int
    minimum_reference_cpm: float
    minimum_shared_genes: int
    minimum_reference_edits: int
    chunk_size: int


@dataclasses.dataclass(frozen=True, kw_only=True)
class _SignatureSource:
    """A cached external signature source and how its rows form drugs."""

    prefix: Path
    drug_col: str
    context_cols: tuple[str, ...]


@dataclasses.dataclass(frozen=True, kw_only=True)
class _RescueProgramSettings:
    """Validated inputs and options for the rescue-program report."""

    gene_sets_path: Path
    rescue_path: Path
    pseudobulk_path: Path
    panel_signatures_path: Path | None
    programs: dict[str, tuple[str, ...]]
    excluded_program: str
    display_state: str
    top_drugs: int
    minimum_shared_genes: int
    minimum_program_genes: int
    donor_col: str
    condition_col: str
    group_col: str | None
    covariate_cols: tuple[str, ...]
    baseline_label: str
    weightloss_label: str
    condition_order: tuple[str, ...]
    sources: dict[str, _SignatureSource]
    positive_controls: tuple[tuple[str, str], ...]


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
            "naive_baselines",
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
    naive_settings = _naive_baseline_settings(config)
    naive_requests = (
        None
        if naive_settings is None
        else _naive_edit_requests(runs, root=root)
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
        if naive_requests is not None:
            logger.info(
                "Naive edit baselines: generic-edit mean for %d edits; "
                "co-expression contrast for %d edits.",
                len(naive_requests),
                naive_requests["coexpression_path"].notna().sum(),
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
    if naive_settings is not None and naive_requests is not None:
        tables.update(
            _naive_edit_baselines(
                combined,
                naive_requests,
                settings=naive_settings,
                config=config,
            )
        )
        identities.extend(
            input_identity(path)
            for path in naive_requests["coexpression_path"].dropna().unique()
        )
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
            "naive_baselines": None
            if naive_settings is None
            else dataclasses.asdict(naive_settings),
            "interpretation": (
                "predicted edit sensitivity; no experimental edited outcomes"
            ),
        },
    )
    return destination


def decompose_rescue_programs(
    config_path: Path,
    *,
    output_dir: Path | None = None,
    dry_run: bool = False,
) -> Path:
    """Split the rescue signature into gene-set programs and score drugs.

    Publishes how every gene set moves in each state's rescue vector,
    donor-level program scores and their paired changes, drug-level
    decompositions of rescue mimicry for each configured signature source
    and the configured positive controls. Program membership comes from the
    gene-set file, never from this data. Descriptive and hypothesis-
    generating: it cannot tell sampling effects from biology on its own.
    """
    config, root = read_report_config(
        config_path,
        allowed_keys=(
            "gene_sets_path",
            "gene_sets_source",
            "rescue_path",
            "pseudobulk_path",
            "panel_signatures_path",
            "output_path",
            "programs",
            "excluded_program",
            "display_state",
            "top_drugs",
            "minimum_shared_genes",
            "minimum_program_genes",
            "donors",
            "signatures",
            "positive_controls",
        ),
    )
    settings = _rescue_program_settings(config, root=root)
    gene_sets = read_gene_sets(settings.gene_sets_path)
    programs = program_genes(gene_sets, settings.programs)
    rescue_table = load_rescue_table(settings.rescue_path)
    rescue_vectors = {
        state: rescue_vector(rescue_table, state=state)
        for state in sorted(rescue_table["state"].astype(str).unique())
    }
    if settings.display_state not in rescue_vectors:
        raise ValueError(
            f"display_state {settings.display_state!r} has no rescue vector"
        )
    destination = output_dir or configured_path(
        config, "output_path", root=root
    )
    if dry_run:
        logger.info(
            "Rescue programs validated: %s; states %s; sources %s; "
            "destination %s",
            {name: len(genes) for name, genes in programs.items()},
            sorted(rescue_vectors),
            sorted(settings.sources),
            destination,
        )
        return destination

    tables = _donor_program_tables(settings, programs)
    tables["hallmark_rescue_summary.csv"] = summarize_rescue_programs(
        rescue_table,
        gene_sets,
        minimum_genes=settings.minimum_program_genes,
    )
    tables["program_genes.csv"] = pd.DataFrame(
        [
            {"program": program, "gene": gene}
            for program, genes in programs.items()
            for gene in sorted(genes)
        ]
    )
    decomposition, source_inputs = _drug_program_decompositions(
        settings, programs, rescue_vectors
    )
    tables["drug_program_decomposition.parquet"] = decomposition
    scores = _wide_program_scores(decomposition)
    tables["drug_program_scores.csv"] = scores
    tables["program_ranking_agreement.csv"] = pd.concat(
        [
            program_ranking_agreement(
                decomposition.loc[decomposition["source"].eq(source)],
                top_n=settings.top_drugs,
            ).assign(source=source)
            for source in settings.sources
        ],
        ignore_index=True,
    )
    tables["positive_controls.csv"] = _positive_control_scores(
        scores, controls=settings.positive_controls
    )

    destination = prepare_report_directory(destination)
    _rescue_program_figures(
        destination,
        tables=tables,
        settings=settings,
        programs=programs,
    )
    publish_report(
        destination,
        tables=tables,
        provenance={
            "task": "rescue_program_decomposition",
            "inputs": {
                "config": input_identity(config_path),
                "gene_sets": input_identity(settings.gene_sets_path)
                | {"source_url": config.get("gene_sets_source")},
                "rescue": input_identity(settings.rescue_path),
                "pseudobulk": input_identity(settings.pseudobulk_path),
                "panel_signatures": input_identity(
                    settings.panel_signatures_path
                )
                if settings.panel_signatures_path is not None
                else None,
                "signatures": source_inputs,
            },
            "parameters": dataclasses.asdict(settings)
            | {
                "program_sizes": {
                    name: len(genes) for name, genes in programs.items()
                },
                "drug_signature": "equal_context_mean_by_drug",
            },
            "interpretation": (
                "descriptive program decomposition; curated gene sets; "
                "cannot separate sampling effects from biology without "
                "sample-collection metadata"
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


def _naive_baseline_settings(
    config: Mapping[str, object],
) -> _NaiveBaselineSettings | None:
    """Validate optional naive-baseline thresholds."""
    values = config.get("naive_baselines")
    if values is None:
        return None
    if not isinstance(values, dict):
        raise TypeError("naive_baselines must be a mapping or null")
    fields = {
        field.name for field in dataclasses.fields(_NaiveBaselineSettings)
    }
    if unknown := set(values) - fields:
        raise ValueError(f"Unknown naive_baselines keys: {sorted(unknown)}")
    condition_col = values.get("condition_col", "condition")
    if not isinstance(condition_col, str) or not condition_col:
        raise ValueError("naive_baselines.condition_col must be a column name")
    return _NaiveBaselineSettings(
        condition_col=condition_col,
        minimum_nuclei=_integer(values, "minimum_nuclei", 20),
        minimum_reference_cpm=_number(values, "minimum_reference_cpm", 1.0),
        minimum_shared_genes=_integer(
            values, "minimum_shared_genes", 50, minimum=2
        ),
        minimum_reference_edits=_integer(values, "minimum_reference_edits", 2),
        chunk_size=_integer(values, "chunk_size", 4096),
    )


def _naive_edit_requests(runs: list[object], *, root: Path) -> pd.DataFrame:
    """Validate the edit metadata each naive predictor needs."""
    rows: list[dict[str, object]] = []
    for run in runs:
        if not isinstance(run, dict):
            raise TypeError("Each run must be a mapping")
        label = f"{run['run_id']}/{run['edit_id']}"
        if run.get("edit_mode") not in {"mask", "pad", "delete", "overexpress"}:
            raise ValueError(f"{label}: naive baselines require edit_mode")
        if run.get("edit_sequence") not in {"src", "tgt"}:
            raise ValueError(f"{label}: naive baselines require edit_sequence")
        path = condition = None
        if run.get("coexpression_h5ad_path") is not None:
            path = configured_path(run, "coexpression_h5ad_path", root=root)
            condition = run.get("edited_condition")
            if not path.is_file():
                raise FileNotFoundError(f"{label}: missing {path}")
            if not isinstance(condition, str) or not condition:
                raise ValueError(
                    f"{label}: coexpression_h5ad_path requires the "
                    "edited_condition label of the edited sequence"
                )
        rows.append(
            {
                "run_id": str(run["run_id"]),
                "edit_id": str(run["edit_id"]),
                "edit_mode": run["edit_mode"],
                "edit_sequence": run["edit_sequence"],
                "edited_genes": tuple(run["edited_genes"]),
                "coexpression_path": path,
                "edited_condition": condition,
            }
        )
    return pd.DataFrame(rows)


def _naive_edit_baselines(
    effects: pd.DataFrame,
    requests: pd.DataFrame,
    *,
    settings: _NaiveBaselineSettings,
    config: Mapping[str, object],
) -> dict[str, pd.DataFrame]:
    """Compare model edits with co-expression and generic-edit predictions."""
    model = effects.loc[
        effects["model"].str.startswith("perturbgen:")
        & effects["scale"].isin(
            ("shared_unedited_log1p_cpm", "log2_ratio_of_nucleus_means_plus1")
        )
    ].merge(
        requests[["run_id", "edit_id", "edit_mode", "edit_sequence"]],
        on=["run_id", "edit_id"],
        validate="many_to_one",
    )
    reference = pd.Series(np.nan, index=model.index)
    for column in ("unedited_prediction_count", "mean_unperturbed_count"):
        if column in model:
            reference = reference.fillna(model[column])
    model["reference_count"] = reference
    contrast = _coexpression_contrasts(
        requests, settings=settings, config=config
    )

    predictions = [contrast]
    agreements: list[pd.DataFrame] = []
    for keys, group in model.groupby(
        ["run_id", "edit_mode", "edit_sequence", "scale"], sort=True
    ):
        run_id, _, _, scale = (str(key) for key in keys)
        generic = generic_edit_effects(
            group, minimum_reference_edits=settings.minimum_reference_edits
        ).assign(run_id=run_id, scale=scale)
        predictions.append(generic)
        for comparator, naive in (
            ("generic-edit-mean", generic),
            (
                "coexpression-contrast",
                contrast.loc[contrast["run_id"].eq(run_id)],
            ),
        ):
            agreements.append(
                edit_effect_agreement(
                    group,
                    naive,
                    minimum_nuclei=settings.minimum_nuclei,
                    minimum_reference_cpm=settings.minimum_reference_cpm,
                    minimum_shared_genes=settings.minimum_shared_genes,
                ).assign(run_id=run_id, comparator=comparator, scale=scale)
            )

    # Report why an edit has no contrast instead of a generic missing status.
    agreement = pd.concat(agreements, ignore_index=True)
    reasons = {
        (row.run_id, row.edit_id): "unsupported_edit_mode"
        if row.edit_mode == "overexpress"
        else "not_configured"
        for row in requests.itertuples()
        if row.edit_mode == "overexpress" or pd.isna(row.coexpression_path)
    }
    reason = pd.Series(
        [
            reasons.get(key)
            for key in zip(
                agreement["run_id"], agreement["edit_id"], strict=True
            )
        ],
        index=agreement.index,
        dtype=object,
    ).where(agreement["comparator"].eq("coexpression-contrast"))
    agreement["status"] = reason.fillna(agreement["status"])
    agreement["comparator_status"] = reason.fillna(
        agreement["comparator_status"]
    )

    summary = donor_metric_summary(
        agreement,
        value_columns=("agreement_pearson", "agreement_spearman"),
        group_columns=("run_id", "comparator", "scale", "edit_id", "state"),
        status_columns={
            "agreement_pearson": "status",
            "agreement_spearman": "status",
        },
        random_seed=_integer(config, "random_seed", 42, minimum=0),
        bootstrap_draws=_integer(config, "bootstrap_draws", 2000),
        confidence_level=_number(config, "confidence_level", 0.95),
    )
    return {
        "naive_edit_baselines.parquet": pd.concat(
            predictions, ignore_index=True
        ),
        "edit_baseline_agreement.csv": agreement,
        "edit_baseline_agreement_summary.csv": summary,
    }


def _coexpression_contrasts(
    requests: pd.DataFrame,
    *,
    settings: _NaiveBaselineSettings,
    config: Mapping[str, object],
) -> pd.DataFrame:
    """Read each configured expression input once per run and condition."""
    contrasts: list[pd.DataFrame] = []
    knockouts = requests.loc[
        requests["coexpression_path"].notna()
        & requests["edit_mode"].ne("overexpress")
    ]
    for (run_id, path, condition), edits in knockouts.groupby(
        ["run_id", "coexpression_path", "edited_condition"], sort=True
    ):
        contrast = coexpression_edit_effects(
            Path(str(path)),
            edited_genes=dict(
                zip(edits["edit_id"], edits["edited_genes"], strict=True)
            ),
            condition=str(condition),
            donor_col=str(config.get("donor_col", "Donor")),
            state_col=str(config.get("state_col", "cell_states_adipocytes")),
            condition_col=settings.condition_col,
            minimum_nuclei=settings.minimum_nuclei,
            chunk_size=settings.chunk_size,
        )
        contrasts.append(contrast.assign(run_id=str(run_id)))
    if not contrasts:
        return pd.DataFrame(
            columns=[
                "run_id",
                "edit_id",
                "donor_id",
                "state",
                "gene_id",
                "delta",
                "status",
            ]
        )
    return pd.concat(contrasts, ignore_index=True)


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
            status_columns={
                "rmse_improvement": "comparison_status",
                "mae_improvement": "comparison_status",
            },
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


def _rescue_program_settings(
    config: Mapping[str, object],
    *,
    root: Path,
) -> _RescueProgramSettings:
    """Validate the rescue-program report options without loading data."""
    programs = config.get("programs")
    if not isinstance(programs, dict) or not programs:
        raise ValueError("programs must map program names to gene-set lists")
    program_sets = {str(name): _texts(programs, name) for name in programs}
    excluded = _text(config, "excluded_program")
    if excluded not in program_sets:
        raise ValueError(
            f"excluded_program {excluded!r} must name one of "
            f"{sorted(program_sets)}"
        )

    donors = config.get("donors")
    if not isinstance(donors, dict):
        raise ValueError("donors must map the donor and condition settings")

    signatures = config.get("signatures", {})
    if not isinstance(signatures, dict):
        raise ValueError("signatures must map source names to options")
    sources: dict[str, _SignatureSource] = {}
    for name, options in signatures.items():
        if not isinstance(options, dict):
            raise ValueError(f"Signature source {name!r} needs options")
        prefix = configured_path(options, "prefix", root=root)
        if not Path(f"{prefix}.npz").is_file():
            raise FileNotFoundError(f"Signature cache not found: {prefix}.npz")
        sources[str(name)] = _SignatureSource(
            prefix=prefix,
            drug_col=_text(options, "drug_col"),
            context_cols=_texts(options, "context_cols"),
        )

    controls = config.get("positive_controls", [])
    if not isinstance(controls, list) or not all(
        isinstance(item, dict) for item in controls
    ):
        raise ValueError("positive_controls must list source and drug pairs")

    panel_path = (
        configured_path(config, "panel_signatures_path", root=root)
        if config.get("panel_signatures_path") is not None
        else None
    )
    return _RescueProgramSettings(
        gene_sets_path=configured_path(config, "gene_sets_path", root=root),
        rescue_path=configured_path(config, "rescue_path", root=root),
        pseudobulk_path=configured_path(config, "pseudobulk_path", root=root),
        panel_signatures_path=panel_path,
        programs=program_sets,
        excluded_program=excluded,
        display_state=_text(config, "display_state", default="AD_ALL"),
        top_drugs=_integer(config, "top_drugs", 15),
        minimum_shared_genes=_integer(
            config, "minimum_shared_genes", 10, minimum=3
        ),
        minimum_program_genes=_integer(config, "minimum_program_genes", 5),
        donor_col=_text(donors, "donor_col"),
        condition_col=_text(donors, "condition_col"),
        group_col=_text(donors, "group_col")
        if donors.get("group_col") is not None
        else None,
        covariate_cols=_texts(donors, "covariate_cols", default=()),
        baseline_label=_text(donors, "baseline_label"),
        weightloss_label=_text(donors, "weightloss_label"),
        condition_order=_texts(donors, "condition_order"),
        sources=sources,
        positive_controls=tuple(
            (_text(item, "source"), _text(item, "drug")) for item in controls
        ),
    )


def _donor_program_tables(
    settings: _RescueProgramSettings,
    programs: Mapping[str, frozenset[str]],
) -> dict[str, pd.DataFrame]:
    """Score programs per donor and condition, then summarize paired changes.

    Conditions outside the paired contrast, such as Lean, are scored for
    reference only and never enter the paired changes.
    """
    group = [settings.group_col] if settings.group_col is not None else []
    scores = donor_program_scores(
        ad.read_h5ad(settings.pseudobulk_path),
        programs,
        donor_col=settings.donor_col,
        condition_col=settings.condition_col,
        carried_cols=[*group, *settings.covariate_cols],
        minimum_genes=settings.minimum_program_genes,
    )
    changes = paired_program_changes(
        scores,
        baseline_label=settings.baseline_label,
        weightloss_label=settings.weightloss_label,
        covariate_cols=settings.covariate_cols,
    )
    return {
        "donor_program_scores.csv": scores,
        "donor_program_changes.csv": changes,
        "program_change_summary.csv": summarize_program_changes(
            changes, group_col=settings.group_col
        ),
        "program_change_correlations.csv": program_change_correlations(
            changes, covariate_cols=settings.covariate_cols
        ),
    }


def _drug_program_decompositions(
    settings: _RescueProgramSettings,
    programs: Mapping[str, frozenset[str]],
    rescue_vectors: Mapping[str, pd.Series],
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Decompose drug-level and panel signatures against every state.

    Each external source is reduced to one equal-context mean signature per
    drug, as in the mean-drug baseline, so contributions add up exactly.
    Panel compounds use their study signature on genes with status "ok".

    Returns:
      The long decomposition with `source` and `drug`, and the identity of
      every signature input.
    """
    frames: list[pd.DataFrame] = []
    inputs: dict[str, object] = {}
    for name, source in settings.sources.items():
        signatures = load_perturb_signatures(source.prefix)
        inputs[name] = {
            "cache": file_identity(Path(f"{source.prefix}.npz")),
            "provenance": signatures.provenance,
        }
        drug_means = mean_drug_signatures(
            signatures,
            drug_col=source.drug_col,
            context_col=source.context_cols,
        )
        del signatures
        frames.append(
            decompose_program_mimicry(
                drug_means,
                rescue_vectors,
                programs,
                excluded_program=settings.excluded_program,
                label_col=source.drug_col,
                minimum_shared_genes=settings.minimum_shared_genes,
            )
            .rename(columns={source.drug_col: "drug"})
            .assign(source=name)
        )

    if settings.panel_signatures_path is not None:
        panel = pd.read_parquet(settings.panel_signatures_path)
        for (study, compound), rows in panel.loc[
            panel["status"].eq("ok")
        ].groupby(["study_id", "compound"], sort=True):
            signature = PerturbSignatures(
                delta=rows["delta"].to_numpy(dtype=float)[np.newaxis, :],
                genes=rows["gene_id"].astype(str).tolist(),
                meta=pd.DataFrame(
                    {"drug": [str(compound)]},
                    index=pd.Index([f"panel_{study}_{compound}"]),
                ),
            )
            frames.append(
                decompose_program_mimicry(
                    signature,
                    rescue_vectors,
                    programs,
                    excluded_program=settings.excluded_program,
                    minimum_shared_genes=settings.minimum_shared_genes,
                ).assign(source=f"panel:{study}")
            )
    if not frames:
        raise ValueError("Configure at least one signature source or panel")
    return pd.concat(frames, ignore_index=True), inputs


def _wide_program_scores(decomposition: pd.DataFrame) -> pd.DataFrame:
    """Pivot the decomposition to one row per source, drug and state."""
    keys = ["source", "drug", "state"]
    named = decomposition.assign(
        column=np.where(
            decomposition["measure"].eq("full"),
            "full",
            decomposition["measure"] + "_" + decomposition["part"],
        )
    )
    values = named.pivot(index=keys, columns="column", values="value")
    support = (
        named.loc[named["measure"].eq("full")]
        .set_index(keys)[["status", "n_genes"]]
        .rename(columns={"status": "full_status", "n_genes": "n_shared_genes"})
    )
    return support.join(values).reset_index()


def _positive_control_scores(
    scores: pd.DataFrame,
    *,
    controls: Sequence[tuple[str, str]],
) -> pd.DataFrame:
    """Return the scores of each configured control, flagging absent ones.

    A control missing from its source is reported as "not_available", never
    replaced by a similarly named compound.
    """
    frames: list[pd.DataFrame] = []
    for source, drug in controls:
        rows = scores.loc[scores["source"].eq(source) & scores["drug"].eq(drug)]
        if rows.empty:
            rows = pd.DataFrame(
                {
                    "source": [source],
                    "drug": [drug],
                    "full_status": ["not_available"],
                }
            )
        frames.append(rows)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _rescue_program_figures(
    destination: Path,
    *,
    tables: Mapping[str, pd.DataFrame],
    settings: _RescueProgramSettings,
    programs: Mapping[str, frozenset[str]],
) -> None:
    """Draw the Hallmark, donor, decomposition and control figures."""
    state = settings.display_state
    excluded = settings.excluded_program
    summary = tables["hallmark_rescue_summary.csv"]
    shifts = summary.pivot(
        index="gene_set", columns="state", values="mean_t_shift"
    )
    shifts = shifts.loc[shifts[state].sort_values(na_position="last").index]
    tags = {
        name: [p for p, sets in settings.programs.items() if name in sets]
        for name in shifts.index
    }
    shifts.index = [
        name.removeprefix("HALLMARK_")
        + (f" [{', '.join(tags[name])}]" if tags[name] else "")
        for name in shifts.index
    ]
    figure, _ = plot_program_matrix(
        shifts,
        color_label="Mean rescue t, set minus other tested genes",
        title=(
            "Hallmark gene sets in the paired rescue vector "
            f"(weight loss - baseline); rows by {state}"
        ),
    )
    _save_figure(figure, destination / "hallmark_rescue_shift.pdf", png=True)

    figure, _ = plot_donor_program_scores(
        tables["donor_program_scores.csv"],
        programs=list(programs),
        condition_order=settings.condition_order,
        group_col=settings.group_col,
    )
    _save_figure(figure, destination / "donor_program_scores.pdf", png=True)

    scores = tables["drug_program_scores.csv"]
    columns = {
        "full": "full",
        f"restricted_without_{excluded}": f"without {excluded}",
        **{f"restricted_{program}": f"{program} only" for program in programs},
    }
    decomposition = tables["drug_program_decomposition.parquet"]
    for source in settings.sources:
        figure, _ = plot_score_contributions(
            decomposition.loc[decomposition["source"].eq(source)],
            state=state,
            part_order=[*programs, "shared", "other"],
            top_n=settings.top_drugs,
        )
        _save_figure(
            figure, destination / f"{source}_score_contributions.pdf", png=True
        )

        rows = scores.loc[
            scores["source"].eq(source)
            & scores["state"].eq(state)
            & scores["full_status"].eq("ok")
        ].set_index("drug")
        selected = list(
            dict.fromkeys(
                drug
                for column in [
                    "full",
                    *(f"restricted_{p}" for p in programs if p != excluded),
                ]
                for drug in rows[column].nlargest(settings.top_drugs).index
            )
        )
        matrix = rows.loc[selected, list(columns)].rename(columns=columns)
        figure, _ = plot_program_matrix(
            matrix.sort_values("full", ascending=False),
            color_label="Pearson mimicry\n(positive follows rescue)",
            title=(
                f"{source}, {state}: top drugs by full or program-only "
                "rescue score"
            ),
        )
        _save_figure(
            figure, destination / f"{source}_program_scores.pdf", png=True
        )

    controls = tables["positive_controls.csv"]
    if "state" in controls:
        shown = controls.loc[controls["state"].eq(state)].copy()
        if not shown.empty:
            shown.index = shown["drug"] + " (" + shown["source"] + ")"
            figure, _ = plot_program_matrix(
                shown[list(columns)].rename(columns=columns),
                color_label="Pearson mimicry\n(positive follows rescue)",
                title=f"Positive controls, {state}",
            )
            _save_figure(
                figure, destination / "positive_controls.pdf", png=True
            )


def _text(
    values: Mapping[str, object],
    key: str,
    *,
    default: str | None = None,
) -> str:
    """Read a nonempty string option; a missing option uses `default`."""
    value = values.get(key, default)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a nonempty string")
    return value


def _texts(
    values: Mapping[str, object],
    key: str,
    *,
    default: Sequence[str] | None = None,
) -> tuple[str, ...]:
    """Read a list of nonempty strings; a missing option uses `default`."""
    value = values.get(key, default)
    if not isinstance(value, (list, tuple)) or not all(
        isinstance(item, str) and item for item in value
    ):
        raise ValueError(f"{key} must be a list of nonempty strings")
    return tuple(value)


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
