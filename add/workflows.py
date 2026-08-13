"""Executable workflows for adipose rescue and drug ranking."""

from __future__ import annotations

import importlib.metadata
import logging
import re
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import anndata as ad  # type: ignore[import]
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from add.baselines import build_adipose_starting_expression
from add.baselines import evaluate_pca_ridge
from add.baselines import evaluate_perturbed_mean
from add.baselines import fit_pca_ridge
from add.baselines import mean_drug_signatures
from add.baselines import predict_pca_ridge
from add.baselines import score_cmap
from add.baselines import score_mean_drug
from add.baselines import score_pca_ridge
from add.baselines import score_perturbed_mean
from add.data import AnalysisConfig
from add.data import file_identity
from add.data import load_adipose
from add.data import load_rescue_table
from add.data import write_provenance
from add.perturb import PerturbSignatures
from add.perturb import load_lincs_signatures
from add.perturb import load_perturb_signatures
from add.perturb import load_tahoe_signatures
from add.perturb import save_perturb_signatures
from add.pseudobulk import adipose_group_support
from add.pseudobulk import build_adipose_pseudobulk
from add.rescue import estimate_rescue
from add.rescue import rescue_vector
from add.rescue import select_matched_pairs
from add.utils import get_physical_cores
from add.visualization import plot_context_variability
from add.visualization import plot_drug_state_scores
from add.visualization import plot_rescue_summary
from add.visualization import plot_signature_scatter
from add.visualization import plot_top_rankings
from add.visualization import set_matplotlib_publication_parameters


logger = logging.getLogger(__name__)


def create_pseudobulk(
    config: AnalysisConfig,
    *,
    input_path: Path | None = None,
    output_dir: Path | None = None,
    chunk_size: int = 4096,
) -> Path:
    """Build and persist donor-level state and pooled count profiles."""
    resolved_input_path = _argument_path(
        input_path,
        fallback=config.adipose_path,
    )
    resolved_output_dir = _stage_output_dir(
        output_dir,
        config=config,
        stage="pseudobulk",
    )
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    adata = load_adipose(
        resolved_input_path,
        count_layer=config.count_layer,
        required_obs=(
            config.donor_col,
            config.condition_col,
            config.state_col,
        ),
        backed=True,
    )
    try:
        support = adipose_group_support(
            adata,
            donor_col=config.donor_col,
            condition_col=config.condition_col,
            state_col=config.state_col,
            min_cells=config.min_cells,
            unassigned_label=config.unassigned_label,
        )
        pseudobulk = build_adipose_pseudobulk(
            adata,
            count_layer=config.count_layer,
            donor_col=config.donor_col,
            condition_col=config.condition_col,
            state_col=config.state_col,
            min_cells=config.min_cells,
            metadata_cols=config.metadata_cols,
            unassigned_label=config.unassigned_label,
            chunk_size=chunk_size,
        )
    finally:
        if adata.isbacked:
            adata.file.close()

    pseudobulk_path = resolved_output_dir / "adipose_pseudobulk.h5ad"
    pseudobulk.write_h5ad(pseudobulk_path, compression="gzip")
    support.to_csv(resolved_output_dir / "group_support.csv", index=False)
    write_provenance(
        resolved_output_dir / "provenance.json",
        command="add pseudobulk",
        config=config,
        inputs={"adipose": file_identity(resolved_input_path)},
        parameters={
            "biological_unit": "donor",
            "aggregation": "sum_raw_counts",
            "count_layer": config.count_layer,
            "grouping": [
                config.donor_col,
                config.condition_col,
                config.state_col,
            ],
            "min_cells": config.min_cells,
            "chunk_size": chunk_size,
        },
    )
    logger.info(
        "Wrote %d pseudobulk profiles to %s",
        pseudobulk.n_obs,
        pseudobulk_path,
    )
    return pseudobulk_path


def estimate_rescue_vectors(
    config: AnalysisConfig,
    *,
    pseudobulk_path: Path | None = None,
    output_dir: Path | None = None,
) -> Path:
    """Estimate and persist paired rescue vectors for estimable states."""
    resolved_pseudobulk_path = _argument_path(
        pseudobulk_path,
        fallback=config.output_path / "pseudobulk" / "adipose_pseudobulk.h5ad",
    )
    if not resolved_pseudobulk_path.is_file():
        raise FileNotFoundError(
            f"Pseudobulk input not found: {resolved_pseudobulk_path}. "
            "Run add pseudobulk."
        )
    resolved_output_dir = _stage_output_dir(
        output_dir,
        config=config,
        stage="rescue",
    )
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    pseudobulk = ad.read_h5ad(resolved_pseudobulk_path)
    states = sorted(
        pseudobulk.obs[config.state_col].dropna().astype(str).unique().tolist()
    )
    if not states:
        raise ValueError("Pseudobulk input contains no adipocyte states.")

    rescue_tables: list[pd.DataFrame] = []
    pair_tables: list[pd.DataFrame] = []
    diagnostics: list[dict[str, object]] = []
    for state in states:
        paired = select_matched_pairs(
            pseudobulk,
            state=state,
            donor_col=config.donor_col,
            condition_col=config.condition_col,
            state_col=config.state_col,
            baseline_label=config.baseline_label,
            weightloss_label=config.weightloss_label,
        )
        n_pairs = paired.n_obs // 2
        pair_table = cast(pd.DataFrame, paired.obs).reset_index().copy()
        pair_table.insert(0, "state", state)
        pair_tables.append(pair_table)
        if n_pairs < 2:
            diagnostics.append(
                {
                    "state": state,
                    "n_pairs": n_pairs,
                    "n_tested_genes": 0,
                    "status": "insufficient_paired_donors",
                }
            )
            continue

        rescue = estimate_rescue(
            pseudobulk,
            state=state,
            donor_col=config.donor_col,
            condition_col=config.condition_col,
            state_col=config.state_col,
            baseline_label=config.baseline_label,
            weightloss_label=config.weightloss_label,
        )
        rescue["contrast"] = (
            f"{config.weightloss_label}-{config.baseline_label}"
        )
        rescue["direction"] = "positive_mimics_rescue"
        rescue_tables.append(rescue)
        diagnostics.append(
            {
                "state": state,
                "n_pairs": n_pairs,
                "n_tested_genes": int(
                    cast(pd.Series, rescue["tested"]).astype(bool).sum()
                ),
                "status": "ok",
            }
        )

    if not rescue_tables:
        raise ValueError(
            "No state has at least two matched donors; rescue is non-estimable."
        )
    rescue_results = pd.concat(rescue_tables, ignore_index=True)
    rescue_path = resolved_output_dir / "rescue_vectors.csv.gz"
    rescue_results.to_csv(rescue_path, index=False)
    pd.concat(pair_tables, ignore_index=True).to_csv(
        resolved_output_dir / "matched_pairs.csv",
        index=False,
    )
    pd.DataFrame(diagnostics).to_csv(
        resolved_output_dir / "diagnostics.csv",
        index=False,
    )
    for state, table in rescue_results.groupby("state", sort=False):
        table.to_csv(
            resolved_output_dir / f"rescue_{_safe_name(str(state))}.csv.gz",
            index=False,
        )

    set_matplotlib_publication_parameters()
    figure, _ = plot_rescue_summary(rescue_results)
    figure.savefig(
        resolved_output_dir / "rescue_summary.pdf",
        bbox_inches="tight",
    )
    plt.close(figure)
    write_provenance(
        resolved_output_dir / "provenance.json",
        command="add rescue",
        config=config,
        inputs={"pseudobulk": file_identity(resolved_pseudobulk_path)},
        parameters={
            "independent_unit": "donor",
            "design": "~ Donor + condition",
            "contrast": f"{config.weightloss_label}-{config.baseline_label}",
            "methods": ["filterByExpr", "TMM", "voom", "limma", "eBayes"],
            "implementation": "native_python",
            "pylimma_version": importlib.metadata.version("pylimma"),
        },
    )
    logger.info(
        "Wrote rescue vectors for %d states to %s",
        rescue_results["state"].nunique(),
        rescue_path,
    )
    return rescue_path


def cache_perturbation_signatures(
    config: AnalysisConfig,
    *,
    source: str,
    input_path: Path | None = None,
    matrix_path: Path | None = None,
    metadata_path: Path | None = None,
    gene_metadata_path: Path | None = None,
    output_prefix: Path | None = None,
) -> dict[str, Path]:
    """Process one configured external perturbation source."""
    if source not in {"tahoe", "lincs"}:
        raise ValueError("source must be 'tahoe' or 'lincs'.")
    section = config.tahoe if source == "tahoe" else config.lincs
    configured_prefix = _required_value(section, "processed_prefix", source)
    resolved_output_prefix = (
        output_prefix.expanduser().resolve()
        if output_prefix is not None
        else config.resolve_path(
            configured_prefix,
            name=f"{source}.processed_prefix",
        )
    )

    if source == "tahoe":
        resolved_input_path = (
            input_path.expanduser().resolve()
            if input_path is not None
            else config.resolve_path(
                _required_value(section, "input_path", "tahoe"),
                name="tahoe.input_path",
            )
        )
        signatures = load_tahoe_signatures(
            resolved_input_path,
            count_layer=_optional_string(section.get("count_layer")),
            drug_col=_string_value(section, "drug_col", "tahoe"),
            vehicle_label=_string_value(section, "vehicle_label", "tahoe"),
            context_cols=_string_sequence(section, "context_cols", "tahoe"),
            sample_col=_optional_string(section.get("sample_col")),
            dose_col=_optional_string(section.get("dose_col")),
            time_col=_optional_string(section.get("time_col")),
            target_col=_optional_string(section.get("target_col")),
            mechanism_col=_optional_string(section.get("mechanism_col")),
            chunk_size=_integer_value(section, "chunk_size", "tahoe"),
        )
    else:
        resolved_matrix_path = _path_override_or_config(
            matrix_path,
            config=config,
            section=section,
            key="matrix_path",
            section_name="lincs",
        )
        resolved_metadata_path = _path_override_or_config(
            metadata_path,
            config=config,
            section=section,
            key="metadata_path",
            section_name="lincs",
        )
        resolved_gene_metadata_path = gene_metadata_path
        if (
            resolved_gene_metadata_path is None
            and section.get("gene_metadata_path") is not None
        ):
            resolved_gene_metadata_path = config.resolve_path(
                section["gene_metadata_path"],
                name="lincs.gene_metadata_path",
            )
        signatures = load_lincs_signatures(
            resolved_matrix_path,
            resolved_metadata_path,
            signature_id_col=_string_value(
                section,
                "signature_id_col",
                "lincs",
            ),
            gene_metadata_path=resolved_gene_metadata_path,
            gene_col=_string_value(section, "gene_col", "lincs"),
            landmark_col=_string_value(section, "landmark_col", "lincs"),
            gene_id_col=_optional_string(section.get("gene_id_col")),
            perturbation_type_col=_optional_string(
                section.get("perturbation_type_col")
            ),
            compound_type=_optional_string(section.get("compound_type")),
            chunk_size=_integer_value(section, "chunk_size", "lincs"),
        )

    paths = save_perturb_signatures(signatures, resolved_output_prefix)
    logger.info(
        "Cached %d %s signatures across %d genes at %s",
        signatures.n_signatures,
        source,
        signatures.n_genes,
        paths["matrix"],
    )
    return paths


def rank_drug_candidates(
    config: AnalysisConfig,
    *,
    model: str,
    signature_prefix: Path | None = None,
    rescue_path: Path | None = None,
    pseudobulk_path: Path | None = None,
    output_dir: Path | None = None,
    workers: int | None = None,
) -> Path:
    """Run the selected baseline using shared scoring infrastructure."""
    supported_models = {"perturbed-mean", "pca-ridge", "mean-drug", "cmap"}
    if model not in supported_models:
        raise ValueError(f"Unsupported baseline model: {model!r}.")
    resolved_workers = get_physical_cores() if workers is None else workers
    if resolved_workers < 1:
        raise ValueError("workers must be at least 1")
    source_name = "lincs" if model == "cmap" else "tahoe"
    source_section = config.lincs if source_name == "lincs" else config.tahoe
    resolved_signature_prefix = (
        signature_prefix.expanduser().resolve()
        if signature_prefix is not None
        else config.resolve_path(
            _required_value(
                source_section,
                "processed_prefix",
                source_name,
            ),
            name=f"{source_name}.processed_prefix",
        )
    )
    resolved_rescue_path = _argument_path(
        rescue_path,
        fallback=config.output_path / "rescue" / "rescue_vectors.csv.gz",
    )
    signatures = load_perturb_signatures(resolved_signature_prefix)
    rescue_results = load_rescue_table(resolved_rescue_path)
    rescue_vectors = {
        str(state): rescue_vector(rescue_results, state=str(state))
        for state in sorted(rescue_results["state"].astype(str).unique())
    }
    minimum_shared_genes = _integer_value(
        config.gene_selection,
        "minimum_shared_genes",
        "gene_selection",
    )
    resolved_output_dir = (
        output_dir.expanduser().resolve()
        if output_dir is not None
        else config.output_path / "baselines" / model
    )
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    outputs: tuple[
        pd.DataFrame,
        pd.DataFrame | None,
        pd.DataFrame | None,
        PerturbSignatures,
        Path | None,
    ]
    if model == "perturbed-mean":
        outputs = _perturbed_mean_outputs(
            signatures,
            rescue_vectors=rescue_vectors,
            minimum_shared_genes=minimum_shared_genes,
            config=config,
        )
    elif model == "pca-ridge":
        outputs = _pca_ridge_outputs(
            signatures,
            rescue_vectors=rescue_vectors,
            minimum_shared_genes=minimum_shared_genes,
            output_dir=resolved_output_dir,
            pseudobulk_override=pseudobulk_path,
            config=config,
        )
    elif model == "mean-drug":
        outputs = _mean_drug_outputs(
            signatures,
            rescue_vectors=rescue_vectors,
            minimum_shared_genes=minimum_shared_genes,
            workers=resolved_workers,
            config=config,
        )
    else:
        outputs = _cmap_outputs(
            signatures,
            rescue_vectors=rescue_vectors,
            minimum_shared_genes=minimum_shared_genes,
            workers=resolved_workers,
            config=config,
        )
    (
        ranked,
        evaluation,
        context_scores,
        scatter_signatures,
        resolved_model_pseudobulk,
    ) = outputs

    ranked_path = resolved_output_dir / "ranked.csv"
    ranked.to_csv(ranked_path, index=False)
    if evaluation is not None:
        evaluation.to_csv(
            resolved_output_dir / "evaluation.csv",
            index=False,
        )
    if context_scores is not None:
        context_scores.to_csv(
            resolved_output_dir / "context_scores.csv",
            index=False,
        )
    _write_baseline_figures(
        ranked,
        rescue_vectors=rescue_vectors,
        scatter_signatures=scatter_signatures,
        context_scores=context_scores,
        output_dir=resolved_output_dir,
        random_seed=config.random_seed,
    )
    inputs: dict[str, object] = {
        "signature_cache": file_identity(
            _cache_matrix_path(resolved_signature_prefix)
        ),
        "rescue": file_identity(resolved_rescue_path),
    }
    if model == "pca-ridge":
        if resolved_model_pseudobulk is None:
            raise RuntimeError("PCA + ridge did not resolve pseudobulk input")
        inputs["pseudobulk"] = file_identity(resolved_model_pseudobulk)
    write_provenance(
        resolved_output_dir / "provenance.json",
        command=f"add baseline --model {model}",
        config=config,
        inputs=inputs,
        parameters={
            "model": model,
            "source": source_name,
            "minimum_shared_genes": minimum_shared_genes,
            "score_direction": "positive_mimics_weightloss_minus_baseline",
            "workers": resolved_workers,
        },
    )
    logger.info("Wrote %s baseline rankings to %s", model, ranked_path)
    return ranked_path


def _perturbed_mean_outputs(
    signatures: PerturbSignatures,
    *,
    rescue_vectors: Mapping[str, pd.Series],
    minimum_shared_genes: int,
    config: AnalysisConfig,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    None,
    PerturbSignatures,
    None,
]:
    """Evaluate and score the generic perturbation-response null."""
    context_cols = _string_sequence(
        config.tahoe,
        "context_cols",
        "tahoe",
    )
    drug_col = _string_value(config.tahoe, "drug_col", "tahoe")
    evaluation = evaluate_perturbed_mean(
        signatures,
        context_col=context_cols,
        drug_col=drug_col,
        test_fraction=_float_value(
            config.pca_ridge,
            "test_fraction",
            "pca_ridge",
        ),
        random_seed=config.random_seed,
        minimum_shared_genes=minimum_shared_genes,
    )
    ranked = score_perturbed_mean(
        signatures,
        rescue_vectors,
        context_col=context_cols,
        minimum_shared_genes=minimum_shared_genes,
        source="tahoe",
    )
    return ranked, evaluation, None, signatures, None


def _pca_ridge_outputs(
    signatures: PerturbSignatures,
    *,
    rescue_vectors: Mapping[str, pd.Series],
    minimum_shared_genes: int,
    output_dir: Path,
    pseudobulk_override: Path | None,
    config: AnalysisConfig,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    None,
    PerturbSignatures,
    Path,
]:
    """Fit externally, infer from obese adipose, and score PCA + ridge."""
    pseudobulk_path = _argument_path(
        pseudobulk_override,
        fallback=(
            config.output_path / "pseudobulk" / "adipose_pseudobulk.h5ad"
        ),
    )
    if not pseudobulk_path.is_file():
        raise FileNotFoundError(
            f"Pseudobulk input not found: {pseudobulk_path}. "
            "Run add pseudobulk."
        )
    pseudobulk = ad.read_h5ad(pseudobulk_path)
    paired_donors = _paired_donors_by_state(
        pseudobulk,
        states=rescue_vectors,
        config=config,
    )
    starting_expression = build_adipose_starting_expression(
        pseudobulk,
        donor_col=config.donor_col,
        condition_col=config.condition_col,
        state_col=config.state_col,
        baseline_label=config.baseline_label,
        paired_donors=paired_donors,
    )
    starting_expression.to_csv(
        output_dir / "adipose_starting_expression.csv.gz",
        index=True,
    )
    model_genes = _shared_pca_ridge_genes(
        signatures,
        rescue_vectors=rescue_vectors,
        starting_expression=starting_expression,
        minimum_shared_genes=minimum_shared_genes,
    )
    context_cols = _string_sequence(
        config.tahoe,
        "context_cols",
        "tahoe",
    )
    drug_col = _string_value(config.tahoe, "drug_col", "tahoe")
    n_components = _integer_value(
        config.pca_ridge,
        "n_components",
        "pca_ridge",
    )
    ridge_alpha = _float_value(
        config.pca_ridge,
        "ridge_alpha",
        "pca_ridge",
    )
    maximum_genes = _integer_value(
        config.gene_selection,
        "max_model_genes",
        "gene_selection",
    )
    target_col = _optional_string(config.tahoe.get("target_col"))
    mechanism_col = _optional_string(config.tahoe.get("mechanism_col"))
    evaluation = evaluate_pca_ridge(
        signatures,
        drug_col=drug_col,
        context_col=context_cols,
        n_components=n_components,
        ridge_alpha=ridge_alpha,
        model_genes=model_genes,
        max_model_genes=maximum_genes,
        target_col=target_col,
        mechanism_col=mechanism_col,
        test_fraction=_float_value(
            config.pca_ridge,
            "test_fraction",
            "pca_ridge",
        ),
        random_seed=config.random_seed,
        minimum_shared_genes=minimum_shared_genes,
    )
    model = fit_pca_ridge(
        signatures,
        drug_col=drug_col,
        context_col=context_cols,
        n_components=n_components,
        ridge_alpha=ridge_alpha,
        model_genes=model_genes,
        max_model_genes=maximum_genes,
        target_col=target_col,
        mechanism_col=mechanism_col,
        random_seed=config.random_seed,
    )
    drugs = sorted(signatures.meta[drug_col].dropna().astype(str).unique())
    predicted = predict_pca_ridge(
        model,
        starting_expression,
        drug_ids=drugs,
    )
    save_perturb_signatures(
        predicted,
        output_dir / "predicted_signatures",
    )
    ranked = score_pca_ridge(
        predicted,
        rescue_vectors,
        minimum_shared_genes=minimum_shared_genes,
    )
    return ranked, evaluation, None, predicted, pseudobulk_path


def _shared_pca_ridge_genes(
    signatures: PerturbSignatures,
    *,
    rescue_vectors: Mapping[str, pd.Series],
    starting_expression: pd.DataFrame,
    minimum_shared_genes: int,
) -> list[str]:
    """Intersect Tahoe, baseline adipose, and every rescue gene space."""
    common_rescue_genes = set.intersection(
        *(set(vector.index.astype(str)) for vector in rescue_vectors.values())
    )
    adipose_genes = set(starting_expression.columns.astype(str))
    model_genes = [
        str(gene)
        for gene in signatures.genes
        if str(gene) in common_rescue_genes and str(gene) in adipose_genes
    ]
    if len(model_genes) < minimum_shared_genes:
        raise ValueError(
            "PCA + ridge has only "
            f"{len(model_genes)} genes shared by Tahoe, adipose, and all "
            "rescue states. Adjust gene identifiers or selection settings."
        )
    return model_genes


def _mean_drug_outputs(
    signatures: PerturbSignatures,
    *,
    rescue_vectors: Mapping[str, pd.Series],
    minimum_shared_genes: int,
    workers: int,
    config: AnalysisConfig,
) -> tuple[
    pd.DataFrame,
    None,
    pd.DataFrame,
    PerturbSignatures,
    None,
]:
    """Score per-drug equal-context means and retain context scores."""
    context_cols = _string_sequence(
        config.tahoe,
        "context_cols",
        "tahoe",
    )
    drug_col = _string_value(config.tahoe, "drug_col", "tahoe")
    ranked, context_scores = score_mean_drug(
        signatures,
        rescue_vectors,
        drug_col=drug_col,
        context_col=context_cols,
        minimum_shared_genes=minimum_shared_genes,
        source="tahoe",
        target_col=_optional_string(config.tahoe.get("target_col")),
        mechanism_col=_optional_string(config.tahoe.get("mechanism_col")),
        workers=workers,
    )
    means = mean_drug_signatures(
        signatures,
        drug_col=drug_col,
        context_col=context_cols,
    )
    return ranked, None, context_scores, means, None


def _cmap_outputs(
    signatures: PerturbSignatures,
    *,
    rescue_vectors: Mapping[str, pd.Series],
    minimum_shared_genes: int,
    workers: int,
    config: AnalysisConfig,
) -> tuple[
    pd.DataFrame,
    None,
    pd.DataFrame,
    PerturbSignatures,
    None,
]:
    """Score measured LINCS signatures using positive-mimic connectivity."""
    drug_col = _string_value(config.lincs, "drug_col", "lincs")
    context_cols = _configured_existing_columns(
        signatures.meta,
        config.lincs,
        keys=("cell_line_col", "dose_col", "time_col"),
    )
    ranked, context_scores = score_cmap(
        signatures,
        rescue_vectors,
        drug_col=drug_col,
        context_col=context_cols,
        query_genes_per_direction=_integer_value(
            config.gene_selection,
            "cmap_query_genes",
            "gene_selection",
        ),
        minimum_query_genes=_integer_value(
            config.gene_selection,
            "cmap_minimum_query_genes",
            "gene_selection",
        ),
        minimum_shared_genes=minimum_shared_genes,
        source="lincs",
        target_col=_optional_string(config.lincs.get("target_col")),
        mechanism_col=_optional_string(config.lincs.get("mechanism_col")),
        workers=workers,
    )
    return ranked, None, context_scores, signatures, None


def _stage_output_dir(
    override: Path | None,
    *,
    config: AnalysisConfig,
    stage: str,
) -> Path:
    """Resolve a stage output directory from an override or configuration."""
    if override is not None:
        return override.expanduser().resolve()
    return config.output_path / stage


def _argument_path(value: Path | None, *, fallback: Path) -> Path:
    """Resolve a path-valued CLI override."""
    return value.expanduser().resolve() if value is not None else fallback


def _path_override_or_config(
    override: Path | None,
    *,
    config: AnalysisConfig,
    section: Mapping[str, object],
    key: str,
    section_name: str,
) -> Path:
    """Resolve one required external path from CLI or YAML."""
    if override is not None:
        return override.expanduser().resolve()
    value = _required_value(section, key, section_name)
    return config.resolve_path(value, name=f"{section_name}.{key}")


def _required_value(
    section: Mapping[str, object],
    key: str,
    section_name: str,
) -> object:
    """Return a required configured value with its full key in errors."""
    value = section.get(key)
    if value is None or value == "":
        raise ValueError(
            f"Configuration value {section_name}.{key} is required."
        )
    return value


def _string_value(
    section: Mapping[str, object],
    key: str,
    section_name: str,
) -> str:
    """Return a required string from a source configuration section."""
    value = _required_value(section, key, section_name)
    if not isinstance(value, str):
        raise ValueError(f"{section_name}.{key} must be a string.")
    return value


def _optional_string(value: object) -> str | None:
    """Normalize a nullable configured column name."""
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError("Optional column names must be strings or null.")
    return value


def _string_sequence(
    section: Mapping[str, object],
    key: str,
    section_name: str,
) -> tuple[str, ...]:
    """Return a required sequence of configured column names."""
    value = _required_value(section, key, section_name)
    if isinstance(value, str) or not isinstance(value, list | tuple):
        raise ValueError(f"{section_name}.{key} must be a list of strings.")
    if not all(isinstance(item, str) and item for item in value):
        raise ValueError(f"{section_name}.{key} must contain only strings.")
    return tuple(value)


def _integer_value(
    section: Mapping[str, object],
    key: str,
    section_name: str,
) -> int:
    """Return a required integer from a source configuration section."""
    value = _required_value(section, key, section_name)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{section_name}.{key} must be an integer.")
    return value


def _float_value(
    section: Mapping[str, object],
    key: str,
    section_name: str,
) -> float:
    """Return a required finite number from a configuration section."""
    value = _required_value(section, key, section_name)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{section_name}.{key} must be numeric.")
    resolved = float(value)
    if not np.isfinite(resolved):
        raise ValueError(f"{section_name}.{key} must be finite.")
    return resolved


def _paired_donors_by_state(
    pseudobulk: ad.AnnData,
    *,
    states: Mapping[str, pd.Series],
    config: AnalysisConfig,
) -> dict[str, tuple[str, ...]]:
    """Return exactly the donor sets defining each rescue vector."""
    paired_donors: dict[str, tuple[str, ...]] = {}
    for state in states:
        paired = select_matched_pairs(
            pseudobulk,
            state=state,
            donor_col=config.donor_col,
            condition_col=config.condition_col,
            state_col=config.state_col,
            baseline_label=config.baseline_label,
            weightloss_label=config.weightloss_label,
        )
        if donors := tuple(
            paired.obs[config.donor_col].astype(str).drop_duplicates().tolist()
        ):
            paired_donors[state] = donors
        else:
            raise ValueError(
                f"No paired donors remain for rescue state {state!r}."
            )
    return paired_donors


def _configured_existing_columns(
    metadata: pd.DataFrame,
    section: Mapping[str, object],
    *,
    keys: tuple[str, ...],
) -> tuple[str, ...]:
    """Resolve configured optional context columns present in metadata."""
    columns: list[str] = []
    for key in keys:
        value = section.get(key)
        if value is None:
            continue
        if not isinstance(value, str) or not value:
            raise ValueError(f"lincs.{key} must be a string or null.")
        if value in metadata:
            columns.append(value)
    if not columns:
        raise ValueError(
            "No configured LINCS context column is present in signature "
            "metadata."
        )
    return tuple(dict.fromkeys(columns))


def _write_baseline_figures(
    rankings: pd.DataFrame,
    *,
    rescue_vectors: Mapping[str, pd.Series],
    scatter_signatures: PerturbSignatures,
    context_scores: pd.DataFrame | None,
    output_dir: Path,
    random_seed: int,
) -> None:
    """Render deterministic summaries from machine-readable baseline results."""
    numeric_scores = cast(
        pd.Series,
        pd.to_numeric(
            cast(pd.Series, rankings["score"]),
            errors="coerce",
        ),
    )
    finite_mask = np.isfinite(numeric_scores.to_numpy(dtype=float))
    finite = rankings.loc[finite_mask]
    if finite.empty:
        logger.warning("No finite baseline scores are available for figures.")
        return
    observed_states = set(finite["state"].astype(str))
    state = (
        "AD_ALL" if "AD_ALL" in observed_states else sorted(observed_states)[0]
    )

    set_matplotlib_publication_parameters()
    state_count = int((finite["state"].astype(str) == state).sum())
    figure, _ = plot_top_rankings(
        rankings,
        adipocyte_state=state,
        top_n=min(15, state_count),
    )
    figure.savefig(output_dir / "top_rankings.pdf", bbox_inches="tight")
    plt.close(figure)

    n_drugs = int(finite["drug"].nunique(dropna=True))
    figure, _ = plot_drug_state_scores(
        rankings,
        top_drugs=min(20, max(1, n_drugs)),
    )
    figure.savefig(output_dir / "drug_state_scores.pdf", bbox_inches="tight")
    plt.close(figure)

    if context_scores is not None:
        context_values = cast(
            pd.Series,
            pd.to_numeric(
                context_scores["score_mimic"],
                errors="coerce",
            ),
        )
        if np.isfinite(context_values.to_numpy(dtype=float)).any():
            figure, _ = plot_context_variability(
                context_scores,
                adipocyte_state=state,
                random_seed=random_seed,
            )
            figure.savefig(
                output_dir / "context_variability.pdf",
                bbox_inches="tight",
            )
            plt.close(figure)

    candidate = _top_candidate_series(
        finite,
        state=state,
        signatures=scatter_signatures,
        context_scores=context_scores,
    )
    if candidate is None:
        return
    candidate_label, candidate_delta = candidate
    figure, _ = plot_signature_scatter(
        candidate_delta,
        rescue_vectors[state],
        candidate_label=candidate_label,
        adipocyte_state=state,
    )
    figure.savefig(
        output_dir / "top_signature_scatter.pdf",
        bbox_inches="tight",
    )
    plt.close(figure)


def _top_candidate_series(
    finite_rankings: pd.DataFrame,
    *,
    state: str,
    signatures: PerturbSignatures,
    context_scores: pd.DataFrame | None,
) -> tuple[str, pd.Series] | None:
    """Resolve the measured or predicted vector behind the top drug row."""
    state_rows = finite_rankings.loc[
        finite_rankings["state"].astype(str) == state
    ]
    if state_rows.empty:
        return None
    top = state_rows.sort_values("score", ascending=False).iloc[0]
    if pd.isna(top.get("drug")):
        return None
    drug = str(top["drug"])
    positions = np.arange(len(signatures.meta))

    if context_scores is not None and "signature_id" in context_scores:
        numeric_context_scores = cast(
            pd.Series,
            pd.to_numeric(
                cast(pd.Series, context_scores["score"]),
                errors="coerce",
            ),
        )
        finite_context_scores = np.isfinite(
            numeric_context_scores.to_numpy(dtype=float)
        )
        scored_contexts = context_scores.loc[
            (context_scores["state"].astype(str) == state)
            & (context_scores["drug"].astype(str) == drug)
            & finite_context_scores
        ]
        if not scored_contexts.empty:
            signature_id = str(
                scored_contexts.sort_values("score", ascending=False).iloc[0][
                    "signature_id"
                ]
            )
            identifiers = (
                signatures.meta["signature_id"].astype(str).to_numpy()
                if "signature_id" in signatures.meta
                else signatures.meta.index.astype(str).to_numpy()
            )
            positions = np.flatnonzero(identifiers == signature_id)
    if len(positions) != 1:
        drug_column = "drug" if "drug" in signatures.meta else "drug_name"
        if drug_column not in signatures.meta:
            return None
        mask = signatures.meta[drug_column].astype(str).eq(drug).to_numpy()
        if "state" in signatures.meta:
            state_mask = (
                signatures.meta["state"].astype(str).eq(state).to_numpy()
            )
            mask = mask & state_mask
        positions = np.flatnonzero(mask)
    if len(positions) == 0:
        return None
    position = int(positions[0])
    values = np.asarray(signatures.delta[position], dtype=float).reshape(-1)
    genes = pd.Index(signatures.genes, name="gene")
    return drug, pd.Series(values, index=genes)


def _cache_matrix_path(prefix: Path) -> Path:
    """Return the NPZ matrix path owned by a perturbation cache prefix."""
    return prefix if prefix.suffix == ".npz" else Path(f"{prefix}.npz")


def _safe_name(value: str) -> str:
    """Return a deterministic filesystem component for a biological label."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return cleaned or "unnamed_state"
