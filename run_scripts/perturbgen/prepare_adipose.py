"""Prepare paired adipocyte counts for PerturbGen."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import cast

import anndata as ad  # type: ignore[import]
import numpy as np
import pandas as pd
from scipy import sparse  # type: ignore[import]

from run_scripts.perturbgen.config import PerturbGenConfig
from run_scripts.perturbgen.config import PrepareConfig
from run_scripts.perturbgen.config import load_perturbgen_config


logger = logging.getLogger(__name__)


def prepare_adipose_adata(config: PerturbGenConfig) -> Path:
    """Prepare paired raw-count adipocytes for PerturbGen tokenization."""
    # Load config and adata
    settings = config.prepare
    adata = _load_adipose(settings)

    # Select paired donors without copying the complete AnnData
    selected, paired_donors = _select_paired_donors(adata, settings)
    raw_counts = cast(
        np.ndarray | sparse.csr_matrix,
        selected.layers[settings.raw_count_layer],
    )
    _validate_raw_counts(raw_counts, layer_name=settings.raw_count_layer)

    # Map genes to Ensembl IDs
    final_counts, final_var = _map_genes(
        raw_counts,
        source_genes=selected.var_names,
        settings=settings,
    )

    # Prep the adata for tokenization
    prepared = _build_prepared_anndata(
        selected,
        final_counts=final_counts,
        final_var=final_var,
        settings=settings,
    )

    output_path = config.output_h5ad_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prepared.write_h5ad(output_path, compression="gzip")
    logger.info(
        "Wrote %d cells from %d paired donors and %d genes to %s",
        prepared.n_obs,
        len(paired_donors),
        prepared.n_vars,
        output_path,
    )
    return output_path


def _load_adipose(settings: PrepareConfig) -> ad.AnnData:
    """Load the configured H5AD and validate its preparation schema."""
    if not settings.source_h5ad_path.is_file():
        raise FileNotFoundError(
            f"Adipose H5AD not found: {settings.source_h5ad_path}"
        )
    if not settings.gene_annotation_path.is_file():
        raise FileNotFoundError(
            f"Gene annotation table not found: {settings.gene_annotation_path}"
        )

    adata = ad.read_h5ad(settings.source_h5ad_path)
    adata_obs = cast(pd.DataFrame, adata.obs)
    if missing_obs := [
        column
        for column in (
            settings.donor_col,
            settings.condition_col,
            settings.source_state_col,
        )
        if column not in adata_obs
    ]:
        raise KeyError(f"Adipose H5AD is missing obs columns: {missing_obs}")
    if settings.raw_count_layer not in adata.layers:
        raise KeyError(
            f"Adipose H5AD has no raw-count layer {settings.raw_count_layer!r}."
        )
    return adata


def _select_paired_donors(
    adata: ad.AnnData,
    settings: PrepareConfig,
) -> tuple[ad.AnnData, set[str]]:
    """Select cells from donors represented at both baseline and weight loss."""
    adata_obs = cast(pd.DataFrame, adata.obs)
    condition = adata_obs[settings.condition_col].astype(str)
    paired_conditions = condition.isin(
        [settings.source_baseline_label, settings.source_weightloss_label]
    )
    candidate_obs = adata_obs.loc[paired_conditions]

    baseline_donors = set(
        candidate_obs.loc[
            candidate_obs[settings.condition_col].astype(str)
            == settings.source_baseline_label,
            settings.donor_col,
        ].astype(str)
    )

    weightloss_donors = set(
        candidate_obs.loc[
            candidate_obs[settings.condition_col].astype(str)
            == settings.source_weightloss_label,
            settings.donor_col,
        ].astype(str)
    )

    paired_donors: set[str] = baseline_donors.intersection(weightloss_donors)
    if not paired_donors:
        raise ValueError(
            "No donor is represented at both baseline and weight loss."
        )

    keep_cells = paired_conditions & adata_obs[settings.donor_col].astype(
        str
    ).isin(sorted(paired_donors))
    selected = cast(ad.AnnData, adata[keep_cells])

    return selected, paired_donors


def _map_genes(
    raw_counts: np.ndarray | sparse.csr_matrix,
    *,
    source_genes: pd.Index,
    settings: PrepareConfig,
) -> tuple[np.ndarray | sparse.csr_matrix, pd.DataFrame]:
    """Map input genes to unique Ensembl IDs without changing count order."""
    annotation = pd.read_csv(settings.gene_annotation_path, sep="\t")
    symbol_col = settings.annotation_gene_symbol_col
    ensembl_col = settings.annotation_ensembl_col

    if missing_annotation := [
        column
        for column in (symbol_col, ensembl_col)
        if column not in annotation
    ]:
        raise KeyError(
            f"Gene annotation is missing columns: {missing_annotation}"
        )

    annotation = annotation.dropna(subset=[symbol_col, ensembl_col]).copy()
    symbol_to_ensembl = dict(
        zip(
            annotation[symbol_col].astype(str),
            annotation[ensembl_col].astype(str),
            strict=False,
        )
    )
    mapped_genes = pd.Index(
        [
            gene if gene.startswith("ENSG") else symbol_to_ensembl.get(gene)
            for gene in source_genes.astype(str)
        ]
    )

    mapped_mask = np.asarray(mapped_genes.notna(), dtype=bool)
    if not mapped_mask.any():
        raise ValueError("No input gene could be mapped to an Ensembl ID.")

    mapped_ids = mapped_genes[mapped_mask].astype(str)
    unique_mask = ~mapped_ids.duplicated(keep="first")
    selected_positions = np.flatnonzero(mapped_mask)[unique_mask]
    final_ids = mapped_ids[unique_mask]
    final_counts = cast(
        np.ndarray | sparse.csr_matrix,
        raw_counts[:, selected_positions],
    )
    final_var = pd.DataFrame(
        {"ensembl_id": final_ids},
        index=pd.Index(final_ids, name="gene_id"),
    )

    return final_counts, final_var


def _build_prepared_anndata(
    selected: ad.AnnData,
    *,
    final_counts: np.ndarray | sparse.csr_matrix,
    final_var: pd.DataFrame,
    settings: PrepareConfig,
) -> ad.AnnData:
    """Build the PerturbGen tokenization input."""
    prepared_obs = cast(pd.DataFrame, selected.obs).copy()
    prepared_obs[settings.state_alias_col] = prepared_obs[
        settings.source_state_col
    ]
    prepared_obs[settings.condition_col] = (
        prepared_obs[settings.condition_col]
        .astype(str)
        .replace(
            {settings.source_baseline_label: (settings.prepared_baseline_label)}
        )
    )
    prepared = ad.AnnData(
        X=final_counts,
        obs=prepared_obs,
        var=final_var,
    )

    if not prepared.obs_names.is_unique:
        prepared.obs_names_make_unique()

    return prepared


def _validate_raw_counts(
    matrix: np.ndarray | sparse.csr_matrix,
    *,
    layer_name: str,
) -> None:
    """Reject non-finite, negative, or non-integer count values."""
    values = np.asarray(
        cast(np.ndarray, matrix.data)  # type: ignore[reportUnknownMemberType]
        if isinstance(matrix, sparse.csr_matrix)
        else matrix
    )
    if not np.isfinite(values).all():
        raise ValueError(
            f"Count layer {layer_name!r} contains non-finite values."
        )
    if np.any(values < 0):
        raise ValueError(
            f"Count layer {layer_name!r} contains negative values."
        )
    if not np.allclose(values, np.rint(values)):
        raise ValueError(f"Count layer {layer_name!r} is not integer-valued.")


def _parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("config.yaml"),
    )
    return parser.parse_args()


def main() -> None:
    """Prepare the configured adipose AnnData."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    arguments = _parse_arguments()
    prepare_adipose_adata(load_perturbgen_config(arguments.config))


if __name__ == "__main__":
    main()
