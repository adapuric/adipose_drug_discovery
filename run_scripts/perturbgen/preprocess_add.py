
"""Preprocess AnnData before the PerturbGen workflow."""

import argparse
import logging
from pathlib import Path
from typing import cast

import anndata as ad  # type: ignore[import]
import pandas as pd
import scanpy as sc  # type: ignore[import]
import numpy as np

logger = logging.getLogger(__name__)


def preprocess_adipose_adata(
    input_adata: Path,
    output_adata: Path,
    max_pct_mito: float | None = None,
    max_pct_ribo: float | None = None,
    remove_ensembl: bool = False,
    remove_lncrnas: bool = False,
    select_hvgs: bool = False,
    highly_variable_gene_col: str = "highly_variable",
    batch_key: str | None = None,
    n_top_genes: int | None = None,
    max_pct_cells: float | None = None,
    state_col: str | None = None, 
    min_pct_in_state: float | None = None,
) -> None:
    """Preprocess adipocyte AnnData for PerturbGen tokenization.

    Args:
      input_adata: Path to the input H5AD file.
      output_adata: Path for the preprocessed H5AD file.
      max_pct_mito: Maximum mitochondrial percentage, or None to skip filtering.
      max_pct_ribo: Maximum ribosomal percentage, or None to skip filtering.
      remove_ensembl: Whether to remove unmapped Ensembl identifiers.
      remove_lncrnas: Whether to remove antisense lncRNA genes.
      select_hvgs: Whether to subset to highly variable genes.
      highly_variable_gene_col: Column in adata.var marking highly variable
        genes.
      n_top_genes: Number of top highly variable genes to select, or None to use
        all genes marked highly variable in the source H5AD.
      batch_key: Column in adata.obs marking batch membership, or None to ignore
        batch when selecting highly variable genes.
      max_pct_cells: Maximum allowed percentage of cells expressing a gene, computed globally across all cells. Genes above this threshold are
        removed as ubiquitously expressed (ex: housekeeping genes).
      state_col: Column in adata.obs identifying the annotated cell state per cell. Required if min_pct_in_state is set.
      min_pct_in_state: Minimum percentage of cells expressing a gene required in at least one state. 
    """
    adata = ad.read_h5ad(input_adata)

    if max_pct_mito is not None:
        adata = _filter_cells_by_mito_pct(adata, max_pct_mito)

    if max_pct_ribo is not None:
        adata = _filter_cells_by_ribo_pct(adata, max_pct_ribo)

    if remove_ensembl:
        adata = _remove_ensembl_genes(adata)

    if remove_lncrnas:
        adata = _remove_lncrna_genes(adata)

    if select_hvgs:
        adata = _select_highly_variable_genes(
            adata=adata,
            highly_variable_gene_col=highly_variable_gene_col,
            batch_key=batch_key,
            n_top_genes=n_top_genes,
        )
    if max_pct_cells is not None:
        adata = _remove_wide_genes(adata, max_pct_cells)
    if min_pct_in_state is not None:
        if state_col is None:
            raise ValueError("state_col is required when min_pct_in_state is set.")
        adata = _filter_genes_by_expression_in_cellstates(adata, state_col, min_pct_in_state)


    output_adata.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(output_adata)

    logger.info(
        "Wrote %d cells and %d genes to %s",
        adata.n_obs,
        adata.n_vars,
        output_adata,
    )


def _filter_cells_by_mito_pct(
    adata: ad.AnnData,
    max_pct_mito: float,
) -> ad.AnnData:
    """Remove cells whose mitochondrial count percentage exceeds a threshold.

    Args:
      adata: Annotated data matrix with cells as observations.
      max_pct_mito: Maximum allowed percentage of counts from mt genes

    Returns:
      AnnData containing only cells that pass the filter.
    """
    keep = adata.obs["pct_counts_mt"] <= max_pct_mito
    return adata[keep]


def _filter_cells_by_ribo_pct(
    adata: ad.AnnData,
    max_pct_ribo: float,
) -> ad.AnnData:
    """Remove cells whose ribosomal count percentage exceeds a threshold.

    Args:
      adata: Annotated data matrix with cells as observations.
      max_pct_ribo: Maximum allowed percentage of counts from ribosomal gene

    Returns:
      AnnData containing only cells that pass the filter.
    """
    keep = adata.obs["pct_counts_ribo"] <= max_pct_ribo
    return adata[keep]


def _remove_ensembl_genes(adata: ad.AnnData) -> ad.AnnData:
    """Remove genes whose only identifier is an unmapped Ensembl ID.

    Args:
      adata: Annotated data matrix with cells as observations.

    Returns:
      AnnData containing only mapped genes
    """
    is_ensembl = adata.var.index.to_series().str.startswith("ENS")
    return adata[:, ~is_ensembl]


def _remove_lncrna_genes(adata: ad.AnnData) -> ad.AnnData:
    """Remove antisense lncRNA genes ending with -AS<number>.

    Args:
      adata: Annotated data matrix with cells as observations.

    Returns:
      AnnData containing no antisense lncRNAs.
    """
    name = (
        adata.var["gene_symbols"].astype(str)
        if "gene_symbols" in adata.var.columns
        else adata.var.index.to_series().astype(str)
    )
    is_antisense = name.str.contains(r"-AS\d*$", regex=True)
    return adata[:, ~is_antisense]


def _select_highly_variable_genes(
    adata: ad.AnnData,
    highly_variable_gene_col: str,
    batch_key: str | None = None,
    n_top_genes: int | None = None,
) -> ad.AnnData:
    """Subset to genes marked highly variable in the source H5AD."""
    sc.pp.highly_variable_genes(
        adata, batch_key=batch_key, n_top_genes=n_top_genes
    )
    adata_var = cast(pd.DataFrame, adata.var)
    if highly_variable_gene_col not in adata_var:
        raise KeyError(
            f"Adipose H5AD is missing var column {highly_variable_gene_col!r}."
        )

    highly_variable = adata_var[highly_variable_gene_col]
    if highly_variable.isna().to_numpy().any():
        raise ValueError(
            f"Gene-selection column {highly_variable_gene_col!r} "
            "contains missing values."
        )
    if not pd.api.types.is_bool_dtype(highly_variable.dtype):
        raise TypeError(
            f"Gene-selection column {highly_variable_gene_col!r} "
            "must be boolean."
        )

    keep_genes = highly_variable.to_numpy(dtype=bool)
    if not keep_genes.any():
        raise ValueError(
            f"Gene-selection column {highly_variable_gene_col!r} "
            "selects no genes."
        )
    return cast(ad.AnnData, adata[:, keep_genes])





def _remove_wide_genes(
    adata: ad.AnnData, 
    max_pct_cells: float, 
) -> ad.AnnData: 
    """Filter out genes that are expressed in over 95% of nuclei as they are ubiquitous 

    Args: 
    adata: Annotated data matrix with cells as observations.
    max_pct_cells: % expression of genes in nuclei to retain 

    Output: 
    AnnData containing no genes expressed in OVER 95% of nuclei   
    
    """
    counts = adata.layers["raw"]
    pct_expressing = np.asarray((counts > 0).sum(axis=0)).ravel() / adata.n_obs * 100
    keep = pct_expressing <= max_pct_cells
    return adata[:, keep]



def _filter_genes_by_expression_in_cellstates(
    adata: ad.AnnData, 
    state_col: str,
    min_pct_in_state: float,
) -> ad.AnnData:
    """Subset to genes expressed in at least 10% of cells, in at least one of the cell states 

    Args: 
    adata: Annotated data matrix with cells as observations.
    state_col: annotation column of cell stes in AnnData
    min_pct_in_state: % expression 

    Output: 
    AnnData with removed genes that are expressed in 95% of nuclei / cells (uniquotusu) 
    
    """
    counts = adata.layers["raw"] # take raw count matrix 
    max_pct_across_states = np.zeros(adata.n_vars) # each entry holds that gene's highest % expressing value across all states

    for state in adata.obs[state_col].unique(): # loop, one check per state 
        state_mask = (adata.obs[state_col] == state).to_numpy() # checks if nuclei belongs to the cell state 
        state_counts = counts[state_mask] # subsets count matrix to only the cells in this state
        pct_expressing = (
            np.asarray((state_counts > 0).sum(axis=0)).ravel() / state_mask.sum() * 100
        )
        # per gene comparison btw current best and this state's computed %s
        max_pct_across_states = np.maximum(max_pct_across_states, pct_expressing) 
        # after all states are processed, each gene's final value is its single highest % expressing score

    keep = max_pct_across_states >= min_pct_in_state
    return adata[:, keep]


    

    


def _parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("input_adata", type=Path)
    parser.add_argument("output_adata", type=Path)
    parser.add_argument("--max-pct-mito", type=float)
    parser.add_argument("--max-pct-ribo", type=float)
    parser.add_argument("--remove-ensembl", action="store_true")
    parser.add_argument("--remove-lncrnas", action="store_true")
    parser.add_argument("--select-hvgs", action="store_true")
    parser.add_argument("--batch-key", default=None)
    parser.add_argument("--n-top-genes", type=int, default=None)
    parser.add_argument("--max-pct-cells", type=float, default=None)
    parser.add_argument("--state-col", default=None)
    parser.add_argument("--min-pct-in-state", type=float, default=None)
    
    parser.add_argument(
        "--highly-variable-gene-col",
        default="highly_variable",
    )
    return parser.parse_args()


def main() -> None:
    """Run adipose AnnData preprocessing."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    arguments = _parse_arguments()

    preprocess_adipose_adata(
        input_adata=arguments.input_adata,
        output_adata=arguments.output_adata,
        max_pct_mito=arguments.max_pct_mito,
        max_pct_ribo=arguments.max_pct_ribo,
        remove_ensembl=arguments.remove_ensembl,
        remove_lncrnas=arguments.remove_lncrnas,
        select_hvgs=arguments.select_hvgs,
        highly_variable_gene_col=arguments.highly_variable_gene_col,
        batch_key=arguments.batch_key,
        n_top_genes=arguments.n_top_genes,
        max_pct_cells=arguments.max_pct_cells,
        state_col=arguments.state_col,
        min_pct_in_state=arguments.min_pct_in_state,
    )


if __name__ == "__main__":
    main()
 
