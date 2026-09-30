"""Scale-aware human adipose array validation without replicate inflation."""

from __future__ import annotations

import gzip
import io
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from add.perturb import PerturbSignatures
from add.scoring import score_mimicry
from add.scoring import weighted_cmap_connectivity


def read_geo_table(path: str | Path, *, table_kind: str) -> pd.DataFrame:
    """Read the explicitly delimited table from a GEO matrix or SOFT file."""
    source = Path(path)
    if table_kind not in {"series_matrix", "platform"}:
        raise ValueError("table_kind must be series_matrix or platform")
    opener = gzip.open if source.suffix == ".gz" else open
    with opener(source, "rt") as handle:
        content = handle.read()
    begin = f"!{table_kind}_table_begin"
    end = f"!{table_kind}_table_end"
    if content.count(begin) != 1 or content.count(end) != 1:
        raise ValueError(
            f"Expected one delimited {table_kind} table in {source}"
        )
    table = content.split(begin, 1)[1].split(end, 1)[0].strip()
    return pd.read_csv(
        io.StringIO(table), sep="\t", dtype={"ID_REF": str, "ID": str}
    )


def map_array_expression(
    expression: pd.DataFrame,
    annotation: pd.DataFrame,
    *,
    input_scale: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Map unambiguous probes and median-collapse normalized log expression."""
    if input_scale not in {
        "log2_expression",
        "log2_reference_ratio",
        "linear_expression",
        "linear_reference_ratio",
    }:
        raise ValueError(
            "Unverified input scale; establish the GEO processing scale first"
        )
    if not expression.index.is_unique or not expression.columns.is_unique:
        raise ValueError("Array probe and sample IDs must be unique")
    if not {"probe_id", "gene_id"} <= set(annotation):
        raise ValueError("Annotation requires probe_id and gene_id")
    mapping = (
        annotation[["probe_id", "gene_id"]]
        .dropna()
        .astype(str)
        .drop_duplicates()
    )
    # One probe with multiple candidate genes is excluded, not expanded.
    candidates = mapping.groupby("probe_id")["gene_id"].agg(
        lambda x: tuple(sorted(set(x)))
    )
    audit = pd.DataFrame({"probe_id": expression.index.astype(str)})
    audit["gene_id"] = audit["probe_id"].map(
        candidates.map(
            lambda x: x[0] if isinstance(x, tuple) and len(x) == 1 else pd.NA
        )
    )
    audit["status"] = np.where(
        audit["gene_id"].notna(), "mapped", "unmapped_or_ambiguous"
    )
    retained = audit.loc[audit["status"].eq("mapped")]
    if retained.empty:
        raise ValueError("No unambiguous array probes map to canonical genes")
    values = expression.loc[retained["probe_id"].tolist()].apply(
        pd.to_numeric, errors="raise"
    )
    if np.isinf(values.to_numpy()).any():
        raise ValueError("Array expression contains infinite values")
    if input_scale.startswith("linear"):
        if (values <= 0).any().any():
            raise ValueError("Linear array values must be positive before log2")
        values = pd.DataFrame(
            np.log2(values.to_numpy()),
            index=values.index,
            columns=values.columns,
        )
    values.index = pd.Index(retained["gene_id"].astype(str), name="gene_id")
    return values.groupby(level="gene_id", sort=True).median(), audit


def adipose_panel_signatures(
    expression: pd.DataFrame,
    samples: pd.DataFrame,
    *,
    pairing: str,
    minimum_units: int = 2,
) -> dict[str, pd.DataFrame]:
    """Build donor-paired or unpaired experimental-unit compound contrasts."""
    required = {
        "study_id",
        "sample_id",
        "compound",
        "unit_id",
        "unit_type",
        "condition",
    }
    if (
        not required <= set(samples)
        or samples[list(required)].isna().any().any()
    ):
        raise ValueError(
            "Sample manifest lacks complete experimental-unit metadata"
        )
    if (
        samples["sample_id"].duplicated().any()
        or samples["study_id"].nunique() != 1
    ):
        raise ValueError("Supply one study with unique sample IDs")
    if pairing not in {"paired", "unpaired"} or minimum_units < 2:
        raise ValueError(
            "Declare paired/unpaired replication with at least two units"
        )
    if not set(samples["sample_id"]) <= set(expression.columns):
        raise ValueError(
            "Manifest samples are absent from the expression matrix"
        )
    if not set(samples["condition"]) <= {"vehicle", "treated"}:
        raise ValueError("Conditions must be vehicle or treated")
    if samples["unit_type"].nunique() != 1:
        raise ValueError("A study cannot mix biological-unit types")
    study_id = str(samples["study_id"].iloc[0])
    measurements = (
        expression.loc[:, samples["sample_id"].tolist()]
        .rename_axis("gene_id")
        .reset_index()
        .melt(id_vars="gene_id", var_name="sample_id", value_name="expression")
    )
    measurements = measurements.merge(
        samples, on="sample_id", validate="many_to_one"
    )
    unit_keys = [
        "study_id",
        "unit_type",
        "unit_id",
        "compound",
        "condition",
        "gene_id",
    ]
    units = (
        measurements.groupby(unit_keys, observed=True, sort=True)["expression"]
        .mean()
        .reset_index()
    )
    controls = units.loc[units["condition"].eq("vehicle")]
    if controls.empty:
        raise ValueError("Study contains no matched-condition vehicle samples")
    results: list[pd.DataFrame] = []
    contrasts: list[pd.DataFrame] = []
    for compound, treated in units.loc[
        units["condition"].eq("treated")
    ].groupby("compound", sort=True):
        if pairing == "paired":
            paired = treated.merge(
                controls,
                on=["study_id", "unit_type", "unit_id", "gene_id"],
                suffixes=("", "_vehicle"),
                how="left",
                validate="one_to_one",
            )
            paired["delta"] = (
                paired["expression"] - paired["expression_vehicle"]
            )
            paired["status"] = np.where(
                paired["delta"].notna(), "ok", "missing_matched_vehicle"
            )
            contrasts.append(
                paired[
                    [
                        "study_id",
                        "compound",
                        "unit_type",
                        "unit_id",
                        "gene_id",
                        "delta",
                        "status",
                    ]
                ]
            )
            summary = (
                paired.groupby("gene_id")["delta"]
                .agg(["mean", "count"])
                .rename(columns={"mean": "delta", "count": "n_units_treated"})
            )
            summary["n_units_control"] = summary["n_units_treated"]
        else:
            summary = (
                treated.groupby("gene_id")["expression"]
                .agg(["mean", "count"])
                .join(
                    controls.groupby("gene_id")["expression"].agg(
                        ["mean", "count"]
                    ),
                    lsuffix="_treated",
                    rsuffix="_control",
                )
            )
            summary["delta"] = summary["mean_treated"] - summary["mean_control"]
            summary = summary.rename(
                columns={
                    "count_treated": "n_units_treated",
                    "count_control": "n_units_control",
                }
            )[["delta", "n_units_treated", "n_units_control"]]
        supported = (
            summary[["n_units_treated", "n_units_control"]]
            .min(axis=1)
            .ge(minimum_units)
        )
        summary["status"] = np.where(
            supported, "ok", "insufficient_independent_units"
        )
        summary.loc[~supported, "delta"] = np.nan
        summary["study_id"] = study_id
        summary["compound"] = str(compound)
        summary["unit_type"] = str(samples["unit_type"].iloc[0])
        summary["pairing"] = pairing
        summary["scale"] = "treated_minus_vehicle_log2"
        results.append(summary.reset_index())
    if not results:
        raise ValueError("Study contains no treated compound samples")
    return {
        "study_signatures": pd.concat(results, ignore_index=True),
        "unit_signatures": pd.concat(contrasts, ignore_index=True)
        if contrasts
        else pd.DataFrame(
            columns=[
                "study_id",
                "compound",
                "unit_type",
                "unit_id",
                "gene_id",
                "delta",
                "status",
            ]
        ),
        "unit_measurements": units,
    }


def compare_adipose_panel(
    panel: pd.DataFrame,
    *,
    external: Mapping[str, PerturbSignatures],
    compound_mapping: pd.DataFrame,
    rescue_vectors: Mapping[str, pd.Series],
    minimum_shared_genes: int = 50,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare exact compounds across sources and separately against rescue."""
    required = {"source", "compound", "drug_col", "external_drug"}
    if not required <= set(compound_mapping):
        raise ValueError(
            "Compound mapping requires source, compound, "
            "drug_col, external_drug"
        )
    if compound_mapping.duplicated(["source", "compound"]).any():
        raise ValueError(
            "Compound mappings must be unique per source and compound"
        )
    agreements: list[dict[str, object]] = []
    rescues: list[dict[str, object]] = []
    for (study, compound), group in panel.groupby(
        ["study_id", "compound"], sort=True
    ):
        vector = group.set_index("gene_id")["delta"]
        if not vector.index.is_unique:
            raise ValueError("Panel study/compound/gene keys must be unique")
        for source, signatures in external.items():
            mapped = compound_mapping.loc[
                compound_mapping["source"].eq(source)
                & compound_mapping["compound"].eq(str(compound))
            ]
            base: dict[str, object] = {
                "study_id": study,
                "compound": compound,
                "source": source,
                "comparison_type": "exact_compound_response",
            }
            if mapped.empty:
                agreements.append(base | {"status": "compound_mapping_missing"})
                continue
            mapping = mapped.iloc[0]
            drug_col = str(mapping["drug_col"])
            if drug_col not in signatures.meta:
                raise ValueError(f"External drug column absent: {drug_col}")
            positions = np.flatnonzero(
                signatures.meta[drug_col]
                .eq(mapping["external_drug"])
                .to_numpy()
            )
            if not len(positions):
                agreements.append(base | {"status": "compound_not_available"})
            for position in positions:
                row = signatures.delta[int(position)]
                values = (
                    sp.csr_matrix(row).toarray()
                    if sp.issparse(row)
                    else np.asarray(row)
                ).reshape(-1)
                score = score_mimicry(
                    values,
                    signatures.genes,
                    vector.to_numpy(),
                    vector.index.tolist(),
                    minimum_shared_genes=minimum_shared_genes,
                )
                context = signatures.meta.iloc[position]
                aligned = (
                    pd.Series(values, index=list(signatures.genes))
                    .rename("external")
                    .to_frame()
                    .join(vector.rename("panel"), how="inner")
                    .dropna()
                )
                agreement = (
                    float(
                        np.mean(
                            np.sign(aligned["external"])
                            == np.sign(aligned["panel"])
                        )
                    )
                    if score.n_shared >= minimum_shared_genes
                    else np.nan
                )
                agreements.append(
                    base
                    | {
                        "signature_id": str(
                            context.get(
                                "signature_id",
                                signatures.meta.index[int(position)],
                            )
                        ),
                        "context_metadata": context.to_json(),
                        "pearson": score.pearson,
                        "spearman": score.spearman,
                        "directional_agreement": agreement,
                        "n_shared_genes": score.n_shared,
                        "n_panel_genes": len(vector),
                        "status": score.status,
                    }
                )
        for state, rescue in rescue_vectors.items():
            score = score_mimicry(
                vector.to_numpy(),
                vector.index.tolist(),
                rescue.to_numpy(),
                rescue.index.tolist(),
                minimum_shared_genes=minimum_shared_genes,
            )
            connectivity = weighted_cmap_connectivity(
                vector.to_numpy(),
                vector.index.tolist(),
                rescue.to_numpy(),
                rescue.index.tolist(),
                minimum_shared_genes=minimum_shared_genes,
            )
            rescues.append(
                {
                    "study_id": study,
                    "compound": compound,
                    "state": state,
                    "comparison_type": "surgery_alignment",
                    "pearson": score.pearson,
                    "spearman": score.spearman,
                    "connectivity": connectivity.score_connectivity,
                    "n_shared_genes": score.n_shared,
                    "status": score.status,
                    "connectivity_status": connectivity.status,
                }
            )
    return pd.DataFrame(agreements), pd.DataFrame(rescues)
