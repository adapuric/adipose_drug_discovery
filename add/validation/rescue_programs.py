"""Split the paired rescue signature into named gene-set programs.

The weightloss - baseline rescue vector can mix biological programs, such as
lipid synthesis, with changes in how samples were collected, such as an
immediate-early stress response. These functions describe which programs
carry the rescue signal, follow program scores within donors, and split a
drug's rescue-mimicry correlation into additive contributions from disjoint
gene groups. Program membership comes from curated gene sets, never from this
data. All results are descriptive and hypothesis-generating.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from add.perturb import PerturbSignatures
from add.rescue import rescue_vector


def read_gene_sets(path: str | Path) -> dict[str, tuple[str, ...]]:
    """Read a GMT file into gene-set names mapped to unique gene symbols.

    Each line holds a set name, a description and the set's genes, separated
    by tabs. Duplicate set names and empty sets are rejected.
    """
    gene_sets: dict[str, tuple[str, ...]] = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 3 or not fields[0]:
                raise ValueError(
                    f"GMT line {line_number} needs a name, description and "
                    f"genes: {path}"
                )
            name = fields[0]
            genes = tuple(dict.fromkeys(gene for gene in fields[2:] if gene))
            if name in gene_sets:
                raise ValueError(f"Duplicate gene set {name!r} in {path}")
            if not genes:
                raise ValueError(f"Gene set {name!r} has no genes in {path}")
            gene_sets[name] = genes

    if not gene_sets:
        raise ValueError(f"No gene sets were read from {path}")
    return gene_sets


def program_genes(
    gene_sets: Mapping[str, Sequence[str]],
    programs: Mapping[str, Sequence[str]],
) -> dict[str, frozenset[str]]:
    """Return each program's genes as the union of its named gene sets.

    "shared" and "other" are reserved for the decomposition's gene groups.
    """
    resolved: dict[str, frozenset[str]] = {}
    for program, names in programs.items():
        if program in {"shared", "other"}:
            raise ValueError(f"Program name {program!r} is reserved")
        if not names:
            raise ValueError(f"Program {program!r} names no gene sets")
        if missing := [name for name in names if name not in gene_sets]:
            raise KeyError(
                f"Program {program!r} names unknown gene sets: {missing}"
            )
        resolved[program] = frozenset(
            gene for name in names for gene in gene_sets[name]
        )
    return resolved


def summarize_rescue_programs(
    rescue_results: pd.DataFrame,
    gene_sets: Mapping[str, Sequence[str]],
    *,
    minimum_genes: int = 5,
) -> pd.DataFrame:
    """Describe how each gene set moves in each state's rescue vector.

    Uses the tested moderated t statistics that drug scores are computed
    against. `mean_t_shift` is the set's mean t minus the mean t of tested
    genes outside it: negative means the set falls after weightloss relative
    to other genes. Genes are not independent, so no p-values are given.
    """
    rows: list[dict[str, object]] = []
    for state in sorted(rescue_results["state"].astype(str).unique()):
        vector = rescue_vector(rescue_results, state=state)
        for name, genes in gene_sets.items():
            member = vector.index.isin(list(genes))
            row: dict[str, object] = {
                "state": state,
                "gene_set": name,
                "n_set_genes": len(set(genes)),
                "n_tested": int(member.sum()),
                "mean_t": np.nan,
                "median_t": np.nan,
                "fraction_negative": np.nan,
                "background_mean_t": np.nan,
                "mean_t_shift": np.nan,
                "status": "insufficient_genes",
            }
            if member.sum() >= minimum_genes and (~member).any():
                inside = vector.loc[member]
                outside = vector.loc[~member]
                row.update(
                    mean_t=float(inside.mean()),
                    median_t=float(inside.median()),
                    fraction_negative=float((inside < 0).mean()),
                    background_mean_t=float(outside.mean()),
                    mean_t_shift=float(inside.mean() - outside.mean()),
                    status="ok",
                )
            rows.append(row)
    return pd.DataFrame(rows)


def donor_program_scores(
    pseudobulk: ad.AnnData,
    programs: Mapping[str, frozenset[str]],
    *,
    donor_col: str,
    condition_col: str,
    carried_cols: Sequence[str] = (),
    count_layer: str | None = None,
    minimum_genes: int = 5,
) -> pd.DataFrame:
    """Score programs in each donor's whole-adipocyte profile per condition.

    Raw counts are summed over all adipocyte-state rows of a donor and
    condition, then converted to log1p CPM. A program score is the mean, over
    the program's variable genes, of each gene's z score across all profiles,
    so scores compare profiles, not genes. `carried_cols` (for example study
    or BMI) must have one value per donor and condition.

    Returns:
      Long table with `donor_id`, `condition`, the carried columns,
      `program`, `score`, `n_genes` and `status`.
    """
    obs = cast(pd.DataFrame, pseudobulk.obs)
    keys = [donor_col, condition_col]
    if missing := [c for c in [*keys, *carried_cols] if c not in obs]:
        raise KeyError(f"Pseudobulk metadata lacks columns: {missing}")
    if obs[keys].isna().any().any():
        raise ValueError("Donor and condition labels must not be missing")

    labels = obs[keys].astype(str)
    group_codes = labels.groupby(keys, sort=True).ngroup().to_numpy()
    metadata = (
        labels.drop_duplicates()
        .sort_values(keys, ignore_index=True)
        .set_axis(["donor_id", "condition"], axis="columns")
    )
    profiles = _summed_log1p_cpm(
        pseudobulk, group_codes, n_groups=len(metadata), layer=count_layer
    )
    for column in carried_cols:
        values = obs[column].groupby(group_codes, sort=True)
        if (values.nunique(dropna=True) > 1).any():
            raise ValueError(
                f"{column} varies within a donor/condition profile"
            )
        metadata[column] = values.first().to_numpy()

    spread = profiles.std(axis=0, ddof=1)
    variable = spread > 0
    z_scores = (
        profiles.loc[:, variable] - profiles.loc[:, variable].mean()
    ) / (spread.loc[variable])

    frames: list[pd.DataFrame] = []
    for program, genes in programs.items():
        members = z_scores.columns[z_scores.columns.isin(list(genes))]
        supported = len(members) >= minimum_genes
        frames.append(
            metadata.assign(
                program=program,
                score=z_scores[members].mean(axis=1).to_numpy()
                if supported
                else np.nan,
                n_genes=len(members),
                status="ok" if supported else "insufficient_genes",
            )
        )
    return pd.concat(frames, ignore_index=True)


def paired_program_changes(
    scores: pd.DataFrame,
    *,
    baseline_label: str,
    weightloss_label: str,
    covariate_cols: Sequence[str] = (),
) -> pd.DataFrame:
    """Return each donor's weightloss - baseline program score change.

    Only donors scored in both conditions are kept, so the donor is the unit
    of every change. Numeric covariates such as BMI change in the same
    direction and appear as `<covariate>_change`; other carried columns must
    agree between the two conditions.
    """
    paired = scores.loc[
        scores["condition"].isin([baseline_label, weightloss_label])
    ]
    constant_cols = [
        column
        for column in scores.columns
        if column
        not in {
            "donor_id",
            "condition",
            "program",
            "score",
            "n_genes",
            "status",
            *covariate_cols,
        }
    ]
    by_condition = paired.set_index(["donor_id", "program", "condition"])
    baseline = by_condition.xs(baseline_label, level="condition")
    weightloss = by_condition.xs(weightloss_label, level="condition")
    joined = baseline.join(
        weightloss, how="inner", lsuffix="_baseline", rsuffix="_weightloss"
    )
    for column in constant_cols:
        if not joined[f"{column}_baseline"].equals(
            joined[f"{column}_weightloss"]
        ):
            raise ValueError(
                f"{column} differs between conditions for a paired donor"
            )

    result = joined[[f"{column}_baseline" for column in constant_cols]]
    result.columns = constant_cols
    result = result.assign(
        change=joined["score_weightloss"] - joined["score_baseline"]
    )
    for column in covariate_cols:
        result[f"{column}_change"] = pd.to_numeric(
            joined[f"{column}_weightloss"]
        ) - pd.to_numeric(joined[f"{column}_baseline"])
    return result.reset_index()


def summarize_program_changes(
    changes: pd.DataFrame,
    *,
    group_col: str | None = None,
) -> pd.DataFrame:
    """Summarize paired program changes over donors, overall and by group.

    Rows labelled "all" pool every paired donor; others restrict to one value
    of `group_col`, such as a study.
    """
    frames = [changes.assign(_group="all")]
    if group_col is not None:
        frames.append(changes.assign(_group=changes[group_col].astype(str)))

    rows: list[dict[str, object]] = []
    for (group, program), values in pd.concat(frames).groupby(
        ["_group", "program"], sort=True
    ):
        change = values["change"].dropna()
        rows.append(
            {
                "group": group,
                "program": program,
                "n_donors": len(change),
                "mean_change": float(change.mean()),
                "median_change": float(change.median()),
                "fraction_decreasing": float((change < 0).mean()),
                "fraction_increasing": float((change > 0).mean()),
            }
        )
    return pd.DataFrame(rows)


def program_change_correlations(
    changes: pd.DataFrame,
    *,
    covariate_cols: Sequence[str] = (),
) -> pd.DataFrame:
    """Correlate program and covariate changes across paired donors.

    Each pair of quantities gets a Spearman correlation over the donors with
    both values; the donor is the unit. Descriptive only.
    """
    wide = changes.pivot(index="donor_id", columns="program", values="change")
    for column in covariate_cols:
        wide[f"{column}_change"] = changes.groupby("donor_id")[
            f"{column}_change"
        ].first()

    rows: list[dict[str, object]] = []
    columns = list(wide.columns)
    for position, first in enumerate(columns):
        for second in columns[position + 1 :]:
            pair = wide[[first, second]].dropna()
            rho = np.nan
            if len(pair) >= 3 and pair.nunique().min() > 1:
                rho = float(pair[first].corr(pair[second], method="spearman"))
            rows.append(
                {
                    "first": first,
                    "second": second,
                    "spearman": rho,
                    "n_donors": len(pair),
                }
            )
    return pd.DataFrame(rows)


def decompose_program_mimicry(
    signatures: PerturbSignatures,
    rescue_vectors: Mapping[str, pd.Series],
    programs: Mapping[str, frozenset[str]],
    *,
    excluded_program: str,
    label_col: str = "drug",
    minimum_shared_genes: int = 10,
) -> pd.DataFrame:
    """Split each signature's rescue-mimicry correlation by gene group.

    In each state the signature and rescue vector are joined on gene ID.
    Every shared gene belongs to one group: the program containing it,
    "shared" when several programs contain it, or "other". With both vectors
    centred over all shared genes, the Pearson correlation is a sum over
    genes, so the per-group "contribution" rows add up exactly to the "full"
    score. "restricted" rows correlate only one program's genes (re-centred),
    and the "without_<excluded_program>" row drops that program's genes.
    Positive values mimic the rescue.

    Returns:
      Long table with `signature_id`, `label_col`, `state`, `measure`
      ("full", "contribution" or "restricted"), `part`, `value`, `n_genes`
      and `status`.
    """
    if excluded_program not in programs:
        raise KeyError(f"Excluded program {excluded_program!r} is undefined")
    if label_col not in signatures.meta:
        raise KeyError(f"Signature metadata lacks {label_col!r}")
    if minimum_shared_genes < 3:
        raise ValueError("minimum_shared_genes must be at least 3")

    genes = pd.Index(signatures.genes)
    identity = {
        "signature_id": signatures.meta.index.astype(str).to_numpy(),
        label_col: signatures.meta[label_col].astype(str).to_numpy(),
    }
    frames: list[pd.DataFrame] = []
    for state, rescue in rescue_vectors.items():
        aligned = rescue.reindex(genes)
        shared = np.isfinite(aligned.to_numpy(dtype=float))
        matrix = np.asarray(signatures.delta)[:, shared]
        if not np.isfinite(matrix).all():
            raise ValueError("Signatures contain non-finite shared genes")
        target = aligned.to_numpy(dtype=float)[shared]
        shared_genes = genes[shared]

        masks = {
            program: _program_mask(shared_genes, members)
            for program, members in programs.items()
        }
        masks[f"without_{excluded_program}"] = ~masks[excluded_program]
        parts = _correlation_parts(
            matrix,
            target,
            groups=_gene_groups(shared_genes, programs),
            minimum_genes=minimum_shared_genes,
        )
        parts += _restricted_parts(
            matrix, target, masks=masks, minimum_genes=minimum_shared_genes
        )
        frames.extend(
            pd.DataFrame(identity | {"state": str(state)} | part)
            for part in parts
        )
    return pd.concat(frames, ignore_index=True)


def program_ranking_agreement(
    decomposition: pd.DataFrame,
    *,
    label_col: str = "drug",
    top_n: int = 20,
) -> pd.DataFrame:
    """Compare drug rankings by the full score and each restricted score.

    Spearman correlation and top-`top_n` overlap use drugs estimable under
    both scores in a state. High agreement means removing or isolating that
    program barely changes which drugs look rescue-like.
    """
    full = decomposition.loc[decomposition["measure"].eq("full")]
    restricted = decomposition.loc[decomposition["measure"].eq("restricted")]
    rows: list[dict[str, object]] = []
    for (state, part), scores in restricted.groupby(["state", "part"]):
        merged = full.loc[full["state"].eq(str(state))].merge(
            scores, on=[label_col, "state"], suffixes=("_full", "")
        )
        both = merged.loc[
            merged["status_full"].eq("ok") & merged["status"].eq("ok")
        ]
        compared = min(top_n, len(both))
        overlap = len(
            set(both.nlargest(compared, "value_full")[label_col])
            & set(both.nlargest(compared, "value")[label_col])
        )
        rows.append(
            {
                "state": state,
                "restricted_part": part,
                "n_drugs": len(both),
                "spearman_with_full": float(
                    both["value_full"].corr(both["value"], method="spearman")
                )
                if len(both) >= 3
                else np.nan,
                "top_n": compared,
                "top_n_overlap": overlap / compared if compared else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _summed_log1p_cpm(
    pseudobulk: ad.AnnData,
    group_codes: np.ndarray,
    *,
    n_groups: int,
    layer: str | None,
) -> pd.DataFrame:
    """Sum raw counts by group with a sparse indicator, then log1p CPM."""
    counts = pseudobulk.X if layer is None else pseudobulk.layers[layer]
    if counts is None:
        raise ValueError("Pseudobulk contains no count matrix")
    indicator = sp.csr_matrix(
        (
            np.ones(len(group_codes)),
            (group_codes, np.arange(len(group_codes))),
        ),
        shape=(n_groups, len(group_codes)),
    )
    summed = np.asarray(
        sp.csr_matrix(indicator @ sp.csr_matrix(counts)).toarray(), dtype=float
    )
    if not np.isfinite(summed).all() or (summed < 0).any():
        raise ValueError("Pseudobulk counts must be finite and nonnegative")

    sizes = summed.sum(axis=1, keepdims=True)
    if (sizes <= 0).any():
        raise ValueError("A donor/condition profile has an empty library")
    return pd.DataFrame(
        np.log1p(summed / sizes * 1_000_000),
        columns=pseudobulk.var_names.astype(str),
    )


def _program_mask(genes: pd.Index, program: frozenset[str]) -> np.ndarray:
    """Return which genes belong to one program."""
    return genes.isin(list(program))


def _gene_groups(
    genes: pd.Index, programs: Mapping[str, frozenset[str]]
) -> np.ndarray:
    """Assign each gene to its single program, "shared" or "other"."""
    memberships = np.column_stack(
        [_program_mask(genes, members) for members in programs.values()]
    )
    names = np.asarray(list(programs), dtype=object)
    counts = memberships.sum(axis=1)
    groups = np.full(len(genes), "other", dtype=object)
    single = counts == 1
    groups[single] = names[memberships[single].argmax(axis=1)]
    groups[counts > 1] = "shared"
    return groups


def _correlation_parts(
    matrix: np.ndarray,
    target: np.ndarray,
    *,
    groups: np.ndarray,
    minimum_genes: int,
) -> list[dict[str, object]]:
    """Return the full correlation and its additive group contributions."""
    values, status = _row_correlations(
        matrix, target, minimum_genes=minimum_genes
    )
    centred = matrix - matrix.mean(axis=1, keepdims=True)
    reference = target - target.mean()
    scale = np.linalg.norm(centred, axis=1) * np.linalg.norm(reference)
    usable = status == "ok"

    parts: list[dict[str, object]] = [
        {
            "measure": "full",
            "part": "full",
            "value": values,
            "n_genes": len(target),
            "status": status,
        }
    ]
    for group in np.unique(groups):
        member = groups == group
        contribution = np.full(len(matrix), np.nan)
        contribution[usable] = (
            centred[usable][:, member] @ reference[member] / scale[usable]
        )
        parts.append(
            {
                "measure": "contribution",
                "part": str(group),
                "value": contribution,
                "n_genes": int(member.sum()),
                "status": status,
            }
        )
    return parts


def _restricted_parts(
    matrix: np.ndarray,
    target: np.ndarray,
    *,
    masks: Mapping[str, np.ndarray],
    minimum_genes: int,
) -> list[dict[str, object]]:
    """Return Pearson correlations re-centred within each gene mask."""
    parts: list[dict[str, object]] = []
    for name, mask in masks.items():
        values, status = _row_correlations(
            matrix[:, mask], target[mask], minimum_genes=minimum_genes
        )
        parts.append(
            {
                "measure": "restricted",
                "part": name,
                "value": values,
                "n_genes": int(mask.sum()),
                "status": status,
            }
        )
    return parts


def _row_correlations(
    matrix: np.ndarray,
    target: np.ndarray,
    *,
    minimum_genes: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Correlate every row with `target`, keeping non-estimable rows NaN."""
    values = np.full(len(matrix), np.nan)
    status = np.full(len(matrix), "ok", dtype=object)
    if len(target) < minimum_genes:
        status[:] = "insufficient_shared_genes"
        return values, status
    if np.ptp(target) == 0:
        status[:] = "constant_rescue"
        return values, status

    constant = np.ptp(matrix, axis=1) == 0
    status[constant] = "constant_candidate"
    centred = matrix[~constant] - matrix[~constant].mean(axis=1, keepdims=True)
    reference = target - target.mean()
    values[~constant] = (
        centred
        @ reference
        / (np.linalg.norm(centred, axis=1) * np.linalg.norm(reference))
    )
    return values, status
