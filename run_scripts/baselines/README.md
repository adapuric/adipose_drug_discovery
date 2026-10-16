# Baseline PBS jobs

Cluster layout:

```text
add/
├── adipose_drug_discovery/
├── data/
│   ├── adipocytes_annotated_step2.h5ad
│   ├── tahoe/
│   └── GSE70138_Broad_LINCS_*
├── baselines/
└── job_out/
```

Submit from the repository root:

```bash
PROJECT_DIR=/rds/general/user/sho3/projects/lms-scott-raw/live/steve/add
REPO_DIR="${PROJECT_DIR}/adipose_drug_discovery"
CONFIG_PATH="${REPO_DIR}/config/baselines.yaml"

qsub "${REPO_DIR}/run_scripts/baselines/download_data.pbs"
qsub -v CONFIG_PATH="${CONFIG_PATH}" \
  "${REPO_DIR}/run_scripts/baselines/prepare_rescue.pbs"

# After prepare_rescue.pbs finishes:
qsub -v CONFIG_PATH="${CONFIG_PATH}" \
  "${REPO_DIR}/run_scripts/baselines/run_tahoe.pbs"
qsub -v CONFIG_PATH="${CONFIG_PATH}" \
  "${REPO_DIR}/run_scripts/baselines/run_lincs.pbs"
qsub -v REPORT=transition \
  "${REPO_DIR}/run_scripts/baselines/run_report.pbs"
```

- `download_data.pbs`: Tahoe plates to `data/tahoe/`, LINCS files to `data/`.
  Needs only `wget` and `gzip`.
- `prepare_rescue.pbs`: donor pseudobulk and paired rescue vectors.
- `run_tahoe.pbs`: train-mean, PCA-ridge and mean-drug.
- `run_lincs.pbs`: CMap.
- `run_report.pbs`: one CPU report, picked by `REPORT`: `transition` (reads
  `baselines/pseudobulk/`), `panel` or `perturbgen`. `panel` and `perturbgen`
  need an absolute `CONFIG_PATH`. `DRY_RUN=1` only validates inputs. Writes to
  `baselines/reports/<REPORT>/<job id>/` unless `REPORT_DIR` is set.

`CONFIG_PATH` must be absolute and defaults to the repo config above. Logs go
to `job_out/<job-name>.<job-id>.live.log`.

Jobs use the RDS env `software/envs/perturbgen` through `~/pbs_common.sh`; no
GPU. Install the repo into it once:

```bash
source /rds/general/user/sho3/projects/lms-scott-raw/live/steve/software/conda/base/etc/profile.d/conda.sh
conda activate /rds/general/user/sho3/projects/lms-scott-raw/live/steve/software/envs/perturbgen
python -m pip install -e "${REPO_DIR}"
add --help
```

## Methods

- `train-mean`: average training response, no drug identity.
- `mean-drug`: each drug's measured responses, equal weight per context.
- `pca-ridge`: additive model of control-expression PCs plus drug identity. No
  drug-by-context interactions.
- `cmap`: weighted bidirectional connectivity, median over eligible measured
  signatures. Not Broad normalized tau.

Positive scores mimic the paired weightloss - baseline rescue. Rank ties keep
gene-ID order; constant candidates are non-estimable. CMap applies
`minimum_shared_genes` to its primary score and keeps invalid contexts as
missing, with a reason.

The CMap scatter is
the best measured context, not the median score; the mean-drug scatter is the
drug mean. Train-mean has no drug rows, so it only writes
`figure_manifest.json`; mean-drug and PCA-ridge rankings draw it as a
reference line.

### Evaluation

`pca_ridge.evaluation_group_cols` holds out whole cell lines, separately from
the vehicle-matching `tahoe.context_cols`. Every evaluated model writes
`evaluation_split.csv`. PCA-ridge also writes:

- `matched_evaluation.csv`: two comparators on its test rows and genes,
  `train_mean_*` and `drug_mean_*` (the drug's mean over its training contexts;
  `unseen_drug` if none). PCA-ridge has to beat the drug mean for control
  expression to matter.
- `evaluation_summary.csv`: comparator - model RMSE and model - comparator
  Pearson, so positive favours PCA-ridge. Held-out cell lines are the units;
  `pca_ridge.bootstrap_draws` resamples them for a
  `pca_ridge.confidence_level` interval.

This tests known-drug transfer across cell lines, not adipose transfer or
unseen drugs.

### Comparing baselines

Scores are only comparable on the same genes. Mean-drug and PCA-ridge rankings
carry `train_mean_score` and `score_minus_train_mean` (positive: more
rescue-like than a generic Tahoe perturbation; descriptive). The PCA-ridge job
rescores mean-drug and train-mean on its genes and writes:

- `matched_rankings.csv`: one row per state and drug.
- `matched_ranking_agreement.csv`: Spearman of drug scores, overlap of the top
  `pca_ridge.comparison_top_n`, and the fraction above train-mean.

CMap isn't matched: LINCS compounds would need a validated name mapping to
Tahoe.

### Data notes

- LINCS QC filters are opt-in: `quality_col` + `minimum_quality` and/or
  `high_quality_col` + `high_quality_value`. GSE70138 has no QC fields; join
  extra QC by signature ID first. Bad config fails, missing QC values fail an
  active filter, and provenance says `not_configured` without one. Level-5
  signatures keep their scale and landmark filtering.
- Tahoe `drugname_drugconc` stays an exact label, not a numeric dose. Sample,
  dose and time grouping are separate from vehicle matching. Mean-drug averages
  treatment conditions within a context. Missing targets/MOA stay missing.
- All signatures come from cancer cell lines.

## Human adipose panel

Config: `config/adipose_validation.yaml`, with the sample/compound manifests
next to it. Download, then run:

```bash
mkdir -p data/adipose_validation/GSE122721 data/adipose_validation/GSE71293
curl -fL https://ftp.ncbi.nlm.nih.gov/geo/series/GSE122nnn/GSE122721/matrix/GSE122721_series_matrix.txt.gz -o data/adipose_validation/GSE122721/series_matrix.txt.gz
curl -fL https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPL16nnn/GPL16522/soft/GPL16522_family.soft.gz -o data/adipose_validation/GSE122721/platform.soft.gz
curl -fL https://ftp.ncbi.nlm.nih.gov/geo/series/GSE71nnn/GSE71293/matrix/GSE71293_series_matrix.txt.gz -o data/adipose_validation/GSE71293/series_matrix.txt.gz
curl -fL https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPL13nnn/GPL13497/soft/GPL13497_family.soft.gz -o data/adipose_validation/GSE71293/platform.soft.gz
python -m run_scripts.baselines.validate_adipose_panel --dry-run
python -m run_scripts.baselines.validate_adipose_panel
```

- GSE71293: hMADS adipocytes, normalized log2 sample/reference ratios.
  Unpaired, four culture experiments per condition.
- GSE122721: two donors (technical duplicates averaged within donor), but
  blocked. The values don't match the stated log2 scale and the sample
  descriptions contradict the drug labels. Needs corrected metadata or raw-array
  processing, plus a gene-symbol mapping for its Entrez probes. Keep
  `blocked_reason` until then.

Set `external` (source to cache prefix) and `rescue_path` to compare with the
external sources. Compounds must match exactly; missing ones are reported,
never substituted. Outputs: study and unit signatures, mapping/QC tables,
every external context comparison, median agreement and rescue alignment.
Provenance records input hashes, protocols, exclusions and study status.

## Donor-held-out surgery benchmark

Set the pseudobulk path in `config/adipose_benchmark.yaml` or pass
`--pseudobulk` (`run_report.pbs` does):

```bash
python -m run_scripts.baselines.evaluate_adipose_transition --dry-run
python -m run_scripts.baselines.evaluate_adipose_transition
```

Leave-one-donor-out; all of a donor's conditions and states share a fold.
Profiles are summed raw counts in log1p-CPM, then paired changes. Predictors:

- no change
- mean shift: average training-donor change
- adipose ridge: training-only PCA/ridge from baseline expression (defaults:
  up to 2,000 genes, 5 PCs, alpha 10; no held-out tuning). Separate from the
  drug `pca-ridge`.

Outputs: `predictions.parquet`, `folds.csv`, `fold_genes.parquet`,
`metrics.csv`, `paired_model_comparisons.csv` and `summary.csv`
(donor-bootstrap intervals, not p-values). Unsupported models stay
non-estimable; error metrics still work when a constant prediction leaves
correlation undefined.

## PerturbGen naive edit baselines

The PerturbGen report compares every edit with three model-free predictions:

- `no-downstream-effect`: zero change outside the edited genes.
- `generic-edit-mean`: per donor, state and gene, the mean effect of the other
  edits in the same run, mode and sequence. Ignores which gene was edited, like
  train-mean.
- `coexpression-contrast`: per donor and state in the edited condition,
  log1p-CPM of summed raw counts from nuclei lacking the edited gene minus
  nuclei expressing it, from the model's input H5AD. Overexpression edits are
  `unsupported_edit_mode`.


Edited
genes are excluded, genes need `minimum_reference_cpm` in the unedited
prediction, and donors are the units. Unedited `pred_counts` aren't compared
with the weightloss profile: inference feeds the observed target tokens to the
decoder, so they're a reconstruction.

Build the run manifest from finished perturbation directories. If HX1 results
aren't visible from CX3, copy the summaries and model input first:

```bash
RUN=adipocytes_obese_weightloss_run7_threshold_noAD1
mkdir -p "${PROJECT_DIR}/pg_results"
rsync -av --prune-empty-dirs \
  --include='*/' \
  --include='prepared_adipocytes.h5ad' \
  --include='perturbation/*/*/summary/donor_state_effects.csv.gz' \
  --exclude='*' \
  "<hx1-login-host>:/gpfs/home/ap5625/add/pg_results/${RUN}" \
  "${PROJECT_DIR}/pg_results/"

cd "${REPO_DIR}"
python -m run_scripts.perturbgen.build_report_manifest \
  --config run_scripts/perturbgen/configs/config_run7_threshold_noAD1.yaml \
  --results-root "${PROJECT_DIR}/pg_results" \
  --output "${PROJECT_DIR}/pg_results/report_runs.json"
```

- Repeat `--config` to combine runs.
- `edited_condition` comes from `reference_time` (source edits) or the
  predicted time point (target edits).
- `prepared_adipocytes.h5ad` becomes `coexpression_h5ad_path`; edits with no
  output are logged.
- `--source native` lists native result H5ADs instead of summaries.
- The co-expression contrast rejects regressed `X` (`prepare.regress_out_obs`);
  point `coexpression_h5ad_path` at an unregressed input with the same nuclei.

Copy `config/perturbgen_visualization.yaml`, set `run_manifest` to the JSON,
and submit:

```bash
qsub -v REPORT=perturbgen,CONFIG_PATH=/abs/path/perturbgen_visualization.yaml \
  "${REPO_DIR}/run_scripts/baselines/run_report.pbs"
```

Adds `naive_edit_baselines.parquet`, per-donor `edit_baseline_agreement.csv`
(Pearson/Spearman, RMS effect sizes and a status) and donor-bootstrap
`edit_baseline_agreement_summary.csv`. Thresholds live under
`naive_baselines`; set it to null to skip. For many native H5ADs, request more
memory (`qsub -l select=1:ncpus=4:mem=128gb`).

## Report scripts

All report scripts take `--config`, `--output-dir` and `--dry-run`. Relative
config paths resolve from the config directory's parent; output overrides
from the current directory. Reports refuse non-empty destinations and write
provenance only after all outputs succeed, so use a new destination when
settings change.
