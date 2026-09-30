# Baseline PBS jobs

The jobs use the cluster project layout:

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

Set the configuration for the run, then submit from the repository root. Wait
for each required stage to finish:

```bash
PROJECT_DIR=/rds/general/user/sho3/projects/lms-scott-raw/live/steve/add
REPO_DIR="${PROJECT_DIR}/adipose_drug_discovery"
CONFIG_PATH="${REPO_DIR}/config/baselines.yaml"

qsub "${REPO_DIR}/run_scripts/baselines/download_data.pbs"
qsub -v CONFIG_PATH="${CONFIG_PATH}" \
  "${REPO_DIR}/run_scripts/baselines/prepare_rescue.pbs"

# These jobs are independent after prepare_rescue.pbs succeeds.
qsub -v CONFIG_PATH="${CONFIG_PATH}" \
  "${REPO_DIR}/run_scripts/baselines/run_tahoe.pbs"
qsub -v CONFIG_PATH="${CONFIG_PATH}" \
  "${REPO_DIR}/run_scripts/baselines/run_lincs.pbs"
```

The download job stores Tahoe plates in `data/tahoe/` and LINCS files directly
in `data/`. `prepare_rescue.pbs` builds donor-level pseudobulk and paired rescue
vectors once. The Tahoe job then runs train-mean, PCA-ridge, and mean-drug;
the LINCS job runs CMap.

The workflow jobs activate the `add` Conda environment. The download job only
requires `wget` and `gzip`, so it does not activate Conda. All jobs stream output
to `job_out/<job-name>.<job-id>.live.log`, and results are written beneath
`baselines/`.

`CONFIG_PATH` must be absolute so its meaning does not depend on the scheduler
working directory. Omitting it uses the absolute repository default shown
above.

## Methods and evaluation

`train-mean` predicts the average external training response without drug
identity. `mean-drug` averages measured drug responses with equal weight per
context. `pca-ridge` is an additive model of control-expression PCs and drug
identity; it does not learn drug-by-context interactions. `cmap` uses this
repository's weighted bidirectional connectivity and a median of eligible
measured signatures, not Broad normalized tau. Nonconstant rank ties retain
the deterministic gene-ID ordering; constant candidates are non-estimable.

Positive scores mimic the paired weightloss-minus-baseline rescue direction.
CMap applies `minimum_shared_genes` to its primary score, and retains invalid
contexts with missing values and a reason. Figures use the primary ranking
score. A best-context scatter is labeled explicitly; it is not the median
drug-level score.

`pca_ridge.evaluation_group_cols` holds out entire cell lines independently of
the vehicle-matching `tahoe.context_cols`. Each evaluated model writes
`evaluation_split.csv`. PCA-ridge additionally writes `matched_evaluation.csv`
with train-mean companion metrics on exactly its test rows and training-selected
genes. This tests known-drug transfer across external cell lines, not adipose
transfer or unseen-drug prediction. The model remains additive.

LINCS quality filters are opt-in: configure both `quality_col` and
`minimum_quality`, and/or `high_quality_col` with `high_quality_value`. Missing
or invalid configured fields fail explicitly. Missing QC values fail an active
filter; without one, provenance says `not_configured`. The configured
GSE70138 signature metadata release has no quality-score fields. Additional
QC metadata must be joined by validated signature IDs before enabling filters.
Level-5 signatures retain their supplied scale and measured-landmark filtering.

Tahoe's `drugname_drugconc` is retained as an exact treatment-condition label;
it is not coerced into an undocumented numeric dose. Sample, dose, and time
grouping remain distinct from vehicle matching. The mean-drug estimator still
averages treatment conditions within contexts. Missing targets/MOA remain
missing, and annotations are not target deconvolution. Cancer-line context
bias is a scientific limitation that annotation alone cannot remove.

## Small human adipose panel

The panel configuration is `config/adipose_validation.yaml`, with curated
sample/compound manifests alongside it. Download the processed expression and
platform tables into the configured paths:

```bash
mkdir -p data/adipose_validation/GSE122721 data/adipose_validation/GSE71293
curl -fL https://ftp.ncbi.nlm.nih.gov/geo/series/GSE122nnn/GSE122721/matrix/GSE122721_series_matrix.txt.gz -o data/adipose_validation/GSE122721/series_matrix.txt.gz
curl -fL https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPL16nnn/GPL16522/soft/GPL16522_family.soft.gz -o data/adipose_validation/GSE122721/platform.soft.gz
curl -fL https://ftp.ncbi.nlm.nih.gov/geo/series/GSE71nnn/GSE71293/matrix/GSE71293_series_matrix.txt.gz -o data/adipose_validation/GSE71293/series_matrix.txt.gz
curl -fL https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPL13nnn/GPL13497/soft/GPL13497_family.soft.gz -o data/adipose_validation/GSE71293/platform.soft.gz
python -m run_scripts.baselines.validate_adipose_panel --dry-run
python -m run_scripts.baselines.validate_adipose_panel
```

GSE71293 contains normalized log2 sample/reference ratios from human hMADS
adipocytes. The default unpaired analysis retains four independent culture
experiments per condition and does not invent donor pairing. GSE122721's two
donors would be paired with technical duplicates averaged within donor, but
its default numerical analysis is blocked: the deposited values conflict with
the stated log2 scale, and sample descriptions conflict with the drug labels.
Resolve these with corrected source metadata or separately validated raw-array
processing, then supply a canonical gene-symbol mapping for its Entrez probes.
Do not remove `blocked_reason` without resolving those conditions.

Set `external` to an explicit source-to-cache-prefix mapping and `rescue_path`
to the rescue table to enable comparisons. Exact compound identity is required;
missing compounds are reported, never replaced with related drugs. Outputs
include study and independent-unit signatures, mapping/QC tables, every external
context comparison, median agreement, and separate rescue alignment. Figures
show experimental-unit effects and external agreement. Input hashes, processing
protocols, exclusions, and study status are recorded in provenance.

This narrow PPAR-related panel is a context check, not broad pharmacological
validation or proof that a compound reproduces surgery.

## Donor-held-out surgery baseline benchmark

Configure the raw-count pseudobulk path in `config/adipose_benchmark.yaml`:

```bash
python -m run_scripts.baselines.evaluate_adipose_transition --dry-run
python -m run_scripts.baselines.evaluate_adipose_transition
```

All conditions/states from a donor share one leave-one-donor-out fold. Profiles
are sums of raw counts normalized to log1p-CPM before taking paired changes.
The three predictors are unchanged baseline, equal-training-donor mean shift,
and a small training-only PCA/ridge fit of change from baseline expression.
The adipose ridge is separate from the external drug `pca-ridge` model.
Default ridge settings are at most 2,000 input genes, five PCs, and alpha 10;
no held-out tuning is performed. Unsupported models remain non-estimable.

`predictions.parquet` contains named donor/state/gene predictions;
`folds.csv`, `fold_genes.parquet`, `metrics.csv`, and
`paired_model_comparisons.csv` make support and comparisons auditable. Error
metrics remain valid when a constant prediction makes correlation undefined.
`summary.csv` gives conditional donor-bootstrap intervals, not calibrated
p-values or confidence in gene-edit causality. Native PerturbGen comparison and
plotting are described in `run_scripts/perturbgen/README.md`.

All new report scripts accept `--config`, `--output-dir`, and `--dry-run`.
Configured relative paths resolve from the parent of the configuration
directory. Output overrides resolve from the current directory. Reports refuse
to overwrite nonempty destinations and publish completion provenance only after
required outputs succeed. Use a fresh destination when settings change.
