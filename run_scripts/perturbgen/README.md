# PerturbGen run scripts

YAML configurations live in `configs/`; submission scripts and PBS jobs live
in `pbs/`. Python entry points remain in this directory and default to
`configs/config.yaml`.

The workflow is:

```text
prepare → tokenize → masking → decoder + embeddings → perturbation
```

From the repository root:

```bash
CONFIG_PATH=/path/to/config.yaml

python -m run_scripts.perturbgen.prepare_adipose --config "${CONFIG_PATH}"
python -m run_scripts.perturbgen.tokenize --config "${CONFIG_PATH}"
python -m run_scripts.perturbgen.train_masking --config "${CONFIG_PATH}"

# Select a masking checkpoint produced by the preceding stage.
python -m run_scripts.perturbgen.train_decoder \
  --config "${CONFIG_PATH}" \
  --checkpoint /path/to/masking.ckpt
python -m run_scripts.perturbgen.extract_embeddings \
  --config "${CONFIG_PATH}" \
  --checkpoint /path/to/masking.ckpt

python -m run_scripts.perturbgen.run_perturbation \
  --config "${CONFIG_PATH}" \
  --checkpoint /path/to/decoder.ckpt \
  --gene ENSG00000131459
```

Every model-launching script supports `--dry-run` which prints the resolved
command without executing it.

## Donor-aware result reports

The reporting script consumes existing native result H5ADs or historical
`summary/donor_state_effects.csv.gz` files. It does not launch the native model:

```bash
python -m run_scripts.perturbgen.plot_perturbation \
  --config config/perturbgen_visualization.yaml --dry-run
python -m run_scripts.perturbgen.plot_perturbation \
  --config config/perturbgen_visualization.yaml \
  --output-dir results/perturbgen_report_run1
```

Copy `config/perturbgen_report_runs.example.json` and configure explicit run/edit
IDs, edited genes in the native gene namespace, and a `result_path` (or
`summary_path`). Set the YAML `run_manifest` to that file. Unknown training
provenance remains unknown. Multiple edits from one run get a common report,
while different runs and expression scales remain separate. An optional
`gene_mapping_path` is a CSV with unique `gene_id` and `symbol` columns.

Native `X` is the edited count prediction and `layers["pred_counts"]` is the
unedited prediction. The adapter sums counts within donor/state, uses the
unedited library size for both profiles, and excludes edited coordinates from
downstream metrics. This prevents compositional renormalization from inventing
a downstream effect when only the edited gene changes. The no-downstream-effect
reference holds the unedited prediction fixed and has exactly zero downstream
change. Missing values remain missing.

Reports include an edit/readout heatmap, donor effects with conditional
donor-bootstrap intervals where supported, downstream magnitude versus zero,
and coverage QC. Each figure has a companion data table and manifest entry.
Readout lists are prespecified when configured; otherwise the top-20 selection
is labeled exploratory. Historical summary plots retain their original
log2-ratio-of-nucleus-means scale. Sampling draws are not independent donors.

`benchmark_path` optionally points to the independent surgery-transition report
produced by `run_scripts.baselines.evaluate_adipose_transition`. It adds donor
prediction-error comparisons and a change scatter. A native transition export
can join only when its manifest documents disjoint donor exposure at masking,
decoder, checkpoint selection, and preprocessing stages, plus absence of target
expression, token/order, and size-factor information. Specify these using
`training_donors_by_stage` and `target_information`; see the adapter's schema.
Each checkpoint/export declares `prediction_kind: unedited_transition`, exactly
one `fold_id`, `profile_identity`
(the reference pseudobulk SHA256), and `population_match_verified: true`.
`transition_predictions_path` must contain unique fold/donor/state/gene rows,
baseline/observed expression, predicted change, and scale matching the reference
benchmark. Gene support and observed values are checked before admission.
Paired improvements and their descriptive donor-bootstrap intervals use common
donor support and appear in `paired_model_comparisons.csv` and
`paired_comparison_summary.csv`.

The current training wrappers request `split=False`; inspected native inference
also receives target information. These outputs support descriptive sensitivity
checks, not a held-out target-blind leaderboard. Merely setting a split flag
does not establish donor isolation or remove target information.
`true_counts` represents ordinary observed target samples, not experimental
gene-edit outcomes. Deviation from the zero-effect reference is not evidence
that an edit is biologically correct. Experimental edited/control data would
be needed for that claim.

Masking and decoder training save one loss-labelled checkpoint per epoch. When
training finishes, each wrapper prints the five newly created checkpoints with
the lowest loss values so that a checkpoint can be selected for the next
stage.

`run.run_name` keeps independent configs separate:

```text
pg_results/<run_name>/
├── prepared_adipocytes.h5ad
├── masking/
├── decoder/
├── embeddings/
└── perturbation/
    └── mask_src/<gene>/
        ├── native_config.yaml
        ├── <PerturbGen result>.h5ad
        └── summary/
```

PerturbGen creates tokenized data in `tokenized_data/<run_name>` beneath its
native sibling directory.

Set `prepare.subset_to_highly_variable_genes` to `true` to retain genes marked
`true` in the source H5AD column named by
`prepare.highly_variable_gene_col`. Use a distinct `run.run_name` when
switching between full-gene and highly-variable-gene inputs.

Checkpoint paths can be written into `config.yaml` or passed with
`--checkpoint`; the command-line value takes precedence.

<br>

## One-time environment setup

PerturbGen currently imports Geneformer while loading the training module,
although this workflow does not otherwise use Geneformer. Install the tested
Geneformer revision into the dedicated PerturbGen environment:

```bash
ROOT=/gpfs/home/ap5625
PG_ENV="${ROOT}/miniforge3/envs/perturbgen"
GENEFORMER_SRC="${ROOT}/software/src/Geneformer"
GENEFORMER_REV=04c2b2e84da7c0f385c3f9ad8f3ec24bab6650e5

mkdir -p "${ROOT}/software/src"

GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 \
  https://huggingface.co/ctheodoris/Geneformer \
  "${GENEFORMER_SRC}"

git -C "${GENEFORMER_SRC}" fetch --depth 1 origin "${GENEFORMER_REV}"

GIT_LFS_SKIP_SMUDGE=1 git -C "${GENEFORMER_SRC}" \
  checkout --detach FETCH_HEAD

"${PG_ENV}/bin/python" -m pip install --no-deps "${GENEFORMER_SRC}"
```

<br>

## PBS submission

To prepare and tokenize the adipocytes, train the masking model, and
then run its downstream stages:

```bash
# Prepare environment variables
PROJECT_DIR=/gpfs/home/ap5625/add
REPO_DIR="${PROJECT_DIR}/adipose_drug_discovery"
CONFIG_PATH="${REPO_DIR}/run_scripts/perturbgen/configs/config.yaml"

# Submit prepare + masking model job
qsub -v CONFIG_PATH="${CONFIG_PATH}" \
  "${REPO_DIR}/run_scripts/perturbgen/pbs/prepare_tokenize_train_masking_model.pbs"

# Select lowest-loss masking checkpoint.
MASKING_CHECKPOINT="${PROJECT_DIR}/pg_results/adipocytes_obese_weightloss_hvg/masking/checkpoints/selected.ckpt"

# Decoder and embedding extraction jobs can be submitted once masking model is complete
# Submit decoder job
qsub \
  -v CONFIG_PATH="${CONFIG_PATH}",MASKING_CHECKPOINT="${MASKING_CHECKPOINT}" \
  "${REPO_DIR}/run_scripts/perturbgen/pbs/train_decoder.pbs"

# Submit embedding extraction job
qsub \
  -v CONFIG_PATH="${CONFIG_PATH}",MASKING_CHECKPOINT="${MASKING_CHECKPOINT}" \
  "${REPO_DIR}/run_scripts/perturbgen/pbs/embedding_extraction.pbs"

# Select lowest-loss decoder checkpoint.
DECODER_CHECKPOINT="${PROJECT_DIR}/pg_results/adipocytes_obese_weightloss_hvg/decoder/checkpoints/selected.ckpt"
PERTURBATION_GENE=ENSG00000131459

# Submit perturbation run
qsub \
  -v CONFIG_PATH="${CONFIG_PATH}",DECODER_CHECKPOINT="${DECODER_CHECKPOINT}",PERTURBATION_GENE="${PERTURBATION_GENE}" \
  "${REPO_DIR}/run_scripts/perturbgen/pbs/run_perturbation.pbs"
```

To run several independent single-gene perturbations sequentially in one PBS
job, pass a colon-separated `PERTURBATION_GENES` value. PBS reserves commas for
separating variables supplied through `qsub -v`.

```bash
PERTURBATION_GENES="ENSG00000131459:ENSG00000123456:ENSG00000198765"

qsub \
  -v CONFIG_PATH="${CONFIG_PATH}",DECODER_CHECKPOINT="${DECODER_CHECKPOINT}",PERTURBATION_GENES="${PERTURBATION_GENES}" \
  "${REPO_DIR}/run_scripts/perturbgen/pbs/run_perturbation.pbs"
```

Use `PERTURBATION_GENE` or `PERTURBATION_GENES`, not both. If neither is set,
the job uses `perturbation.genes_to_perturb` from the YAML configuration.

`PERTURBATION_MODE` and `PERTURBATION_SEQUENCE` override `perturbation.mode`
and `perturbation.sequence` for one job; `run_perturbation` accepts the same
values as `--mode` and `--sequence`. Each mode/sequence pair writes to its own
`perturbation/<mode>_<sequence>/` directory:

```bash
qsub \
  -v CONFIG_PATH="${CONFIG_PATH}",DECODER_CHECKPOINT="${DECODER_CHECKPOINT}",PERTURBATION_GENES="${PERTURBATION_GENES}",PERTURBATION_SEQUENCE=tgt \
  "${REPO_DIR}/run_scripts/perturbgen/pbs/run_perturbation.pbs"
```
