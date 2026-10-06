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

Masking and decoder training save one loss-labelled checkpoint per epoch. When
training finishes, each wrapper prints the five newly created checkpoints with
the lowest loss values so that a checkpoint can be selected for the next
stage.

`run.run_name` keeps independent configs separate:

```text
pg_results/<run_name>/
├── prepared_adipocytes.h5ad    # prepare stage only
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

## Other datasets

To run a non-adipose H5AD, omit the `prepare` section, set
`tokenize.input_h5ad_path`, and leave `prepare` out of `STAGES`. The H5AD needs
raw counts in `X`, Ensembl IDs in `var["ensembl_id"]`, and the obs columns your
config names.

```yaml
run:
  run_name: lps_tutorial
  results_root_directory: /path/to/pg_results

# No prepare section.

tokenize:
  input_h5ad_path: /path/to/full_lps.h5ad
  gene_filtering_mode: hvg
  highly_variable_gene_count: 2000
  time_obs_col: time_after_LPS
  main_pairing_obs: cell_type_harmonized
  reference_time: normal
  time_point_order: [normal, 90m_LPS, 6h_LPS, 10h_LPS]
  # Remaining tokenize keys as in config.yaml.
```

```bash
qsub -v CONFIG_PATH=/path/to/config.yaml,STAGES=tokenize:masking \
  run_scripts/perturbgen/pbs/prepare_tokenize_train_masking_model.pbs
```

Full example: `configs/config_LPS_reproducing_hvg.yaml`.

Notes:

- `highly_variable_gene_count` sets `--n_hvg` (default 2000). In "hvg" mode
  PerturbGen writes `dataset_2000_hvg_*` instead of `dataset_all_*`, and the
  wrappers follow it.
- Set `masking.sampling_keys` if `masking.use_weighted_sampler` is true.
- `conditioning_obs_cols: []` trains without condition tokens.
- `run_perturbation` still needs a `prepare` section for its donor/state
  summaries.


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

# Run selected stages, in workflow order, with a colon-separated STAGES value.
# Command-line -l options override the job's resource directives.
qsub -l walltime=72:00:00 \
  -v CONFIG_PATH="${CONFIG_PATH}",STAGES=tokenize:masking \
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

A `tgt` run edits the gene in the target-condition sequence at every
`model.predicted_time_points` value, passed to PerturbGen as `pert_tps`. For
"mask", "pad", and "delete", PerturbGen keeps only pairs whose target nucleus
expresses the gene at those time points.
