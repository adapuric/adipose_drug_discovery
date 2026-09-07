# PerturbGen run scripts

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

To train the masking model on HX1 and then run its downstream stages:

```bash
PROJECT_DIR=/gpfs/home/ap5625/add
REPO_DIR="${PROJECT_DIR}/adipose_drug_discovery"
CONFIG_PATH="${REPO_DIR}/run_scripts/perturbgen/config.yaml"

qsub -v CONFIG_PATH="${CONFIG_PATH}" \
  "${REPO_DIR}/run_scripts/perturbgen/train_masking_model.pbs"

# Select one of the five lowest-loss masking checkpoints printed by training.
MASKING_CHECKPOINT="${PROJECT_DIR}/pg_results/adipocytes_obese_weightloss_hvg/masking/checkpoints/selected.ckpt"

qsub \
  -v CONFIG_PATH="${CONFIG_PATH}",MASKING_CHECKPOINT="${MASKING_CHECKPOINT}" \
  "${REPO_DIR}/run_scripts/perturbgen/train_decoder.pbs"
qsub \
  -v CONFIG_PATH="${CONFIG_PATH}",MASKING_CHECKPOINT="${MASKING_CHECKPOINT}" \
  "${REPO_DIR}/run_scripts/perturbgen/embedding_extraction.pbs"

# Select one of the five lowest-loss decoder checkpoints printed by training.
DECODER_CHECKPOINT="${PROJECT_DIR}/pg_results/adipocytes_obese_weightloss_hvg/decoder/checkpoints/selected.ckpt"
PERTURBATION_GENE=ENSG00000131459

qsub \
  -v CONFIG_PATH="${CONFIG_PATH}",DECODER_CHECKPOINT="${DECODER_CHECKPOINT}",PERTURBATION_GENE="${PERTURBATION_GENE}" \
  "${REPO_DIR}/run_scripts/perturbgen/run_perturbation.pbs"
```

To run several independent single-gene perturbations sequentially in one PBS
job, pass a colon-separated `PERTURBATION_GENES` value. PBS reserves commas for
separating variables supplied through `qsub -v`.

```bash
PERTURBATION_GENES="ENSG00000131459:ENSG00000123456:ENSG00000198765"

qsub \
  -v CONFIG_PATH="${CONFIG_PATH}",DECODER_CHECKPOINT="${DECODER_CHECKPOINT}",PERTURBATION_GENES="${PERTURBATION_GENES}" \
  "${REPO_DIR}/run_scripts/perturbgen/run_perturbation.pbs"
```

Use `PERTURBATION_GENE` or `PERTURBATION_GENES`, not both. If neither is set,
the job uses `perturbation.genes_to_perturb` from the YAML configuration.
