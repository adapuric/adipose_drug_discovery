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
  --perturbation-config /path/to/perturbation.yaml
```

Every model-launching script supports `--dry-run` which prints the resolved
command without executing it.

`run.run_name` keeps independent configs separate:

```text
pg_results/<run_name>/
├── prepared_adipocytes.h5ad
├── masking/
├── decoder/
└── embeddings/
```

PerturbGen creates tokenized data in `tokenized_data/<run_name>` beneath its
native sibling directory.

Set `prepare.subset_to_highly_variable_genes` to `true` to retain genes marked
`true` in the source H5AD column named by
`prepare.highly_variable_gene_col`. This uses the existing gene annotation; it
does not recalculate highly variable genes. Use a distinct `run.run_name` when
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

# Replace this with the masking checkpoint selected from the preceding job.
MASKING_CHECKPOINT="${PROJECT_DIR}/pg_results/adipocytes_obese_weightloss_hvg/masking/checkpoints/selected.ckpt"

qsub \
  -v CONFIG_PATH="${CONFIG_PATH}",MASKING_CHECKPOINT="${MASKING_CHECKPOINT}" \
  "${REPO_DIR}/run_scripts/perturbgen/train_decoder.pbs"
qsub \
  -v CONFIG_PATH="${CONFIG_PATH}",MASKING_CHECKPOINT="${MASKING_CHECKPOINT}" \
  "${REPO_DIR}/run_scripts/perturbgen/embedding_extraction.pbs"
```

The HX1 PBS jobs use
`/gpfs/home/ap5625/miniforge3/envs/perturbgen` and source shared PBS functions
from `/gpfs/home/sho3/pbs_common.sh`.
`CONFIG_PATH` must be absolute so its meaning does not depend on the scheduler
working directory. Omitting it uses the absolute repository default shown
above.

The configured masking command includes
`--cond_list cell_states_adipocytes`. This is the alias that preparation creates
from the source H5AD's `cell_state_t2d` annotation, and it must remain in
`model.retained_obs_cols` and `model.conditioning_obs_cols`. Confirm the
resolved command before submission with:

```bash
python -m run_scripts.perturbgen.train_masking \
  --config "${CONFIG_PATH}" \
  --dry-run
```

### HX1 masking-training inputs

When reusing the existing tokenization, the following must be present on HX1:

- the `adipose_drug_discovery` checkout and the PerturbGen `adipose` branch;
- `/gpfs/home/ap5625/miniforge3/envs/perturbgen` with the required Geneformer
  revision described above;
- `/gpfs/home/sho3/pbs_common.sh` and the configured OpenMPI module;
- `data/perturbgen_encoder.ckpt` beneath the ADD project directory;
- the complete
  `T_perturb/tokenized_data/adipocytes_obese_weightloss_hvg` directory,
  including `dataset_all_src/obese.dataset`, `dataset_all_tgt`,
  `h5ad_pairing_all_src/obese.h5ad`, `h5ad_pairing_all_tgt`,
  `token_id_to_genename_all.pkl`, and `tokenid_to_rowid_all.pkl`; and
- a writable `pg_results/adipocytes_obese_weightloss_hvg/masking` output path.

The original `adipocytes_annotated_step2.h5ad`, `mart_export.txt`, and the three
tokenization dictionaries are needed only if preparation or tokenization must
be rerun. Do not resume from a masking checkpoint created with a different
vocabulary allocation; leave `masking.resume_checkpoint_path` as `null` for a
fresh compatible model.

Model jobs default to offline Weights & Biases logging, so they do not require
an API key. After configuring W&B authentication, override this with
`qsub -v WANDB_MODE=online,...`; use `WANDB_MODE=disabled` to turn logging off.

Decoder training and embedding extraction both use the masking checkpoint, so
their jobs can run at the same time. After they finish, run the perturbation
command shown above with the native PerturbGen YAML.
