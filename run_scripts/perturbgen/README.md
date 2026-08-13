# PerturbGen run scripts

The workflow is:

```text
prepare → tokenize → masking → decoder + embeddings → perturbation
```

From the repository root:

```bash
python -m run_scripts.perturbgen.prepare_adipose
python -m run_scripts.perturbgen.tokenize
python -m run_scripts.perturbgen.train_masking

# Select a masking checkpoint produced by the preceding stage.
python -m run_scripts.perturbgen.train_decoder \
  --checkpoint /path/to/masking.ckpt
python -m run_scripts.perturbgen.extract_embeddings \
  --checkpoint /path/to/masking.ckpt

python -m run_scripts.perturbgen.run_perturbation \
  --perturbation-config /path/to/perturbation.yaml
```

Every model-launching script supports `--dry-run` which prints the resolved
command without executing it. 

Every script accepts `--config` for a
different workflow YAML. `run.run_name` keeps independent runs separate:

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

## PBS submission

```bash
qsub adipose_drug_discovery/run_scripts/perturbgen/pbs/prepare_and_tokenize.pbs
qsub adipose_drug_discovery/run_scripts/perturbgen/pbs/train_masking_model_subset.pbs

# Replace this with the masking checkpoint selected from the preceding job.
MASKING_CHECKPOINT=/path/to/masking.ckpt

qsub -v MASKING_CHECKPOINT="${MASKING_CHECKPOINT}" \
  adipose_drug_discovery/run_scripts/perturbgen/pbs/train_count_decoder.pbs
qsub -v MASKING_CHECKPOINT="${MASKING_CHECKPOINT}" \
  adipose_drug_discovery/run_scripts/perturbgen/pbs/gene_embd_extraction_subset.pbs
```

Decoder training and embedding extraction both use the masking checkpoint, so
their jobs can run at the same time. After they finish, run the perturbation
command shown above with the native PerturbGen YAML.
