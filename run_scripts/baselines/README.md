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
CONFIG_PATH=/path/to/baselines.yaml

qsub run_scripts/baselines/download_data.pbs
qsub -v CONFIG_PATH="$CONFIG_PATH" run_scripts/baselines/prepare_rescue.pbs

# These jobs are independent after prepare_rescue.pbs succeeds.
qsub -v CONFIG_PATH="$CONFIG_PATH" run_scripts/baselines/run_tahoe.pbs
qsub -v CONFIG_PATH="$CONFIG_PATH" run_scripts/baselines/run_lincs.pbs
```

The download job stores Tahoe plates in `data/tahoe/` and LINCS files directly
in `data/`. `prepare_rescue.pbs` builds donor-level pseudobulk and paired rescue
vectors once. The Tahoe job then runs train-mean, PCA-ridge, and mean-drug;
the LINCS job runs CMap.

The workflow jobs activate the `add` Conda environment. The download job only
requires `wget` and `gzip`, so it does not activate Conda. All jobs stream output
to `job_out/<job-name>.<job-id>.live.log`, and results are written beneath
`baselines/`.
