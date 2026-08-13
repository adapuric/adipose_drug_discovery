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

Submit from the repository root. Wait for each required stage to finish:

```bash
qsub run_scripts/baselines/pbs/download_data.pbs
qsub run_scripts/baselines/pbs/prepare_rescue.pbs

# These jobs are independent after prepare_rescue.pbs succeeds.
qsub run_scripts/baselines/pbs/run_tahoe.pbs
qsub run_scripts/baselines/pbs/run_lincs.pbs
```

The download job stores Tahoe plates in `data/tahoe/` and LINCS files directly
in `data/`. `prepare_rescue.pbs` builds donor-level pseudobulk and paired rescue
vectors once. The Tahoe job then runs perturbed-mean, PCA-ridge, and mean-drug;
the LINCS job runs CMap.

Outputs are written beneath `baselines/`.
