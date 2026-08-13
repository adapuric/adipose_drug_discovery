# Adipose Drug Discovery

*In-silico* perturbation modelling to identify drivers of adipose tissue dysfunction and candidate drug targets.

<br>

## Installation

```bash
conda create -n add python=3.11
conda activate add
python -m pip install -e .
```

<br>

## Data

Baseline perturbation and drug-prioritization models use Tahoe-100M and LINCS.
Keep Tahoe plates in their own directory so the baseline does not ingest the
adipose H5AD as a Tahoe shard.

To download the required drug-database files, run:

```bash
qsub run_scripts/baselines/pbs/download_data.pbs
```

See [`download_data.pbs`](run_scripts/baselines/pbs/download_data.pbs) for the
download job. All 14 Tahoe plates require approximately 328 GB.

<br>

## CLI for model baselines

```bash
add pseudobulk
add rescue

add perturb --source tahoe
add baseline --model perturbed-mean
add baseline --model pca-ridge
add baseline --model mean-drug

add perturb --source lincs
add baseline --model cmap
```

Mean-drug and CMap scoring use physical CPU. Pass
`--workers N` to match an HPC allocation or force a serial run with
`--workers 1`.

Use `add <command> --help` for path overrides.

## HPC

- [Baseline PBS jobs](run_scripts/baselines/README.md)
- [PerturbGen PBS jobs](run_scripts/perturbgen/README.md)
