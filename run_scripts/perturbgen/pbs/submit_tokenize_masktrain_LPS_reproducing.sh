#!/bin/bash

PROJECT_DIR=/gpfs/home/ap5625/add
REPO_DIR="${PROJECT_DIR}/adipose_drug_discovery"

CONFIG_PATHS=(
  "${REPO_DIR}/run_scripts/perturbgen/configs/config_LPS_reproducing_hvg.yaml"
)

# The LPS config has no prepare section, so tokenization reads
# tokenize.input_h5ad_path directly.
for CONFIG_PATH in "${CONFIG_PATHS[@]}"; do
  qsub \
    -N LPS_tokenize_train_masking \
    -l select=1:ncpus=16:mem=256gb:ngpus=1:gpu_type=A100 \
    -l walltime=72:00:00 \
    -v CONFIG_PATH="${CONFIG_PATH}",STAGES=tokenize:masking \
    "${REPO_DIR}/run_scripts/perturbgen/pbs/prepare_tokenize_train_masking_model.pbs"
done
