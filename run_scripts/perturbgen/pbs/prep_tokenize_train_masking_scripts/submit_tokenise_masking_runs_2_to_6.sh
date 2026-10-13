#!/bin/bash

PROJECT_DIR=/gpfs/home/ap5625/add
REPO_DIR="${PROJECT_DIR}/adipose_drug_discovery"

CONFIG_PATHS=(
  "${REPO_DIR}/run_scripts/perturbgen/configs/config_run2_HVG_clean.yaml"
  "${REPO_DIR}/run_scripts/perturbgen/configs/config_run3_HVG_mergedannot.yaml"
  "${REPO_DIR}/run_scripts/perturbgen/configs/config_run4_HVG_noAD1.yaml"
  "${REPO_DIR}/run_scripts/perturbgen/configs/config_run5_threshold.yaml"
  "${REPO_DIR}/run_scripts/perturbgen/configs/config_run6_threshold_noAD1.yaml"
)

for CONFIG_PATH in "${CONFIG_PATHS[@]}"; do
  qsub -v CONFIG_PATH="${CONFIG_PATH}" "${REPO_DIR}/run_scripts/perturbgen/pbs/prepare_tokenize_train_masking_model.pbs"
done
