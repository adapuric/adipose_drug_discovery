#!/bin/bash

PROJECT_DIR=/gpfs/home/ap5625/add
REPO_DIR="${PROJECT_DIR}/adipose_drug_discovery"

CONFIG_PATHS=(
  "${REPO_DIR}/run_scripts/perturbgen/configs/config_run2B_HVG_cellstate_pairing.yaml"
)

MASKING_CHECKPOINTS=(
  "${PROJECT_DIR}/pg_results/adipocytes_run2B_obese_weightloss_hvg_cellstate_pairing/masking/checkpoints/20261008_1128_cellgen_train_masking_lr_0.0001_wd_0.0001_batch_64_ptime_pos_sin_m_pow_tp_1_s_42-epoch_19-loss_4.75989103317.ckpt"
)

for i in "${!CONFIG_PATHS[@]}"; do
  CONFIG_PATH="${CONFIG_PATHS[$i]}"
  MASKING_CHECKPOINT="${MASKING_CHECKPOINTS[$i]}"
  qsub -v CONFIG_PATH="${CONFIG_PATH}",MASKING_CHECKPOINT="${MASKING_CHECKPOINT}" "${REPO_DIR}/run_scripts/perturbgen/pbs/train_decoder.pbs"
  qsub -v CONFIG_PATH="${CONFIG_PATH}",MASKING_CHECKPOINT="${MASKING_CHECKPOINT}" "${REPO_DIR}/run_scripts/perturbgen/pbs/embedding_extraction.pbs"
done
