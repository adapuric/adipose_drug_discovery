#!/bin/bash

PROJECT_DIR=/gpfs/home/ap5625/add
REPO_DIR="${PROJECT_DIR}/adipose_drug_discovery"

CONFIG_PATHS=(
  "${REPO_DIR}/run_scripts/perturbgen/configs/config_run6B_threshold_pairing.yaml"
  "${REPO_DIR}/run_scripts/perturbgen/configs/config_run7B_thresh_pairing_noAD1.yaml"
)

MASKING_CHECKPOINTS=(
 "${PROJECT_DIR}/pg_results/adipocytes_run6B_obese_weightloss_thresh_cellstate_pairing/masking/checkpoints/20261008_1136_cellgen_train_masking_lr_0.0001_wd_0.0001_batch_32_ptime_pos_sin_m_pow_tp_1_s_42-epoch_19-loss_5.81047439575.ckpt"
 "${PROJECT_DIR}/pg_results/adipocytes_obese_weightloss_run7B_thresh_noAD1_cellstate_pairing/masking/checkpoints/20261008_1154_cellgen_train_masking_lr_0.0001_wd_0.0001_batch_32_ptime_pos_sin_m_pow_tp_1_s_42-epoch_18-loss_6.00010347366.ckpt"


)

for i in "${!CONFIG_PATHS[@]}"; do
  CONFIG_PATH="${CONFIG_PATHS[$i]}"
  MASKING_CHECKPOINT="${MASKING_CHECKPOINTS[$i]}"
  qsub -v CONFIG_PATH="${CONFIG_PATH}",MASKING_CHECKPOINT="${MASKING_CHECKPOINT}" "${REPO_DIR}/run_scripts/perturbgen/pbs/train_decoder_72hrs.pbs"
  qsub -v CONFIG_PATH="${CONFIG_PATH}",MASKING_CHECKPOINT="${MASKING_CHECKPOINT}" "${REPO_DIR}/run_scripts/perturbgen/pbs/embedding_extraction_highmem.pbs"
done
