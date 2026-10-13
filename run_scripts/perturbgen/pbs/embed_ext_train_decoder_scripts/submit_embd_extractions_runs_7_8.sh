#!/bin/bash
PROJECT_DIR=/gpfs/home/ap5625/add
REPO_DIR="${PROJECT_DIR}/adipose_drug_discovery"

CONFIG_PATHS=(
   "${REPO_DIR}/run_scripts/perturbgen/configs/config_run7_threshold_noAD1.yaml"
   "${REPO_DIR}/run_scripts/perturbgen/configs/config_run8_threshold_merged_noAD1.yaml"
)
MASKING_CHECKPOINTS=(
   "${PROJECT_DIR}/pg_results/adipocytes_obese_weightloss_run7_threshold_noAD1/masking/checkpoints/20260928_1414_cellgen_train_masking_lr_0.0001_wd_0.0001_batch_32_ptime_pos_sin_m_pow_tp_1_s_42-epoch_19-loss_6.50106811523.ckpt"
   "${PROJECT_DIR}/pg_results/adipocytes_obese_weightloss_run8_threshold_merged_noAD1/masking/checkpoints/20260928_1433_cellgen_train_masking_lr_0.0001_wd_0.0001_batch_32_ptime_pos_sin_m_pow_tp_1_s_42-epoch_19-loss_6.05125665665.ckpt"
)
for i in "${!CONFIG_PATHS[@]}"; do
  CONFIG_PATH="${CONFIG_PATHS[$i]}"
  MASKING_CHECKPOINT="${MASKING_CHECKPOINTS[$i]}"
  qsub -v CONFIG_PATH="${CONFIG_PATH}",MASKING_CHECKPOINT="${MASKING_CHECKPOINT}" "${REPO_DIR}/run_scripts/perturbgen/pbs/embedding_extraction_highmem.pbs"
done
