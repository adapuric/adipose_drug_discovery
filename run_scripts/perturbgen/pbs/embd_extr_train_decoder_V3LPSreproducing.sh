#!/bin/bash

PROJECT_DIR=/gpfs/home/ap5625/add
REPO_DIR="${PROJECT_DIR}/adipose_drug_discovery"

CONFIG_PATHS=(
  "${REPO_DIR}/run_scripts/perturbgen/configs/config_V3_LPS_90m_source.yaml"
)

MASKING_CHECKPOINTS=(
  "${PROJECT_DIR}/pg_results/V3_LPS_90m_source_perturbation/masking/checkpoints/20261016_1829_cellgen_train_masking_lr_0.0001_wd_0.0001_batch_64_ptime_pos_sin_m_pow_tp_1-2_s_0-epoch_19-loss_4.17091941833.ckpt"
)

for i in "${!CONFIG_PATHS[@]}"; do
  CONFIG_PATH="${CONFIG_PATHS[$i]}"
  MASKING_CHECKPOINT="${MASKING_CHECKPOINTS[$i]}"
  qsub -v CONFIG_PATH="${CONFIG_PATH}",MASKING_CHECKPOINT="${MASKING_CHECKPOINT}" "${REPO_DIR}/run_scripts/perturbgen/pbs/train_decoder_72hrs.pbs"
  qsub -v CONFIG_PATH="${CONFIG_PATH}",MASKING_CHECKPOINT="${MASKING_CHECKPOINT}" "${REPO_DIR}/run_scripts/perturbgen/pbs/embedding_extraction_highmem.pbs"
done
