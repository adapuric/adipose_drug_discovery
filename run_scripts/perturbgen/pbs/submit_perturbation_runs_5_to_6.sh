#!/bin/bash

PROJECT_DIR=/gpfs/home/ap5625/add
REPO_DIR="${PROJECT_DIR}/adipose_drug_discovery"

CONFIG_PATHS=(
  "${REPO_DIR}/run_scripts/perturbgen/configs/config_run5_threshold.yaml"
  "${REPO_DIR}/run_scripts/perturbgen/configs/config_run6_threshold_noAD1.yaml"
)

DECODER_CHECKPOINTS=(
  "${PROJECT_DIR}/pg_results/adipocytes_obese_weightloss_run5_threshold/decoder/checkpoints/20260921_1040_cellgen_train_count_lr_0.001_wd_0.001_batch_16_drop_0.1_zinb_tp_1_s_42_pos_time_pos_sin_m_cosine-epoch_03-loss_6844.890625.ckpt"
  "${PROJECT_DIR}/pg_results/adipocytes_obese_weightloss_run6_noAD1_threshold/decoder/checkpoints/20260921_1039_cellgen_train_count_lr_0.001_wd_0.001_batch_16_drop_0.1_zinb_tp_1_s_42_pos_time_pos_sin_m_cosine-epoch_02-loss_6948.07177734.ckpt"
)

PERTURBATION_GENES="ENSG00000105835:ENSG00000124762:ENSG00000131459"
# these are: NAMPT : CDKN1A : GFPT2 - all submitted in parallel 

for i in "${!CONFIG_PATHS[@]}"; do
  CONFIG_PATH="${CONFIG_PATHS[$i]}"
  DECODER_CHECKPOINT="${DECODER_CHECKPOINTS[$i]}"
  qsub -v CONFIG_PATH="${CONFIG_PATH}",DECODER_CHECKPOINT="${DECODER_CHECKPOINT}",PERTURBATION_GENES="${PERTURBATION_GENES}" "${REPO_DIR}/run_scripts/perturbgen/pbs/run_perturbation.pbs"
done
