PROJECT_DIR=/gpfs/home/ap5625/add
REPO_DIR="${PROJECT_DIR}/adipose_drug_discovery"

CONFIG_PATHS=(
  "${REPO_DIR}/run_scripts/perturbgen/config_run2_HVG_clean.yaml"
  "${REPO_DIR}/run_scripts/perturbgen/config_run3_HVG_mergedannot.yaml"
  "${REPO_DIR}/run_scripts/perturbgen/config_run4_HVG_noAD1.yaml"
)

MASKING_CHECKPOINTS=(
  "${PROJECT_DIR}/pg_results/adipocytes_run2_obese_weightloss_hvg_clean/masking/checkpoints/20260915_2111_cellgen_train_masking_lr_0.0001_wd_0.0001_batch_64_ptime_pos_sin_m_pow_tp_1_s_42-epoch_19-loss_4.79041194916.ckpt"
  "${PROJECT_DIR}/pg_results/adipocytes_run3_obese_weightloss_hvg_mergedannot/masking/checkpoints/20260915_2111_cellgen_train_masking_lr_0.0001_wd_0.0001_batch_64_ptime_pos_sin_m_pow_tp_1_s_42-epoch_19-loss_4.74240875244.ckpt"
  "${PROJECT_DIR}/pg_results/adipocytes_run4_obese_weightloss_hvg_noAD1/masking/checkpoints/20260915_2110_cellgen_train_masking_lr_0.0001_wd_0.0001_batch_64_ptime_pos_sin_m_pow_tp_1_s_42-epoch_19-loss_4.86603927612.ckpt"
)

for i in "${!CONFIG_PATHS[@]}"; do
  CONFIG_PATH="${CONFIG_PATHS[$i]}"
  MASKING_CHECKPOINT="${MASKING_CHECKPOINTS[$i]}"
  qsub -v CONFIG_PATH="${CONFIG_PATH}",MASKING_CHECKPOINT="${MASKING_CHECKPOINT}" "${REPO_DIR}/run_scripts/perturbgen/train_decoder.pbs"
  qsub -v CONFIG_PATH="${CONFIG_PATH}",MASKING_CHECKPOINT="${MASKING_CHECKPOINT}" "${REPO_DIR}/run_scripts/perturbgen/embedding_extraction.pbs"
done
