PROJECT_DIR=/gpfs/home/ap5625/add
REPO_DIR="${PROJECT_DIR}/adipose_drug_discovery"

CONFIG_PATHS=(
  "${REPO_DIR}/run_scripts/perturbgen/config_run2_HVG_clean.yaml"
  "${REPO_DIR}/run_scripts/perturbgen/config_run3_HVG_mergedannot.yaml"
  "${REPO_DIR}/run_scripts/perturbgen/config_run4_HVG_noAD1.yaml"
  "${REPO_DIR}/run_scripts/perturbgen/config_run5_threshold.yaml"
  "${REPO_DIR}/run_scripts/perturbgen/config_run6_threshold_noAD1.yaml"
)

MASKING_CHECKPOINTS=(
  "${PROJECT_DIR}/pg_results/adipocytes_run2_obese_weightloss_hvg_clean/masking/checkpoints/selected.ckpt"
  "${PROJECT_DIR}/pg_results/adipocytes_run3_obese_weightloss_hvg_mergedannot/masking/checkpoints/selected.ckpt"
  "${PROJECT_DIR}/pg_results/adipocytes_run4_obese_weightloss_hvg_noAD1/masking/checkpoints/selected.ckpt"
  "${PROJECT_DIR}/pg_results/adipocytes_obese_weightloss_run5_threshold/masking/checkpoints/selected.ckpt"
  "${PROJECT_DIR}/pg_results/adipocytes_obese_weightloss_run6_noAD1_threshold/masking/checkpoints/selected.ckpt"
)

for CONFIG_PATH in "${CONFIG_PATHS[@]}"; do
  qsub -v CONFIG_PATH="${CONFIG_PATH}" "${REPO_DIR}/run_scripts/perturbgen/prepare_tokenize_train_masking_model.pbs"
done

for i in "${!CONFIG_PATHS[@]}"; do
  CONFIG_PATH="${CONFIG_PATHS[$i]}"
  MASKING_CHECKPOINT="${MASKING_CHECKPOINTS[$i]}"
  qsub -v CONFIG_PATH="${CONFIG_PATH}",MASKING_CHECKPOINT="${MASKING_CHECKPOINT}" "${REPO_DIR}/run_scripts/perturbgen/train_decoder.pbs"
  qsub -v CONFIG_PATH="${CONFIG_PATH}",MASKING_CHECKPOINT="${MASKING_CHECKPOINT}" "${REPO_DIR}/run_scripts/perturbgen/embedding_extraction.pbs"
done
