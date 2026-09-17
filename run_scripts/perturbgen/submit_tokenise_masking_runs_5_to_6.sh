PROJECT_DIR=/gpfs/home/ap5625/add
REPO_DIR="${PROJECT_DIR}/adipose_drug_discovery"

CONFIG_PATHS=(
  "${REPO_DIR}/run_scripts/perturbgen/config_run5_threshold.yaml"
  "${REPO_DIR}/run_scripts/perturbgen/config_run6_threshold_noAD1.yaml"
)

for CONFIG_PATH in "${CONFIG_PATHS[@]}"; do
  qsub -v CONFIG_PATH="${CONFIG_PATH}" "${REPO_DIR}/run_scripts/perturbgen/prepare_tokenize_train_masking_model_72hrs.pbs"
done
