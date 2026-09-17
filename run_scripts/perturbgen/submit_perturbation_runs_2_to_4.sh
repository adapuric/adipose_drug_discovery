PROJECT_DIR=/gpfs/home/ap5625/add
REPO_DIR="${PROJECT_DIR}/adipose_drug_discovery"

CONFIG_PATHS=(
  "${REPO_DIR}/run_scripts/perturbgen/config_run2_HVG_clean.yaml"
  "${REPO_DIR}/run_scripts/perturbgen/config_run3_HVG_mergedannot.yaml"
  "${REPO_DIR}/run_scripts/perturbgen/config_run4_HVG_noAD1.yaml"
)

DECODER_CHECKPOINTS=(
  "${PROJECT_DIR}/pg_results/adipocytes_run2_obese_weightloss_hvg_clean/decoder/checkpoints/20260916_1004_cellgen_train_count_lr_0.001_wd_0.001_batch_16_drop_0.1_zinb_tp_1_s_42_pos_time_pos_sin_m_cosine-epoch_01-loss_1231.67663574.ckpt"
  "${PROJECT_DIR}/pg_results/adipocytes_run3_obese_weightloss_hvg_mergedannot/decoder/checkpoints/20260916_1004_cellgen_train_count_lr_0.001_wd_0.001_batch_16_drop_0.1_zinb_tp_1_s_42_pos_time_pos_sin_m_cosine-epoch_01-loss_1232.62097168.ckpt"
  "${PROJECT_DIR}/pg_results/adipocytes_run4_obese_weightloss_hvg_noAD1/decoder/checkpoints/20260916_1004_cellgen_train_count_lr_0.001_wd_0.001_batch_16_drop_0.1_zinb_tp_1_s_42_pos_time_pos_sin_m_cosine-epoch_02-loss_1294.24951172.ckpt"
)

PERTURBATION_GENES="ENSG00000105835:ENSG00000124762:ENSG00000131459"
# these are: NAMPT : CDKN1A : GFPT2 - all submitted in parallel 

for i in "${!CONFIG_PATHS[@]}"; do
  CONFIG_PATH="${CONFIG_PATHS[$i]}"
  DECODER_CHECKPOINT="${DECODER_CHECKPOINTS[$i]}"
  qsub -v CONFIG_PATH="${CONFIG_PATH}",DECODER_CHECKPOINT="${DECODER_CHECKPOINT}",PERTURBATION_GENES="${PERTURBATION_GENES}" "${REPO_DIR}/run_scripts/perturbgen/run_perturbation.pbs"
done
