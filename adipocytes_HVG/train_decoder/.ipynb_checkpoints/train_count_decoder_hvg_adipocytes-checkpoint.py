
import subprocess
import sys


# GPU required
COUNT_OUTPUT_DIR = "/rds/general/project/lms-scott-raw/live/Ada/perturbation_modeling/T_perturb/res_hvg/count"  # relative to repo root, here set the output directory

# this is the fine tuned masking checkpoint (from the masking step output)
CKPT_MASKING_PATH = "/rds/general/project/lms-scott-raw/live/Ada/perturbation_modeling/T_perturb/res_hvg/checkpoints/20260626_1513_cellgen_train_masking_lr_0.0001_wd_0.0001_batch_64_ptime_pos_sin_m_pow_tp_1_s_0-epoch=19.ckpt" # should be selected based on best masking model from previous step

SRC_DATASET = "/rds/general/project/lms-scott-raw/live/Ada/perturbation_modeling/T_perturb/tokenized_data/hvg_adipocytes_obese_WL/dataset_all_src/obese.dataset" # path/to/src_dataset.dataset from tokenization step
TGT_DATASET_FOLDER = "/rds/general/project/lms-scott-raw/live/Ada/perturbation_modeling/T_perturb/tokenized_data/hvg_adipocytes_obese_WL/dataset_all_tgt" # path/to/tgt_datasets_folder from tokenization step
SRC_ADATA = "/rds/general/project/lms-scott-raw/live/Ada/perturbation_modeling/T_perturb/tokenized_data/hvg_adipocytes_obese_WL/h5ad_pairing_all_src/obese.h5ad" # path/to/src_adata.h5ad" from tokenization step
TGT_ADATA_FOLDER = "/rds/general/project/lms-scott-raw/live/Ada/perturbation_modeling/T_perturb/tokenized_data/hvg_adipocytes_obese_WL/h5ad_pairing_all_tgt"  # path/to/tgt_adata_folder from tokenization step
MAPPING_DICT_PATH = "/rds/general/project/lms-scott-raw/live/Ada/perturbation_modeling/T_perturb/tokenized_data/hvg_adipocytes_obese_WL/token_id_to_genename_all.pkl" # path/to/mapping_dict.pkl from tokenization step

# this is different to the checkpoint above - this is the original pre-trained Geneformer Checkpoint 
ENCODER_PATH = "/rds/general/project/lms-scott-raw/live/Ada/perturbation_modeling/Perturbgen/pretraining_cohort/20250709_1223_cellgen_train_masking_lr_5e-05_wd_1e-06_batch_64_ptime_pos_sin_m_pow_tp_1-2-3_s_42-epoch=00.ckpt" # path to pretrained encoder checkpoint provided with Perturbgen

BATCH_SIZE = 16 # Model training batch size
EPOCHS = 16 # number of training epochs - same as in their tutorial 
COUNT_LR = 0.001 # learning rate for count model
CELLGEN_LR = 0.0001 # learning rate for masking part, not useful here
CELLGEN_WD = 0.0001 # weight decay for masking part, not useful here
COUNT_WD = 0.001 # weight decay for count model
MLM_PROB = 0.30 # masking probability
N_WORKERS = 16 # number of data loading workers - it was 32 from tutorial - try 16 (match it in pbs)
NUM_LAYERS = 6 # number of transformer layers
D_FF = 32 # feedforward dimension
LOSS_MODE = "zinb" # loss mode for count model, could be mse, nb, or zinb
PRED_TPS = ["1"] # time points to train on and predict 

VAR_LIST = ["cell_states_adipocytes", "condition"] # list of obs retained in adata.vars after preprocessing

COUNT_DROPOUT = 0.1 # dropout for count model
USE_POSITIONAL_ENCODING = "False" # whether to use positional encoding in count model
LAYER_NORM = "True" # whether to use layer normalization in count model
CONTEXT_MODE = "True" # whether to use context tokens
POS_ENCODING_MODE = "time_pos_sin" # positional encoding mode
MASK_SCHEDULER = "cosine" # masking scheduler type
NUM_NODE = 1 # number of nodes if using distributed training
D_MODEL = 768 # model dimension
ADD_CELL_TIME = "False"
D_CONDC = 64
D_CONDT = 768





cmd = [
    sys.executable,
    "-m",
    "perturbgen",
    "train-decoder",
    "--train_mode", "count",
    "--split", "False",
    "--splitting_mode", "stratified",
    "--split_obs", "cell_states_adipocytes",  # add here
    "--output_dir", COUNT_OUTPUT_DIR,
    "--ckpt_masking_path", CKPT_MASKING_PATH,
    "--src_dataset", SRC_DATASET,
    "--tgt_dataset_folder", TGT_DATASET_FOLDER,
    "--src_adata", SRC_ADATA,
    "--tgt_adata_folder", TGT_ADATA_FOLDER,
    "--mapping_dict_path", MAPPING_DICT_PATH,
    "--batch_size", str(BATCH_SIZE),
    "--epochs", str(EPOCHS),
    "--count_lr", str(COUNT_LR),
    "--cellgen_lr", str(CELLGEN_LR),
    "--cellgen_wd", str(CELLGEN_WD),
    "--count_wd", str(COUNT_WD),
    "--mlm_prob", str(MLM_PROB),
    "--n_workers", str(N_WORKERS),
    "--num_layers", str(NUM_LAYERS),
    "--d_ff", str(D_FF),
    "--loss_mode", LOSS_MODE,
    "--pred_tps", *PRED_TPS,
    "--var_list", *VAR_LIST,
    "--encoder", "scmaskgit",
    "--cond_list", "cell_states_adipocytes",
    "--count_dropout", str(COUNT_DROPOUT),
    "--use_positional_encoding", USE_POSITIONAL_ENCODING,
    "--layer_norm", LAYER_NORM,
    "--context_mode", CONTEXT_MODE,
    "--encoder_path", ENCODER_PATH,
    "--pos_encoding_mode", POS_ENCODING_MODE,
    "--mask_scheduler", MASK_SCHEDULER,
    "--num_node", str(NUM_NODE),
    "--d_model", str(D_MODEL),
    "--seed", "0",
    "--ckpt_every_n_epochs", "5",
]
print(" ".join(cmd))



subprocess.run(cmd, check=True)
