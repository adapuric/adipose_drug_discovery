import subprocess
import sys

OUTPUT_DIR = "/rds/general/project/lms-scott-raw/live/Ada/perturbation_modeling/T_perturb/res"  # relative to repo root, here set the output directory
SRC_DATASET = "/rds/general/project/lms-scott-raw/live/Ada/perturbation_modeling/T_perturb/tokenized_data/adipocytes_subset_obese_WL/dataset_all_src/obese.dataset"  # path/to/src_dataset.dataset from tokenization step
TGT_DATASET_FOLDER = "/rds/general/project/lms-scott-raw/live/Ada/perturbation_modeling/T_perturb/tokenized_data/adipocytes_subset_obese_WL/dataset_all_tgt"  # path/to/tgt_datasets_folder from tokenization step
SRC_ADATA = "/rds/general/project/lms-scott-raw/live/Ada/perturbation_modeling/T_perturb/tokenized_data/adipocytes_subset_obese_WL/h5ad_pairing_all_src/obese.h5ad"  # path/to/src_adata.h5ad" from tokenization step
TGT_ADATA_FOLDER = "/rds/general/project/lms-scott-raw/live/Ada/perturbation_modeling/T_perturb/tokenized_data/adipocytes_subset_obese_WL/h5ad_pairing_all_tgt"  # path/to/tgt_adata_folder from tokenization step
MAPPING_DICT_PATH = "/rds/general/project/lms-scott-raw/live/Ada/perturbation_modeling/T_perturb/tokenized_data/adipocytes_subset_obese_WL/token_id_to_genename_all.pkl"  # path/to/mapping_dict.pkl from tokenization step

ENCODER_PATH = "/rds/general/project/lms-scott-raw/live/Ada/perturbation_modeling/Perturbgen/pretraining_cohort/20250709_1223_cellgen_train_masking_lr_5e-05_wd_1e-06_batch_64_ptime_pos_sin_m_pow_tp_1-2-3_s_42-epoch=00.ckpt"  # path to pretrained encoder checkpoint provided with Perturbgen

BATCH_SIZE = 64  # Model training batch size
EPOCHS = 20  # number of training epochs
CELLGEN_LR = 1e-4  # learning rate
CELLGEN_WD = 1e-4  # weight decay
N_WORKERS = 4  # number of data loading workers
NUM_LAYERS = 6  # number of transformer layers
D_FF = 32  # feedforward dimension
PRED_TPS = [
    "1"
]  # time points to train on and predict (for LPS: "1"=90m, "2"=6h, "3"=10h) - in my case 1 is the only non reference time point (i.e. weight loss)

VAR_LIST = [
    "cell_states_adipocytes",
    "condition",
]  # list of obs retained in adata.vars after preprocessing

SEED = 0  # random seed for reproducibility
CONTEXT_MODE = "True"  # whether to use context tokens
POS_ENCODING_MODE = "time_pos_sin"  # positional encoding mode
MASK_SCHEDULER = "pow"  # masking scheduler type
NUM_NODE = 1  # number of nodes if using distributed training
D_MODEL = 768  # model dimension

CKPT_MASKING_PATH = None  # optional path to checkpoint to resume training from
USE_WEIGHTED_SAMPLER = (
    "False"  # whether to use weighted sampler during training
)

# MAX_LEN = 344      # actual max sequence length in your dataset
# TGT_VOCAB_SIZE = 1466


cmd = [
    sys.executable,
    "-m",
    "perturbgen",
    "train-mask",
    "--train_mode",
    "masking",
    "--split",
    "False",
    "--encoder",
    "scmaskgit",
    "--splitting_mode",
    "stratified",
    "--split_obs",
    "cell_states_adipocytes",
    "--cond_list",
    "cell_states_adipocytes",
    "--output_dir",
    OUTPUT_DIR,
    "--src_dataset",
    SRC_DATASET,
    "--tgt_dataset_folder",
    TGT_DATASET_FOLDER,
    "--src_adata",
    SRC_ADATA,
    "--tgt_adata_folder",
    TGT_ADATA_FOLDER,
    "--mapping_dict_path",
    MAPPING_DICT_PATH,
    "--batch_size",
    str(BATCH_SIZE),
    "--epochs",
    str(EPOCHS),
    "--cellgen_lr",
    str(CELLGEN_LR),
    "--cellgen_wd",
    str(CELLGEN_WD),
    "--n_workers",
    str(N_WORKERS),
    "--num_layers",
    str(NUM_LAYERS),
    "--d_ff",
    str(D_FF),
    "--pred_tps",
    *PRED_TPS,
    "--var_list",
    *VAR_LIST,
    "--encoder_path",
    ENCODER_PATH,
    "--seed",
    str(SEED),
    "--context_mode",
    CONTEXT_MODE,
    "--pos_encoding_mode",
    POS_ENCODING_MODE,
    "--mask_scheduler",
    MASK_SCHEDULER,
    "--num_node",
    str(NUM_NODE),
    "--d_model",
    str(D_MODEL),
    "--use_weighted_sampler",
    USE_WEIGHTED_SAMPLER,
]

if CKPT_MASKING_PATH:
    cmd += ["--ckpt_masking_path", CKPT_MASKING_PATH]

print(" ".join(cmd))


subprocess.run(cmd, check=True)
