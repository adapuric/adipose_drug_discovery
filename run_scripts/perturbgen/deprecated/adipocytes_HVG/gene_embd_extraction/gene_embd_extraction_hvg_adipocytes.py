import subprocess
import sys


OUTPUT_DIR = "/rds/general/project/lms-scott-raw/live/Ada/perturbation_modeling/T_perturb/res_hvg/masking"  # relative to repo root, here set the output directory
CKPT_MASKING_PATH = "/rds/general/project/lms-scott-raw/live/Ada/perturbation_modeling/T_perturb/res_hvg/checkpoints/20260626_1513_cellgen_train_masking_lr_0.0001_wd_0.0001_batch_64_ptime_pos_sin_m_pow_tp_1_s_0-epoch=19.ckpt"  # same as in decoder training - path to load masking model checkpoint from masking training of notebook 03

SRC_DATASET = "/rds/general/project/lms-scott-raw/live/Ada/perturbation_modeling/T_perturb/tokenized_data/hvg_adipocytes_obese_WL/dataset_all_src/obese.dataset"  # path/to/src_dataset.dataset from tokenization step
TGT_DATASET_FOLDER = "/rds/general/project/lms-scott-raw/live/Ada/perturbation_modeling/T_perturb/tokenized_data/hvg_adipocytes_obese_WL/dataset_all_tgt"  # path/to/tgt_datasets_folder from tokenization step
SRC_ADATA = "/rds/general/project/lms-scott-raw/live/Ada/perturbation_modeling/T_perturb/tokenized_data/hvg_adipocytes_obese_WL/h5ad_pairing_all_src/obese.h5ad"  # path/to/src_adata.h5ad" from tokenization step
TGT_ADATA_FOLDER = "/rds/general/project/lms-scott-raw/live/Ada/perturbation_modeling/T_perturb/tokenized_data/hvg_adipocytes_obese_WL/h5ad_pairing_all_tgt"  # path/to/tgt_adata_folder from tokenization step
MAPPING_DICT_PATH = "/rds/general/project/lms-scott-raw/live/Ada/perturbation_modeling/T_perturb/tokenized_data/hvg_adipocytes_obese_WL/token_id_to_genename_all.pkl"  # path/to/mapping_dict.pkl from tokenization step

TOKENID_TO_ROWID = "/rds/general/project/lms-scott-raw/live/Ada/perturbation_modeling/T_perturb/tokenized_data/hvg_adipocytes_obese_WL/tokenid_to_rowid_all.pkl"  # path/to/tokenid_to_rowid.pkl from tokenization step

BATCH_SIZE = 64  # model batch size
CELLGEN_LR = 1e-4  # learning rate
CELLGEN_WD = 1e-4  # weight decay
COUNT_LR = 0.001  # learning rate for count head
COUNT_WD = 0.001  # weight decay for count head
D_FF = 32  # feedforward dimension
NUM_LAYERS = 6  # number of transformer layers
N_WORKERS = 16  # number of workers for data loading - this is same number as ncpus in the pbs file - will use more for full dataset
PRED_TPS = ["1"]  # predicted time points

VAR_LIST = ["cell_pairing_index", "cell_states_adipocytes", "condition"]

ENCODER = "scmaskgit"  # encoder type
ENCODER_PATH = "/rds/general/project/lms-scott-raw/live/Ada/perturbation_modeling/Perturbgen/pretraining_cohort/20250709_1223_cellgen_train_masking_lr_5e-05_wd_1e-06_batch_64_ptime_pos_sin_m_pow_tp_1-2-3_s_42-epoch=00.ckpt"  # path to pretrained encoder checkpoint provided with Perturbgen
CONTEXT_MODE = "True"  # whether to use context tokens
MASK_SCHEDULER = "pow"  # masking scheduler type
RETURN_EMBED = "True"  # whether to save cell embeddings
RETURN_ATTN = "False"  # whether to return attention weights
GENERATE = "False"  # whether to perform generation
RETURN_GENE_EMBS = "True"  # whether to return gene embeddings
GENE_EMBS_CONDITION = "condition"  # condition for gene embeddings
POS_ENCODING_MODE = "time_pos_sin"  # positional encoding mode
D_MODEL = 768  # model dimension


cmd = [
    sys.executable,  # python executable
    "-m",
    "perturbgen",
    "extract-embedding",
    "--test_mode",
    "masking",
    "--split",
    "False",
    "--splitting_mode",
    "stratified",
    "--cond_list",
    "cell_states_adipocytes",
    "--return_embed",
    RETURN_EMBED,
    "--return_attn",
    RETURN_ATTN,
    "--generate",
    GENERATE,
    "--ckpt_masking_path",
    CKPT_MASKING_PATH,
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
    "--cellgen_lr",
    str(CELLGEN_LR),
    "--cellgen_wd",
    str(CELLGEN_WD),
    "--count_lr",
    str(COUNT_LR),
    "--count_wd",
    str(COUNT_WD),
    "--d_ff",
    str(D_FF),
    "--num_layers",
    str(NUM_LAYERS),
    "--n_workers",
    str(N_WORKERS),
    "--pred_tps",
    *PRED_TPS,
    "--var_list",
    *VAR_LIST,
    "--tokenid_to_rowid",
    TOKENID_TO_ROWID,
    "--encoder",
    ENCODER,
    "--encoder_path",
    ENCODER_PATH,
    "--context_mode",
    CONTEXT_MODE,
    "--mask_scheduler",
    MASK_SCHEDULER,
    "--return_gene_embs",
    RETURN_GENE_EMBS,
    "--gene_embs_condition",
    GENE_EMBS_CONDITION,
    "--pos_encoding_mode",
    POS_ENCODING_MODE,
    "--d_model",
    str(D_MODEL),
]

print(" ".join(cmd))
subprocess.run(cmd, check=True)
