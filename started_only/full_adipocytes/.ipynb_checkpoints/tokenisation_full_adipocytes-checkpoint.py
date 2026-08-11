

import subprocess
import sys


H5AD_PATH = "/rds/general/user/ap5625/projects/lms-scott-raw/live/Ada/perturbation_modeling/adipocytes_full_perturbgen_preprocessed_paired.h5ad"

DATASET_NAME = "full_adipocytes_ob_wl" # choose a name for the dataset
GENE_FILTERING_MODE = "all"  # one of: hvg, degs, all
HVG_MODE = "before_tokenisation"  # before_tokenisation or after_tokenisation


VAR_LIST = [
    "cell_states_adipocytes",
    "condition",
] # list of obs to retain in adata.vars after preprocessing


PAIRING_MODE = "stratified"  # stratified, random, mapping. We select the pairing strategy here, for more info please read the paper.
TIME_OBS = "condition" # the obs that contains the time point information for pairing
PAIRING_FILE = "path/to/pairing.csv"  # only if PAIRING_MODE == 'mapping'
MAIN_PAIRING_OBS = "Donor" # in previous iteration i did main pairing obs = cell_states - i think donor should make more sense biologically 
OPT_PAIRING_OBS = []  # optional additional obs


NPROC = 8 # number of parallel processes to use
#N_HVG = 2000 # number of highly variable genes to select if HVG filtering is used
TIME_POINT_ORDER = ["obese", "weightloss"]
REFERENCE_TIME = "obese" # the reference time point for pairing, usually the control or untreated condition


GENE_MEDIAN_PATH = "/rds/general/user/ap5625/projects/lms-scott-raw/live/Ada/perturbation_modeling/Perturbgen_clean/Perturbgen/perturbgen/pp/gene_median_dict_gftokens_gc95M.pkl"  # path to gene median dictionary, provided with Perturbgen for pretrained model
TOKEN_DICT_PATH  = "/rds/general/user/ap5625/projects/lms-scott-raw/live/Ada/perturbation_modeling/Perturbgen_clean/Perturbgen/perturbgen/pp/token_dict_gftokens_gc95M.pkl"  # path to token dictionary, provided with Perturbgen for pretrained model
GENE_MAPPING_PATH = "/rds/general/user/ap5625/projects/lms-scott-raw/live/Ada/perturbation_modeling/Perturbgen_clean/Perturbgen/perturbgen/pp/ensembl_mapping_dict_gc95M.pkl"  # path to gene mapping dictionary, Geneformer 95M mapping dictionary provided with Perturbgen



cmd = [
    sys.executable,
    "-m",
    "perturbgen",
    "tokenise",
    "--h5ad_path", H5AD_PATH,
    "--dataset", DATASET_NAME,
    "--gene_filtering_mode", GENE_FILTERING_MODE,
    "--hvg_mode", HVG_MODE,
    "--var_list", *VAR_LIST,
    "--pairing_mode", PAIRING_MODE,
    "--time_obs", TIME_OBS,
    "--main_pairing_obs", MAIN_PAIRING_OBS,
    "--nproc", str(NPROC),
    "--reference_time", REFERENCE_TIME,
    "--time_point_order", *TIME_POINT_ORDER,
    "--gene_median_path", GENE_MEDIAN_PATH,
    "--token_dict_path", TOKEN_DICT_PATH,
    "--gene_mapping_path", GENE_MAPPING_PATH,
]



subprocess.run(cmd, check=True)

