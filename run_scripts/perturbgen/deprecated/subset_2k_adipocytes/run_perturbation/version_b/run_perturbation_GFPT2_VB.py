import subprocess


VAL_SCRIPT = "/rds/general/user/ap5625/home/perturbation_modeling/perturbGEN/subset_2k_adipocytes/run_perturbation/version_b/perturb_val_adipocytes.py"
CONFIG_PATH = "/rds/general/user/ap5625/home/perturbation_modeling/perturbGEN/subset_2k_adipocytes/run_perturbation/version_b/perturbation_adipocytes_GFPT2_versionb.yaml"

cmd = [
    "python",
    VAL_SCRIPT,
    "--config",
    CONFIG_PATH,
]

print(" ".join(cmd))


subprocess.run(cmd, check=True)
