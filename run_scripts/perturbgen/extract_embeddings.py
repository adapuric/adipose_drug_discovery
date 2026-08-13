"""Extract gene and cell embeddings from a PerturbGen masking checkpoint."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from run_scripts.perturbgen.config import PerturbGenConfig
from run_scripts.perturbgen.config import load_perturbgen_config
from run_scripts.perturbgen.utils import bool_text
from run_scripts.perturbgen.utils import checkpoint_path
from run_scripts.perturbgen.utils import model_data_arguments
from run_scripts.perturbgen.utils import run_command
from run_scripts.perturbgen.utils import shared_model_arguments


def build_embedding_command(
    config: PerturbGenConfig,
    *,
    masking_checkpoint: str | Path | None = None,
    python_executable: str | None = None,
) -> list[str]:
    """Build the PerturbGen gene/cell embedding extraction command."""
    model = config.model
    embedding = config.embedding
    checkpoint = checkpoint_path(
        embedding.masking_checkpoint_path,
        override=masking_checkpoint,
    )
    return [
        python_executable or sys.executable,
        "-m",
        "perturbgen",
        "extract-embedding",
        "--test_mode",
        "masking",
        "--split",
        "False",
        "--splitting_mode",
        "stratified",
        "--return_embeddings",
        bool_text(embedding.return_cell_embeddings),
        "--return_attn",
        bool_text(embedding.return_attention),
        "--generate",
        bool_text(embedding.generate),
        "--ckpt_masking_path",
        str(checkpoint),
        "--output_dir",
        str(config.embedding_output_directory),
        *model_data_arguments(config),
        "--tokenid_to_rowid_path",
        str(config.token_to_row_mapping_path),
        "--batch_size",
        str(embedding.batch_size),
        "--n_workers",
        str(embedding.data_loader_workers),
        "--cellgen_lr",
        str(embedding.masking_learning_rate),
        "--cellgen_wd",
        str(embedding.masking_weight_decay),
        "--count_lr",
        str(embedding.count_learning_rate),
        "--count_wd",
        str(embedding.count_weight_decay),
        *shared_model_arguments(
            model,
            metadata_cols=embedding.retained_obs_cols,
        ),
        "--mask_scheduler",
        embedding.mask_scheduler,
        "--return_gene_embs",
        bool_text(embedding.return_gene_embeddings),
        "--gene_embs_condition",
        embedding.gene_embedding_condition_obs,
    ]


def _parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("config.yaml"),
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run configured PerturbGen embedding extraction."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    arguments = _parse_arguments()
    config = load_perturbgen_config(arguments.config)
    run_command(
        build_embedding_command(
            config,
            masking_checkpoint=arguments.checkpoint,
        ),
        perturbgen_directory=config.project.perturbgen_directory,
        dry_run=arguments.dry_run,
    )


if __name__ == "__main__":
    main()
