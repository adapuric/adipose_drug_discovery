"""Build the curated human adipose compound-validation panel."""

import argparse
import logging
from pathlib import Path

from add.validation.workflows import validate_adipose_panel


def _parse_arguments() -> argparse.Namespace:
    """Parse panel configuration without mixing science into the CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("config/adipose_validation.yaml")
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Validate the panel or write its independently replicated signatures."""
    args = _parse_arguments()
    logging.basicConfig(level=logging.INFO)
    validate_adipose_panel(
        args.config, output_dir=args.output_dir, dry_run=args.dry_run
    )


if __name__ == "__main__":
    main()
