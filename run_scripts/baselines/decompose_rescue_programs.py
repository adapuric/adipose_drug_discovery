"""Split the rescue signature into Hallmark programs and score drugs."""

import argparse
import logging
from pathlib import Path

from add.validation.workflows import decompose_rescue_programs


def _parse_arguments() -> argparse.Namespace:
    """Parse the program-report configuration and output path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("config/rescue_programs.yaml")
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Validate inputs or publish the rescue-program report."""
    args = _parse_arguments()
    logging.basicConfig(level=logging.INFO)
    decompose_rescue_programs(
        args.config, output_dir=args.output_dir, dry_run=args.dry_run
    )


if __name__ == "__main__":
    main()
