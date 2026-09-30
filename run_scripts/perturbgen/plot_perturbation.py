"""Report existing PerturbGen edit effects and independent baseline checks."""

import argparse
import logging
from pathlib import Path

from add.validation.workflows import perturbgen_report


def _parse_arguments() -> argparse.Namespace:
    """Parse explicit result selection and report destination."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/perturbgen_visualization.yaml"),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Validate provenance and render donor-aware sensitivity figures."""
    args = _parse_arguments()
    logging.basicConfig(level=logging.INFO)
    perturbgen_report(
        args.config, output_dir=args.output_dir, dry_run=args.dry_run
    )


if __name__ == "__main__":
    main()
