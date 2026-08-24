"""CLI entry point for a two-or-more-source tracking run."""

from __future__ import annotations

import argparse
import json
import logging

from .config import MultiStreamConfig
from .multi_pipeline import MultiStreamPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run independent local tracking on multiple camera feeds.")
    parser.add_argument("--config", required=True, help="Path to a multi-stream JSON configuration file.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    summary = MultiStreamPipeline(MultiStreamConfig.from_file(args.config)).run()
    logging.info("Multi-stream run complete: %s", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
