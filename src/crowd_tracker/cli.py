"""Command-line entry point."""

from __future__ import annotations

import argparse
import json
import logging

from .config import AppConfig
from .pipeline import OfflinePipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a YOLO + ByteTrack crowd tracking baseline.")
    parser.add_argument("--config", required=True, help="Path to a JSON configuration file.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = AppConfig.from_file(args.config)
    summary = OfflinePipeline(config).run()
    logging.info("Run complete: %s", json.dumps(summary, indent=2))
