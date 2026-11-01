"""The `astro-data-lab` command: the only entry point you need."""
from __future__ import annotations

import argparse
import json
import os
import signal
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="astro-data-lab",
        description="Flag complete TESS light curves that contain a transit-like signal. "
                    "Typical order: sync -> train -> evaluate -> predict.",
    )
    commands = parser.add_subparsers(dest="command", metavar="COMMAND")

    sync = commands.add_parser("sync", help="1. update planet catalogues and the light-curve manifest")
    sync.add_argument("--only", choices=("catalogues", "manifest"))
    sync.add_argument("--max-sectors", type=int, default=2, help="new sectors to index (default 2)")

    train = commands.add_parser("train", help="2. train (or resume) the experiment in a TOML file")
    train.add_argument("config", type=Path, help="experiment file, e.g. experiments/transit_cnn.toml")
    train.add_argument("--rounds", type=int, help="train this many more rounds, then stop")
    train.add_argument("--curves-per-round", type=int, help="override (useful for a quick test)")

    status = commands.add_parser("status", help="show a run's progress")
    status.add_argument("config", type=Path)

    evaluate = commands.add_parser("evaluate", help="3. score the best model on unseen test stars")
    evaluate.add_argument("config", type=Path)
    evaluate.add_argument("--split", choices=("validation", "test"), default="test")
    evaluate.add_argument("--checkpoint", type=Path, help="default: the run's best checkpoint")

    predict = commands.add_parser("predict", help="4. classify one FITS light curve")
    predict.add_argument("checkpoint", type=Path)
    predict.add_argument("fits", type=Path)
    predict.add_argument("--device", choices=("cpu", "cuda"))

    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return
    if getattr(args, "rounds", None) is not None and args.rounds < 1:
        parser.error("--rounds must be positive")
    if args.command == "sync" and args.max_sectors < 1:
        parser.error("--max-sectors must be positive")

    # Imports are deferred so `--help` is instant and needs no database or GPU.
    from astro_data_lab.config import Experiment, Settings

    settings = Settings.from_env()
    if args.command == "sync":
        from astro_data_lab.database import ensure_schema

        ensure_schema()
        if args.only != "manifest":
            from astro_data_lab.data.catalogues import synchronise_catalogues

            synchronise_catalogues()
        if args.only != "catalogues":
            from astro_data_lab.data.manifest import inventory_lock, sync_incremental

            directory = Path(os.environ["MAST_INVENTORY_DIR"])
            with inventory_lock(directory):
                sync_incremental(directory, 1, args.max_sectors)
    elif args.command == "train":
        from astro_data_lab.training.trainer import train as run_training

        experiment = Experiment.load(args.config, curves_per_round=args.curves_per_round)
        signal.signal(signal.SIGTERM, signal.default_int_handler)  # `kill` stops like Ctrl+C
        run_training(experiment, settings, rounds=args.rounds)
    elif args.command == "status":
        from astro_data_lab.training.trainer import status as show_status

        show_status(Experiment.load(args.config).name)
    elif args.command == "evaluate":
        from astro_data_lab.training.evaluation import evaluate as run_evaluation

        run_evaluation(Experiment.load(args.config).name, settings, args.split, args.checkpoint)
    else:
        from astro_data_lab.predict import predict as run_prediction

        print(json.dumps(run_prediction(args.checkpoint, args.fits, args.device), indent=2))
