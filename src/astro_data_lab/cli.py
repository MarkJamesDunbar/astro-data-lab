"""The `astro-data-lab` command. Everything else is done from Python (see the notebook)."""
from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="astro-data-lab",
                                     description="Housekeeping for the TESS light-curve lab.")
    commands = parser.add_subparsers(dest="command", metavar="COMMAND")
    sync = commands.add_parser("sync", help="update planet catalogues and the light-curve manifest")
    sync.add_argument("--only", choices=("catalogues", "manifest"))
    sync.add_argument("--max-sectors", type=int, default=2, help="new sectors to index (default 2)")
    sync.add_argument("--refresh-one", action="store_true",
                      help="also refresh the oldest previously indexed sector")
    commands.add_parser("cache", help="show how many light curves are cached")
    args = parser.parse_args(argv)

    if args.command == "sync":
        if args.max_sectors < 1:
            parser.error("--max-sectors must be positive")
        if args.refresh_one and args.only == "catalogues":
            parser.error("--refresh-one requires manifest sync")
        from astro_data_lab.sync import sync as run_sync

        run_sync(catalogues=args.only != "manifest", manifest=args.only != "catalogues",
                 max_sectors=args.max_sectors, refresh_one=args.refresh_one)
    elif args.command == "cache":
        from astro_data_lab.lightcurves import cache_info

        info = cache_info()
        print(f"{info['curves']:,} curves, {info['size_gib']:.2f} of {info['limit_gib']:.0f} GiB, "
              f"in {info['directory']}")
    else:
        parser.print_help()
