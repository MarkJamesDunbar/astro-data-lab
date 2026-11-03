"""Run lab housekeeping: python -m astro_data_lab.lab sync, cache, or status."""
from __future__ import annotations

import argparse
import subprocess
import sys


def unit_properties(unit: str) -> dict[str, str]:
    result = subprocess.run(
        ["systemctl", "--user", "show", unit,
         "--property=LoadState,ActiveState,UnitFileState,Result,ExecMainStatus,"
         "ExecMainExitTimestamp,LastTriggerUSec,NextElapseUSecRealtime"],
        capture_output=True, text=True, check=True, timeout=15,
    )
    return dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)


def check_status() -> bool:
    """Check timer readiness and last service outcomes without starting a sync."""
    healthy = True
    for name in ("astro-catalogues", "astro-mast"):
        timer = unit_properties(f"{name}.timer")
        service = unit_properties(f"{name}.service")
        timer_ok = (
            timer.get("LoadState") == "loaded"
            and timer.get("ActiveState") == "active"
            and timer.get("UnitFileState") == "enabled"
            and timer.get("NextElapseUSecRealtime", "") not in ("", "n/a")
        )
        running = service.get("ActiveState") in ("active", "activating")
        last_exit = service.get("ExecMainExitTimestamp", "")
        last_ok = (
            bool(last_exit) and service.get("Result") == "success"
            and service.get("ExecMainStatus") == "0"
        )
        service_ok = (
            service.get("LoadState") == "loaded"
            and (running or (service.get("ActiveState") == "inactive" and last_ok))
        )
        ok = timer_ok and service_ok
        healthy = healthy and ok
        state = "ATTENTION" if not ok else "RUNNING" if running else "OK"
        print(f"{name}: {state}")
        print(f"  Timer: {timer.get('UnitFileState', 'unknown')}, "
              f"{timer.get('ActiveState', 'unknown')} "
              f"(load: {timer.get('LoadState', 'unknown')})")
        print(f"  Next run: {timer.get('NextElapseUSecRealtime') or 'not scheduled'}")
        print(f"  Last triggered: {timer.get('LastTriggerUSec') or 'never'}")
        print(f"  Last completed: {last_exit or 'no recorded completion'}")
        print(f"  Last result: {service.get('Result', 'unknown')}, "
              f"exit status {service.get('ExecMainStatus', 'unknown')}")
        if not ok:
            print(f"  Inspect: journalctl --user -u {name}.service -n 50 --no-pager")
    return healthy


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m astro_data_lab.lab",
                                     description="Housekeeping for the TESS light-curve lab.")
    commands = parser.add_subparsers(dest="command", metavar="COMMAND")
    sync = commands.add_parser("sync", help="update planet catalogues and the light-curve manifest")
    sync.add_argument("--only", choices=("catalogues", "manifest"))
    sync.add_argument("--max-sectors", type=int, default=2, help="new sectors to index (default 2)")
    sync.add_argument("--refresh-one", action="store_true",
                      help="also refresh the oldest previously indexed sector")
    commands.add_parser("cache", help="show how many light curves are cached")
    commands.add_parser("status", help="check weekly timers and last sync results (no sync started)")
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
    elif args.command == "status":
        try:
            healthy = check_status()
        except (OSError, subprocess.SubprocessError) as error:
            print(f"Cannot check weekly jobs: {error}", file=sys.stderr)
            if isinstance(error, subprocess.CalledProcessError) and error.stderr:
                print(error.stderr.strip(), file=sys.stderr)
            raise SystemExit(1) from error
        if not healthy:
            raise SystemExit(1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
