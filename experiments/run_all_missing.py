"""
Launch all missing benchmark×condition runs in parallel as subprocesses.
Each run saves its own JSON independently, so crashes are isolated.

Usage:
  python run_all_missing.py              # launch all missing
  python run_all_missing.py --dry-run    # just print what would run
  python run_all_missing.py --max-parallel 4  # limit concurrency
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
RUNNER = Path(__file__).parent / "run_benchmarks.py"

FULL_GRID: dict[str, list[str]] = {
    "ifeval":      ["baseline", "retry_feedback", "forge", "cadg", "sentinel",
                    "forge_cadg", "forge_sentinel", "cadg_sentinel", "full_stack"],
    "mosaic":      ["baseline", "retry_feedback", "forge", "cadg", "sentinel",
                    "forge_cadg", "forge_sentinel", "cadg_sentinel", "full_stack"],
    "ifbench":     ["baseline", "retry_feedback", "forge", "cadg", "sentinel",
                    "forge_cadg", "forge_sentinel", "cadg_sentinel", "full_stack"],
    "followbench": ["baseline", "retry_feedback", "forge", "cadg", "sentinel",
                    "forge_cadg", "forge_sentinel", "cadg_sentinel", "full_stack"],
    "sysprompt":   ["baseline", "retry_feedback", "forge", "cadg", "sentinel",
                    "forge_cadg", "forge_sentinel", "cadg_sentinel", "full_stack"],
    "toolsel":     ["baseline", "retry_feedback", "forge", "cadg", "sentinel",
                    "forge_cadg", "forge_sentinel", "cadg_sentinel", "full_stack"],
}

MIN_COUNTS = {
    "ifeval": 541,
    "mosaic": 200,
    "ifbench": 300,
    "followbench": 50,
    "sysprompt": 40,
    "toolsel": 52,
}


def find_existing(results_dir: Path) -> dict[tuple[str, str], int]:
    """Return {(benchmark, condition): n_results} for existing files."""
    existing = {}
    for p in results_dir.glob("*.json"):
        if p.name == "summary.json":
            continue
        try:
            data = json.loads(p.read_text())
            bench = data.get("benchmark", "")
            cond = data.get("condition", "")
            n = data.get("n_results", 0)
            if bench and cond:
                existing[(bench, cond)] = n
        except Exception:
            continue
    return existing


def find_missing(existing: dict) -> list[tuple[str, str]]:
    """Return list of (benchmark, condition) pairs that need running.
    Sorted smallest benchmarks first so they finish and save quickly."""
    missing = []
    for bench, conditions in FULL_GRID.items():
        min_n = MIN_COUNTS.get(bench, 10)
        for cond in conditions:
            n = existing.get((bench, cond), 0)
            if n < min_n:
                missing.append((bench, cond))
    missing.sort(key=lambda bc: MIN_COUNTS.get(bc[0], 999))
    return missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-parallel", type=int, default=6)
    ap.add_argument("--skip-ifeval", action="store_true",
                    help="Skip IFEval (long) to finish smaller benchmarks first")
    args = ap.parse_args()

    existing = find_existing(RESULTS_DIR)
    missing = find_missing(existing)

    if args.skip_ifeval:
        missing = [(b, c) for b, c in missing if b != "ifeval"]

    if not missing:
        print("All benchmark×condition pairs are complete!")
        return

    print(f"Missing {len(missing)} runs:")
    for b, c in missing:
        cur = existing.get((b, c), 0)
        target = MIN_COUNTS.get(b, 0)
        print(f"  {b:12s} / {c:18s}  ({cur}/{target})")

    if args.dry_run:
        print("\n--dry-run: not launching anything.")
        return

    procs: list[tuple[str, str, subprocess.Popen, Path]] = []

    for bench, cond in missing:
        while len([p for _, _, p, _ in procs if p.poll() is None]) >= args.max_parallel:
            time.sleep(5)

        log_path = RESULTS_DIR / f"run_{bench}_{cond}.log"
        cmd = [
            sys.executable, str(RUNNER),
            "-b", bench, "-c", cond,
        ]
        print(f"[{time.strftime('%H:%M:%S')}] Launching: {bench} / {cond}  -> {log_path.name}")
        with open(log_path, "w") as log_f:
            proc = subprocess.Popen(
                cmd, stdout=log_f, stderr=subprocess.STDOUT,
                cwd=str(Path(__file__).parent),
            )
        procs.append((bench, cond, proc, log_path))

    print(f"\n[{time.strftime('%H:%M:%S')}] All {len(procs)} processes launched. Waiting...")

    while any(p.poll() is None for _, _, p, _ in procs):
        running = [(b, c) for b, c, p, _ in procs if p.poll() is None]
        done = [(b, c, p.returncode) for b, c, p, _ in procs if p.poll() is not None]
        print(f"  [{time.strftime('%H:%M:%S')}] running={len(running)} done={len(done)}", flush=True)
        time.sleep(30)

    print(f"\n{'='*60}")
    print("ALL DONE")
    print(f"{'='*60}")
    for bench, cond, proc, log_path in procs:
        status = "OK" if proc.returncode == 0 else f"FAIL (exit {proc.returncode})"
        print(f"  {bench:12s} / {cond:18s}  {status}  -> {log_path.name}")

    failed = [(b, c) for b, c, p, _ in procs if p.returncode != 0]
    if failed:
        print(f"\n{len(failed)} FAILED runs. Check logs for details.")
    else:
        print(f"\nAll {len(procs)} runs succeeded!")


if __name__ == "__main__":
    main()
