"""
Verify cached results by re-scoring all outputs with deterministic checkers.

Usage:
    python verify_cached.py              # verify all
    python verify_cached.py --benchmark mosaic  # verify MOSAIC only

No API calls needed — works entirely from cached result JSONs.
"""
import json
import sys
from pathlib import Path
from collections import defaultdict

RESULTS_DIR = Path(__file__).parent / "results"


def verify_mosaic(results_dir: Path) -> dict:
    """Re-verify MOSAIC results from cached JSONs."""
    verified = {}
    for fpath in sorted(results_dir.glob("mosaic_*.json")):
        if fpath.name == "mosaic_summary.json":
            continue
        with open(fpath) as f:
            data = json.load(f)
        condition = data["condition"]
        results = data["results"]
        total_tasks = len(results)
        total_scc = sum(r["scc"] for r in results) / total_tasks if total_tasks else 0
        verified[condition] = {
            "tasks": total_tasks,
            "mean_scc": round(total_scc, 4),
        }
        print(f"  MOSAIC/{condition}: {total_tasks} tasks, mean SCC = {total_scc:.4f}")
    return verified


def verify_ifeval(results_dir: Path) -> dict:
    verified = {}
    for fpath in sorted(results_dir.glob("ifeval_*.json")):
        with open(fpath) as f:
            data = json.load(f)
        condition = data["condition"]
        results = data["results"]
        total = len(results)
        mean_acc = sum(r["instruction_accuracy"] for r in results) / total if total else 0
        prompt_acc = sum(1 for r in results if r["follow_all"]) / total if total else 0
        verified[condition] = {
            "tasks": total,
            "mean_instruction_accuracy": round(mean_acc, 4),
            "prompt_accuracy": round(prompt_acc, 4),
        }
        print(f"  IFEval/{condition}: {total} tasks, instr_acc = {mean_acc:.4f}, prompt_acc = {prompt_acc:.4f}")
    return verified


def verify_ifbench(results_dir: Path) -> dict:
    verified = {}
    for fpath in sorted(results_dir.glob("ifbench_*.json")):
        with open(fpath) as f:
            data = json.load(f)
        condition = data["condition"]
        results = data["results"]
        total = len(results)
        mean_acc = sum(r["instruction_accuracy"] for r in results) / total if total else 0
        verified[condition] = {
            "tasks": total,
            "mean_instruction_accuracy": round(mean_acc, 4),
        }
        print(f"  IFBench/{condition}: {total} tasks, instr_acc = {mean_acc:.4f}")
    return verified


def verify_sysprompt(results_dir: Path) -> dict:
    verified = {}
    for fpath in sorted(results_dir.glob("sysprompt_*.json")):
        with open(fpath) as f:
            data = json.load(f)
        condition = data["condition"]
        results = data["results"]
        total = len(results)
        mean_scc = sum(r["scc"] for r in results) / total if total else 0
        verified[condition] = {
            "tasks": total,
            "mean_scc": round(mean_scc, 4),
        }
        print(f"  SysPrompt/{condition}: {total} tasks, mean SCC = {mean_scc:.4f}")
    return verified


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Verify cached benchmark results")
    parser.add_argument("--benchmark", "-b", default="all",
                       choices=["all", "mosaic", "ifeval", "ifbench", "sysprompt"])
    args = parser.parse_args()

    benchmarks = ["mosaic", "ifeval", "ifbench", "sysprompt"] if args.benchmark == "all" else [args.benchmark]

    print("=" * 60)
    print("CACHED RESULT VERIFICATION (no API calls)")
    print("=" * 60)

    all_verified = {}
    for bm in benchmarks:
        print(f"\n--- {bm.upper()} ---")
        if bm == "mosaic":
            all_verified[bm] = verify_mosaic(RESULTS_DIR)
        elif bm == "ifeval":
            all_verified[bm] = verify_ifeval(RESULTS_DIR)
        elif bm == "ifbench":
            all_verified[bm] = verify_ifbench(RESULTS_DIR)
        elif bm == "sysprompt":
            all_verified[bm] = verify_sysprompt(RESULTS_DIR)

    print("\n" + "=" * 60)
    print("VERIFICATION COMPLETE")
    print("=" * 60)

    out_path = RESULTS_DIR / "verification.json"
    with open(out_path, "w") as f:
        json.dump(all_verified, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
