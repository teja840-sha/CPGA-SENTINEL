"""
Bootstrap 95% confidence intervals for saved benchmark JSON files in results/.

Usage:
  python analyze_bootstrap.py
  python analyze_bootstrap.py --results-dir ./results --n-bootstrap 2000 --seed 42
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


def _bootstrap_ci(
    values: np.ndarray,
    statistic: str = "mean",
    n_boot: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Return (point_estimate, lo, hi) for mean or proportion."""
    rng = np.random.default_rng(seed)
    n = len(values)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    if statistic == "mean":
        point = float(np.mean(values))

        def stat_fn(sample: np.ndarray) -> float:
            return float(np.mean(sample))
    else:
        point = float(np.mean(values))

        def stat_fn(sample: np.ndarray) -> float:
            return float(np.mean(sample))

    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        stats.append(stat_fn(values[idx]))
    stats_arr = np.array(stats)
    lo = float(np.quantile(stats_arr, alpha / 2))
    hi = float(np.quantile(stats_arr, 1 - alpha / 2))
    return point, lo, hi


def load_result_files(results_dir: Path) -> list[tuple[str, Path]]:
    """Pairs (benchmark, path) for files like ifeval_baseline.json."""
    out = []
    for p in sorted(results_dir.glob("*.json")):
        if p.name == "summary.json":
            continue
        m = re.match(
            r"^(ifeval|ifbench|mosaic|sysprompt|toolsel|followbench)_([a-z_]+)\.json$",
            p.name,
        )
        if m:
            out.append((m.group(1), p))
    return out


def extract_metrics(benchmark: str, rows: list[dict]) -> dict[str, np.ndarray]:
    """Per-task metrics as 1d arrays."""
    if benchmark in ("ifeval", "ifbench"):
        return {
            "instruction_accuracy": np.array([r["instruction_accuracy"] for r in rows], dtype=float),
            "prompt_strict": np.array([1.0 if r["follow_all"] else 0.0 for r in rows], dtype=float),
        }
    if benchmark == "mosaic":
        return {"scc": np.array([r["scc"] for r in rows], dtype=float)}
    if benchmark == "sysprompt":
        return {"scc": np.array([r["scc"] for r in rows], dtype=float)}
    if benchmark == "toolsel":
        return {"correct": np.array([1.0 if r["correct"] else 0.0 for r in rows], dtype=float)}
    if benchmark == "followbench":
        return {"ssr": np.array([r["ssr"] for r in rows], dtype=float)}
    return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=Path(__file__).parent / "results")
    ap.add_argument("--n-bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rd = args.results_dir
    if not rd.is_dir():
        print(f"No results directory: {rd}")
        return

    files = load_result_files(rd)
    if not files:
        print(f"No benchmark JSON files in {rd}")
        return

    by_benchmark: dict[str, list[tuple[str, Path]]] = {}
    for bench, path in files:
        by_benchmark.setdefault(bench, []).append((path.stem.split("_", 1)[-1], path))

    print(f"Bootstrap 95% CI (percentile method), n_bootstrap={args.n_bootstrap}\n")
    for bench in sorted(by_benchmark.keys()):
        print(f"## {bench.upper()}")
        print(f"{'condition':<18} {'metric':<22} {'mean':>8} {'95% CI':>24} {'n':>6}")
        for cond, path in sorted(by_benchmark[bench], key=lambda x: x[0]):
            with open(path) as f:
                data = json.load(f)
            rows = data.get("results", [])
            metrics = extract_metrics(bench, rows)
            for metric_name, arr in metrics.items():
                pt, lo, hi = _bootstrap_ci(
                    arr, n_boot=args.n_bootstrap, seed=args.seed + abs(hash(cond)) % 100000,
                )
                ci = f"[{lo:.4f}, {hi:.4f}]"
                print(f"{cond:<18} {metric_name:<22} {pt:8.4f} {ci:>24} {len(arr):6d}")
        print()


if __name__ == "__main__":
    main()
