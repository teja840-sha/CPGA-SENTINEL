"""Extract per-task raw score vectors from cached result JSONs.

Produces one CSV per benchmark with columns:
  task_id, baseline, cadg, sentinel, forge, cadg_sentinel,
  forge_cadg, forge_sentinel, full_stack, retry_feedback

Each cell is the primary metric (SCC for MOSAIC/SysPrompt,
instruction_accuracy for IFEval/IFBench).
"""

import json, csv, pathlib, sys

RESULTS_DIR = pathlib.Path(__file__).parent / "results"
OUTPUT_DIR = RESULTS_DIR / "score_vectors"

BENCHMARKS = ["mosaic", "ifeval", "ifbench", "sysprompt"]
CONDITIONS = [
    "baseline", "cadg", "sentinel", "forge",
    "cadg_sentinel", "forge_cadg", "forge_sentinel",
    "full_stack", "retry_feedback",
]

METRIC_MAP = {
    "mosaic": "scc",
    "sysprompt": "scc",
    "ifeval": "instruction_accuracy",
    "ifbench": "instruction_accuracy",
}

METRIC_FALLBACKS = {
    "instruction_accuracy": ["instruction_accuracy", "accuracy", "score", "scc"],
    "scc": ["scc", "score"],
}


def get_score(result: dict, metric: str) -> float | None:
    for key in METRIC_FALLBACKS.get(metric, [metric]):
        if key in result:
            return result[key]
    return None


def load_results(benchmark: str, condition: str) -> dict[str, float]:
    fname = RESULTS_DIR / f"{benchmark}_{condition}.json"
    if not fname.exists():
        return {}
    with open(fname) as f:
        data = json.load(f)
    metric = METRIC_MAP[benchmark]
    out = {}
    for r in data.get("results", []):
        tid = r.get("task_id", "")
        score = get_score(r, metric)
        if score is not None:
            out[tid] = score
    return out


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    for bench in BENCHMARKS:
        all_tasks: set[str] = set()
        cond_scores: dict[str, dict[str, float]] = {}
        for cond in CONDITIONS:
            scores = load_results(bench, cond)
            cond_scores[cond] = scores
            all_tasks.update(scores.keys())

        if not all_tasks:
            print(f"  {bench}: no data, skipping")
            continue

        tasks = sorted(all_tasks)
        outpath = OUTPUT_DIR / f"{bench}_score_vectors.csv"
        with open(outpath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["task_id"] + CONDITIONS)
            for tid in tasks:
                row = [tid] + [
                    cond_scores[c].get(tid, "") for c in CONDITIONS
                ]
                writer.writerow(row)
        print(f"  {bench}: {len(tasks)} tasks -> {outpath.name}")

    print(f"\nScore vectors written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
