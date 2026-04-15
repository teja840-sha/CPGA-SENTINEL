"""
CPGA+SENTINEL Public Benchmark Runner

Runs experimental conditions across benchmarks (deterministic scoring only).

  Conditions: baseline, retry_feedback, forge, cadg, sentinel, forge_cadg,
    forge_sentinel, cadg_sentinel, full_stack
  Benchmarks: mosaic, ifeval, followbench, ifbench, sysprompt, toolsel

  Post-hoc: python analyze_bootstrap.py (95% bootstrap CIs on results/*.json)

Usage:
  python run_benchmarks.py --benchmark mosaic --condition baseline
  python run_benchmarks.py --benchmark all --condition all
  python run_benchmarks.py --benchmark mosaic --condition all --dry-run
  python run_benchmarks.py --benchmark ifeval --condition baseline --max-tasks 50

Constraint counts (MOSAIC): 5, 10, 15, 20
IFEval: all 541 prompts (or --max-tasks subset)
FollowBench: levels 1-5

All scoring uses deterministic Python checkers. No LLM judges for metrics.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from cpga_harness import (
    LLMClient, load_config, score_output, save_results,
    aggregate_results, GenerationResult, ScoringResult,
)
from cpga_methods import (
    forge_classify, forge_filter_prompt,
    cadg_generate, cadg_select_best,
    permute_constraints,
    run_sentinel_swarm, run_full_stack,
    run_forge_cadg, run_forge_sentinel, run_cadg_sentinel,
)
from adapters.mosaic_adapter import MOSAICAdapter, load_mosaic_from_repo
from adapters.ifeval_adapter import IFEvalAdapter
from adapters.followbench_adapter import FollowBenchAdapter
from adapters.ifbench_adapter import IFBenchAdapter
from adapters.system_prompt_adapter import SystemPromptAdapter
from adapters.tool_selection_adapter import ToolSelectionAdapter


RESULTS_DIR = Path(__file__).parent / "results"
REPOS_DIR = Path(__file__).parent / "repos"

CONDITIONS = [
    "baseline",
    "retry_feedback",
    "retry_feedback_5",
    "forge",
    "cadg",
    "sentinel",
    "forge_cadg",
    "forge_sentinel",
    "cadg_sentinel",
    "full_stack",
]
BENCHMARKS = ["mosaic", "ifeval", "followbench", "ifbench", "sysprompt", "toolsel"]


def _log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _extra_condition_kwargs(condition: str, exp: dict, n_candidates: int, seed: int, forge_mode: str = "type") -> dict:
    """CADG-style conditions need n_candidates/seed; retry_feedback needs max_rounds."""
    kw: dict = {}
    if condition in ("cadg", "full_stack", "forge_cadg", "cadg_sentinel"):
        kw["n_candidates"] = n_candidates
        kw["seed"] = seed
    if condition == "retry_feedback":
        kw["max_rounds"] = int(exp.get("retry_max_rounds", 2))
    if condition == "retry_feedback_5":
        kw["max_rounds"] = 5
    if condition in ("forge", "full_stack", "forge_cadg", "forge_sentinel"):
        kw["forge_mode"] = forge_mode
    return kw


# ---------------------------------------------------------------------------
# Condition runners
# ---------------------------------------------------------------------------

def run_baseline(
    client: LLMClient,
    task: dict,
    provider: str = "anthropic",
    model: str | None = None,
    max_tokens: int = 4096,
) -> tuple[str, dict]:
    """Baseline: all constraints in prompt -> generate -> score."""
    from cpga_harness import run_single_task

    result = run_single_task(
        client=client,
        task_id=task["task_id"],
        prompt=task["prompt"],
        condition="baseline",
        system=task.get("system", ""),
        provider=provider,
        model=model,
        max_tokens=max_tokens,
    )
    return result.output, {"generation": asdict(result)}


def run_forge(
    client: LLMClient,
    task: dict,
    provider: str = "anthropic",
    model: str | None = None,
    max_tokens: int = 4096,
    forge_mode: str = "type",
) -> tuple[str, dict]:
    """FORGE: classify constraints, remove Layer 0/1, generate with residual."""
    from cpga_harness import run_single_task

    constraints = task["constraints"]
    filtered_prompt, layers = forge_filter_prompt(
        task["prompt"], constraints, forge_mode=forge_mode,
        client=client, provider=provider, model=model,
    )

    layer_counts = {k: len(v) for k, v in layers.items()}
    _log(f"  FORGE layers: {layer_counts}")

    result = run_single_task(
        client=client,
        task_id=task["task_id"],
        prompt=filtered_prompt,
        condition="forge",
        system=task.get("system", ""),
        provider=provider,
        model=model,
        max_tokens=max_tokens,
    )
    return result.output, {
        "generation": asdict(result),
        "forge_layers": layer_counts,
    }


def run_cadg(
    client: LLMClient,
    task: dict,
    n_candidates: int = 5,
    seed: int = 42,
    provider: str = "anthropic",
    model: str | None = None,
    max_tokens: int = 4096,
) -> tuple[str, dict]:
    """CADG: permute constraint order across N candidates -> score -> select best."""
    constraints = task["constraints"]
    candidates = cadg_generate(
        client=client,
        task_id=task["task_id"],
        base_prompt=task.get("base_prompt", task["prompt"]),
        constraints=constraints,
        n_candidates=n_candidates,
        seed=seed,
        provider=provider,
        model=model,
        max_tokens=max_tokens,
    )

    best_output, best_score, best_idx = cadg_select_best(candidates, constraints)
    _log(f"  CADG best: candidate {best_idx}, score {best_score:.3f}")

    return best_output, {
        "cadg_candidates": n_candidates,
        "cadg_best_index": best_idx,
        "cadg_best_score": best_score,
    }


def run_sentinel(
    client: LLMClient,
    task: dict,
    provider: str = "anthropic",
    model: str | None = None,
    max_tokens: int = 4096,
) -> tuple[str, dict]:
    """SENTINEL: generate with constraints -> run full Tier 2/3/4 swarm.

    Tier 4 = P13 Hyper-Isolated Verification: one LLM call per failing
    constraint with ~150 tokens context. No attention competition.
    """
    from cpga_harness import run_single_task

    result = run_single_task(
        client=client,
        task_id=task["task_id"],
        prompt=task["prompt"],
        condition="sentinel",
        system=task.get("system", ""),
        provider=provider,
        model=model,
        max_tokens=max_tokens,
    )

    fixed_output, verdicts, decision = run_sentinel_swarm(
        result.output, task["constraints"],
        enable_tier2=True, enable_tier4=True,
        client=client, provider=provider, model=model,
    )

    n_fixed = sum(1 for v in verdicts if v.auto_fixed)
    n_t4 = sum(1 for v in verdicts if v.tier == "tier4" and v.auto_fixed)
    _log(f"  SENTINEL: {decision}, {n_fixed} fixed ({n_t4} via P13)")

    return fixed_output, {
        "generation": asdict(result),
        "sentinel_decision": decision,
        "sentinel_auto_fixed": n_fixed,
        "sentinel_verdicts": len(verdicts),
    }


def run_full_stack_condition(
    client: LLMClient,
    task: dict,
    n_candidates: int = 5,
    seed: int = 42,
    provider: str = "anthropic",
    model: str | None = None,
    max_tokens: int = 4096,
    forge_mode: str = "type",
) -> tuple[str, dict]:
    """Full stack: FORGE + CADG + SENTINEL composed."""
    output, score, meta = run_full_stack(
        client=client,
        task_id=task["task_id"],
        base_prompt=task.get("base_prompt", task["prompt"]),
        constraints=task["constraints"],
        n_candidates=n_candidates,
        seed=seed,
        provider=provider,
        model=model,
        max_tokens=max_tokens,
        forge_mode=forge_mode,
    )
    _log(f"  Full stack: score {score:.3f}")
    return output, meta


def run_retry_feedback(
    client: LLMClient,
    task: dict,
    provider: str = "anthropic",
    model: str | None = None,
    max_tokens: int = 4096,
    max_rounds: int = 2,
) -> tuple[str, dict]:
    """Strong baseline: generate, score with deterministic checkers, append failure
    feedback and regenerate (up to max_rounds retries after the initial attempt).
    """
    from cpga_harness import run_single_task

    base = task["prompt"]
    prompt = base
    last_output = ""
    attempts = 0
    for attempt in range(max_rounds + 1):
        attempts = attempt + 1
        result = run_single_task(
            client=client,
            task_id=f"{task['task_id']}_retry{attempt}",
            prompt=prompt,
            condition="retry_feedback",
            system=task.get("system", ""),
            provider=provider,
            model=model,
            max_tokens=max_tokens,
        )
        last_output = result.output
        sr = score_output(task["task_id"], "retry_feedback", last_output, task["constraints"])
        if sr.constraints_total <= 0 or sr.constraints_satisfied >= sr.constraints_total:
            break
        if attempt >= max_rounds:
            break
        failed_lines = [
            f"- {pc.get('text', pc.get('id', '?'))}"
            for pc in sr.per_constraint
            if not pc.get("passed", False)
        ]
        fb = (
            "\n\nYour previous answer did not satisfy all listed requirements. "
            "Rewrite the COMPLETE response from scratch so every requirement is met.\n"
            "Requirements that still failed verification:\n"
            + "\n".join(failed_lines)
        )
        prompt = base + fb

    _log(f"  retry_feedback: {attempts} generation(s)")
    return last_output, {"retry_attempts": attempts, "retry_max_rounds": max_rounds}


def run_forge_cadg_condition(
    client: LLMClient,
    task: dict,
    n_candidates: int = 5,
    seed: int = 42,
    provider: str = "anthropic",
    model: str | None = None,
    max_tokens: int = 4096,
    forge_mode: str = "type",
) -> tuple[str, dict]:
    output, meta = run_forge_cadg(
        client=client,
        task_id=task["task_id"],
        base_prompt=task.get("base_prompt", task["prompt"]),
        constraints=task["constraints"],
        n_candidates=n_candidates,
        seed=seed,
        provider=provider,
        model=model,
        max_tokens=max_tokens,
        forge_mode=forge_mode,
    )
    _log(f"  forge_cadg: best_idx={meta.get('cadg_best_index')}")
    return output, meta


def run_forge_sentinel_condition(
    client: LLMClient,
    task: dict,
    provider: str = "anthropic",
    model: str | None = None,
    max_tokens: int = 4096,
    forge_mode: str = "type",
) -> tuple[str, dict]:
    output, meta = run_forge_sentinel(
        client=client,
        task_id=task["task_id"],
        task_prompt=task["prompt"],
        constraints=task["constraints"],
        system=task.get("system", ""),
        provider=provider,
        model=model,
        max_tokens=max_tokens,
        forge_mode=forge_mode,
    )
    _log(f"  forge_sentinel: {meta.get('sentinel_decision')}")
    return output, meta


def run_cadg_sentinel_condition(
    client: LLMClient,
    task: dict,
    n_candidates: int = 5,
    seed: int = 42,
    provider: str = "anthropic",
    model: str | None = None,
    max_tokens: int = 4096,
) -> tuple[str, dict]:
    output, meta = run_cadg_sentinel(
        client=client,
        task_id=task["task_id"],
        base_prompt=task.get("base_prompt", task["prompt"]),
        constraints=task["constraints"],
        n_candidates=n_candidates,
        seed=seed,
        provider=provider,
        model=model,
        max_tokens=max_tokens,
    )
    _log(f"  cadg_sentinel: {meta.get('sentinel_decision')}")
    return output, meta


def run_retry_feedback_5(
    client: LLMClient,
    task: dict,
    provider: str = "anthropic",
    model: str | None = None,
    max_tokens: int = 4096,
    max_rounds: int = 5,
) -> tuple[str, dict]:
    """Guardrails-proxy baseline: 5 retries with feedback (matches Guardrails AI loop)."""
    return run_retry_feedback(
        client, task, provider=provider, model=model,
        max_tokens=max_tokens, max_rounds=max_rounds,
    )


CONDITION_RUNNERS = {
    "baseline": run_baseline,
    "retry_feedback": run_retry_feedback,
    "retry_feedback_5": run_retry_feedback_5,
    "forge": run_forge,
    "cadg": run_cadg,
    "sentinel": run_sentinel,
    "forge_cadg": run_forge_cadg_condition,
    "forge_sentinel": run_forge_sentinel_condition,
    "cadg_sentinel": run_cadg_sentinel_condition,
    "full_stack": run_full_stack_condition,
}


# ---------------------------------------------------------------------------
# Benchmark runners
# ---------------------------------------------------------------------------

def run_mosaic(
    client: LLMClient,
    conditions: list[str],
    config: dict,
    dry_run: bool = False,
    max_tasks: int | None = None,
    provider: str = "anthropic",
    model_override: str | None = None,
    forge_mode: str = "type",
    result_suffix: str | None = None,
) -> dict:
    """Run MOSAIC benchmark across specified conditions."""
    exp = config.get("experiment", {})
    seed = exp.get("seed", 42)
    constraint_counts = exp.get("constraint_counts", [5, 10, 15, 20])
    tasks_per = exp.get("tasks_per_condition", 15)
    if max_tasks:
        tasks_per = max(1, max_tasks // len(constraint_counts))
    n_candidates = exp.get("cadg_candidates", 5)
    model = model_override or config.get(provider, {}).get("model")
    max_tokens = exp.get("max_tokens", 4096)

    adapter = MOSAICAdapter(seed=seed)
    mosaic_repo = REPOS_DIR / "mosaic"
    if mosaic_repo.exists():
        _log("Using real MOSAIC dataset from repo")
        max_per_size = max_tasks // len(constraint_counts) if max_tasks else tasks_per
        tasks = load_mosaic_from_repo(mosaic_repo, constraint_counts, max_per_size)
    else:
        tasks = adapter.load_tasks(
            constraint_counts=constraint_counts,
            tasks_per_condition=tasks_per,
        )
    if max_tasks:
        tasks = tasks[:max_tasks]
    _log(f"MOSAIC: {len(tasks)} tasks loaded")

    all_results = {}
    for condition in conditions:
        _log(f"\n=== MOSAIC / {condition.upper()} ===")
        results = []

        for i, task in enumerate(tasks):
            _log(f"  Task {i+1}/{len(tasks)}: {task['task_id']} ({task['constraint_count']}c)")

            if dry_run:
                output = f"[DRY RUN] Would generate for {task['task_id']}"
                meta = {"dry_run": True}
            else:
                runner = CONDITION_RUNNERS[condition]
                kwargs = {"client": client, "task": task, "provider": provider,
                          "model": model, "max_tokens": max_tokens}
                kwargs.update(_extra_condition_kwargs(condition, exp, n_candidates, seed, forge_mode))
                output, meta = runner(**kwargs)

            score_result = adapter.score(output, task)
            results.append({
                "task_id": task["task_id"],
                "condition": condition,
                "constraint_count": task["constraint_count"],
                "scc": score_result["scc"],
                "pa": score_result["pa"],
                "satisfied": score_result["satisfied"],
                "total_checkable": score_result["total_checkable"],
                "meta": meta,
            })
            _log(f"    SCC={score_result['scc']:.3f} PA={score_result['pa']:.3f} "
                 f"({score_result['satisfied']}/{score_result['total_checkable']})")

        all_results[condition] = results
        suffix = f"_{result_suffix}" if result_suffix else ""
        out_path = RESULTS_DIR / f"mosaic_{condition}{suffix}.json"
        _save_benchmark_results("mosaic", condition, results, out_path)

    return all_results


def run_ifeval(
    client: LLMClient,
    conditions: list[str],
    config: dict,
    dry_run: bool = False,
    max_tasks: int | None = None,
    provider: str = "anthropic",
    model_override: str | None = None,
    forge_mode: str = "type",
    result_suffix: str | None = None,
) -> dict:
    """Run IFEval benchmark across specified conditions."""
    exp = config.get("experiment", {})
    model = model_override or config.get(provider, {}).get("model")
    max_tokens = exp.get("max_tokens", 4096)
    seed = exp.get("seed", 42)
    n_candidates = exp.get("cadg_candidates", 5)

    adapter = IFEvalAdapter()
    tasks = adapter.load_tasks(max_tasks=max_tasks)
    _log(f"IFEval: {len(tasks)} tasks loaded")

    all_results = {}
    for condition in conditions:
        _log(f"\n=== IFEval / {condition.upper()} ===")
        results = []
        strict_scores_for_cat: list[dict] = []

        for i, task in enumerate(tasks):
            _log(f"  Task {i+1}/{len(tasks)}: {task['task_id']} ({task['constraint_count']}c)")

            if dry_run:
                output = f"[DRY RUN] Would generate for {task['task_id']}"
            else:
                runner = CONDITION_RUNNERS[condition]
                kwargs = {"client": client, "task": task, "provider": provider,
                          "model": model, "max_tokens": max_tokens}
                kwargs.update(_extra_condition_kwargs(condition, exp, n_candidates, seed, forge_mode))
                output, _ = runner(**kwargs)

            score_result = adapter.score(output, task)
            if not dry_run:
                strict_scores_for_cat.append(score_result)
            results.append({
                "task_id": task["task_id"],
                "condition": condition,
                "constraint_count": task["constraint_count"],
                "follow_all": score_result["follow_all_instructions"],
                "instruction_accuracy": score_result["instruction_accuracy"],
                "prompt_accuracy": score_result["prompt_accuracy"],
                "n_passed": score_result["n_passed"],
                "n_total": score_result["n_total"],
            })
            _log(f"    follow_all={score_result['follow_all_instructions']} "
                 f"instr_acc={score_result['instruction_accuracy']:.3f}")

        all_results[condition] = results
        suffix = f"_{result_suffix}" if result_suffix else ""
        out_path = RESULTS_DIR / f"ifeval_{condition}{suffix}.json"
        _save_benchmark_results("ifeval", condition, results, out_path)

        cat_agg = adapter.aggregate_by_category(strict_scores_for_cat) if strict_scores_for_cat else {}
        if cat_agg:
            _log(f"  Per-category: {json.dumps(cat_agg, indent=2)}")

    return all_results


def run_ifbench(
    client: LLMClient,
    conditions: list[str],
    config: dict,
    dry_run: bool = False,
    max_tasks: int | None = None,
    provider: str = "anthropic",
    model_override: str | None = None,
    forge_mode: str = "type",
    result_suffix: str | None = None,
) -> dict:
    """Run IFBench (Allen AI, NeurIPS 2025) across specified conditions."""
    exp = config.get("experiment", {})
    model = model_override or config.get(provider, {}).get("model")
    max_tokens = exp.get("max_tokens", 4096)
    seed = exp.get("seed", 42)
    n_candidates = exp.get("cadg_candidates", 5)

    ifbench_repo = REPOS_DIR / "ifbench"
    adapter = IFBenchAdapter(repo_path=ifbench_repo if ifbench_repo.exists() else None)
    tasks = adapter.load_tasks(max_tasks=max_tasks)
    _log(f"IFBench: {len(tasks)} tasks loaded")

    all_results = {}
    for condition in conditions:
        _log(f"\n=== IFBench / {condition.upper()} ===")
        results = []

        for i, task in enumerate(tasks):
            _log(f"  Task {i+1}/{len(tasks)}: {task['task_id']} ({task['constraint_count']}c)")

            if dry_run:
                output = f"[DRY RUN] Would generate for {task['task_id']}"
            else:
                runner = CONDITION_RUNNERS[condition]
                kwargs = {"client": client, "task": task, "provider": provider,
                          "model": model, "max_tokens": max_tokens}
                kwargs.update(_extra_condition_kwargs(condition, exp, n_candidates, seed, forge_mode))
                output, _ = runner(**kwargs)

            score_result = adapter.score(output, task)
            results.append({
                "task_id": task["task_id"],
                "condition": condition,
                "constraint_count": task["constraint_count"],
                "follow_all": score_result["follow_all_instructions"],
                "instruction_accuracy": score_result["instruction_accuracy"],
                "prompt_accuracy": score_result["prompt_accuracy"],
                "n_passed": score_result["n_passed"],
                "n_total": score_result["n_total"],
            })
            _log(f"    follow_all={score_result['follow_all_instructions']} "
                 f"instr_acc={score_result['instruction_accuracy']:.3f}")

        all_results[condition] = results
        suffix = f"_{result_suffix}" if result_suffix else ""
        out_path = RESULTS_DIR / f"ifbench_{condition}{suffix}.json"
        _save_benchmark_results("ifbench", condition, results, out_path)

    return all_results


def run_sysprompt(
    client: LLMClient,
    conditions: list[str],
    config: dict,
    dry_run: bool = False,
    max_tasks: int | None = None,
    provider: str = "anthropic",
    model_override: str | None = None,
    forge_mode: str = "type",
    result_suffix: str | None = None,
) -> dict:
    """Run System Prompt Adherence test (T2)."""
    exp = config.get("experiment", {})
    model = model_override or config.get(provider, {}).get("model")
    max_tokens = exp.get("max_tokens", 4096)
    seed = exp.get("seed", 42)
    n_candidates = exp.get("cadg_candidates", 5)

    adapter = SystemPromptAdapter(seed=seed)
    tasks = adapter.load_tasks(max_tasks=max_tasks)
    _log(f"SystemPrompt: {len(tasks)} tasks loaded")

    all_results = {}
    for condition in conditions:
        _log(f"\n=== SystemPrompt / {condition.upper()} ===")
        results = []
        for i, task in enumerate(tasks):
            _log(f"  Task {i+1}/{len(tasks)}: {task['task_id']} ({task['constraint_count']}g)")
            if dry_run:
                output = f"[DRY RUN]"
            else:
                runner = CONDITION_RUNNERS[condition]
                kwargs = {"client": client, "task": task, "provider": provider,
                          "model": model, "max_tokens": max_tokens}
                kwargs.update(_extra_condition_kwargs(condition, exp, n_candidates, seed, forge_mode))
                output, _ = runner(**kwargs)
            score_result = adapter.score(output, task)
            results.append({
                "task_id": task["task_id"],
                "condition": condition,
                "constraint_count": task["constraint_count"],
                "scc": score_result["scc"],
                "satisfied": score_result["satisfied"],
                "total_checkable": score_result["total_checkable"],
            })
            _log(f"    SCC={score_result['scc']:.3f} ({score_result['satisfied']}/{score_result['total_checkable']})")
        all_results[condition] = results
        suffix = f"_{result_suffix}" if result_suffix else ""
        out_path = RESULTS_DIR / f"sysprompt_{condition}{suffix}.json"
        _save_benchmark_results("sysprompt", condition, results, out_path)
    return all_results


def run_toolsel(
    client: LLMClient,
    conditions: list[str],
    config: dict,
    dry_run: bool = False,
    max_tasks: int | None = None,
    provider: str = "anthropic",
    model_override: str | None = None,
    forge_mode: str = "type",
    result_suffix: str | None = None,
) -> dict:
    """Run Multi-Tool Selection test (T4)."""
    exp = config.get("experiment", {})
    model = model_override or config.get(provider, {}).get("model")
    max_tokens = exp.get("max_tokens", 4096)
    seed = exp.get("seed", 42)
    n_candidates = exp.get("cadg_candidates", 5)

    adapter = ToolSelectionAdapter(seed=seed)
    tasks = adapter.load_tasks(max_tasks=max_tasks)
    _log(f"ToolSelection: {len(tasks)} tasks loaded")

    all_results = {}
    for condition in conditions:
        _log(f"\n=== ToolSelection / {condition.upper()} ===")
        results = []
        for i, task in enumerate(tasks):
            _log(f"  Task {i+1}/{len(tasks)}: {task['task_id']} ({task['n_tools']}t)")
            if dry_run:
                output = f"[DRY RUN]"
            else:
                runner = CONDITION_RUNNERS[condition]
                kwargs = {"client": client, "task": task, "provider": provider,
                          "model": model, "max_tokens": max_tokens}
                kwargs.update(_extra_condition_kwargs(condition, exp, n_candidates, seed, forge_mode))
                output, _ = runner(**kwargs)
            score_result = adapter.score(output, task)
            results.append({
                "task_id": task["task_id"],
                "condition": condition,
                "n_tools": task["n_tools"],
                "correct": score_result["correct"],
                "correct_tool": task["correct_tool"],
                "correct_position": task["correct_position"],
            })
            _log(f"    correct={score_result['correct']} (tool={task['correct_tool']} pos={task['correct_position']})")
        all_results[condition] = results
        suffix = f"_{result_suffix}" if result_suffix else ""
        out_path = RESULTS_DIR / f"toolsel_{condition}{suffix}.json"
        _save_benchmark_results("toolsel", condition, results, out_path)
    return all_results


def run_followbench(
    client: LLMClient,
    conditions: list[str],
    config: dict,
    dry_run: bool = False,
    max_tasks: int | None = None,
    repo_path: str | None = None,
    provider: str = "anthropic",
    model_override: str | None = None,
    forge_mode: str = "type",
    result_suffix: str | None = None,
) -> dict:
    """Run FollowBench across specified conditions."""
    exp = config.get("experiment", {})
    model = model_override or config.get(provider, {}).get("model")
    max_tokens = exp.get("max_tokens", 4096)
    seed = exp.get("seed", 42)
    n_candidates = exp.get("cadg_candidates", 5)

    if repo_path is None:
        fb_repo = REPOS_DIR / "followbench"
        if fb_repo.exists():
            repo_path = str(fb_repo)
            _log("Using real FollowBench dataset from repo")
    adapter = FollowBenchAdapter(repo_path=repo_path)
    tasks = adapter.load_tasks(levels=[1, 2, 3, 4, 5], max_tasks=max_tasks)
    _log(f"FollowBench: {len(tasks)} tasks loaded (levels 1-5)")

    all_results = {}
    for condition in conditions:
        _log(f"\n=== FollowBench / {condition.upper()} ===")
        results = []

        for i, task in enumerate(tasks):
            _log(f"  Task {i+1}/{len(tasks)}: {task['task_id']} (L{task.get('level', '?')})")

            if dry_run:
                output = f"[DRY RUN] Would generate for {task['task_id']}"
            else:
                runner = CONDITION_RUNNERS[condition]
                kwargs = {"client": client, "task": task, "provider": provider,
                          "model": model, "max_tokens": max_tokens}
                kwargs.update(_extra_condition_kwargs(condition, exp, n_candidates, seed, forge_mode))
                output, _ = runner(**kwargs)

            score_result = adapter.score(output, task)
            results.append({
                "task_id": task["task_id"],
                "condition": condition,
                "level": task.get("level", 0),
                "hsr": score_result["hsr"],
                "ssr": score_result["ssr"],
                "satisfied": score_result["satisfied"],
                "total_checkable": score_result["total_checkable"],
            })
            _log(f"    HSR={score_result['hsr']:.1f} SSR={score_result['ssr']:.3f}")

        all_results[condition] = results
        suffix = f"_{result_suffix}" if result_suffix else ""
        out_path = RESULTS_DIR / f"followbench_{condition}{suffix}.json"
        _save_benchmark_results("followbench", condition, results, out_path)

        level_agg = adapter.aggregate_by_level(
            [{"level": r["level"], "hsr": r["hsr"], "ssr": r["ssr"]} for r in results]
        )
        _log(f"  Per-level: {json.dumps(level_agg, indent=2)}")

    return all_results


def _save_benchmark_results(benchmark: str, condition: str, results: list, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "benchmark": benchmark,
        "condition": condition,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_results": len(results),
        "results": results,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    _log(f"  Saved to {path}")


# ---------------------------------------------------------------------------
# Summary & analysis
# ---------------------------------------------------------------------------

def print_summary(all_results: dict):
    """Print a formatted summary table."""
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    for benchmark, conditions in all_results.items():
        print(f"\n--- {benchmark.upper()} ---")
        for condition, results in conditions.items():
            if not results:
                continue

            if benchmark == "mosaic":
                from collections import defaultdict
                by_count = defaultdict(list)
                for r in results:
                    by_count[r["constraint_count"]].append(r["scc"])
                print(f"  {condition:12s}:", end="")
                for nc in sorted(by_count.keys()):
                    import numpy as np
                    vals = np.array(by_count[nc])
                    print(f"  {nc}c={np.mean(vals):.3f}±{np.std(vals):.3f}", end="")
                print()

            elif benchmark in ("ifeval", "ifbench"):
                import numpy as np
                accs = [r["instruction_accuracy"] for r in results]
                follow_all = sum(1 for r in results if r["follow_all"]) / len(results)
                print(f"  {condition:12s}: instr_acc={np.mean(accs):.3f}  "
                      f"prompt_acc={follow_all:.3f}")

            elif benchmark == "sysprompt":
                from collections import defaultdict
                import numpy as np
                by_count = defaultdict(list)
                for r in results:
                    by_count[r["constraint_count"]].append(r["scc"])
                print(f"  {condition:12s}:", end="")
                for nc in sorted(by_count.keys()):
                    vals = np.array(by_count[nc])
                    print(f"  {nc}g={np.mean(vals):.3f}", end="")
                print()

            elif benchmark == "toolsel":
                from collections import defaultdict
                by_count = defaultdict(list)
                for r in results:
                    by_count[r["n_tools"]].append(1.0 if r["correct"] else 0.0)
                print(f"  {condition:12s}:", end="")
                for nc in sorted(by_count.keys()):
                    import numpy as np
                    vals = np.array(by_count[nc])
                    print(f"  {nc}t={np.mean(vals):.3f}", end="")
                print()

            elif benchmark == "followbench":
                from collections import defaultdict
                by_level = defaultdict(list)
                for r in results:
                    by_level[r["level"]].append(r["ssr"])
                print(f"  {condition:12s}:", end="")
                for lvl in sorted(by_level.keys()):
                    import numpy as np
                    vals = np.array(by_level[lvl])
                    print(f"  L{lvl}={np.mean(vals):.3f}", end="")
                print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="CPGA+SENTINEL Public Benchmark Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--benchmark", "-b",
        choices=BENCHMARKS + ["all"],
        default="all",
        help="Which benchmark to run",
    )
    parser.add_argument(
        "--condition", "-c",
        choices=CONDITIONS + ["all"],
        default="all",
        help="Which condition to run",
    )
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Skip LLM calls, test pipeline")
    parser.add_argument("--max-tasks", type=int, default=None, help="Limit tasks per benchmark")
    parser.add_argument("--provider", choices=["anthropic", "openai"], default="anthropic")
    parser.add_argument("--model", type=str, default=None, help="Override model name from config")
    parser.add_argument("--forge-mode", choices=["type", "full"], default="type",
                        help="FORGE classification mode: type (heuristic) or full (LLM-probe)")
    parser.add_argument("--followbench-repo", type=str, default=None, help="Path to cloned FollowBench repo")
    parser.add_argument("--mosaic-repo", type=str, default=None, help="Path to cloned MOSAIC repo")
    parser.add_argument("--result-suffix", type=str, default=None,
                        help="Suffix appended to result filenames (e.g. 'openai' -> ifeval_baseline_openai.json)")

    args = parser.parse_args()

    config = load_config(args.config)
    client = LLMClient(config)

    if args.model:
        config.setdefault(args.provider, {})["model"] = args.model

    benchmarks = BENCHMARKS if args.benchmark == "all" else [args.benchmark]
    conditions = CONDITIONS if args.condition == "all" else [args.condition]

    _log(f"Running: benchmarks={benchmarks}, conditions={conditions}")
    _log(f"Dry run: {args.dry_run}")
    if args.max_tasks:
        _log(f"Max tasks: {args.max_tasks}")

    all_results = {}

    common_kw = dict(
        provider=args.provider,
        model_override=args.model,
        forge_mode=args.forge_mode,
        result_suffix=args.result_suffix,
    )

    for benchmark in benchmarks:
        _log(f"\n{'='*60}")
        _log(f"BENCHMARK: {benchmark.upper()}")
        _log(f"{'='*60}")

        if benchmark == "mosaic":
            results = run_mosaic(client, conditions, config, args.dry_run, args.max_tasks, **common_kw)
        elif benchmark == "ifeval":
            results = run_ifeval(client, conditions, config, args.dry_run, args.max_tasks, **common_kw)
        elif benchmark == "followbench":
            results = run_followbench(
                client, conditions, config, args.dry_run,
                args.max_tasks, args.followbench_repo, **common_kw,
            )
        elif benchmark == "ifbench":
            results = run_ifbench(client, conditions, config, args.dry_run, args.max_tasks, **common_kw)
        elif benchmark == "sysprompt":
            results = run_sysprompt(client, conditions, config, args.dry_run, args.max_tasks, **common_kw)
        elif benchmark == "toolsel":
            results = run_toolsel(client, conditions, config, args.dry_run, args.max_tasks, **common_kw)
        else:
            continue

        all_results[benchmark] = results

    print_summary(all_results)

    _log(f"\nTotal API cost: ${client.total_cost:.2f}")
    _log(f"Total API calls: {client.total_calls}")

    summary_path = RESULTS_DIR / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "benchmarks": benchmarks,
        "conditions": conditions,
        "total_cost_usd": client.total_cost,
        "total_calls": client.total_calls,
        "dry_run": args.dry_run,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    return all_results


if __name__ == "__main__":
    main()
