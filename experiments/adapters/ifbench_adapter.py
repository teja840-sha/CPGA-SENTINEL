"""
IFBench Benchmark Adapter (Allen AI, NeurIPS 2025)

300 test prompts with 58 OOD constraint types and built-in deterministic
Python verification functions. Significantly harder than IFEval — top model
achieves only 69.3% vs IFEval's 95%+.

Reference: https://github.com/allenai/IFBench
Dataset: https://huggingface.co/datasets/allenai/IFBench_test
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable


def _load_ifbench_modules(repo_path: Path):
    """Import IFBench's native instruction classes and registry."""
    repo_str = str(repo_path)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)

    import importlib
    instructions_mod = importlib.import_module("instructions")
    registry_mod = importlib.import_module("instructions_registry")
    eval_lib = importlib.import_module("evaluation_lib")
    return instructions_mod, registry_mod, eval_lib


class IFBenchAdapter:
    """Adapter for Allen AI's IFBench using their native verification functions."""

    def __init__(self, repo_path: str | Path | None = None):
        if repo_path is None:
            repo_path = Path(__file__).parent.parent / "repos" / "ifbench"
        self.repo_path = Path(repo_path)
        self._instructions_mod = None
        self._registry = None
        self._eval_lib = None

    def _ensure_loaded(self):
        if self._registry is None:
            self._instructions_mod, reg_mod, self._eval_lib = _load_ifbench_modules(self.repo_path)
            self._registry = reg_mod.INSTRUCTION_DICT

    def load_tasks(self, max_tasks: int | None = None) -> list[dict]:
        """Load IFBench from HuggingFace or local JSONL."""
        self._ensure_loaded()

        tasks = []
        # Try HuggingFace first
        try:
            from datasets import load_dataset
            ds = load_dataset("allenai/IFBench_test", split="train")
            source = "huggingface"
        except Exception:
            # Fall back to local JSONL
            jsonl_path = self.repo_path / "data" / "IFBench_test.jsonl"
            if not jsonl_path.exists():
                print(f"IFBench data not found at {jsonl_path}")
                return []
            ds = []
            with open(jsonl_path, encoding="utf-8") as f:
                for line in f:
                    ds.append(json.loads(line))
            source = "local"

        for i, row in enumerate(ds):
            if max_tasks and i >= max_tasks:
                break

            prompt = row["prompt"]
            instruction_ids = row["instruction_id_list"]
            kwargs_list = row["kwargs"]

            constraints = []
            for j, (iid, kw) in enumerate(zip(instruction_ids, kwargs_list)):
                # Filter None values from kwargs
                clean_kw = {k: v for k, v in kw.items() if v is not None}

                # Build checker using IFBench's native verification
                check_fn = self._build_native_checker(iid, clean_kw, prompt)
                constraints.append({
                    "id": f"ifbench_{i}_{iid}_{j}",
                    "instruction_id": iid,
                    "text": f"[{iid}]",
                    "check_fn": check_fn,
                    "kwargs": clean_kw,
                })

            tasks.append({
                "task_id": f"ifbench_{i}",
                "prompt": prompt,
                "base_prompt": prompt,
                "constraints": constraints,
                "constraint_count": len(constraints),
                "instruction_ids": instruction_ids,
                "source": source,
            })

        print(f"Loaded {len(tasks)} IFBench tasks from {source}")
        return tasks

    def _build_native_checker(
        self, instruction_id: str, kwargs: dict, prompt: str
    ) -> Callable[[str], bool] | None:
        """Build a checker using IFBench's own instruction classes."""
        try:
            cls = self._registry.get(instruction_id)
            if cls is None:
                return None

            instruction = cls(instruction_id)
            instruction.build_description(**kwargs)

            args = instruction.get_instruction_args()
            if args and "prompt" in args:
                instruction.build_description(prompt=prompt)

            def checker(response: str, _inst=instruction) -> bool:
                if not response or not response.strip():
                    return False
                return _inst.check_following(response)

            return checker
        except Exception:
            return None

    def extract_constraints(self, task: dict) -> list[dict]:
        return task["constraints"]

    def score(self, output: str, task: dict) -> dict:
        """Score using IFBench's native verification functions."""
        constraints = task["constraints"]
        per_instruction = []
        all_passed = True

        for c in constraints:
            check_fn = c.get("check_fn")
            if check_fn is None:
                per_instruction.append({
                    "id": c["id"],
                    "instruction_id": c.get("instruction_id", ""),
                    "passed": None,
                    "skipped": True,
                })
                continue
            try:
                passed = bool(check_fn(output))
            except Exception:
                passed = False
            per_instruction.append({
                "id": c["id"],
                "instruction_id": c.get("instruction_id", ""),
                "passed": passed,
            })
            if not passed:
                all_passed = False

        checkable = [p for p in per_instruction if not p.get("skipped")]
        n_passed = sum(1 for p in checkable if p["passed"])
        n_total = len(checkable)

        return {
            "follow_all_instructions": all_passed,
            "instruction_accuracy": n_passed / n_total if n_total > 0 else 0.0,
            "prompt_accuracy": 1.0 if all_passed else 0.0,
            "n_passed": n_passed,
            "n_total": n_total,
            "per_instruction": per_instruction,
        }

    def aggregate_by_category(self, results: list[dict]) -> dict:
        """Aggregate by instruction type prefix."""
        from collections import defaultdict
        stats: dict[str, dict] = defaultdict(lambda: {"passed": 0, "total": 0})

        for r in results:
            for pi in r.get("per_instruction", []):
                if pi.get("skipped"):
                    continue
                iid = pi.get("instruction_id", pi["id"])
                cat = iid.split(":")[0] if ":" in iid else "other"
                stats[cat]["total"] += 1
                if pi["passed"]:
                    stats[cat]["passed"] += 1

        return {
            cat: {
                "accuracy": s["passed"] / s["total"] if s["total"] > 0 else 0.0,
                **s,
            }
            for cat, s in sorted(stats.items())
        }
