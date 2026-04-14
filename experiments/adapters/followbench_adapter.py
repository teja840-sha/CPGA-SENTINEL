"""
FollowBench Benchmark Adapter

Adapts FollowBench (ACL 2024) for CPGA+SENTINEL evaluation.

FollowBench uses a multi-level constraint ladder: Level 0 (base) through
Level 5 (maximum bundled difficulty). Each level adds one additional
fine-grained requirement. Constraint types: content, situation, style,
format, example.

Reference: https://aclanthology.org/2024.acl-long.257/
Repo: https://github.com/YJiangcm/FollowBench
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Any, Callable


# ---------------------------------------------------------------------------
# FollowBench constraint type checkers (deterministic where possible)
# ---------------------------------------------------------------------------

def _check_word_limit(text: str, limit: int, relation: str = "at most") -> bool:
    count = len(text.split())
    if relation == "at most":
        return count <= limit
    elif relation == "at least":
        return count >= limit
    return count == limit


def _check_sentence_limit(text: str, limit: int, relation: str = "at most") -> bool:
    sentences = re.split(r'[.!?]+\s*', text.strip())
    sentences = [s for s in sentences if s.strip()]
    count = len(sentences)
    if relation == "at most":
        return count <= limit
    elif relation == "at least":
        return count >= limit
    return count == limit


def _check_contains_phrase(text: str, phrase: str) -> bool:
    return phrase.lower() in text.lower()


def _check_starts_with(text: str, prefix: str) -> bool:
    return text.strip().startswith(prefix)


def _check_ends_with(text: str, suffix: str) -> bool:
    return text.strip().endswith(suffix)


def _check_paragraph_count(text: str, count: int) -> bool:
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    return len(paras) == count


def _check_has_heading(text: str) -> bool:
    return bool(re.search(r"^#{1,6}\s+", text, re.MULTILINE))


def _check_bullet_list(text: str) -> bool:
    return bool(re.search(r"^\s*[\-\*•]\s+", text, re.MULTILINE))


def _check_numbered_list(text: str) -> bool:
    return bool(re.search(r"^\s*\d+[.)]\s+", text, re.MULTILINE))


def _check_all_lowercase(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    return all(c.islower() for c in letters) if letters else False


def _check_all_uppercase(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    return all(c.isupper() for c in letters) if letters else False


def _check_no_comma(text: str) -> bool:
    return "," not in text


def _check_json_format(text: str) -> bool:
    try:
        stripped = text.strip()
        if stripped.startswith("```"):
            lines = stripped.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            stripped = "\n".join(lines).strip()
        json.loads(stripped)
        return True
    except (json.JSONDecodeError, ValueError):
        return False


FORMAT_CHECKERS: dict[str, Callable] = {
    "word_limit": _check_word_limit,
    "sentence_limit": _check_sentence_limit,
    "paragraph_count": _check_paragraph_count,
    "heading": lambda text: _check_has_heading(text),
    "bullet_list": lambda text: _check_bullet_list(text),
    "numbered_list": lambda text: _check_numbered_list(text),
    "lowercase": lambda text: _check_all_lowercase(text),
    "uppercase": lambda text: _check_all_uppercase(text),
    "no_comma": lambda text: _check_no_comma(text),
    "json_format": lambda text: _check_json_format(text),
    "starts_with": _check_starts_with,
    "ends_with": _check_ends_with,
    "contains_phrase": _check_contains_phrase,
}


# ---------------------------------------------------------------------------
# Synthetic FollowBench tasks (for when repo is not cloned)
# ---------------------------------------------------------------------------

SYNTHETIC_TASKS = [
    {
        "example_id": 1,
        "source": "synthetic",
        "levels": {
            0: {
                "instruction": "Write a short story about a robot learning to paint.",
                "constraints": [],
            },
            1: {
                "instruction": "Write a short story about a robot learning to paint. The story must be exactly 3 paragraphs.",
                "new_constraint": {"type": "format", "text": "The story must be exactly 3 paragraphs.",
                                   "check_fn": lambda text: _check_paragraph_count(text, 3)},
            },
            2: {
                "instruction": "Write a short story about a robot learning to paint. The story must be exactly 3 paragraphs. Include the phrase 'digital brushstrokes' at least once.",
                "new_constraint": {"type": "content", "text": "Include the phrase 'digital brushstrokes' at least once.",
                                   "check_fn": lambda text: _check_contains_phrase(text, "digital brushstrokes")},
            },
            3: {
                "instruction": "Write a short story about a robot learning to paint. The story must be exactly 3 paragraphs. Include the phrase 'digital brushstrokes' at least once. Write in a melancholic tone.",
                "new_constraint": {"type": "style", "text": "Write in a melancholic tone.", "check_fn": None},
            },
            4: {
                "instruction": "Write a short story about a robot learning to paint. The story must be exactly 3 paragraphs. Include the phrase 'digital brushstrokes' at least once. Write in a melancholic tone. Each paragraph must start with a time reference.",
                "new_constraint": {"type": "format", "text": "Each paragraph must start with a time reference.", "check_fn": None},
            },
            5: {
                "instruction": "Write a short story about a robot learning to paint. The story must be exactly 3 paragraphs. Include the phrase 'digital brushstrokes' at least once. Write in a melancholic tone. Each paragraph must start with a time reference. Do not use any commas.",
                "new_constraint": {"type": "format", "text": "Do not use any commas.",
                                   "check_fn": lambda text: _check_no_comma(text)},
            },
        },
    },
    {
        "example_id": 2,
        "source": "synthetic",
        "levels": {
            0: {
                "instruction": "Explain how photosynthesis works.",
                "constraints": [],
            },
            1: {
                "instruction": "Explain how photosynthesis works. Your response must be at most 150 words.",
                "new_constraint": {"type": "format", "text": "Your response must be at most 150 words.",
                                   "check_fn": lambda text: _check_word_limit(text, 150)},
            },
            2: {
                "instruction": "Explain how photosynthesis works. Your response must be at most 150 words. Use a numbered list for the steps.",
                "new_constraint": {"type": "format", "text": "Use a numbered list for the steps.",
                                   "check_fn": lambda text: _check_numbered_list(text)},
            },
            3: {
                "instruction": "Explain how photosynthesis works. Your response must be at most 150 words. Use a numbered list for the steps. Include the word 'chlorophyll' at least 3 times.",
                "new_constraint": {"type": "content", "text": "Include the word 'chlorophyll' at least 3 times.",
                                   "check_fn": lambda text: text.lower().count("chlorophyll") >= 3},
            },
            4: {
                "instruction": "Explain how photosynthesis works. Your response must be at most 150 words. Use a numbered list for the steps. Include the word 'chlorophyll' at least 3 times. Write as if explaining to a 5-year-old.",
                "new_constraint": {"type": "style", "text": "Write as if explaining to a 5-year-old.", "check_fn": None},
            },
            5: {
                "instruction": "Explain how photosynthesis works. Your response must be at most 150 words. Use a numbered list for the steps. Include the word 'chlorophyll' at least 3 times. Write as if explaining to a 5-year-old. Start each numbered step with an action verb.",
                "new_constraint": {"type": "format", "text": "Start each numbered step with an action verb.", "check_fn": None},
            },
        },
    },
    {
        "example_id": 3,
        "source": "synthetic",
        "levels": {
            0: {
                "instruction": "Write a product description for a new smartphone.",
                "constraints": [],
            },
            1: {
                "instruction": "Write a product description for a new smartphone. Use exactly 5 bullet points.",
                "new_constraint": {"type": "format", "text": "Use exactly 5 bullet points.",
                                   "check_fn": lambda text: len(re.findall(r"^\s*[\-\*•]\s+", text, re.MULTILINE)) == 5},
            },
            2: {
                "instruction": "Write a product description for a new smartphone. Use exactly 5 bullet points. Include the phrase 'next generation' somewhere.",
                "new_constraint": {"type": "content", "text": "Include the phrase 'next generation' somewhere.",
                                   "check_fn": lambda text: _check_contains_phrase(text, "next generation")},
            },
            3: {
                "instruction": "Write a product description for a new smartphone. Use exactly 5 bullet points. Include the phrase 'next generation' somewhere. Write in an enthusiastic and persuasive tone.",
                "new_constraint": {"type": "style", "text": "Write in an enthusiastic and persuasive tone.", "check_fn": None},
            },
            4: {
                "instruction": "Write a product description for a new smartphone. Use exactly 5 bullet points. Include the phrase 'next generation' somewhere. Write in an enthusiastic and persuasive tone. End the description with a call to action.",
                "new_constraint": {"type": "content", "text": "End the description with a call to action.", "check_fn": None},
            },
            5: {
                "instruction": "Write a product description for a new smartphone. Use exactly 5 bullet points. Include the phrase 'next generation' somewhere. Write in an enthusiastic and persuasive tone. End the description with a call to action. Each bullet point must be at most 15 words.",
                "new_constraint": {"type": "format", "text": "Each bullet point must be at most 15 words.",
                                   "check_fn": lambda text: all(
                                       len(m.group().split()) <= 15
                                       for m in re.finditer(r"^\s*[\-\*•]\s+(.+)$", text, re.MULTILINE)
                                   )},
            },
        },
    },
]


def _build_constraint_chain(task: dict, level: int) -> list[dict]:
    """Build cumulative constraint list up to the given level."""
    constraints = []
    for lvl in range(1, level + 1):
        level_data = task["levels"].get(lvl)
        if level_data and "new_constraint" in level_data:
            nc = level_data["new_constraint"]
            constraints.append({
                "id": f"fb_{task['example_id']}_L{lvl}_{nc['type']}",
                "text": nc["text"],
                "check_fn": nc.get("check_fn"),
                "type": nc["type"],
                "level": lvl,
            })
    return constraints


class FollowBenchAdapter:
    """Adapter for FollowBench multi-level constraint evaluation."""

    def __init__(self, repo_path: str | Path | None = None):
        self.repo_path = Path(repo_path) if repo_path else None

    def load_tasks(
        self,
        levels: list[int] | None = None,
        max_tasks: int | None = None,
    ) -> list[dict]:
        """Load FollowBench tasks. Tries repo first, falls back to synthetic."""
        if levels is None:
            levels = [1, 2, 3, 4, 5]

        if self.repo_path and self._repo_exists():
            return self._load_from_repo(levels, max_tasks)
        return self._load_synthetic(levels, max_tasks)

    def _repo_exists(self) -> bool:
        if not self.repo_path:
            return False
        return (self.repo_path / "data").exists()

    def _load_from_repo(
        self,
        levels: list[int],
        max_tasks: int | None,
    ) -> list[dict]:
        """Load from cloned FollowBench repo.

        Groups items by (source, example_id) to build the constraint ladder.
        For each example at level N, the constraints are the diffs between
        instructions at levels 0..N.
        """
        data_dir = self.repo_path / "data"

        # Load all items grouped by (file, example_id)
        all_items: dict[str, dict[int, dict]] = {}
        for json_file in sorted(data_dir.glob("*.json")):
            try:
                with open(json_file, encoding="utf-8") as f:
                    items = json.load(f)
            except (json.JSONDecodeError, OSError, UnicodeDecodeError):
                continue
            if not isinstance(items, list):
                continue
            for item in items:
                eid = item.get("example_id", 0)
                key = f"{json_file.stem}_{eid}"
                if key not in all_items:
                    all_items[key] = {}
                all_items[key][item.get("level", 0)] = item

        tasks = []
        for key, level_map in all_items.items():
            for lvl in sorted(level_map.keys()):
                if lvl not in levels:
                    continue
                item = level_map[lvl]
                instruction = item.get("instruction", "")
                category = item.get("category", "")

                # Build constraints by diffing instruction at each step
                constraints = []
                prev_instr = level_map.get(0, {}).get("instruction", "")
                for step in range(1, lvl + 1):
                    step_item = level_map.get(step)
                    if not step_item:
                        continue
                    step_instr = step_item.get("instruction", "")
                    # The new constraint = what was added at this level
                    diff = step_instr.replace(prev_instr, "").strip()
                    if not diff:
                        diff = step_instr  # fallback
                    step_cat = step_item.get("category", "unknown")
                    if "," in step_cat:
                        cats = step_cat.split(",")
                        step_cat = cats[min(step - 1, len(cats) - 1)].strip()
                    check_fn = self._try_build_checker(diff)
                    constraints.append({
                        "id": f"fb_{key}_L{step}_{step_cat}",
                        "text": diff,
                        "check_fn": check_fn,
                        "type": step_cat,
                        "level": step,
                    })
                    prev_instr = step_instr

                tasks.append({
                    "task_id": f"followbench_{key}_L{lvl}",
                    "prompt": instruction,
                    "base_prompt": level_map.get(0, {}).get("instruction", instruction),
                    "constraints": constraints,
                    "constraint_count": len(constraints),
                    "level": lvl,
                    "category": category,
                    "example_id": item.get("example_id", 0),
                    "source": key.rsplit("_", 1)[0],
                })

                if max_tasks and len(tasks) >= max_tasks:
                    return tasks

        print(f"Loaded {len(tasks)} FollowBench tasks from repo")
        return tasks

    def _extract_repo_constraints(
        self,
        instruction: str,
        categories: list[str],
        level: int,
        id_prefix: str,
    ) -> list[dict]:
        """Extract constraints from a FollowBench repo instruction.

        Heuristic: split numbered items or sentence-level requirements.
        """
        constraints = []
        sentences = re.split(r'(?<=[.!?])\s+', instruction)

        if len(sentences) > 1:
            for i, sent in enumerate(sentences[1:], 1):
                cat = categories[min(i - 1, len(categories) - 1)] if categories else "unknown"
                cat = cat.strip()
                check_fn = self._try_build_checker(sent)
                constraints.append({
                    "id": f"{id_prefix}_s{i}_{cat}",
                    "text": sent.strip(),
                    "check_fn": check_fn,
                    "type": cat,
                    "level": min(i, level),
                })

        if not constraints:
            constraints.append({
                "id": f"{id_prefix}_full",
                "text": instruction,
                "check_fn": None,
                "type": "mixed",
                "level": level,
            })

        return constraints

    def _try_build_checker(self, constraint_text: str) -> Callable | None:
        """Try to build a deterministic checker from constraint text."""
        text = constraint_text.lower().strip()

        m = re.search(r"(?:at most|no more than|maximum of)\s+(\d+)\s+words?", text)
        if m:
            limit = int(m.group(1))
            return lambda t, _l=limit: _check_word_limit(t, _l)

        m = re.search(r"(?:at least|minimum of)\s+(\d+)\s+words?", text)
        if m:
            limit = int(m.group(1))
            return lambda t, _l=limit: _check_word_limit(t, _l, "at least")

        m = re.search(r"exactly\s+(\d+)\s+paragraphs?", text)
        if m:
            count = int(m.group(1))
            return lambda t, _c=count: _check_paragraph_count(t, _c)

        if "do not use" in text and "comma" in text:
            return lambda t: _check_no_comma(t)

        if "json" in text and "format" in text:
            return lambda t: _check_json_format(t)

        m = re.search(r"include\s+(?:the\s+)?(?:phrase|word)\s+['\"](.+?)['\"]", text)
        if m:
            phrase = m.group(1)
            return lambda t, _p=phrase: _check_contains_phrase(t, _p)

        if "bullet" in text and "list" in text:
            return lambda t: _check_bullet_list(t)

        if "numbered list" in text:
            return lambda t: _check_numbered_list(t)

        if "lowercase" in text and ("entire" in text or "all" in text):
            return lambda t: _check_all_lowercase(t)

        if "uppercase" in text and ("entire" in text or "all" in text):
            return lambda t: _check_all_uppercase(t)

        return None

    def _load_synthetic(
        self,
        levels: list[int],
        max_tasks: int | None,
    ) -> list[dict]:
        """Generate synthetic FollowBench-style tasks."""
        tasks = []
        for task_def in SYNTHETIC_TASKS:
            for level in levels:
                if level not in task_def["levels"]:
                    continue

                level_data = task_def["levels"][level]
                constraints = _build_constraint_chain(task_def, level)

                tasks.append({
                    "task_id": f"followbench_syn_{task_def['example_id']}_L{level}",
                    "prompt": level_data["instruction"],
                    "base_prompt": task_def["levels"][0]["instruction"],
                    "constraints": constraints,
                    "constraint_count": len(constraints),
                    "level": level,
                    "example_id": task_def["example_id"],
                    "source": "synthetic",
                })

                if max_tasks and len(tasks) >= max_tasks:
                    return tasks

        return tasks

    def extract_constraints(self, task: dict) -> list[dict]:
        return task["constraints"]

    def score(self, output: str, task: dict) -> dict:
        """Score with HSR, SSR, and per-constraint results."""
        constraints = task["constraints"]
        per_constraint = []
        satisfied = 0

        for c in constraints:
            check_fn = c.get("check_fn")
            if check_fn is None:
                per_constraint.append({
                    "id": c["id"],
                    "type": c.get("type", ""),
                    "level": c.get("level", 0),
                    "passed": None,
                    "skipped": True,
                })
                continue

            try:
                passed = bool(check_fn(output))
            except Exception:
                passed = False

            if passed:
                satisfied += 1
            per_constraint.append({
                "id": c["id"],
                "type": c.get("type", ""),
                "level": c.get("level", 0),
                "passed": passed,
            })

        checkable = [p for p in per_constraint if not p.get("skipped")]
        n_checkable = len(checkable)
        n_passed = sum(1 for p in checkable if p["passed"])

        all_passed = all(p["passed"] for p in checkable) if checkable else True
        hsr = 1.0 if all_passed else 0.0
        ssr = n_passed / n_checkable if n_checkable > 0 else 0.0

        return {
            "hsr": hsr,
            "ssr": ssr,
            "satisfied": n_passed,
            "total_checkable": n_checkable,
            "total_constraints": len(constraints),
            "level": task.get("level", 0),
            "per_constraint": per_constraint,
        }

    def compute_csl(self, per_level_results: dict[int, dict]) -> int:
        """Compute Consistent Satisfaction Levels (CSL).

        Number of consecutive levels from 1 where HSR=1.
        """
        csl = 0
        for lvl in range(1, 6):
            result = per_level_results.get(lvl)
            if result and result.get("hsr", 0) == 1.0:
                csl = lvl
            else:
                break
        return csl

    def aggregate_by_level(self, results: list[dict]) -> dict:
        """Aggregate results by constraint level."""
        from collections import defaultdict
        level_stats: dict[int, dict] = defaultdict(lambda: {"hsr_sum": 0.0, "ssr_sum": 0.0, "count": 0})

        for r in results:
            lvl = r.get("level", 0)
            level_stats[lvl]["hsr_sum"] += r.get("hsr", 0.0)
            level_stats[lvl]["ssr_sum"] += r.get("ssr", 0.0)
            level_stats[lvl]["count"] += 1

        return {
            lvl: {
                "mean_hsr": s["hsr_sum"] / s["count"] if s["count"] > 0 else 0.0,
                "mean_ssr": s["ssr_sum"] / s["count"] if s["count"] > 0 else 0.0,
                "n_tasks": s["count"],
            }
            for lvl, s in sorted(level_stats.items())
        }

    def aggregate_by_type(self, results: list[dict]) -> dict:
        """Aggregate results by constraint type."""
        from collections import defaultdict
        type_stats: dict[str, dict] = defaultdict(lambda: {"passed": 0, "total": 0})

        for r in results:
            for pc in r.get("per_constraint", []):
                if pc.get("skipped"):
                    continue
                ctype = pc.get("type", "unknown")
                type_stats[ctype]["total"] += 1
                if pc["passed"]:
                    type_stats[ctype]["passed"] += 1

        return {
            ctype: {
                "accuracy": s["passed"] / s["total"] if s["total"] > 0 else 0.0,
                "passed": s["passed"],
                "total": s["total"],
            }
            for ctype, s in sorted(type_stats.items())
        }
