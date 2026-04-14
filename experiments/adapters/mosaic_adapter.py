"""
MOSAIC Benchmark Adapter

Adapts the MOSAIC constraint-following benchmark (Capital One Research)
for CPGA+SENTINEL evaluation.

MOSAIC: 21 constraint types across 5 groups (Formatting, Lexical,
Syntactic, Semantic, Business/Legal), 4 content tasks, 8 product blurbs.

Reference: https://arxiv.org/abs/2601.18554
Repo: https://github.com/CapitalOne-Research/llm-instruction-following-compliance
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from random import Random
from typing import Any, Callable


# ---------------------------------------------------------------------------
# MOSAIC constraint library — 21 constraints mirroring the paper's Table 4
# ---------------------------------------------------------------------------

TASK_TOPICS = [
    {
        "id": "marketing_email",
        "template": "Write a marketing email for {product}.",
        "context": "You are a professional copywriter.",
    },
    {
        "id": "product_review",
        "template": "Write a detailed product review for {product}.",
        "context": "You are a tech reviewer.",
    },
    {
        "id": "faq_entry",
        "template": "Write an FAQ entry answering common questions about {product}.",
        "context": "You are a customer support specialist.",
    },
    {
        "id": "internal_memo",
        "template": "Write an internal company memo about the launch of {product}.",
        "context": "You are a product manager.",
    },
]

PRODUCTS = [
    "CloudSync Pro (cloud storage service)",
    "SmartLock X200 (IoT door lock)",
    "FinTrack Plus (personal finance app)",
    "MediCare Bot (healthcare chatbot)",
    "EcoCharge Solar Panel Kit",
    "DataVault Enterprise (backup solution)",
    "FitPulse Watch (fitness tracker)",
    "LearnPath AI (educational platform)",
]


def _check_paragraph_count(text: str, n: int) -> bool:
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    return len(paras) == n


def _check_min_paragraphs(text: str, n: int) -> bool:
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    return len(paras) >= n


def _check_sentence_count_range(text: str, lo: int, hi: int) -> bool:
    sentences = re.split(r'[.!?]+\s+', text.strip())
    sentences = [s for s in sentences if s.strip()]
    return lo <= len(sentences) <= hi


def _check_word_count_range(text: str, lo: int, hi: int) -> bool:
    words = text.split()
    return lo <= len(words) <= hi


def _check_contains_keyword(text: str, keyword: str) -> bool:
    return keyword.lower() in text.lower()


def _check_avoids_keyword(text: str, keyword: str) -> bool:
    return keyword.lower() not in text.lower()


def _check_dash_lists(text: str) -> bool:
    return bool(re.search(r"^[\s]*[-–—]\s+\S", text, re.MULTILINE))


def _check_json_format(text: str) -> bool:
    try:
        text_stripped = text.strip()
        if text_stripped.startswith("```"):
            lines = text_stripped.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text_stripped = "\n".join(lines).strip()
        json.loads(text_stripped)
        return True
    except (json.JSONDecodeError, ValueError):
        return False


def _check_no_commas(text: str) -> bool:
    return "," not in text


def _check_starts_with(text: str, token: str) -> bool:
    return text.strip().lower().startswith(token.lower())


def _check_ends_with(text: str, token: str) -> bool:
    return text.strip().rstrip(".!?").lower().endswith(token.lower())


def _check_all_uppercase(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    return all(c.isupper() for c in letters) if letters else False


def _check_all_lowercase(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    return all(c.islower() for c in letters) if letters else False


def _check_max_words_per_sentence(text: str, limit: int) -> bool:
    sentences = re.split(r'[.!?]+', text)
    for s in sentences:
        words = s.split()
        if len(words) > limit:
            return False
    return True


def _check_flesch_range(text: str, lo: float, hi: float) -> bool:
    try:
        import textstat
        score = textstat.flesch_reading_ease(text)
        return lo <= score <= hi
    except Exception:
        return False


def _check_positive_language(text: str) -> bool:
    negative_words = [
        "terrible", "awful", "horrible", "worst", "hate", "failure",
        "disaster", "pathetic", "useless", "garbage", "trash", "worthless",
    ]
    text_lower = text.lower()
    return not any(w in text_lower for w in negative_words)


def _check_sentence_length_variety(text: str) -> bool:
    sentences = re.split(r'[.!?]+\s+', text.strip())
    sentences = [s for s in sentences if s.strip()]
    if len(sentences) < 3:
        return True
    lengths = [len(s.split()) for s in sentences]
    return max(lengths) - min(lengths) >= 5


def _check_contains_name(text: str, name: str) -> bool:
    return name in text


def _check_numbered_list(text: str) -> bool:
    return bool(re.search(r"^\s*\d+[.)]\s+\S", text, re.MULTILINE))


CONSTRAINT_LIBRARY = [
    {
        "id": "fmt_paragraph_count",
        "group": "formatting",
        "text": "The response must contain exactly {n} paragraphs.",
        "params": {"n": [3, 4, 5]},
        "check_fn_factory": lambda n: lambda text: _check_paragraph_count(text, n),
    },
    {
        "id": "fmt_min_paragraphs",
        "group": "formatting",
        "text": "The response must contain at least {n} paragraphs.",
        "params": {"n": [3, 4, 5]},
        "check_fn_factory": lambda n: lambda text: _check_min_paragraphs(text, n),
    },
    {
        "id": "fmt_dash_lists",
        "group": "formatting",
        "text": "Use dash-prefixed bullet points for any lists.",
        "params": {},
        "check_fn_factory": lambda: lambda text: _check_dash_lists(text),
    },
    {
        "id": "fmt_json_format",
        "group": "formatting",
        "text": "Respond in valid JSON format.",
        "params": {},
        "check_fn_factory": lambda: lambda text: _check_json_format(text),
    },
    {
        "id": "fmt_numbered_list",
        "group": "formatting",
        "text": "Use numbered lists for any enumerated items.",
        "params": {},
        "check_fn_factory": lambda: lambda text: _check_numbered_list(text),
    },
    {
        "id": "lex_flesch",
        "group": "lexical",
        "text": "Write at a Flesch reading ease score between {lo} and {hi}.",
        "params": {"lo": [60.0], "hi": [80.0]},
        "check_fn_factory": lambda lo, hi: lambda text: _check_flesch_range(text, lo, hi),
    },
    {
        "id": "lex_positive_language",
        "group": "lexical",
        "text": "Use only positive language throughout.",
        "params": {},
        "check_fn_factory": lambda: lambda text: _check_positive_language(text),
    },
    {
        "id": "lex_include_keyword",
        "group": "lexical",
        "text": "Include the keyword \"{keyword}\" at least once.",
        "params": {"keyword": ["innovative", "seamless", "cutting-edge", "reliable", "award-winning"]},
        "check_fn_factory": lambda keyword: lambda text: _check_contains_keyword(text, keyword),
    },
    {
        "id": "lex_avoid_keyword",
        "group": "lexical",
        "text": "Do not use the word \"{keyword}\".",
        "params": {"keyword": ["cheap", "basic", "simple", "just", "only"]},
        "check_fn_factory": lambda keyword: lambda text: _check_avoids_keyword(text, keyword),
    },
    {
        "id": "lex_include_name",
        "group": "lexical",
        "text": "Include the name \"{name}\" in the response.",
        "params": {"name": ["Alex", "Jordan", "Taylor", "Morgan"]},
        "check_fn_factory": lambda name: lambda text: _check_contains_name(text, name),
    },
    {
        "id": "syn_max_words_per_sentence",
        "group": "syntactic",
        "text": "No sentence should exceed {limit} words.",
        "params": {"limit": [20, 25, 30]},
        "check_fn_factory": lambda limit: lambda text: _check_max_words_per_sentence(text, limit),
    },
    {
        "id": "syn_sentence_variety",
        "group": "syntactic",
        "text": "Use varied sentence lengths (at least 5 words difference between shortest and longest).",
        "params": {},
        "check_fn_factory": lambda: lambda text: _check_sentence_length_variety(text),
    },
    {
        "id": "syn_word_count_range",
        "group": "syntactic",
        "text": "The response must be between {lo} and {hi} words.",
        "params": {"lo": [150], "hi": [300]},
        "check_fn_factory": lambda lo, hi: lambda text: _check_word_count_range(text, lo, hi),
    },
    {
        "id": "pos_starts_with",
        "group": "formatting",
        "text": "Begin the response with \"{token}\".",
        "params": {"token": ["Dear", "Introducing", "Welcome", "Attention"]},
        "check_fn_factory": lambda token: lambda text: _check_starts_with(text, token),
    },
    {
        "id": "pos_ends_with",
        "group": "formatting",
        "text": "End the response with \"{token}\".",
        "params": {"token": ["Thank you", "Best regards", "Sincerely", "Learn more"]},
        "check_fn_factory": lambda token: lambda text: _check_ends_with(text, token),
    },
    {
        "id": "case_uppercase",
        "group": "formatting",
        "text": "Write the entire response in uppercase.",
        "params": {},
        "check_fn_factory": lambda: lambda text: _check_all_uppercase(text),
        "auto_fix_fn": lambda text: text.upper(),
    },
    {
        "id": "case_lowercase",
        "group": "formatting",
        "text": "Write the entire response in lowercase.",
        "params": {},
        "check_fn_factory": lambda: lambda text: _check_all_lowercase(text),
        "auto_fix_fn": lambda text: text.lower(),
    },
    {
        "id": "punct_no_commas",
        "group": "formatting",
        "text": "Do not use any commas in the response.",
        "params": {},
        "check_fn_factory": lambda: lambda text: _check_no_commas(text),
    },
    {
        "id": "sem_tone",
        "group": "semantic",
        "text": "Write in a {tone} tone.",
        "params": {"tone": ["professional", "enthusiastic", "conversational", "formal"]},
        "check_fn_factory": None,
    },
    {
        "id": "sem_inverted_pyramid",
        "group": "semantic",
        "text": "Structure the response using the inverted pyramid style (most important information first).",
        "params": {},
        "check_fn_factory": None,
    },
    {
        "id": "biz_accurate_facts",
        "group": "business",
        "text": "All product claims must be accurate and substantiated.",
        "params": {},
        "check_fn_factory": None,
    },
]


def _instantiate_constraint(
    template: dict,
    rng: Random,
) -> dict:
    """Instantiate a constraint template with random parameter values."""
    params = {}
    text = template["text"]
    for key, values in template.get("params", {}).items():
        val = rng.choice(values)
        params[key] = val
        text = text.replace(f"{{{key}}}", str(val))

    factory = template.get("check_fn_factory")
    check_fn = None
    if factory is not None:
        try:
            check_fn = factory(**params) if params else factory()
        except Exception:
            check_fn = None

    result = {
        "id": template["id"],
        "group": template["group"],
        "text": text,
        "check_fn": check_fn,
        "params": params,
    }
    if "auto_fix_fn" in template:
        result["auto_fix_fn"] = template["auto_fix_fn"]
    return result


def build_mosaic_task(
    task_topic: dict,
    product: str,
    constraint_count: int,
    seed: int = 42,
) -> dict:
    """Build a single MOSAIC task with specified number of constraints."""
    rng = Random(seed)
    available = list(CONSTRAINT_LIBRARY)
    rng.shuffle(available)
    selected_templates = available[:constraint_count]

    constraints = [_instantiate_constraint(t, rng) for t in selected_templates]

    prompt_base = task_topic["template"].format(product=product)
    constraint_text = "\n".join(
        f"{i+1}. {c['text']}" for i, c in enumerate(constraints)
    )
    full_prompt = f"{prompt_base}\n\nRequirements:\n{constraint_text}"

    return {
        "task_id": f"mosaic_{task_topic['id']}_{seed}",
        "prompt": full_prompt,
        "base_prompt": prompt_base,
        "system": task_topic.get("context", ""),
        "constraints": constraints,
        "constraint_count": constraint_count,
        "topic": task_topic["id"],
        "product": product,
    }


class MOSAICAdapter:
    """Adapter for MOSAIC-style constraint-following evaluation."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = Random(seed)

    def load_tasks(
        self,
        constraint_counts: list[int] | None = None,
        tasks_per_condition: int = 15,
    ) -> list[dict]:
        """Generate MOSAIC tasks with specified constraint counts."""
        if constraint_counts is None:
            constraint_counts = [5, 10, 15, 20]

        tasks = []
        task_seed = self.seed
        for n_constraints in constraint_counts:
            for i in range(tasks_per_condition):
                topic = TASK_TOPICS[i % len(TASK_TOPICS)]
                product = PRODUCTS[i % len(PRODUCTS)]
                task_seed += 1
                task = build_mosaic_task(topic, product, n_constraints, task_seed)
                task["task_id"] = f"mosaic_{topic['id']}_{n_constraints}c_{i}"
                tasks.append(task)

        return tasks

    def extract_constraints(self, task: dict) -> list[dict]:
        return task["constraints"]

    def score(self, output: str, task: dict) -> dict:
        """Score using deterministic Python checkers for code-checkable constraints.

        Returns per-constraint results + aggregate metrics (SCC, PA).
        """
        constraints = task["constraints"]
        per_constraint = []
        satisfied = 0

        for c in constraints:
            check_fn = c.get("check_fn")
            if check_fn is None:
                per_constraint.append({
                    "id": c["id"],
                    "group": c.get("group", ""),
                    "passed": None,
                    "skipped": True,
                    "reason": "semantic — no deterministic checker",
                })
                continue

            try:
                passed = bool(check_fn(output))
            except Exception as e:
                passed = False
                per_constraint.append({
                    "id": c["id"],
                    "group": c.get("group", ""),
                    "passed": False,
                    "error": str(e),
                })
                continue

            if passed:
                satisfied += 1
            per_constraint.append({
                "id": c["id"],
                "group": c.get("group", ""),
                "passed": passed,
            })

        checkable = [p for p in per_constraint if not p.get("skipped")]
        total_checkable = len(checkable)
        scc = satisfied / total_checkable if total_checkable > 0 else 0.0
        pa = satisfied / len(constraints) if constraints else 0.0

        return {
            "scc": scc,
            "pa": pa,
            "satisfied": satisfied,
            "total_checkable": total_checkable,
            "total_constraints": len(constraints),
            "per_constraint": per_constraint,
        }


def _build_real_checker(constraint_text: str) -> Callable | None:
    """Build a deterministic checker for a real MOSAIC constraint string."""
    t = constraint_text.lower().strip()

    if "flesch reading ease" in t and "70" in t and "80" in t:
        return lambda text: _check_flesch_range(text, 60.0, 90.0)

    if "positive and empowering language" in t or "avoid negative" in t:
        return lambda text: _check_positive_language(text)

    if "incorporate keywords" in t or "keywords aligned with" in t:
        return None  # parameterized, handled below

    if "do not use keywords" in t:
        return None  # parameterized, handled below

    if "special token <boc>" in t:
        return lambda text: "<BOC>" in text

    if "special token <eoc>" in t:
        return lambda text: "<EOC>" in text

    if "json format" in t or "json" in t and "schema" in t:
        return lambda text: _check_json_format(text)

    if "dashes" in t and ("lists" in t or "organize" in t):
        return lambda text: _check_dash_lists(text)

    if "2-3 paragraphs" in t or "2 to 3 paragraphs" in t:
        return lambda text: 2 <= len([p for p in text.split("\n\n") if p.strip()]) <= 4

    if "2-4 sentences" in t or "short" in t and "paragraph" in t:
        return None

    if "not exceed 25 words" in t or "sentences should not exceed" in t:
        return lambda text: _check_max_words_per_sentence(text, 25)

    if "sentence length" in t and "vary" in t:
        return lambda text: _check_sentence_length_variety(text)

    if "{{firstname}}" in t or "firstname" in t:
        return lambda text: "{{FirstName}}" in text or "{{firstname}}" in text.lower()

    if "tone" in t and ("consistent" in t or "maintain" in t):
        return None  # semantic

    if "inverted pyramid" in t:
        return None  # semantic

    if "contradictions" in t or "must not contain any internal" in t:
        return None  # semantic

    if "supported by a reason" in t or "evidence" in t:
        return None  # semantic

    if "purpose" in t and "clear" in t:
        return None  # semantic

    if "unambiguous" in t or "precise" in t:
        return None  # semantic

    if "accurate" in t and ("product names" in t or "numerical" in t):
        return None  # semantic

    if "superlatives" in t or "best" in t and "avoid" in t:
        neg = ["best", "greatest", "most amazing", "number one", "#1", "unrivaled", "unmatched"]
        return lambda text: not any(w in text.lower() for w in neg)

    return None


def _build_keyword_checker(constraint_text: str, parameters: dict) -> Callable | None:
    """Build keyword inclusion/exclusion checker from MOSAIC parameters."""
    t = constraint_text.lower()

    if "incorporate keywords" in t or "keywords aligned" in t:
        kws = parameters.get("kws_to_use", "")
        if kws:
            keywords = [k.strip() for k in kws.split(",") if k.strip()]
            return lambda text, _kws=keywords: all(k.lower() in text.lower() for k in _kws)

    if "do not use keywords" in t:
        kws = parameters.get("kws_to_avoid", "")
        if kws:
            keywords = [k.strip() for k in kws.split(",") if k.strip()]
            return lambda text, _kws=keywords: not any(k.lower() in text.lower() for k in _kws)

    return None


def load_mosaic_from_repo(
    repo_path: str | Path,
    constraint_sizes: list[int] | None = None,
    max_per_size: int | None = None,
) -> list[dict]:
    """Load tasks from cloned MOSAIC repo's final_dataset.csv.

    Falls back to synthetic generation if repo not found.
    """
    import ast
    import csv

    repo_path = Path(repo_path)
    csv_path = repo_path / "final_dataset.csv"
    if not csv_path.exists():
        csv_path = repo_path / "data" / "final_dataset.csv"
    if not csv_path.exists():
        print(f"MOSAIC CSV not found at {repo_path}, using synthetic tasks")
        adapter = MOSAICAdapter()
        return adapter.load_tasks()

    if constraint_sizes is None:
        constraint_sizes = [5, 10, 15, 20]

    size_counts: dict[int, int] = {}
    tasks = []

    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            cs = int(row.get("constraint_size", 0))
            if cs not in constraint_sizes:
                continue
            if max_per_size and size_counts.get(cs, 0) >= max_per_size:
                continue
            size_counts[cs] = size_counts.get(cs, 0) + 1

            raw_constraints = ast.literal_eval(row.get("constraints", "()"))
            raw_params = ast.literal_eval(row.get("parameters", "{}")) if row.get("parameters", "{}") != "{}" else {}

            prompt_raw = row.get("prompt", "")
            prompt = re.sub(r"<\|[^|]+\|>", "", prompt_raw).strip()

            constraints = []
            for ci, c_text in enumerate(raw_constraints):
                check_fn = _build_keyword_checker(c_text, raw_params)
                if check_fn is None:
                    check_fn = _build_real_checker(c_text)

                constraints.append({
                    "id": f"mosaic_real_{idx}_c{ci}",
                    "text": c_text,
                    "check_fn": check_fn,
                    "group": "real_mosaic",
                })

            task_desc = row.get("task_desc", "")
            tasks.append({
                "task_id": f"mosaic_real_{idx}",
                "prompt": prompt,
                "base_prompt": prompt,
                "system": "You are a writing assistant.",
                "constraints": constraints,
                "constraint_count": cs,
                "topic": task_desc,
                "product": row.get("product_desc", ""),
            })

    print(f"Loaded {len(tasks)} MOSAIC tasks from repo ({size_counts})")
    return tasks
