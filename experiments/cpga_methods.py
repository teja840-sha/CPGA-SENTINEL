"""
CPGA+SENTINEL Methods — Benchmark-Oriented Implementations

Implements FORGE (constraint filtering), CADG (permutation-based selection),
and SENTINEL (post-generation checking/fixing) adapted for public benchmarks.

These are the core CPGA components extracted from the article-generator-v3
pipeline and generalized for arbitrary constraint-satisfaction tasks.
"""

from __future__ import annotations

import re
import json
from dataclasses import dataclass, field
from random import Random
from typing import Any, Callable


# ---------------------------------------------------------------------------
# FORGE: Constraint classification into enforcement layers
# ---------------------------------------------------------------------------

LAYER_NAMES = {
    0: "compiled_regex",
    1: "template_structural",
    3: "prompt_residual",
    4: "post_gen_judge",
}

TOKEN_BAN_PATTERNS = [
    (r"\b(?:must\s+not|do\s+not|don't|avoid|never)\s+(?:use|include|contain|mention|write)\b", "negation_constraint"),
    (r"\b(?:exactly|precisely)\s+\d+\b", "exact_count"),
    (r"\b(?:at\s+least|at\s+most|no\s+more\s+than|no\s+fewer\s+than)\s+\d+\b", "bound_count"),
    (r"\b(?:begin|start|end)\s+with\b", "positional"),
    (r"\bJSON\s*(?:format|schema|object)\b", "json_format"),
    (r"\b(?:bullet|numbered)\s+list\b", "list_format"),
    (r"\b(?:all\s+)?(?:uppercase|lowercase|capital)\b", "case_constraint"),
    (r"\bno\s+comma\b", "punctuation"),
]

SEMANTIC_KEYWORDS = [
    "tone", "style", "formal", "informal", "humorous", "persuasive",
    "creative", "engaging", "professional", "casual", "empathetic",
    "assertive", "nuanced", "balanced", "objective", "subjective",
    "compelling", "vivid", "concise", "elaborate",
]

CODE_CHECKABLE_PATTERNS = [
    r"\b(?:paragraph|sentence|word|character)\s*(?:count|length|limit)\b",
    r"\b\d+\s*(?:paragraphs?|sentences?|words?|characters?|lines?|bullets?)\b",
    r"\b(?:include|contain|mention)\s+(?:the\s+)?(?:word|phrase|keyword)\b",
    r"\bplaceholder\b",
    r"\bpostscript\b",
    r"\bP\.?S\.?\b",
    r"\b(?:title|heading|header|section)\b",
    r"\b(?:highlight|bold|italic|underline)\b",
]


@dataclass
class ClassifiedConstraint:
    id: str
    text: str
    layer: int
    check_fn: Callable[[str], bool] | None = None
    probe_hit: str = ""
    auto_fix_fn: Callable[[str], str] | None = None
    original: dict = field(default_factory=dict)


def _probe_token_ban(text: str) -> str | None:
    for pat, name in TOKEN_BAN_PATTERNS:
        if re.search(pat, text, re.IGNORECASE):
            return name
    return None


def _probe_code_checkable(text: str) -> bool:
    for pat in CODE_CHECKABLE_PATTERNS:
        if re.search(pat, text, re.IGNORECASE):
            return True
    return False


def _probe_semantic(text: str) -> bool:
    text_lower = text.lower()
    hits = sum(1 for kw in SEMANTIC_KEYWORDS if kw in text_lower)
    return hits >= 2


def classify_constraint(constraint: dict) -> ClassifiedConstraint:
    """Classify a single constraint into a FORGE layer.

    Layer 0: Can be checked with regex/code deterministically
    Layer 1: Structural template (JSON schema, list format)
    Layer 3: Must stay in prompt (needs LLM reasoning)
    Layer 4: Post-generation judge (semantic quality)
    """
    cid = constraint["id"]
    text = constraint["text"]
    check_fn = constraint.get("check_fn")

    token_ban = _probe_token_ban(text)
    if token_ban:
        return ClassifiedConstraint(
            id=cid, text=text, layer=0,
            check_fn=check_fn, probe_hit=f"token_ban:{token_ban}",
            original=constraint,
        )

    if _probe_code_checkable(text):
        return ClassifiedConstraint(
            id=cid, text=text, layer=0 if check_fn else 1,
            check_fn=check_fn, probe_hit="code_checkable",
            original=constraint,
        )

    if _probe_semantic(text):
        return ClassifiedConstraint(
            id=cid, text=text, layer=4,
            check_fn=check_fn, probe_hit="semantic",
            original=constraint,
        )

    return ClassifiedConstraint(
        id=cid, text=text, layer=3,
        check_fn=check_fn, probe_hit="residual",
        original=constraint,
    )


def classify_constraint_full(
    constraint: dict,
    client: Any,
    provider: str = "openai",
    model: str | None = None,
) -> ClassifiedConstraint:
    """Full FORGE LLM-probe classification (production mode).

    Three-phase protocol:
    1. LLM classifies constraint as token/structural/semantic/holistic
    2. LLM generates a Python checker function
    3. Checker is validated against 3 test outputs; demoted to Layer 3 on failure
    """
    cid = constraint["id"]
    text = constraint["text"]
    check_fn = constraint.get("check_fn")

    classify_prompt = (
        "Classify this constraint into exactly one category.\n\n"
        f"Constraint: \"{text}\"\n\n"
        "Categories:\n"
        "- TOKEN: Can be verified by regex or simple string matching (word bans, "
        "exact counts, positional checks, case requirements, punctuation rules)\n"
        "- STRUCTURAL: Can be verified by structural parsing (JSON schema, list "
        "format, section structure, markdown formatting)\n"
        "- SEMANTIC: Requires understanding meaning, tone, style, persuasiveness, "
        "creativity — cannot be checked by code\n"
        "- HOLISTIC: Requires evaluating overall quality, coherence, or "
        "cross-section consistency\n\n"
        "Respond with ONLY the category name (TOKEN, STRUCTURAL, SEMANTIC, or HOLISTIC)."
    )
    try:
        category_text, _, _ = client.generate(
            classify_prompt, provider=provider, model=model,
            max_tokens=20, temperature=0.0,
        )
        category = category_text.strip().upper()
    except Exception:
        category = "SEMANTIC"

    layer_map = {"TOKEN": 0, "STRUCTURAL": 1, "SEMANTIC": 4, "HOLISTIC": 4}
    layer = layer_map.get(category, 3)

    if layer in (0, 1) and check_fn is None:
        checker_prompt = (
            "Write a Python function `check(output: str) -> bool` that returns True "
            "if the output satisfies this constraint, False otherwise.\n\n"
            f"Constraint: \"{text}\"\n\n"
            "Return ONLY the function body, no imports needed (re is available). "
            "The function must be self-contained."
        )
        try:
            code_text, _, _ = client.generate(
                checker_prompt, provider=provider, model=model,
                max_tokens=300, temperature=0.0,
            )
            code_clean = code_text.strip()
            if "```" in code_clean:
                code_clean = code_clean.split("```")[1]
                if code_clean.startswith("python"):
                    code_clean = code_clean[6:]
                code_clean = code_clean.strip()

            ns: dict[str, Any] = {"re": __import__("re")}
            exec(code_clean, ns)
            generated_fn = ns.get("check")

            if generated_fn and callable(generated_fn):
                test_outputs = [
                    "This is a simple test output for validation.",
                    "Another test output with numbers 123 and UPPERCASE.",
                    "A third output.\n\nWith multiple paragraphs.\n1. And a list.",
                ]
                valid = True
                for to in test_outputs:
                    try:
                        result = generated_fn(to)
                        if not isinstance(result, bool):
                            valid = False
                            break
                    except Exception:
                        valid = False
                        break

                if valid:
                    check_fn = generated_fn
                else:
                    layer = 3
            else:
                layer = 3
        except Exception:
            layer = 3

    return ClassifiedConstraint(
        id=cid, text=text, layer=layer,
        check_fn=check_fn, probe_hit=f"llm_probe:{category}",
        original=constraint,
    )


def forge_classify(
    constraints: list[dict],
    mode: str = "type",
    client: Any = None,
    provider: str = "openai",
    model: str | None = None,
) -> dict[int, list[ClassifiedConstraint]]:
    """Classify all constraints into FORGE layers. Returns {layer: [constraints]}.

    mode='type': heuristic type-based routing (default, benchmark mode)
    mode='full': LLM-probe three-phase classification (production mode)
    """
    layers: dict[int, list[ClassifiedConstraint]] = {0: [], 1: [], 3: [], 4: []}
    for c in constraints:
        if mode == "full" and client is not None:
            classified = classify_constraint_full(c, client, provider, model)
        else:
            classified = classify_constraint(c)
        layers[classified.layer].append(classified)
    return layers


def forge_filter_prompt(
    task_prompt: str,
    constraints: list[dict],
    forge_mode: str = "type",
    client: Any = None,
    provider: str = "openai",
    model: str | None = None,
) -> tuple[str, dict[int, list[ClassifiedConstraint]]]:
    """FORGE: Remove Layer 0/1 constraints from prompt, keep Layer 3 residual.

    Returns (filtered_prompt, classified_layers).
    Layer 0/1 constraints are enforced post-generation by code.
    Layer 3 stays in the prompt.
    Layer 4 is evaluated post-generation by judge.
    """
    layers = forge_classify(constraints, mode=forge_mode, client=client,
                            provider=provider, model=model)

    removable_texts = set()
    for layer_idx in (0, 1):
        for cc in layers[layer_idx]:
            removable_texts.add(cc.text.strip())

    filtered = task_prompt
    for txt in removable_texts:
        escaped = re.escape(txt)
        filtered = re.sub(
            rf"[-•*\d.)\s]*{escaped}\s*\n?",
            "",
            filtered,
            count=1,
        )

    filtered = re.sub(r"\n{3,}", "\n\n", filtered).strip()
    return filtered, layers


# ---------------------------------------------------------------------------
# CADG: Constraint-order permutation across N candidates
# ---------------------------------------------------------------------------

def permute_constraints(
    constraints: list[dict],
    candidate_index: int,
    seed: int = 42,
    failure_weights: dict[str, float] | None = None,
) -> list[dict]:
    """CADG: Reorder constraints for a given candidate.

    Candidate 0: Sort by descending failure weight (hardest first — primacy bias).
    Candidates 1+: Weighted shuffle without replacement.
    """
    if not constraints:
        return constraints

    items = list(constraints)

    if candidate_index == 0:
        if failure_weights:
            items.sort(
                key=lambda c: failure_weights.get(c["id"], 0.5),
                reverse=True,
            )
        return items

    rng = Random(seed + candidate_index)
    if failure_weights:
        remaining = list(items)
        result = []
        while remaining:
            weights = [failure_weights.get(c["id"], 0.5) + 0.1 for c in remaining]
            total = sum(weights)
            probs = [w / total for w in weights]
            r = rng.random()
            cumulative = 0.0
            chosen_idx = len(remaining) - 1
            for i, p in enumerate(probs):
                cumulative += p
                if r <= cumulative:
                    chosen_idx = i
                    break
            result.append(remaining.pop(chosen_idx))
        return result
    else:
        rng.shuffle(items)
        return items


def cadg_generate(
    client: Any,
    task_id: str,
    base_prompt: str,
    constraints: list[dict],
    n_candidates: int = 5,
    seed: int = 42,
    failure_weights: dict[str, float] | None = None,
    provider: str = "anthropic",
    model: str | None = None,
    max_tokens: int = 4096,
    temperature: float = 1.0,
) -> list[tuple[str, list[dict]]]:
    """Generate N candidates with permuted constraint orders.

    Returns list of (output_text, permuted_constraints) tuples.
    """
    from cpga_harness import run_single_task

    candidates = []
    for ci in range(n_candidates):
        permuted = permute_constraints(constraints, ci, seed, failure_weights)
        constraint_block = "\n".join(
            f"{i+1}. {c['text']}" for i, c in enumerate(permuted)
        )
        prompt = f"{base_prompt}\n\nRequirements:\n{constraint_block}"

        result = run_single_task(
            client=client,
            task_id=f"{task_id}_c{ci}",
            prompt=prompt,
            condition="cadg",
            provider=provider,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            candidate_index=ci,
        )
        candidates.append((result.output, permuted))

    return candidates


def cadg_select_best(
    candidates: list[tuple[str, list[dict]]],
    constraints: list[dict],
) -> tuple[str, float, int]:
    """Select best candidate by constraint satisfaction.

    Returns (best_output, best_score, best_index).
    """
    best_output = ""
    best_score = -1.0
    best_idx = 0

    for i, (output, _) in enumerate(candidates):
        satisfied = 0
        for c in constraints:
            check_fn = c.get("check_fn")
            if check_fn:
                try:
                    if check_fn(output):
                        satisfied += 1
                except Exception:
                    pass
            else:
                satisfied += 1
        score = satisfied / len(constraints) if constraints else 1.0
        if score > best_score:
            best_score = score
            best_output = output
            best_idx = i

    return best_output, best_score, best_idx


# ---------------------------------------------------------------------------
# SENTINEL: Post-generation checking and fixing
# ---------------------------------------------------------------------------

@dataclass
class SentinelVerdict:
    rule_id: str
    tier: str
    passed: bool
    auto_fixed: bool = False
    evidence: str = ""
    fixed_text: str = ""


def sentinel_tier2_fix(
    output: str,
    constraints: list[dict],
) -> tuple[str, list[SentinelVerdict]]:
    """Tier 2: Apply deterministic regex/code fixes.

    For constraints with check_fn and auto_fix_fn, apply fixes.
    """
    verdicts = []
    fixed = output

    for c in constraints:
        check_fn = c.get("check_fn")
        fix_fn = c.get("auto_fix_fn")
        if not check_fn:
            continue

        try:
            passed = check_fn(fixed)
        except Exception:
            passed = False

        if not passed and fix_fn:
            try:
                fixed = fix_fn(fixed)
                re_passed = check_fn(fixed)
                verdicts.append(SentinelVerdict(
                    rule_id=c["id"], tier="tier2",
                    passed=re_passed, auto_fixed=re_passed,
                    evidence="auto-fixed by tier2",
                ))
            except Exception as e:
                verdicts.append(SentinelVerdict(
                    rule_id=c["id"], tier="tier2",
                    passed=False, evidence=f"fix failed: {e}",
                ))
        else:
            verdicts.append(SentinelVerdict(
                rule_id=c["id"], tier="tier2",
                passed=passed,
            ))

    return fixed, verdicts


def sentinel_tier3_check(
    output: str,
    constraints: list[dict],
) -> list[SentinelVerdict]:
    """Tier 3: Programmatic checks without fixing."""
    verdicts = []
    for c in constraints:
        check_fn = c.get("check_fn")
        if not check_fn:
            verdicts.append(SentinelVerdict(
                rule_id=c["id"], tier="tier3",
                passed=True,
                evidence="no checker — assumed pass",
            ))
            continue
        try:
            passed = check_fn(output)
        except Exception as e:
            passed = False
            verdicts.append(SentinelVerdict(
                rule_id=c["id"], tier="tier3",
                passed=False, evidence=f"check error: {e}",
            ))
            continue
        verdicts.append(SentinelVerdict(
            rule_id=c["id"], tier="tier3", passed=passed,
        ))
    return verdicts


def sentinel_tier4_fix(
    client: Any,
    output: str,
    failing_constraints: list[dict],
    all_constraints: list[dict] | None = None,
    provider: str = "anthropic",
    model: str | None = None,
    max_repair_rounds: int = 2,
) -> tuple[str, list[SentinelVerdict]]:
    """Tier 4: P13 Hyper-Isolated Verification + Repair.

    For each failing constraint, make ONE focused LLM call with minimal
    context (~output + single rule). The LLM sees only the text and one
    rule — no attention competition. If it can fix the output, we re-check
    with the deterministic checker.
    """
    if all_constraints is None:
        all_constraints = failing_constraints
    verdicts = []
    fixed = output

    for round_i in range(max_repair_rounds):
        still_failing = []
        for c in failing_constraints:
            check_fn = c.get("check_fn")
            if not check_fn:
                continue
            try:
                if check_fn(fixed):
                    continue
            except Exception:
                pass
            still_failing.append(c)

        if not still_failing:
            break

        for c in still_failing:
            repair_prompt = (
                f"Here is a text:\n\n---\n{fixed}\n---\n\n"
                f"This text FAILS the following requirement:\n"
                f"\"{c['text']}\"\n\n"
                f"Rewrite the COMPLETE text so it satisfies this requirement "
                f"while preserving all other content as much as possible. "
                f"Output ONLY the rewritten text, nothing else."
            )
            try:
                repaired, _, _ = client.generate(
                    repair_prompt,
                    provider=provider,
                    model=model,
                    max_tokens=4096,
                    temperature=0.3,
                )
                check_fn = c.get("check_fn")
                target_passes = check_fn and check_fn(repaired)

                # Only accept if net compliance across ALL constraints improves
                if target_passes:
                    old_score = sum(1 for cc in all_constraints if cc.get("check_fn") and cc["check_fn"](fixed))
                    new_score = sum(1 for cc in all_constraints if cc.get("check_fn") and cc["check_fn"](repaired))
                    if new_score > old_score:
                        fixed = repaired
                        verdicts.append(SentinelVerdict(
                            rule_id=c["id"], tier="tier4",
                            passed=True, auto_fixed=True,
                            evidence=f"P13 repair round {round_i+1}, net +{new_score - old_score}",
                        ))
                    else:
                        verdicts.append(SentinelVerdict(
                            rule_id=c["id"], tier="tier4",
                            passed=False,
                            evidence=f"P13 repair would regress ({new_score} < {old_score}), rejected",
                        ))
                else:
                    verdicts.append(SentinelVerdict(
                        rule_id=c["id"], tier="tier4",
                        passed=False,
                        evidence=f"P13 repair attempted but target still fails round {round_i+1}",
                    ))
            except Exception as e:
                verdicts.append(SentinelVerdict(
                    rule_id=c["id"], tier="tier4",
                    passed=False, evidence=f"P13 repair error: {e}",
                ))

    return fixed, verdicts


def run_sentinel_swarm(
    output: str,
    constraints: list[dict],
    enable_tier2: bool = True,
    enable_tier4: bool = False,
    client: Any = None,
    provider: str = "anthropic",
    model: str | None = None,
) -> tuple[str, list[SentinelVerdict], str]:
    """Run the full SENTINEL swarm.

    Returns (possibly_fixed_output, all_verdicts, arbiter_decision).
    Tier 4 (P13 isolated LLM repair) requires a client.
    """
    all_verdicts = []
    fixed_output = output

    # Tier 2: deterministic regex/code fixers
    if enable_tier2:
        fixed_output, t2_verdicts = sentinel_tier2_fix(fixed_output, constraints)
        all_verdicts.extend(t2_verdicts)

    # Tier 3: programmatic checks (identify what still fails)
    t3_verdicts = sentinel_tier3_check(fixed_output, constraints)
    all_verdicts.extend(t3_verdicts)

    # Tier 4: P13 isolated LLM repair for remaining failures
    if enable_tier4 and client:
        t3_failing_ids = {v.rule_id for v in t3_verdicts if not v.passed}
        failing_constraints = [c for c in constraints if c["id"] in t3_failing_ids]
        if failing_constraints:
            fixed_output, t4_verdicts = sentinel_tier4_fix(
                client, fixed_output, failing_constraints,
                all_constraints=constraints,
                provider=provider, model=model,
            )
            all_verdicts.extend(t4_verdicts)
            # Update Tier 3 verdicts for constraints that Tier 4 fixed
            t4_fixed = {v.rule_id for v in t4_verdicts if v.passed}
            for v in all_verdicts:
                if v.tier == "tier3" and v.rule_id in t4_fixed:
                    v.passed = True
                    v.auto_fixed = True
                    v.evidence = "fixed by tier4 P13"

    n_pass = sum(1 for v in all_verdicts if v.passed or v.auto_fixed)
    n_total = len(all_verdicts)
    pass_rate = n_pass / n_total if n_total > 0 else 1.0

    if pass_rate >= 1.0:
        decision = "SHIP"
    elif pass_rate >= 0.95:
        decision = "SHIP_WITH_WARNINGS"
    elif pass_rate >= 0.85:
        decision = "REPAIR"
    else:
        decision = "ESCALATE"

    return fixed_output, all_verdicts, decision


# ---------------------------------------------------------------------------
# Full-stack composition: FORGE + CADG + SENTINEL
# ---------------------------------------------------------------------------

def run_full_stack(
    client: Any,
    task_id: str,
    base_prompt: str,
    constraints: list[dict],
    n_candidates: int = 5,
    seed: int = 42,
    provider: str = "anthropic",
    model: str | None = None,
    max_tokens: int = 4096,
    temperature: float = 1.0,
    forge_mode: str = "type",
) -> tuple[str, float, dict]:
    """FORGE + CADG + SENTINEL composed.

    1. FORGE: Classify constraints, remove Layer 0/1 from prompt
    2. CADG: Generate N candidates with permuted Layer 3 constraints
    3. Score all candidates against ALL constraints (including Layer 0/1)
    4. SENTINEL: Apply Tier 2 fixes to best candidate, re-score
    """
    from cpga_harness import score_output

    layers = forge_classify(constraints, mode=forge_mode, client=client,
                            provider=provider, model=model)
    layer3_constraints = [cc.original for cc in layers[3]]

    # CADG with only Layer 3 in prompt
    candidates = cadg_generate(
        client, task_id, base_prompt, layer3_constraints,
        n_candidates=n_candidates, seed=seed,
        provider=provider, model=model,
        max_tokens=max_tokens, temperature=temperature,
    )

    # Score against ALL constraints
    best_output, best_score, best_idx = cadg_select_best(candidates, constraints)

    # SENTINEL full swarm (Tier 2 + 3 + 4)
    fixed_output, verdicts, decision = run_sentinel_swarm(
        best_output, constraints,
        enable_tier2=True, enable_tier4=True,
        client=client, provider=provider, model=model,
    )

    final_result = score_output(task_id, "full_stack", fixed_output, constraints)

    meta = {
        "forge_layers": {k: len(v) for k, v in layers.items()},
        "cadg_candidates": n_candidates,
        "cadg_best_index": best_idx,
        "cadg_pre_sentinel_score": best_score,
        "sentinel_decision": decision,
        "sentinel_auto_fixed": sum(1 for v in verdicts if v.auto_fixed),
    }

    return fixed_output, final_result.satisfaction_rate, meta


def run_forge_cadg(
    client: Any,
    task_id: str,
    base_prompt: str,
    constraints: list[dict],
    n_candidates: int = 5,
    seed: int = 42,
    provider: str = "anthropic",
    model: str | None = None,
    max_tokens: int = 4096,
    temperature: float = 1.0,
    forge_mode: str = "type",
) -> tuple[str, dict]:
    """FORGE + CADG only (no SENTINEL). CADG permutes Layer-3 residual constraints."""
    layers = forge_classify(constraints, mode=forge_mode, client=client,
                            provider=provider, model=model)
    layer3_constraints = [cc.original for cc in layers[3]]
    candidates = cadg_generate(
        client, task_id, base_prompt, layer3_constraints,
        n_candidates=n_candidates, seed=seed,
        provider=provider, model=model,
        max_tokens=max_tokens, temperature=temperature,
    )
    best_output, best_score, best_idx = cadg_select_best(candidates, constraints)
    meta = {
        "ablation": "forge_cadg",
        "forge_layers": {k: len(v) for k, v in layers.items()},
        "cadg_candidates": n_candidates,
        "cadg_best_index": best_idx,
        "cadg_best_score": best_score,
    }
    return best_output, meta


def run_forge_sentinel(
    client: Any,
    task_id: str,
    task_prompt: str,
    constraints: list[dict],
    system: str = "",
    provider: str = "anthropic",
    model: str | None = None,
    max_tokens: int = 4096,
    temperature: float = 1.0,
    forge_mode: str = "type",
) -> tuple[str, dict]:
    """FORGE + single generation + SENTINEL (no CADG)."""
    from cpga_harness import run_single_task

    filtered_prompt, layers = forge_filter_prompt(
        task_prompt, constraints, forge_mode=forge_mode,
        client=client, provider=provider, model=model,
    )
    result = run_single_task(
        client=client,
        task_id=task_id,
        prompt=filtered_prompt,
        condition="forge_sentinel",
        system=system,
        provider=provider,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    fixed_output, verdicts, decision = run_sentinel_swarm(
        result.output, constraints,
        enable_tier2=True, enable_tier4=True,
        client=client, provider=provider, model=model,
    )
    meta = {
        "ablation": "forge_sentinel",
        "generation": {"task_id": result.task_id, "model": result.model},
        "forge_layers": {k: len(v) for k, v in layers.items()},
        "sentinel_decision": decision,
        "sentinel_auto_fixed": sum(1 for v in verdicts if v.auto_fixed),
    }
    return fixed_output, meta


def run_cadg_sentinel(
    client: Any,
    task_id: str,
    base_prompt: str,
    constraints: list[dict],
    n_candidates: int = 5,
    seed: int = 42,
    provider: str = "anthropic",
    model: str | None = None,
    max_tokens: int = 4096,
    temperature: float = 1.0,
) -> tuple[str, dict]:
    """CADG on full constraint set + SENTINEL on the best candidate (no FORGE filtering)."""
    candidates = cadg_generate(
        client, task_id, base_prompt, constraints,
        n_candidates=n_candidates, seed=seed,
        provider=provider, model=model,
        max_tokens=max_tokens, temperature=temperature,
    )
    best_output, best_score, best_idx = cadg_select_best(candidates, constraints)
    fixed_output, verdicts, decision = run_sentinel_swarm(
        best_output, constraints,
        enable_tier2=True, enable_tier4=True,
        client=client, provider=provider, model=model,
    )
    meta = {
        "ablation": "cadg_sentinel",
        "cadg_candidates": n_candidates,
        "cadg_best_index": best_idx,
        "cadg_pre_sentinel_score": best_score,
        "sentinel_decision": decision,
        "sentinel_auto_fixed": sum(1 for v in verdicts if v.auto_fixed),
    }
    return fixed_output, meta
