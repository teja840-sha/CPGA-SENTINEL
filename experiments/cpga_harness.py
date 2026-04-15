"""
CPGA+SENTINEL Evaluation Harness

Generic harness for running LLM generation with cost estimation across
Anthropic and OpenAI APIs. Supports multiple experimental conditions
(Baseline, FORGE, CADG, SENTINEL, Full Stack) with deterministic scoring.

All scoring uses Python checkers — no LLM judges for metric computation.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable

import yaml

PRICING = {
    "claude-sonnet-4-20250514": {"input": 3.00 / 1e6, "output": 15.00 / 1e6},
    "claude-opus-4-20250514": {"input": 15.00 / 1e6, "output": 75.00 / 1e6},
    "claude-opus-4-6": {"input": 15.00 / 1e6, "output": 75.00 / 1e6},
    "gpt-4o": {"input": 2.50 / 1e6, "output": 10.00 / 1e6},
    "gpt-4o-mini": {"input": 0.15 / 1e6, "output": 0.60 / 1e6},
    "gpt-5.2": {"input": 5.00 / 1e6, "output": 20.00 / 1e6},
    "gpt-5.4": {"input": 5.00 / 1e6, "output": 20.00 / 1e6},
    "meta-llama/Llama-3.3-70B-Instruct-Turbo": {"input": 0.88 / 1e6, "output": 0.88 / 1e6},
}


@dataclass
class GenerationResult:
    task_id: str
    condition: str
    model: str
    output: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    candidate_index: int = 0
    metadata: dict = field(default_factory=dict)


@dataclass
class ScoringResult:
    task_id: str
    condition: str
    constraint_count: int
    constraints_satisfied: int
    constraints_total: int
    satisfaction_rate: float
    per_constraint: list[dict] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


def load_config(config_path: str | Path | None = None) -> dict:
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path) as f:
        raw = f.read()

    for key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "TOGETHER_API_KEY"):
        env_val = os.environ.get(key, "")
        raw = raw.replace(f"${{{key}}}", env_val)

    cfg = yaml.safe_load(raw)

    # Fall back to project-level config/config.yaml for API keys
    project_config = None
    if not cfg.get("anthropic", {}).get("api_key") or not cfg.get("openai", {}).get("api_key"):
        project_config = _try_load_project_config()
        if project_config:
            if not cfg.get("anthropic", {}).get("api_key") and project_config.get("anthropic_key"):
                cfg.setdefault("anthropic", {})["api_key"] = project_config["anthropic_key"]
            if not cfg.get("openai", {}).get("api_key") and project_config.get("openai_key"):
                cfg.setdefault("openai", {})["api_key"] = project_config["openai_key"]

    # For Together AI configs, if api_key is still empty, try OPENAI_API_KEY from project
    if cfg.get("openai", {}).get("base_url") and not cfg.get("openai", {}).get("api_key"):
        if project_config is None:
            project_config = _try_load_project_config()
        together_key = os.environ.get("TOGETHER_API_KEY", "")
        if together_key:
            cfg["openai"]["api_key"] = together_key

    return cfg


def _try_load_project_config() -> dict | None:
    """Try to load API keys from the Mocho project config/config.yaml."""
    try:
        candidates = [
            Path(__file__).resolve().parents[2] / "config" / "config.yaml",
            Path(__file__).resolve().parents[3] / "config" / "config.yaml",
            Path(__file__).resolve().parents[4] / "config" / "config.yaml",
            Path(__file__).resolve().parents[5] / "config" / "config.yaml",
            Path("c:/Cursor/Mocho/config/config.yaml"),
        ]
        for candidate in candidates:
            if candidate.exists():
                with open(candidate) as f:
                    proj_cfg = yaml.safe_load(f)
                result = {}
                anth = proj_cfg.get("anthropic", {})
                if isinstance(anth, dict) and anth.get("api_key"):
                    result["anthropic_key"] = anth["api_key"]
                oai = proj_cfg.get("openai", {})
                if isinstance(oai, dict) and oai.get("api_key"):
                    result["openai_key"] = oai["api_key"]
                if result:
                    return result
    except Exception:
        pass
    return None


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    prices = PRICING.get(model, {"input": 5.0 / 1e6, "output": 15.0 / 1e6})
    return input_tokens * prices["input"] + output_tokens * prices["output"]


class LLMClient:
    """Unified client for Anthropic and OpenAI with cost tracking."""

    def __init__(self, config: dict):
        self.config = config
        self.total_cost = 0.0
        self.total_calls = 0
        self.budget_limit = config.get("experiment", {}).get("budget_limit_usd", 30.0)
        self._anthropic_client = None
        self._openai_client = None

    @property
    def anthropic(self):
        if self._anthropic_client is None:
            import anthropic
            api_key = self.config.get("anthropic", {}).get("api_key", "")
            self._anthropic_client = anthropic.Anthropic(api_key=api_key)
        return self._anthropic_client

    @property
    def openai(self):
        if self._openai_client is None:
            import openai as oai
            oai_cfg = self.config.get("openai", {})
            api_key = oai_cfg.get("api_key", "")
            base_url = oai_cfg.get("base_url")
            kwargs = {"api_key": api_key}
            if base_url:
                kwargs["base_url"] = base_url
            self._openai_client = oai.OpenAI(**kwargs)
        return self._openai_client

    def _check_budget(self):
        if self.total_cost >= self.budget_limit:
            raise RuntimeError(
                f"Budget exhausted: ${self.total_cost:.2f} >= ${self.budget_limit:.2f}"
            )

    def generate_anthropic(
        self,
        prompt: str,
        system: str = "",
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 1.0,
    ) -> tuple[str, int, int]:
        """Returns (text, input_tokens, output_tokens)."""
        self._check_budget()
        model = model or self.config.get("anthropic", {}).get("model", "claude-sonnet-4-20250514")
        messages = [{"role": "user", "content": prompt}]
        kwargs: dict[str, Any] = dict(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=messages,
        )
        if system:
            kwargs["system"] = system

        if not prompt or not prompt.strip():
            return "(empty prompt — skipped)", 0, 0
        resp = self.anthropic.messages.create(**kwargs)
        text = resp.content[0].text
        inp = resp.usage.input_tokens
        out = resp.usage.output_tokens
        cost = estimate_cost(model, inp, out)
        self.total_cost += cost
        self.total_calls += 1
        return text, inp, out

    def generate_openai(
        self,
        prompt: str,
        system: str = "",
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 1.0,
    ) -> tuple[str, int, int]:
        """Returns (text, input_tokens, output_tokens)."""
        self._check_budget()
        model = model or self.config.get("openai", {}).get("model", "gpt-4o")
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        if not prompt or not prompt.strip():
            return "(empty prompt — skipped)", 0, 0
        create_kw: dict[str, Any] = dict(
            model=model,
            messages=messages,
            temperature=temperature,
        )
        if any(tag in model for tag in ("gpt-5", "o1", "o3", "o4")):
            create_kw["max_completion_tokens"] = max_tokens
        else:
            create_kw["max_tokens"] = max_tokens
        resp = self.openai.chat.completions.create(**create_kw)
        text = resp.choices[0].message.content or ""
        usage = resp.usage
        inp = usage.prompt_tokens if usage else 0
        out = usage.completion_tokens if usage else 0
        cost = estimate_cost(model, inp, out)
        self.total_cost += cost
        self.total_calls += 1
        return text, inp, out

    def generate(
        self,
        prompt: str,
        system: str = "",
        provider: str = "anthropic",
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 1.0,
        max_retries: int = 5,
    ) -> tuple[str, int, int]:
        RETRYABLE = (
            "APIConnectionError", "APITimeoutError", "ConnectionError",
            "Timeout", "OverloadedError", "RateLimitError",
            "InternalServerError",
        )
        for attempt in range(max_retries):
            try:
                if provider == "anthropic":
                    return self.generate_anthropic(prompt, system, model, max_tokens, temperature)
                elif provider == "openai":
                    return self.generate_openai(prompt, system, model, max_tokens, temperature)
                else:
                    raise ValueError(f"Unknown provider: {provider}")
            except Exception as e:
                err_name = type(e).__name__
                err_str = str(e)
                retryable = any(tag in err_name for tag in RETRYABLE) or "529" in err_str or "overloaded" in err_str.lower()
                if retryable and attempt < max_retries - 1:
                    wait = min(2 ** attempt * 10, 120)
                    print(f"  [retry {attempt+1}/{max_retries}] {err_name}, waiting {wait}s...", flush=True)
                    time.sleep(wait)
                    continue
                raise
        raise RuntimeError(f"Failed after {max_retries} retries")


def run_single_task(
    client: LLMClient,
    task_id: str,
    prompt: str,
    condition: str,
    system: str = "",
    provider: str = "anthropic",
    model: str | None = None,
    max_tokens: int = 4096,
    temperature: float = 1.0,
    candidate_index: int = 0,
    metadata: dict | None = None,
) -> GenerationResult:
    t0 = time.perf_counter()
    text, inp, out = client.generate(
        prompt, system, provider, model, max_tokens, temperature,
    )
    latency = (time.perf_counter() - t0) * 1000
    m = model or client.config.get(provider, {}).get("model", "unknown")
    return GenerationResult(
        task_id=task_id,
        condition=condition,
        model=m,
        output=text,
        input_tokens=inp,
        output_tokens=out,
        cost_usd=estimate_cost(m, inp, out),
        latency_ms=latency,
        candidate_index=candidate_index,
        metadata=metadata or {},
    )


def score_output(
    task_id: str,
    condition: str,
    output: str,
    constraints: list[dict],
) -> ScoringResult:
    """Score output against constraints using deterministic Python checkers.

    Each constraint: {id: str, text: str, check_fn: Callable[[str], bool]}
    """
    per_constraint = []
    satisfied = 0
    for c in constraints:
        check_fn: Callable[[str], bool] = c["check_fn"]
        try:
            passed = bool(check_fn(output))
        except Exception as e:
            passed = False
            per_constraint.append({
                "id": c["id"],
                "text": c.get("text", ""),
                "passed": False,
                "error": str(e),
            })
            continue
        if passed:
            satisfied += 1
        per_constraint.append({
            "id": c["id"],
            "text": c.get("text", ""),
            "passed": passed,
        })

    total = len(constraints)
    return ScoringResult(
        task_id=task_id,
        condition=condition,
        constraint_count=total,
        constraints_satisfied=satisfied,
        constraints_total=total,
        satisfaction_rate=satisfied / total if total > 0 else 1.0,
        per_constraint=per_constraint,
    )


def save_results(
    results: list[ScoringResult],
    output_path: str | Path,
    extra_meta: dict | None = None,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "num_results": len(results),
        "meta": extra_meta or {},
        "results": [asdict(r) for r in results],
    }
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved {len(results)} results to {output_path}")


def aggregate_results(results: list[ScoringResult]) -> dict:
    """Aggregate scoring results by condition and constraint count."""
    from collections import defaultdict
    buckets: dict[tuple[str, int], list[float]] = defaultdict(list)
    for r in results:
        buckets[(r.condition, r.constraint_count)].append(r.satisfaction_rate)

    summary = {}
    for (cond, n_constraints), rates in sorted(buckets.items()):
        import numpy as np
        arr = np.array(rates)
        key = f"{cond}@{n_constraints}"
        summary[key] = {
            "condition": cond,
            "constraint_count": n_constraints,
            "n_tasks": len(rates),
            "mean_satisfaction": float(np.mean(arr)),
            "std_satisfaction": float(np.std(arr)),
            "min_satisfaction": float(np.min(arr)),
            "max_satisfaction": float(np.max(arr)),
        }
    return summary
