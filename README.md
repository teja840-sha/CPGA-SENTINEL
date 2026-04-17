# CPGA+SENTINEL Experiment Suite

![Visitors](https://api.visitorbadge.io/api/visitors?path=https%3A%2F%2Fgithub.com%2Fteja840-sha%2FCPGA-SENTINEL&label=Visitors&countColor=%23263759)

> Reproducible experiments for **"Context-Partitioned Generic Assembly with Swarm-External Enforcement for Overcoming LLM Context Degradation"**

## Paper

- **PDF**: [`paper/CPGA_SENTINEL.pdf`](paper/CPGA_SENTINEL.pdf) (36 pages, ~540 KB, arXiv-compliant embedded fonts)
- **HTML**: [`paper/CPGA_SENTINEL.html`](paper/CPGA_SENTINEL.html) (source, for diffs and reference)

Author: Siva Teja Narayana (Independent Researcher), April 2026.

## What This Repo Contains

Everything needed to reproduce the benchmark results from the paper:

- **Paper** — PDF + HTML source of the full manuscript
- **Pre-registered analysis plan** — [`ANALYSIS_PLAN.md`](ANALYSIS_PLAN.md), frozen at git tag `analysis-plan-v1` (commit `e03e754`)
- **Experiment harness** — LLM client wrappers, cost tracking, scoring primitives
- **CPGA method implementations** — FORGE classifier, CADG permutation, SENTINEL swarm
- **6 benchmark adapters** — MOSAIC, IFEval, IFBench, FollowBench, SysPrompt, ToolSelection
- **57 result JSONs** — raw results (with cached LLM outputs) for all conditions × benchmarks
- **Per-task score vectors** — [`experiments/results/score_vectors/`](experiments/results/score_vectors/) — CSVs for independent bootstrap verification
- **Bootstrap CI analysis** — statistical significance testing, Holm–Bonferroni corrected
- **4 model configurations** — Claude Opus 4.6 (default), OpenAI (GPT-5.4 / GPT-4o), Together AI (Llama 3.3 70B), local vLLM

## Quick Start

```bash
git clone https://github.com/teja840-sha/CPGA-SENTINEL.git
cd CPGA-SENTINEL

pip install -r experiments/requirements.txt

export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...
```

### Run Experiments

```bash
# Dry run (no API calls — validates the pipeline)
python experiments/run_benchmarks.py --dry-run

# Run MOSAIC with all conditions
python experiments/run_benchmarks.py -b mosaic -c all

# Run IFEval baseline only (first 50 tasks)
python experiments/run_benchmarks.py -b ifeval -c baseline --max-tasks 50

# Run everything
python experiments/run_benchmarks.py -b all -c all

# Bootstrap confidence intervals
python experiments/analyze_bootstrap.py
```

## Repository Structure

```
CPGA-SENTINEL/
├── README.md
├── LICENSE                              # MIT
├── ANALYSIS_PLAN.md                     # Pre-registered analysis plan (frozen)
├── paper/
│   ├── CPGA_SENTINEL.pdf                # Full manuscript (36 pp)
│   └── CPGA_SENTINEL.html               # HTML source of paper
├── config/
│   ├── config.yaml                      # Anthropic Claude (default)
│   ├── config_openai.yaml               # OpenAI GPT-5.4 / GPT-4o
│   ├── config_together.yaml             # Together AI (Llama 3.3 70B)
│   └── config_vllm.yaml                 # Local vLLM (zero API cost)
└── experiments/
    ├── README.md                        # Detailed experiment documentation
    ├── requirements.txt                 # Python dependencies
    ├── cpga_harness.py                  # LLM client, cost tracking, scoring
    ├── cpga_methods.py                  # FORGE, CADG, SENTINEL implementations
    ├── run_benchmarks.py                # CLI orchestrator
    ├── run_all_missing.py               # Fill in missing conditions
    ├── extract_results.py               # Result extraction utilities
    ├── extract_score_vectors.py         # JSON → per-task CSV conversion
    ├── analyze_bootstrap.py             # Bootstrap CI analysis (B=2000, seed=42)
    ├── compute_all_tables.py            # Reproduce all paper tables
    ├── verify_cached.py                 # Re-score cached outputs without API calls
    ├── adapters/
    │   ├── mosaic_adapter.py            # 21-constraint library
    │   ├── ifeval_adapter.py            # 25 instruction checkers
    │   ├── ifbench_adapter.py           # 58 constraint types
    │   ├── followbench_adapter.py       # Multi-level constraint ladder
    │   ├── system_prompt_adapter.py     # SysPrompt compliance
    │   └── tool_selection_adapter.py    # Tool selection accuracy
    └── results/
        ├── *.json                       # 57 raw result files w/ cached outputs
        ├── summary.json                 # Aggregate cost + call totals
        ├── paper_tables.json            # Numbers used in paper tables
        ├── verification.json            # Cached-output verification log
        └── score_vectors/
            ├── mosaic_score_vectors.csv     # per-task × 9 conditions
            ├── ifbench_score_vectors.csv
            ├── ifeval_score_vectors.csv
            └── sysprompt_score_vectors.csv
```

## Experimental Conditions

| Condition | Description |
|-----------|-------------|
| **Baseline** | All constraints in prompt → generate → score |
| **FORGE** | Classify constraints by layer, remove Layer 0/1, generate with residual only |
| **CADG** | Permute constraint order across N candidates → score → select best |
| **SENTINEL** | Generate → post-gen Tier 2/3/4 checking + repair |
| **FORGE+CADG** | FORGE filtering + CADG permutation |
| **FORGE+SENTINEL** | FORGE filtering + SENTINEL enforcement |
| **CADG+SENTINEL** | CADG permutation + SENTINEL enforcement |
| **Full Stack** | FORGE + CADG + SENTINEL composed |
| **Retry-Feedback** | Strong baseline: retry with error feedback (up to 3 attempts) |

## Key Results

| Benchmark | Baseline | Best Composition | Δ | Best Condition |
|-----------|----------|-----------------|---|----------------|
| **MOSAIC** (SCC, n=200) | 0.758 | **0.822** | +6.4pp | Full Stack |
| **IFEval** (Inst. Acc., n=541) | 0.799 | **0.860** | +6.2pp | Full Stack |
| **IFBench** (Inst. Acc., n=300) | 0.397 | **0.497** | +10.0pp | CADG+SENTINEL |
| **SysPrompt** (SCC, n=40) | 0.862 | **0.933** | +7.1pp | CADG+SENTINEL |

**Key finding**: FORGE's value scales with rule taxonomy complexity — Full Stack (with FORGE) wins on MOSAIC/IFEval; CADG+SENTINEL (without FORGE) wins on short-list benchmarks (IFBench/SysPrompt). For production deployments with 100+ rules, the full stack is recommended.

All scoring is **deterministic Python checkers** — no LLM judges for metric computation.

## Reproducibility Protocol

### 1. Verify scores without API calls
The `results/` directory contains raw LLM outputs for every task. Re-run scoring locally:
```bash
python experiments/verify_cached.py   # re-scores all cached outputs, compares to reported numbers
```

### 2. Independent bootstrap verification (no API calls)
All per-task scores are in `experiments/results/score_vectors/*.csv`. Any
bootstrap implementation (numpy, scipy, R) can reproduce the paper's CIs:
```python
import pandas as pd, numpy as np
df = pd.read_csv("experiments/results/score_vectors/mosaic_score_vectors.csv")
# Paired task-level bootstrap, B=2000, seed=42 (matches paper)
rng = np.random.default_rng(42)
deltas = df["full_stack"] - df["baseline"]
boots = [deltas.sample(len(deltas), replace=True, random_state=rng.integers(2**32)).mean()
         for _ in range(2000)]
print("95% CI:", np.percentile(boots, [2.5, 97.5]))
```

### 3. Run with other model families
The harness works with any OpenAI-compatible endpoint:
```bash
# OpenAI GPT-5.4 or GPT-4o (for §6.9.10 cross-model validation)
python experiments/run_benchmarks.py -b mosaic -c all --provider openai \
  --config config/config_openai.yaml

# Together AI (Llama 3.3 70B, open-weight)
export OPENAI_API_KEY=your-together-key
python experiments/run_benchmarks.py -b mosaic -c baseline --provider openai \
  --config config/config_together.yaml

# Local vLLM (zero API cost)
python experiments/run_benchmarks.py -b ifeval -c all --provider openai \
  --config config/config_vllm.yaml
```

### 4. Cost to reproduce
| Scope | API Cost | Time |
|-------|----------|------|
| Bootstrap CIs from cached score vectors | $0 | ~30 sec |
| Re-score all cached LLM outputs (`verify_cached.py`) | $0 | ~2 min |
| Subset (baseline + full_stack, MOSAIC + IFEval) | ~$30 | ~2 hrs |
| Full (9 conditions × 4 benchmarks, Claude) | ~$200 | ~8 hrs |
| Open-weight (vLLM on 1× A100) | $0 API | ~4 hrs |

### 5. Cross-model validation (measured, §6.9.10 of paper)
The harness has been validated on three model families with no code changes:

| Model | Provider | MOSAIC baseline | MOSAIC full-stack | Δ |
|-------|----------|----------------|-------------------|---|
| Claude Opus 4.6 | Anthropic | 0.758 | 0.822 | +6.4 pp |
| GPT-5.4 | OpenAI | 0.601 | 0.739 | +13.8 pp |
| GPT-4o | OpenAI | 0.669 | 0.767 | +9.8 pp |

The monotonic "lower baseline → larger gain" pattern is the predicted signature of external enforcement. See `config/config_openai.yaml` to reproduce the OpenAI runs, and `config/config_together.yaml` / `config/config_vllm.yaml` for open-weight models.

## Design Principles

1. **Deterministic scoring only** — all check functions are pure Python (regex, counts, AST parsers). Reproducible without API calls for scoring.
2. **Real API calls for generation** — uses Claude Opus 4.6 and GPT-5.4 via standard APIs. Provider-agnostic: works with any OpenAI-compatible endpoint.
3. **Modular adapters** — each benchmark is a self-contained adapter; add new benchmarks by implementing the adapter interface.
4. **Full cost tracking** — every run logs token counts, API calls, and dollar costs.
5. **Cached outputs included** — all 52 result files include raw LLM responses for score verification without API access.

## Citation

```bibtex
@techreport{narayana2026cpga,
  author = {Narayana, Siva Teja},
  title  = {Context-Partitioned Generic Assembly with Swarm-External
            Enforcement for Overcoming {LLM} Context Degradation},
  institution = {Independent Researcher},
  year   = {2026},
  month  = apr,
  url    = {https://github.com/teja840-sha/CPGA-SENTINEL},
  note   = {Preprint. PDF: paper/CPGA\_SENTINEL.pdf}
}
```

## License

MIT
