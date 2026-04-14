# CPGA+SENTINEL Experiment Suite

![Visitors](https://api.visitorbadge.io/api/visitors?path=https%3A%2F%2Fgithub.com%2Fteja840-sha%2FCPGA-SENTINEL&label=Visitors&countColor=%23263759)

> Reproducible experiments for **"Context-Partitioned Generic Assembly with Swarm-External Enforcement for Overcoming LLM Context Degradation"**

## What This Repo Contains

Everything needed to reproduce the benchmark results from the paper:

- **Experiment harness** — LLM client wrappers, cost tracking, scoring primitives
- **CPGA method implementations** — FORGE classifier, CADG permutation, SENTINEL swarm
- **6 benchmark adapters** — MOSAIC, IFEval, IFBench, FollowBench, SysPrompt, ToolSelection
- **52 result JSONs** — raw results for all conditions across all benchmarks
- **Bootstrap CI analysis** — statistical significance testing

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
├── experiments/
│   ├── cpga_harness.py            # LLM client, cost tracking, scoring
│   ├── cpga_methods.py            # FORGE, CADG, SENTINEL implementations
│   ├── run_benchmarks.py          # CLI orchestrator
│   ├── run_all_missing.py         # Fill in missing conditions
│   ├── extract_results.py         # Result extraction utilities
│   ├── analyze_bootstrap.py       # Bootstrap CI analysis
│   ├── requirements.txt           # Python dependencies
│   ├── README.md                  # Detailed experiment documentation
│   ├── adapters/
│   │   ├── mosaic_adapter.py      # 21-constraint library
│   │   ├── ifeval_adapter.py      # 25 instruction checkers
│   │   ├── ifbench_adapter.py     # 58 constraint types
│   │   ├── followbench_adapter.py # Multi-level constraint ladder
│   │   ├── system_prompt_adapter.py  # System prompt compliance
│   │   └── tool_selection_adapter.py # Tool selection accuracy
│   └── results/                   # 52 raw result JSONs
│       ├── mosaic_*.json
│       ├── ifeval_*.json
│       ├── ifbench_*.json
│       ├── followbench_*.json
│       ├── sysprompt_*.json
│       ├── toolsel_*.json
│       └── summary.json
└── README.md
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

### 2. Run with open-weight models
The harness works with any OpenAI-compatible endpoint:
```bash
# Together AI (Llama 3.3 70B)
export OPENAI_API_KEY=your-together-key
python experiments/run_benchmarks.py -b mosaic -c baseline --provider openai \
  --config config/config_together.yaml

# Local vLLM
python experiments/run_benchmarks.py -b ifeval -c all --provider openai \
  --config config/config_vllm.yaml
```

### 3. Cost to reproduce
| Scope | API Cost | Time |
|-------|----------|------|
| Subset (baseline + full_stack, MOSAIC + IFEval) | ~$30 | ~2 hrs |
| Full (9 conditions × 4 benchmarks) | ~$200 | ~8 hrs |
| Open-weight (vLLM on 1× A100) | $0 API | ~4 hrs |

### 4. Open-weight projections
Published IFEval baselines for open-weight models (Llama 3.1 70B: ~0.72; Mistral Large: ~0.74) are 3–8pp below the Claude Opus 4.6 baseline (0.799). Because CPGA+SENTINEL gains are architectural (reducing context competition, enforcing externally), we expect comparable relative improvement. Open-weight results planned for v2.

## Design Principles

1. **Deterministic scoring only** — all check functions are pure Python (regex, counts, AST parsers). Reproducible without API calls for scoring.
2. **Real API calls for generation** — uses Claude Opus 4.6 and GPT-5.4 via standard APIs. Provider-agnostic: works with any OpenAI-compatible endpoint.
3. **Modular adapters** — each benchmark is a self-contained adapter; add new benchmarks by implementing the adapter interface.
4. **Full cost tracking** — every run logs token counts, API calls, and dollar costs.
5. **Cached outputs included** — all 52 result files include raw LLM responses for score verification without API access.

## Citation

```bibtex
@article{cpga_sentinel_2026,
  title={Context-Partitioned Generic Assembly with Swarm-External Enforcement
         for Overcoming LLM Context Degradation},
  year={2026},
  url={https://github.com/teja840-sha/CPGA-SENTINEL}
}
```

## License

MIT
