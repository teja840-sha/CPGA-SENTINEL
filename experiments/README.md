# CPGA+SENTINEL Evaluation Suite

Runs the CPGA+SENTINEL architecture against three public instruction-following benchmarks.

## Benchmarks

| Benchmark | Source | What it tests | Scoring |
|-----------|--------|---------------|---------|
| **MOSAIC** | [CapitalOne](https://github.com/CapitalOne-Research/llm-instruction-following-compliance) | 21 constraint types, 5–20 per task | SCC, PA (deterministic + LLM judge) |
| **IFEval** | [Google](https://huggingface.co/datasets/google/IFEval) | 541 prompts, 25 instruction types | Strict/loose accuracy (deterministic) |
| **FollowBench** | [ACL 2024](https://github.com/YJiangcm/FollowBench) | Multi-level constraint ladder (L1–L5) | HSR, SSR, CSL |

## Experimental Conditions

| Condition | Description |
|-----------|-------------|
| **Baseline** | All constraints in prompt → generate → score |
| **FORGE** | Classify constraints by layer, remove Layer 0/1, generate with residual only |
| **CADG** | Permute constraint order across N=5 candidates → score → select best |
| **SENTINEL** | Generate with constraints → post-gen Tier 2/3 checking and fixing |
| **Full Stack** | FORGE + CADG + SENTINEL composed |

## Setup

```bash
pip install -r requirements.txt
```

Set API keys:
```bash
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...
```

Or edit `config/config.yaml` directly.

## Usage

```bash
# Dry run (no API calls, tests the pipeline)
python run_benchmarks.py --dry-run

# Run MOSAIC with all conditions
python run_benchmarks.py -b mosaic -c all

# Run IFEval baseline only (first 50 tasks)
python run_benchmarks.py -b ifeval -c baseline --max-tasks 50

# Run everything
python run_benchmarks.py -b all -c all

# Use OpenAI instead of Anthropic
python run_benchmarks.py -b mosaic -c baseline --provider openai
```

## Key Chart (paper Figure 8)

The critical result: plot satisfaction rate vs constraint count for each condition on MOSAIC.

If CPGA (especially Full Stack) keeps the curve **flat** while Baseline degrades,
that's the publishable finding — constraint-count degradation is the central claim.

## Architecture

```
run_benchmarks.py          # CLI orchestrator
├── cpga_harness.py        # LLM client, cost tracking, scoring primitives
├── cpga_methods.py        # FORGE, CADG, SENTINEL implementations
└── adapters/
    ├── mosaic_adapter.py      # 21-constraint library, task generation
    ├── ifeval_adapter.py      # 25 instruction checkers from IFEval
    └── followbench_adapter.py # Multi-level constraint chain, HSR/SSR/CSL
```

## Results

Saved to `results/` as JSON:
- `results/mosaic_baseline.json`, `mosaic_forge.json`, etc.
- `results/ifeval_baseline.json`, etc.
- `results/followbench_baseline.json`, etc.
- `results/summary.json` (cost + call totals)

## Design Decisions

1. **Deterministic scoring only.** No LLM judges for metric computation. All check functions
   are Python code (regex, counts, parsers). Semantic constraints that require LLM judgment
   (tone, style, accuracy) are marked `skipped` in results and excluded from checkable totals.

2. **Synthetic MOSAIC tasks.** The real MOSAIC dataset requires running their generator.
   Our synthetic library implements 21 constraint types matching their paper's Table 4.
   If you clone the repo, pass `--mosaic-repo path/to/repo` to load their data directly.

3. **IFEval checkers reimplemented.** Google's reference checkers from `instructions.py` are
   reimplemented here to avoid the dependency on their full codebase. All 25 instruction
   types are covered with the same logic.

4. **FollowBench synthetic fallback.** When the repo isn't cloned, 3 synthetic tasks with
   5-level constraint ladders are used. Clone the repo for the full 500+ tasks.
