# CPGA+SENTINEL: Context-Partitioned Generic Assembly with Swarm-External Enforcement

**Overcoming LLM Context Degradation in Instruction-Following Tasks**

## Abstract

Large Language Models lose instruction-following compliance as prompt context grows — a consequence of finite softmax attention. CPGA+SENTINEL addresses this through:

1. **FORGE** (Filtered Orchestration for Rule-Governed Execution) — classifies each context item into the cheapest reliable enforcement layer (token mask, compiled checker, residual prompt, or post-generation judge)
2. **CADG** (Constraint-Aware Diverse Generation) — permutes constraint order across N candidates to de-correlate position-bias failures
3. **SENTINEL** swarm — enforces rules externally via independent per-rule agents across four tiers, removing enforcement context from the LLM prompt entirely

The full stack achieves **+6.4pp** on MOSAIC, **+7.6pp** on SysPrompt, **+5.8pp** on IFEval, and **+14.3pp** in production deployment over baselines.

## Repository Structure

```
CPGA-SENTINEL/
├── paper/
│   ├── CPGA_SENTINEL_PUBLISH_v4.html   # Full paper
│   ├── BENCHMARK_RESULTS.md            # Results summary
│   ├── fig1_architecture.png           # Architecture diagram
│   └── fig2_forge.png                  # FORGE protocol diagram
├── config/
│   └── config.yaml                     # API configuration template
├── experiments/
│   ├── cpga_harness.py                 # Main experiment harness
│   ├── cpga_methods.py                 # CPGA method implementations
│   ├── run_benchmarks.py               # Benchmark runner
│   ├── run_all_missing.py              # Run missing conditions
│   ├── extract_results.py              # Result extraction
│   ├── analyze_bootstrap.py            # Bootstrap CI analysis
│   ├── requirements.txt                # Python dependencies
│   ├── adapters/                       # Benchmark adapters
│   │   ├── ifeval_adapter.py
│   │   ├── ifbench_adapter.py
│   │   ├── mosaic_adapter.py
│   │   ├── followbench_adapter.py
│   │   ├── system_prompt_adapter.py
│   │   └── tool_selection_adapter.py
│   └── results/                        # Raw result JSONs (52 files)
└── README.md
```

## Quick Start

```bash
# Install dependencies
pip install -r experiments/requirements.txt

# Set API keys
export ANTHROPIC_API_KEY=your_key
export OPENAI_API_KEY=your_key

# Run all experiments
python experiments/run_benchmarks.py --all

# Analyze results with bootstrap CIs
python experiments/analyze_bootstrap.py
```

## Benchmarks

| Benchmark | Source | Tasks |
|-----------|--------|-------|
| IFEval | Zhou et al. (2023) | 541 |
| IFBench | Allen AI (2025) | 58 constraint types |
| MOSAIC | EACL 2026 | Multi-constraint |
| FollowBench | Multi-level | 5 difficulty levels |
| SysPrompt | Custom (released here) | System prompt compliance |

## Key Results

| Condition | MOSAIC SCC | IFEval Inst. | IFBench | SysPrompt |
|-----------|-----------|-------------|---------|-----------|
| Baseline | 0.758 | 0.820 | 0.604 | 0.682 |
| Full Stack | **0.822** | **0.878** | **0.696** | **0.759** |
| Δ | +6.4pp | +5.8pp | +9.2pp | +7.6pp |

## Citation

```bibtex
@article{cpga_sentinel_2026,
  title={Context-Partitioned Generic Assembly with Swarm-External Enforcement for Overcoming LLM Context Degradation},
  year={2026},
  note={Preprint}
}
```

## License

MIT
