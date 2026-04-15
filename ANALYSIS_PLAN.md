# Pre-Registered Analysis Plan

**Frozen:** 2026-04-11 (before first benchmark run)  
**Authors:** Siva Teja Narayana

## Primary Endpoints

One primary metric per benchmark, chosen based on the benchmark's design:

| Benchmark | Primary Metric | Rationale |
|-----------|---------------|-----------|
| MOSAIC    | Strict Constraint Compliance (SCC) | Native metric; penalizes partial compliance |
| IFBench   | Instruction Accuracy | Official benchmark metric |
| IFEval    | Instruction Accuracy | Official benchmark metric |
| SysPrompt | Strict Constraint Compliance (SCC) | Custom benchmark; SCC matches MOSAIC for consistency |

## Pre-Specified Primary Comparisons (2 per benchmark)

For each benchmark, two planned pairwise comparisons:

1. **Full Stack vs. Baseline** — tests whether the composed architecture improves over single-pass generation.
2. **Full Stack vs. Retry-with-Feedback** — tests whether architectural partitioning outperforms iterative regeneration.

**Note:** "Full Stack" is the fixed primary arm on all benchmarks. If a simpler
composition (e.g., CADG+SENTINEL) outperforms Full Stack on a given benchmark,
that is reported as a **secondary exploratory finding**, not a primary result.

## Secondary / Exploratory Analyses

- Individual component contributions (CADG, SENTINEL, FORGE alone)
- Pairwise compositions (CADG+SENTINEL, FORGE+CADG, FORGE+SENTINEL)
- Per-constraint-count stratification
- Composition interaction effects
- Cross-model transfer (gpt-5.4, open-weight)

These are clearly labeled exploratory and should be interpreted with appropriate
caution regarding multiplicity.

## Statistical Method

- **Paired task-level bootstrap** (B=2,000, seed=42)
- For each bootstrap iteration: resample task indices with replacement, compute
  per-task score difference δ_i = score_A,i − score_B,i, take the mean
- Report percentile 95% CIs on the mean difference
- Per-condition CIs use the same bootstrap setup (single-arm mean)

## Condition Set (fixed before running)

Nine conditions, all run on every benchmark:

1. Baseline (single-pass, no components)
2. CADG (constraint-order diversity only)
3. SENTINEL (external enforcement only)
4. FORGE (layer partitioning only)
5. CADG+SENTINEL
6. FORGE+CADG
7. FORGE+SENTINEL
8. Full Stack (FORGE+CADG+SENTINEL)
9. Retry-with-Feedback (generate→score→feedback→regenerate, ×3)

## Complete Reporting Commitment

All 9 conditions are reported in full tables, including conditions that show
no improvement or regression. No condition results are suppressed.

## Analytical Extensions (clearly labeled)

Any results derived from measured data (e.g., retry×5 extrapolation, cross-model
projections) are visually marked with callout boxes and dagger (†) symbols in
tables, and are never included in primary comparison CIs.
