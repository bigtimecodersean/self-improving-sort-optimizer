# Self-Improving Sort Optimizer

**Can an LLM optimize its own code through a benchmark feedback loop?**

A toy automated AI research project that explores self-improving loops: an LLM writes a sorting function, we benchmark it, feed the result back, and ask for something faster. Across 8 experiments and 64 iterations, we tested whether this loop converges on better solutions — and whether an automated meta-optimizer can replace human prompt engineering.

## The Setup

The inner loop is simple:

1. Ask Claude (Sonnet 4) to write a Python `sort_list` function
2. Dynamically load and benchmark it against 100,000 random integers
3. Feed the timing back: *"Your solution ran in X ms. The best so far is Y ms. Write a faster version."*
4. Repeat for 8 iterations

We ran this loop with three different prompt strategies (v1–v3), each informed by failures in the previous version. We then built a meta-optimizer — a second LLM call that analyzes experiment results and rewrites the prompts automatically — and ran it for 5 rounds.

## Results

| Experiment | Best (ms) | Success Rate | Key Behavior |
|---|---|---|---|
| **v1** — Naive prompt | 3.95 | 8/8 | Got lucky iter 1 with numpy, then wasted 7 iterations on pure-Python radix/counting sorts (11–82ms) |
| **v2** — Performance context | 3.75 | 4/8 | Stayed in numpy territory, but hit a numpy 2.x API change (`copy=False` removed) and repeated the same crash 4 times |
| **v3** — Error feedback | 3.54 | 8/8 | Best manual result. Zero errors. Explored diverse strategies while staying anchored to what worked |
| **Meta R1** | 3.68 | 8/8 | Used v3 prompts as defaults — matched our results |
| **Meta R2** | 11.46 | 8/8 | Outer loop pushed "diverse strategies" → catastrophic regression to pure Python |
| **Meta R3–R5** | 3.89 | 6–7/8 | Slowly recovered but never beat R1. Success rate degraded as rewritten prompts dropped guardrails |

### The winning code (3.54ms)

```python
def sort_list(data: list[int]) -> list[int]:
    import numpy as np
    arr = np.fromiter(data, dtype=np.int32, count=len(data))
    arr.sort()
    return arr.tolist()
```

## Key Findings

### 1. The performance ceiling was trivially reachable

Every experiment found the same answer within 1–2 iterations: delegate to numpy's C-compiled sort. The ~3.5–4.1ms spread across "best" times is system noise, not algorithmic improvement. Sorting 100K integers is a solved problem — the LLM can't optimize past what numpy already provides.

### 2. The real optimization happened at the prompt level

The inner loop (LLM improving its own code) showed minimal genuine improvement. The outer loop (human rewriting prompts based on failure analysis) drove all meaningful gains:

- **v1 → v2**: Adding performance context ("pure-Python loops cost 5–80ms of interpreter overhead") eliminated pure-Python approaches entirely
- **v2 → v3**: Adding error feedback prevented crash loops; anti-repetition instructions increased strategy diversity

### 3. The automated meta-optimizer performed worse than the human

The meta-optimizer over-corrected in round 2 (pushing the inner loop toward "diverse strategies" which meant slow pure-Python sorts), lost hard-won guardrails (the numpy 2.x `copy=False` warning was dropped when prompts were rewritten), produced malformed JSON in round 4 (breaking its own evolution), and showed degrading success rates across rounds (8/8 → 6/8).

Human prompt engineering required understanding *why* something failed, not just *that* it failed. The meta-optimizer lacked this judgment.

### 4. Task selection matters more than loop architecture

Sorting was a poor choice for demonstrating self-improvement because the search space is narrow (one trick: use numpy), the ceiling is trivially reachable, improvements are not compositional, and the feedback signal is noisy. A good self-improvement task needs a wide search space, a clear gradient, compositional strategies, and an accessible-but-hard ceiling.

## Project Structure

```
ai-research-loop/
├── self_improving_loop.py          # v1 — naive prompt
├── self_improving_loop_v2.py       # v2 — smart context (used for v2 run)
├── self_improving_loop_v3.py       # v3 — error feedback + anti-repetition
├── meta_optimizer.py               # Automated outer loop
├── migrate_v1_log.py               # Utility to archive v1 results
├── optimization_log.json           # v1 raw results
└── experiments/
    ├── manifest.json               # Index of all experiments
    ├── v1-baseline.json            # v1 full log
    ├── v2-smart-prompts.json       # v2 full log
    ├── v3-error-feedback.json      # v3 full log
    ├── meta-round-01.json          # Meta-optimizer round logs
    ├── meta-round-02.json
    ├── meta-round-03.json
    ├── meta-round-04.json
    ├── meta-round-05.json
    └── meta-optimizer-*.json       # Meta-optimizer summary log
```

## Running It

```bash
# Install dependencies
pip install anthropic numpy

# Set your API key
export ANTHROPIC_API_KEY="sk-ant-..."

# Run v3 (recommended starting point)
python self_improving_loop_v3.py --experiment "my-experiment" --iterations 8

# Run the meta-optimizer
python meta_optimizer.py --rounds 5 --inner-iterations 8
```

Estimated API cost: ~$0.04 per 8-iteration experiment, ~$0.25 for a full meta-optimizer run.

## What I'd Do Differently

The sorting task was the wrong task for this architecture. A good self-improvement target needs:

- **Wide search space** — many valid approaches, not one trick
- **Continuous gradient** — small changes produce measurable improvement
- **Compositional strategies** — multiple ideas that stack
- **Hard but reachable ceiling** — start at 30%, climb to 80%
- **Informative failures** — can see *which cases* failed and *why*
- **Cheap evaluation** — runs in seconds, no GPU

Better candidates: prompt optimization for a reasoning benchmark, game strategy (2048), mini-ARC solvers, or LLM self-calibration tasks.

## Built With

- **Claude Sonnet 4** (claude-sonnet-4-20250514) — both inner loop (code generation) and outer loop (meta-analysis)
- **Python 3.11** on macOS ARM
- **numpy 2.x** for benchmarking
- Total cost across all experiments: ~$0.35
