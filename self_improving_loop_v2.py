"""
Self-Improving Sort Optimizer — v2
===================================
Smarter prompts, full history context, experiment-level logging.

Changes from v1:
  - Prompt now explains WHY pure-Python loops are slow (interpreter overhead)
  - LLM sees the FULL scoreboard of all previous attempts, not just the last one
  - Best code so far is always included as a reference
  - Experiment metadata logged for write-up (experiment name, notes, prompt versions)

Usage:
    python self_improving_loop_v2.py [--iterations 8] [--list-size 100000] [--experiment "v2-smart-prompts"]

Requires:
    pip install anthropic
    ANTHROPIC_API_KEY environment variable set
"""

import anthropic
import importlib.util
import json
import os
import random
import re
import statistics
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────

ITERATIONS = int(sys.argv[sys.argv.index("--iterations") + 1]) if "--iterations" in sys.argv else 8
LIST_SIZE = int(sys.argv[sys.argv.index("--list-size") + 1]) if "--list-size" in sys.argv else 100_000
BENCHMARK_REPEATS = 7          # more repeats for tighter medians
MODEL = "claude-sonnet-4-20250514"

# Experiment name for the write-up
EXPERIMENT = (
    sys.argv[sys.argv.index("--experiment") + 1]
    if "--experiment" in sys.argv
    else f"experiment-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
)

# Each experiment gets its own log file
LOG_DIR = Path("experiments")
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / f"{EXPERIMENT}.json"

# Also maintain a combined manifest
MANIFEST_FILE = LOG_DIR / "manifest.json"

# ── Prompt Templates ───────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert Python performance engineer. You write sorting functions \
that are as fast as possible on CPython. You understand that:

1. Pure-Python for-loops over 100K items cost 5-80ms of interpreter overhead alone.
2. The fastest path is delegating to C-compiled code (numpy, built-in sorted(), array ops).
3. Algorithmic complexity (O(n) vs O(n log n)) only matters if the constant factor is small.
4. numpy's np.sort on int32 arrays runs an introsort in C — extremely fast.
5. list.sort() / sorted() use Timsort in C — fast, but list→numpy→list can beat it.
6. The conversion cost (list → np.array → .tolist()) is part of the benchmark.

Your goal: write the FASTEST sort_list function possible. Every millisecond matters.\
"""

INITIAL_PROMPT = f"""\
Write a Python function called `sort_list` that takes a list of integers and returns \
a new sorted list. Make it as fast as possible for a list of {LIST_SIZE:,} random \
integers in the range [-1_000_000, 1_000_000].

Key insight: pure-Python loops are ~20x slower than C-compiled equivalents. \
Prefer numpy, built-in sorted(), or other approaches that stay in compiled code.

Rules:
- Signature MUST be: def sort_list(data: list[int]) -> list[int]
- You may use any standard library or numpy. No subprocess/os.system/ctypes.
- Return ONLY the function inside a ```python code block. No explanation.
"""

IMPROVE_PROMPT_TEMPLATE = """\
Here is the scoreboard from all previous iterations:

{scoreboard}

The current best is {best_ms:.3f} ms from iteration {best_iter}:
```python
{best_code}
```

Your most recent attempt (iteration {latest_iter}) ran in {latest_ms:.3f} ms:
```python
{latest_code}
```

{analysis}

Write a new `sort_list` that is FASTER than {best_ms:.3f} ms. \
Think about what made the best version fast and what made slow versions slow. \
Stay in C-compiled territory (numpy, built-ins) — avoid pure-Python loops over the full data.

Rules:
- Signature: def sort_list(data: list[int]) -> list[int]
- numpy is allowed and encouraged. No subprocess/os.system/ctypes.
- Return ONLY the function inside a ```python code block. No explanation.
"""

def build_scoreboard(iterations: list[dict]) -> str:
    """Build a text scoreboard of all previous iterations."""
    lines = ["  Iter │ Median (ms) │ Strategy"]
    lines.append("  ─────┼─────────────┼─────────────────────────────")
    for it in iterations:
        if it["status"] != "success":
            lines.append(f"  {it['iteration']:4d} │   ERROR     │ {it.get('error', 'unknown')[:40]}")
            continue
        ms = it["stats"]["median_ms"]
        # Extract a short strategy label from the code
        code = it["code"]
        if "np.sort" in code or "numpy" in code:
            strategy = "numpy sort"
        elif "count" in code.lower() and "sort" not in code.lower().split("count")[0][-10:]:
            strategy = "counting sort"
        elif "radix" in code.lower() or ("exp" in code and "base" in code):
            strategy = "radix sort"
        elif "sorted(" in code and code.count("\n") < 5:
            strategy = "built-in sorted()"
        elif ".sort()" in code:
            strategy = "in-place .sort()"
        else:
            strategy = "custom"
        
        marker = " ★" if it.get("is_new_best") else "  "
        lines.append(f" {marker}{it['iteration']:3d} │ {ms:10.3f}  │ {strategy}")
    return "\n".join(lines)


def build_analysis(iterations: list[dict]) -> str:
    """Generate a short analysis hint based on patterns in the history."""
    successful = [it for it in iterations if it["status"] == "success"]
    if len(successful) < 2:
        return ""
    
    # Find what strategies worked and didn't
    numpy_times = [it["stats"]["median_ms"] for it in successful if "np.sort" in it["code"] or "numpy" in it["code"]]
    pure_python_times = [it["stats"]["median_ms"] for it in successful if "np.sort" not in it["code"] and "numpy" not in it["code"]]
    
    hints = []
    if numpy_times and pure_python_times:
        avg_np = statistics.mean(numpy_times)
        avg_pp = statistics.mean(pure_python_times)
        if avg_np < avg_pp:
            hints.append(f"Pattern: numpy approaches averaged {avg_np:.1f} ms vs pure-Python at {avg_pp:.1f} ms.")
    
    # Check if we've been stuck at the same best
    best = min(successful, key=lambda x: x["stats"]["median_ms"])
    recent = successful[-3:] if len(successful) >= 3 else successful
    all_worse = all(it["stats"]["median_ms"] > best["stats"]["median_ms"] * 1.05 for it in recent if it["iteration"] != best["iteration"])
    if all_worse and len(recent) >= 2:
        hints.append("You've been stuck — try something fundamentally different from all previous attempts.")
    
    return " ".join(hints)


# ── Helpers ────────────────────────────────────────────────────────────────

def extract_code(response_text: str) -> str:
    """Pull the python code block out of the LLM response."""
    match = re.search(r"```python\s*\n(.*?)```", response_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"```\s*\n(.*?)```", response_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return response_text.strip()


def load_sort_function(code: str):
    """Dynamically load the sort_list function from a code string."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        f.flush()
        spec = importlib.util.spec_from_file_location("candidate", f.name)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        os.unlink(f.name)
    
    if not hasattr(module, "sort_list"):
        raise AttributeError("Code does not define a `sort_list` function")
    return module.sort_list


def benchmark(sort_fn, test_data: list[int], repeats: int = BENCHMARK_REPEATS) -> dict:
    """Benchmark a sort function. Returns timing stats in milliseconds."""
    # Verify correctness
    result = sort_fn(test_data[:1000])
    expected = sorted(test_data[:1000])
    if result != expected:
        raise ValueError("sort_list returned incorrect results!")
    
    # Warm-up run (especially important for numpy import caching)
    sort_fn(test_data[:1000])
    
    # Benchmark with full data
    times = []
    for _ in range(repeats):
        data_copy = test_data.copy()
        start = time.perf_counter()
        sort_fn(data_copy)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    
    return {
        "median_ms": statistics.median(times),
        "mean_ms": statistics.mean(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "stdev_ms": statistics.stdev(times) if len(times) > 1 else 0,
        "all_ms": times,
    }


def call_llm(client: anthropic.Anthropic, system: str, prompt: str) -> str:
    """Send a prompt to Claude with a system prompt and return the text response."""
    message = client.messages.create(
        model=MODEL,
        max_tokens=2048,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


def update_manifest(experiment_name: str, log_file: Path, log: dict):
    """Update the experiments manifest for easy discovery."""
    manifest = []
    if MANIFEST_FILE.exists():
        try:
            manifest = json.loads(MANIFEST_FILE.read_text())
        except json.JSONDecodeError:
            manifest = []
    
    # Update or add this experiment
    entry = {
        "experiment": experiment_name,
        "file": str(log_file.name),
        "model": log["config"]["model"],
        "list_size": log["config"]["list_size"],
        "iterations": log["config"]["iterations"],
        "started_at": log["config"]["started_at"],
        "best_ms": log["config"].get("best_ms"),
        "best_iter": log["config"].get("best_iter"),
        "prompt_version": log["config"].get("prompt_version"),
    }
    
    # Replace existing or append
    manifest = [e for e in manifest if e["experiment"] != experiment_name]
    manifest.append(entry)
    MANIFEST_FILE.write_text(json.dumps(manifest, indent=2))


# ── Main Loop ──────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  SELF-IMPROVING SORT OPTIMIZER — v2")
    print(f"  Experiment: {EXPERIMENT}")
    print(f"  Model: {MODEL}")
    print(f"  List size: {LIST_SIZE:,} integers")
    print(f"  Iterations: {ITERATIONS}")
    print(f"  Log: {LOG_FILE}")
    print("=" * 70)
    
    client = anthropic.Anthropic()
    
    # Generate a fixed test dataset for fair comparison
    random.seed(42)
    test_data = [random.randint(-1_000_000, 1_000_000) for _ in range(LIST_SIZE)]
    
    log = {
        "config": {
            "model": MODEL,
            "list_size": LIST_SIZE,
            "iterations": ITERATIONS,
            "benchmark_repeats": BENCHMARK_REPEATS,
            "experiment": EXPERIMENT,
            "prompt_version": "v2-smart-context",
            "started_at": datetime.now().isoformat(),
            "changes_from_v1": [
                "System prompt with performance engineering context",
                "Full scoreboard of all iterations shown to LLM",
                "Best code always included as reference",
                "Auto-generated analysis hints (numpy vs pure-python patterns)",
                "7 benchmark repeats (up from 5)",
                "Warm-up run before benchmarking",
            ],
        },
        "iterations": [],
        "prompts_used": [],  # log every prompt for the write-up
    }
    
    best_ms = float("inf")
    best_iter = 0
    best_code = ""
    previous_code = ""
    latest_ms = 0
    
    for i in range(1, ITERATIONS + 1):
        print(f"\n{'─' * 70}")
        print(f"  ITERATION {i}/{ITERATIONS}")
        print(f"{'─' * 70}")
        
        # ── Step 1: Build the prompt ──
        if i == 1:
            prompt = INITIAL_PROMPT
        else:
            scoreboard = build_scoreboard(log["iterations"])
            analysis = build_analysis(log["iterations"])
            prompt = IMPROVE_PROMPT_TEMPLATE.format(
                scoreboard=scoreboard,
                best_ms=best_ms,
                best_iter=best_iter,
                best_code=best_code,
                latest_iter=i - 1,
                latest_ms=latest_ms,
                latest_code=previous_code,
                analysis=analysis,
            )
        
        # Log the prompt
        log["prompts_used"].append({
            "iteration": i,
            "system_prompt": SYSTEM_PROMPT,
            "user_prompt": prompt,
        })
        
        print("  → Asking Claude for a sorting function...")
        response_text = call_llm(client, SYSTEM_PROMPT, prompt)
        code = extract_code(response_text)
        
        print(f"  → Received {len(code)} chars of code")
        print()
        for line in code.splitlines():
            print(f"    {line}")
        print()
        
        # ── Step 2: Load & Validate ──
        try:
            sort_fn = load_sort_function(code)
        except Exception as e:
            print(f"  ✗ Failed to load code: {e}")
            log["iterations"].append({
                "iteration": i,
                "code": code,
                "error": str(e),
                "status": "load_error",
            })
            continue
        
        # ── Step 3: Benchmark ──
        try:
            print(f"  → Benchmarking ({BENCHMARK_REPEATS} runs)...")
            stats = benchmark(sort_fn, test_data)
            latest_ms = stats["median_ms"]
        except Exception as e:
            print(f"  ✗ Benchmark failed: {e}")
            log["iterations"].append({
                "iteration": i,
                "code": code,
                "error": str(e),
                "status": "benchmark_error",
            })
            continue
        
        # ── Step 4: Record Results ──
        is_new_best = latest_ms < best_ms
        if is_new_best:
            best_ms = latest_ms
            best_iter = i
            best_code = code
        
        status_icon = "★" if is_new_best else "·"
        print(f"  {status_icon} Median: {latest_ms:.3f} ms  |  Best: {best_ms:.3f} ms (iter {best_iter})")
        print(f"    Range: [{stats['min_ms']:.3f}, {stats['max_ms']:.3f}] ms  σ={stats['stdev_ms']:.3f}")
        
        if not is_new_best and best_ms > 0:
            pct_slower = ((latest_ms - best_ms) / best_ms) * 100
            print(f"    ({pct_slower:.1f}% slower than best)")
        
        iteration_record = {
            "iteration": i,
            "code": code,
            "status": "success",
            "is_new_best": is_new_best,
            "stats": {
                "median_ms": round(stats["median_ms"], 4),
                "mean_ms": round(stats["mean_ms"], 4),
                "min_ms": round(stats["min_ms"], 4),
                "max_ms": round(stats["max_ms"], 4),
                "stdev_ms": round(stats["stdev_ms"], 4),
                "all_ms": [round(t, 4) for t in stats["all_ms"]],
            },
        }
        log["iterations"].append(iteration_record)
        previous_code = code
        
        # Save log after each iteration (crash-safe)
        log["config"]["best_ms"] = round(best_ms, 4)
        log["config"]["best_iter"] = best_iter
        log["config"]["finished_at"] = datetime.now().isoformat()
        LOG_FILE.write_text(json.dumps(log, indent=2))
        update_manifest(EXPERIMENT, LOG_FILE, log)
    
    # ── Summary ────────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("  FINAL RESULTS")
    print(f"{'=' * 70}")
    
    successful = [it for it in log["iterations"] if it["status"] == "success"]
    if successful:
        print(f"\n  Best time: {best_ms:.3f} ms (iteration {best_iter})")
        print(f"  Iterations completed: {len(successful)}/{ITERATIONS}")
        print(f"\n  Scoreboard:")
        print(build_scoreboard(log["iterations"]))
    
    print(f"\n  Experiment log: {LOG_FILE}")
    print(f"  All experiments: {MANIFEST_FILE}")
    print("=" * 70)


if __name__ == "__main__":
    main()