"""
Self-Improving Sort Optimizer
=============================
An LLM writes a sorting function, we benchmark it, then ask for something faster.
Loop N times and watch it converge (or get creative).

Usage:
    python self_improving_loop.py [--iterations 8] [--list-size 100000]

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
import subprocess
import sys
import tempfile
import textwrap
import time
import timeit
from datetime import datetime
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────

ITERATIONS = int(sys.argv[sys.argv.index("--iterations") + 1]) if "--iterations" in sys.argv else 8
LIST_SIZE = int(sys.argv[sys.argv.index("--list-size") + 1]) if "--list-size" in sys.argv else 100_000
BENCHMARK_REPEATS = 5          # timeit repetitions per measurement
MODEL = "claude-sonnet-4-20250514"
LOG_FILE = Path("optimization_log.json")

# ── Prompt Templates ───────────────────────────────────────────────────────

INITIAL_PROMPT = f"""\
Write a Python function called `sort_list` that takes a list of integers and returns \
a new sorted list. Make it as fast as possible for a list of {LIST_SIZE:,} random \
integers in the range [-1_000_000, 1_000_000].

Rules:
- The function signature MUST be: def sort_list(data: list[int]) -> list[int]
- You may use any standard library or pure-Python technique.
- You may NOT use subprocess, os.system, ctypes, or any C extensions beyond what ships with Python.
- numpy is allowed if you think it helps.
- Return ONLY the function inside a ```python code block. No explanation needed.
"""

IMPROVE_PROMPT_TEMPLATE = """\
Your previous `sort_list` function ran in {latest_ms:.3f} ms (median of {repeats} runs).
The best so far across all iterations is {best_ms:.3f} ms (iteration {best_iter}).

Here is your previous code:
```python
{previous_code}
```

Write an improved version that is FASTER. Try a fundamentally different strategy if \
the current one seems near-optimal. Consider: timsort shortcuts, radix sort, \
counting sort, numpy-based approaches, or hybrid strategies.

Same rules apply:
- Signature: def sort_list(data: list[int]) -> list[int]
- No subprocess/os.system/ctypes/C-extensions (numpy is OK).
- Return ONLY the function inside a ```python code block.
"""

# ── Helpers ────────────────────────────────────────────────────────────────

def extract_code(response_text: str) -> str:
    """Pull the python code block out of the LLM response."""
    # Try ```python ... ``` first
    match = re.search(r"```python\s*\n(.*?)```", response_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback: any ``` block
    match = re.search(r"```\s*\n(.*?)```", response_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Last resort: assume the whole thing is code
    return response_text.strip()


def load_sort_function(code: str):
    """Dynamically load the sort_list function from a code string."""
    # Write to a temp file and import it
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
    # Verify correctness first
    result = sort_fn(test_data[:1000])
    expected = sorted(test_data[:1000])
    if result != expected:
        raise ValueError("sort_list returned incorrect results!")
    
    # Benchmark with full data
    times = []
    for _ in range(repeats):
        data_copy = test_data.copy()  # fresh copy each time
        start = time.perf_counter()
        sort_fn(data_copy)
        elapsed = (time.perf_counter() - start) * 1000  # ms
        times.append(elapsed)
    
    return {
        "median_ms": statistics.median(times),
        "mean_ms": statistics.mean(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "stdev_ms": statistics.stdev(times) if len(times) > 1 else 0,
        "all_ms": times,
    }


def call_llm(client: anthropic.Anthropic, prompt: str) -> str:
    """Send a prompt to Claude and return the text response."""
    message = client.messages.create(
        model=MODEL,
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


# ── Main Loop ──────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  SELF-IMPROVING SORT OPTIMIZER")
    print(f"  Model: {MODEL}")
    print(f"  List size: {LIST_SIZE:,} integers")
    print(f"  Iterations: {ITERATIONS}")
    print("=" * 70)
    
    client = anthropic.Anthropic()  # uses ANTHROPIC_API_KEY env var
    
    # Generate a fixed test dataset for fair comparison
    random.seed(42)
    test_data = [random.randint(-1_000_000, 1_000_000) for _ in range(LIST_SIZE)]
    
    log = {
        "config": {
            "model": MODEL,
            "list_size": LIST_SIZE,
            "iterations": ITERATIONS,
            "benchmark_repeats": BENCHMARK_REPEATS,
            "started_at": datetime.now().isoformat(),
        },
        "iterations": [],
    }
    
    best_ms = float("inf")
    best_iter = 0
    previous_code = ""
    
    for i in range(1, ITERATIONS + 1):
        print(f"\n{'─' * 70}")
        print(f"  ITERATION {i}/{ITERATIONS}")
        print(f"{'─' * 70}")
        
        # ── Step 1: Ask the LLM ──
        if i == 1:
            prompt = INITIAL_PROMPT
        else:
            prompt = IMPROVE_PROMPT_TEMPLATE.format(
                latest_ms=latest_ms,
                repeats=BENCHMARK_REPEATS,
                best_ms=best_ms,
                best_iter=best_iter,
                previous_code=previous_code,
            )
        
        print("  → Asking Claude for a sorting function...")
        response_text = call_llm(client, prompt)
        code = extract_code(response_text)
        
        print(f"  → Received {len(code)} chars of code")
        print()
        # Show the code indented
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
        
        status_icon = "★" if is_new_best else "·"
        print(f"  {status_icon} Median: {latest_ms:.3f} ms  |  Best: {best_ms:.3f} ms (iter {best_iter})")
        print(f"    Range: [{stats['min_ms']:.3f}, {stats['max_ms']:.3f}] ms  σ={stats['stdev_ms']:.3f}")
        
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
        LOG_FILE.write_text(json.dumps(log, indent=2))
    
    # ── Summary ────────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("  FINAL RESULTS")
    print(f"{'=' * 70}")
    
    successful = [it for it in log["iterations"] if it["status"] == "success"]
    if successful:
        print(f"\n  Best time: {best_ms:.3f} ms (iteration {best_iter})")
        print(f"  Iterations completed: {len(successful)}/{ITERATIONS}")
        print(f"\n  Timeline:")
        for it in successful:
            marker = " ★" if it["is_new_best"] else "  "
            print(f"   {marker} Iter {it['iteration']}: {it['stats']['median_ms']:.3f} ms")
    
    print(f"\n  Full log saved to: {LOG_FILE}")
    print(f"  View the dashboard: open the .html artifact in your browser")
    print("=" * 70)


if __name__ == "__main__":
    main()