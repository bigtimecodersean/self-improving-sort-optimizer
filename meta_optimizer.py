"""
Meta-Optimizer — Self-Improving Self-Improvement Loop
=====================================================
Two-level architecture:
  OUTER LOOP: An LLM analyzes experiment results, diagnoses failures,
              and rewrites the system/user prompts for the next experiment.
  INNER LOOP: The existing sort optimizer runs with those prompts.

This is the script that ran as "you" in our v1→v2→v3 conversation —
diagnosing problems, tweaking prompts, and re-running.

Usage:
    python meta_optimizer.py [--rounds 5] [--inner-iterations 8]

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

OUTER_ROUNDS = int(sys.argv[sys.argv.index("--rounds") + 1]) if "--rounds" in sys.argv else 5
INNER_ITERATIONS = int(sys.argv[sys.argv.index("--inner-iterations") + 1]) if "--inner-iterations" in sys.argv else 8
LIST_SIZE = int(sys.argv[sys.argv.index("--list-size") + 1]) if "--list-size" in sys.argv else 100_000
BENCHMARK_REPEATS = 7
INNER_MODEL = "claude-sonnet-4-20250514"   # writes sort functions
OUTER_MODEL = "claude-sonnet-4-20250514"   # analyzes results & rewrites prompts

LOG_DIR = Path("experiments")
LOG_DIR.mkdir(exist_ok=True)
META_LOG_FILE = LOG_DIR / f"meta-optimizer-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"

# ── Fixed test data ────────────────────────────────────────────────────────

random.seed(42)
TEST_DATA = [random.randint(-1_000_000, 1_000_000) for _ in range(LIST_SIZE)]

# ── Default prompts (outer loop will evolve these) ─────────────────────────

DEFAULT_SYSTEM_PROMPT = """\
You are an expert Python performance engineer. You write sorting functions \
that are as fast as possible on CPython. You understand that:

1. Pure-Python for-loops over 100K items cost 5-80ms of interpreter overhead alone.
2. The fastest path is delegating to C-compiled code (numpy, built-in sorted(), array ops).
3. Algorithmic complexity (O(n) vs O(n log n)) only matters if the constant factor is small.
4. numpy's np.sort on int32 arrays runs an introsort in C — extremely fast.
5. list.sort() / sorted() use Timsort in C — fast, but list→numpy→list can beat it.
6. The conversion cost (list → np.array → .tolist()) is part of the benchmark.

CRITICAL environment details:
- numpy 2.x is installed. Do NOT use copy=False in np.array() — it was removed.
  Use np.asarray() if you want to avoid copies, or just np.array().
- Python 3.11 on macOS ARM (Apple Silicon).

Your goal: write the FASTEST sort_list function possible. Every millisecond matters. \
Never submit code identical or nearly identical to a previous attempt.\
"""

DEFAULT_INITIAL_PROMPT = f"""\
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

DEFAULT_IMPROVE_TEMPLATE = """\
Here is the scoreboard from all previous iterations:

{scoreboard}

The current best is {best_ms:.3f} ms from iteration {best_iter}:
```python
{best_code}
```

{error_context}\
Your most recent attempt (iteration {latest_iter}) {latest_result}:
```python
{latest_code}
```

{analysis}

IMPORTANT: Do NOT repeat any previous approach. Each attempt must be meaningfully different.

Write a new `sort_list` that is FASTER than {best_ms:.3f} ms.

Rules:
- Signature: def sort_list(data: list[int]) -> list[int]
- numpy is allowed and encouraged. No subprocess/os.system/ctypes.
- Do NOT use np.array(..., copy=False) — it fails on numpy 2.x.
- Return ONLY the function inside a ```python code block. No explanation.
"""

# ── The Outer Loop Prompt ──────────────────────────────────────────────────

OUTER_ANALYSIS_PROMPT = """\
You are a meta-optimizer. You analyze the results of an AI code optimization experiment \
and rewrite the prompts to get better results next round.

Here is a summary of ALL experiments so far:

{experiment_history}

The inner loop works like this:
1. An LLM (Claude Sonnet) receives a system prompt + user prompt asking it to write a fast sorting function.
2. The code is extracted, benchmarked on {list_size:,} random integers, and the result is fed back.
3. This repeats for {inner_iters} iterations per experiment.

Your job: analyze what went wrong and what went right, then output IMPROVED prompts \
that will produce better results in the next experiment.

Think about:
- Are there strategies the LLM hasn't tried yet? (e.g., specific numpy internals, \
  buffer protocol tricks, parallel approaches, different sort algorithms)
- Did it waste iterations repeating similar approaches?
- Did it crash on API incompatibilities? If so, warn more explicitly.
- Is the feedback it receives sufficient to learn from mistakes?
- Could the improve template be more directive about what to try next?

Output your response as a JSON object with exactly these keys:
```json
{{
  "diagnosis": "2-3 sentence analysis of what happened and why",
  "changes": ["list of specific changes you're making to the prompts"],
  "system_prompt": "the complete new system prompt",
  "initial_prompt": "the complete new initial prompt (use {{list_size}} as placeholder for the list size)",
  "improve_template": "the complete new improve template (must contain these placeholders: {{scoreboard}}, {{best_ms}}, {{best_iter}}, {{best_code}}, {{error_context}}, {{latest_iter}}, {{latest_result}}, {{latest_code}}, {{analysis}})"
}}
```

Return ONLY the JSON. No other text.
"""

# ── Inner Loop Helpers (same as v3) ────────────────────────────────────────

def extract_code(response_text: str) -> str:
    match = re.search(r"```python\s*\n(.*?)```", response_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"```\s*\n(.*?)```", response_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return response_text.strip()


def extract_json(response_text: str) -> dict:
    """Extract JSON from LLM response, handling code fences."""
    # Try to find JSON in code block
    match = re.search(r"```(?:json)?\s*\n(.*?)```", response_text, re.DOTALL)
    text = match.group(1).strip() if match else response_text.strip()
    return json.loads(text)


def load_sort_function(code: str):
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


def benchmark(sort_fn, repeats=BENCHMARK_REPEATS):
    result = sort_fn(TEST_DATA[:1000])
    expected = sorted(TEST_DATA[:1000])
    if result != expected:
        raise ValueError("sort_list returned incorrect results!")
    sort_fn(TEST_DATA[:1000])  # warm-up
    times = []
    for _ in range(repeats):
        data_copy = TEST_DATA.copy()
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


def build_scoreboard(iterations):
    lines = ["  Iter │ Median (ms) │ Strategy"]
    lines.append("  ─────┼─────────────┼─────────────────────────────")
    for it in iterations:
        if it["status"] != "success":
            lines.append(f"  {it['iteration']:4d} │   ERROR     │ {it.get('error', 'unknown')[:40]}")
            continue
        ms = it["stats"]["median_ms"]
        code = it["code"]
        if "np.sort" in code or "numpy" in code:
            strategy = "numpy sort"
        elif "count" in code.lower():
            strategy = "counting sort"
        elif "radix" in code.lower():
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


def build_analysis(iterations):
    successful = [it for it in iterations if it["status"] == "success"]
    if len(successful) < 2:
        return ""
    numpy_times = [it["stats"]["median_ms"] for it in successful if "np" in it["code"]]
    pure_times = [it["stats"]["median_ms"] for it in successful if "np" not in it["code"]]
    hints = []
    if numpy_times and pure_times:
        if statistics.mean(numpy_times) < statistics.mean(pure_times):
            hints.append(f"Pattern: numpy avg {statistics.mean(numpy_times):.1f} ms vs pure-Python {statistics.mean(pure_times):.1f} ms.")
    best = min(successful, key=lambda x: x["stats"]["median_ms"])
    recent = successful[-3:]
    if len(recent) >= 2 and all(it["stats"]["median_ms"] > best["stats"]["median_ms"] * 1.05 for it in recent if it != best):
        hints.append("Stuck — try something fundamentally different.")
    return " ".join(hints)


# ── Inner Loop ─────────────────────────────────────────────────────────────

def run_inner_loop(client, system_prompt, initial_prompt, improve_template, experiment_name):
    """Run one complete inner optimization loop. Returns the experiment log."""
    log = {
        "config": {
            "model": INNER_MODEL,
            "list_size": LIST_SIZE,
            "iterations": INNER_ITERATIONS,
            "benchmark_repeats": BENCHMARK_REPEATS,
            "experiment": experiment_name,
            "started_at": datetime.now().isoformat(),
        },
        "system_prompt": system_prompt,
        "initial_prompt": initial_prompt,
        "improve_template": improve_template,
        "iterations": [],
    }

    best_ms = float("inf")
    best_iter = 0
    best_code = ""
    previous_code = ""
    latest_ms = 0
    last_error = ""

    for i in range(1, INNER_ITERATIONS + 1):
        print(f"    Iter {i}/{INNER_ITERATIONS}", end=" → ")

        # Build prompt
        if i == 1:
            prompt = initial_prompt
        else:
            scoreboard = build_scoreboard(log["iterations"])
            analysis = build_analysis(log["iterations"])
            error_context = ""
            latest_result = f"ran in {latest_ms:.3f} ms"
            if last_error:
                error_context = (
                    f"⚠ Previous attempt CRASHED: {last_error}\n"
                    f"Do NOT repeat this mistake.\n\n"
                )
                latest_result = f"CRASHED ({last_error[:60]})"
            prompt = improve_template.format(
                scoreboard=scoreboard,
                best_ms=best_ms,
                best_iter=best_iter,
                best_code=best_code,
                latest_iter=i - 1,
                latest_ms=latest_ms,
                latest_code=previous_code,
                latest_result=latest_result,
                error_context=error_context,
                analysis=analysis,
            )

        # Call LLM
        try:
            message = client.messages.create(
                model=INNER_MODEL, max_tokens=2048,
                system=system_prompt,
                messages=[{"role": "user", "content": prompt}],
            )
            code = extract_code(message.content[0].text)
        except Exception as e:
            print(f"API error: {e}")
            log["iterations"].append({"iteration": i, "code": "", "error": str(e), "status": "api_error"})
            continue

        # Load
        try:
            sort_fn = load_sort_function(code)
        except Exception as e:
            print(f"load error: {e}")
            last_error = str(e)
            log["iterations"].append({"iteration": i, "code": code, "error": str(e), "status": "load_error"})
            previous_code = code
            continue

        # Benchmark
        try:
            stats = benchmark(sort_fn)
            latest_ms = stats["median_ms"]
        except Exception as e:
            print(f"bench error: {e}")
            last_error = str(e)
            log["iterations"].append({"iteration": i, "code": code, "error": str(e), "status": "benchmark_error"})
            previous_code = code
            continue

        # Record
        last_error = ""
        is_new_best = latest_ms < best_ms
        if is_new_best:
            best_ms = latest_ms
            best_iter = i
            best_code = code

        marker = "★" if is_new_best else "·"
        print(f"{marker} {latest_ms:.3f} ms", end="")
        if is_new_best:
            print(f"  NEW BEST")
        else:
            print(f"  (+{((latest_ms - best_ms) / best_ms * 100):.0f}%)")

        log["iterations"].append({
            "iteration": i, "code": code, "status": "success",
            "is_new_best": is_new_best,
            "stats": {k: round(v, 4) if isinstance(v, float) else [round(x, 4) for x in v] if isinstance(v, list) else v for k, v in stats.items()},
        })
        previous_code = code

    log["config"]["best_ms"] = round(best_ms, 4) if best_ms < float("inf") else None
    log["config"]["best_iter"] = best_iter
    log["config"]["finished_at"] = datetime.now().isoformat()
    log["config"]["success_rate"] = f"{sum(1 for it in log['iterations'] if it['status'] == 'success')}/{INNER_ITERATIONS}"

    # Save individual experiment
    exp_file = LOG_DIR / f"{experiment_name}.json"
    exp_file.write_text(json.dumps(log, indent=2))

    return log


# ── Outer Loop ─────────────────────────────────────────────────────────────

def build_experiment_summary(exp_log):
    """Create a concise summary of an experiment for the outer loop."""
    cfg = exp_log["config"]
    iters = exp_log["iterations"]
    successful = [it for it in iters if it["status"] == "success"]
    errors = [it for it in iters if it["status"] != "success"]

    strategies = []
    for it in iters:
        if it["status"] == "success":
            strategies.append(f"  Iter {it['iteration']}: {it['stats']['median_ms']:.3f} ms — {it['code'][:80]}...")
        else:
            strategies.append(f"  Iter {it['iteration']}: ERROR — {it.get('error', '')[:60]}")

    return f"""Experiment: {cfg['experiment']}
Best: {cfg.get('best_ms', 'N/A')} ms (iter {cfg.get('best_iter', 'N/A')})
Success rate: {cfg.get('success_rate', 'N/A')}
Errors: {len(errors)}
Iterations:
{chr(10).join(strategies)}

System prompt (first 200 chars): {exp_log.get('system_prompt', 'N/A')[:200]}...
"""


def main():
    print("=" * 70)
    print("  META-OPTIMIZER — Self-Improving Self-Improvement")
    print(f"  Outer rounds: {OUTER_ROUNDS}")
    print(f"  Inner iterations per round: {INNER_ITERATIONS}")
    print(f"  Inner model: {INNER_MODEL}")
    print(f"  Outer model: {OUTER_MODEL}")
    print(f"  List size: {LIST_SIZE:,}")
    print("=" * 70)

    client = anthropic.Anthropic()

    meta_log = {
        "config": {
            "outer_rounds": OUTER_ROUNDS,
            "inner_iterations": INNER_ITERATIONS,
            "inner_model": INNER_MODEL,
            "outer_model": OUTER_MODEL,
            "list_size": LIST_SIZE,
            "started_at": datetime.now().isoformat(),
        },
        "rounds": [],
    }

    # Current prompts (will be evolved by outer loop)
    current_system = DEFAULT_SYSTEM_PROMPT
    current_initial = DEFAULT_INITIAL_PROMPT
    current_improve = DEFAULT_IMPROVE_TEMPLATE

    global_best_ms = float("inf")
    global_best_code = ""
    global_best_round = 0

    for round_num in range(1, OUTER_ROUNDS + 1):
        experiment_name = f"meta-round-{round_num:02d}"

        print(f"\n{'━' * 70}")
        print(f"  OUTER ROUND {round_num}/{OUTER_ROUNDS}: {experiment_name}")
        print(f"{'━' * 70}")

        # ── Run inner loop ──
        print(f"\n  Running inner loop ({INNER_ITERATIONS} iterations)...")
        exp_log = run_inner_loop(
            client, current_system, current_initial, current_improve, experiment_name
        )

        best_this_round = exp_log["config"].get("best_ms")
        success_rate = exp_log["config"].get("success_rate", "0/0")
        print(f"\n  Round {round_num} result: {best_this_round} ms ({success_rate} succeeded)")

        if best_this_round and best_this_round < global_best_ms:
            global_best_ms = best_this_round
            global_best_round = round_num
            # Find the best code
            best_iter = min(
                [it for it in exp_log["iterations"] if it["status"] == "success"],
                key=lambda x: x["stats"]["median_ms"]
            )
            global_best_code = best_iter["code"]
            print(f"  ★ NEW GLOBAL BEST: {global_best_ms} ms")

        # ── Record round ──
        round_record = {
            "round": round_num,
            "experiment": experiment_name,
            "best_ms": best_this_round,
            "success_rate": success_rate,
            "is_global_best": best_this_round == global_best_ms if best_this_round else False,
            "system_prompt": current_system,
            "initial_prompt": current_initial,
            "improve_template": current_improve,
        }

        # ── Outer loop: analyze and evolve prompts ──
        if round_num < OUTER_ROUNDS:
            print(f"\n  Outer loop: analyzing results and evolving prompts...")

            # Build history of all experiments
            all_summaries = []
            for prev_round in meta_log["rounds"]:
                # Load the experiment log
                prev_file = LOG_DIR / f"{prev_round['experiment']}.json"
                if prev_file.exists():
                    prev_log = json.loads(prev_file.read_text())
                    all_summaries.append(build_experiment_summary(prev_log))
            all_summaries.append(build_experiment_summary(exp_log))

            experiment_history = "\n---\n".join(all_summaries)

            outer_prompt = OUTER_ANALYSIS_PROMPT.format(
                experiment_history=experiment_history,
                list_size=LIST_SIZE,
                inner_iters=INNER_ITERATIONS,
            )

            try:
                outer_response = client.messages.create(
                    model=OUTER_MODEL, max_tokens=4096,
                    messages=[{"role": "user", "content": outer_prompt}],
                )
                outer_text = outer_response.content[0].text
                result = extract_json(outer_text)

                diagnosis = result.get("diagnosis", "No diagnosis")
                changes = result.get("changes", [])
                new_system = result.get("system_prompt", current_system)
                new_initial = result.get("initial_prompt", current_initial)
                new_improve = result.get("improve_template", current_improve)

                # Replace {list_size} placeholder in initial prompt
                new_initial = new_initial.replace("{list_size}", f"{LIST_SIZE:,}")

                print(f"\n  Diagnosis: {diagnosis}")
                print(f"  Changes ({len(changes)}):")
                for c in changes:
                    print(f"    • {c}")

                round_record["outer_diagnosis"] = diagnosis
                round_record["outer_changes"] = changes

                current_system = new_system
                current_initial = new_initial
                current_improve = new_improve

            except Exception as e:
                print(f"\n  ⚠ Outer loop failed: {e}")
                print(f"    Keeping current prompts for next round.")
                round_record["outer_error"] = str(e)

        meta_log["rounds"].append(round_record)

        # Save meta log after each round
        meta_log["config"]["global_best_ms"] = round(global_best_ms, 4) if global_best_ms < float("inf") else None
        meta_log["config"]["global_best_round"] = global_best_round
        meta_log["config"]["finished_at"] = datetime.now().isoformat()
        META_LOG_FILE.write_text(json.dumps(meta_log, indent=2))

    # ── Final Summary ──────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("  META-OPTIMIZER FINAL RESULTS")
    print(f"{'=' * 70}")
    print(f"\n  Global best: {global_best_ms:.3f} ms (round {global_best_round})")
    print(f"\n  Round-by-round:")
    for r in meta_log["rounds"]:
        marker = " ★" if r.get("is_global_best") else "  "
        print(f"   {marker}Round {r['round']}: {r['best_ms']} ms ({r['success_rate']})")
        if r.get("outer_diagnosis"):
            print(f"      Diagnosis: {r['outer_diagnosis'][:80]}...")

    print(f"\n  Best code:")
    for line in global_best_code.splitlines():
        print(f"    {line}")

    print(f"\n  Meta log: {META_LOG_FILE}")
    print("=" * 70)


if __name__ == "__main__":
    main()