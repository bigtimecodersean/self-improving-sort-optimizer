"""
Run this once to copy your v1 optimization_log.json into the experiments/ folder.
Usage: python migrate_v1_log.py
"""
import json
from pathlib import Path

LOG_DIR = Path("experiments")
LOG_DIR.mkdir(exist_ok=True)

v1_file = Path("optimization_log.json")
if not v1_file.exists():
    print("No optimization_log.json found — skipping migration.")
    exit()

log = json.loads(v1_file.read_text())

# Add experiment metadata
log["config"]["experiment"] = "v1-baseline"
log["config"]["prompt_version"] = "v1-naive"
log["config"]["changes_from_v1"] = []
log.setdefault("prompts_used", [])

# Save to experiments/
dest = LOG_DIR / "v1-baseline.json"
dest.write_text(json.dumps(log, indent=2))

# Update manifest
manifest_file = LOG_DIR / "manifest.json"
manifest = []
if manifest_file.exists():
    try:
        manifest = json.loads(manifest_file.read_text())
    except:
        pass

manifest = [e for e in manifest if e.get("experiment") != "v1-baseline"]
manifest.append({
    "experiment": "v1-baseline",
    "file": "v1-baseline.json",
    "model": log["config"]["model"],
    "list_size": log["config"]["list_size"],
    "iterations": log["config"]["iterations"],
    "started_at": log["config"]["started_at"],
    "best_ms": log["config"].get("best_ms"),
    "best_iter": log["config"].get("best_iter"),
    "prompt_version": "v1-naive",
})
manifest_file.write_text(json.dumps(manifest, indent=2))

print(f"✓ Migrated v1 log to {dest}")
print(f"✓ Updated manifest at {manifest_file}")