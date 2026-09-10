"""Restore the committed LLM cache bundle into the local cache directory.

Called automatically by `python run.py reproduce`. Idempotent.
"""
from __future__ import annotations

from common import DATA, info, ok, warn
from llm import load_cache_bundle

BUNDLE = DATA / "llm_cache_bundle.jsonl"

if __name__ == "__main__":
    if not BUNDLE.exists():
        warn(f"{BUNDLE} not found - nothing to restore.")
        warn("If you are reproducing results, this file should have been committed.")
        raise SystemExit(0)
    n = load_cache_bundle(BUNDLE)
    ok(f"restored {n} cached model calls from {BUNDLE.name}")
    info("evaluation will now replay these instead of calling a model")
