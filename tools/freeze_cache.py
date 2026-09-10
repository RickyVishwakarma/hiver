"""Export the local LLM cache into one committable file.

Run this after `python run.py generate` + `judge`. The resulting bundle is what
makes `python run.py reproduce` work on a grader's machine with no model access.
"""
from __future__ import annotations

from common import DATA, info, ok
from llm import CACHE_DIR, export_cache_bundle

BUNDLE = DATA / "llm_cache_bundle.jsonl"

if __name__ == "__main__":
    n = export_cache_bundle(BUNDLE)
    size = BUNDLE.stat().st_size / 1e6 if BUNDLE.exists() else 0
    ok(f"froze {n} cached model calls -> {BUNDLE} ({size:.1f}MB)")
    info("commit this file: it is the reproduction path")
    if size > 45:
        info("note: >45MB - consider gzipping if GitHub complains")
