"""Local LLM access via Ollama, with a content-addressed disk cache.

The cache is not an optimisation - it is what makes the deliverable reproducible.
Every call is keyed by sha256(model + prompt + options), so:

  - `make generate` populates the cache by actually running the models (slow:
    ~600 calls on a 3B model on a 4GB laptop GPU is 45-60 min)
  - the cache file is COMMITTED to the repo
  - `make reproduce` runs with ANTHILL_CACHE_ONLY=1 and replays it: identical
    numbers, no GPU, no Ollama, no API key, ~2 minutes

Determinism: temperature 0 and a fixed seed. Note that even so, llama.cpp is not
bit-identical across GPU/CPU backends - which is precisely why we ship the cache
rather than asking a grader to regenerate and hope they match.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any

from common import GEN_MODEL, OLLAMA_HOST, SEED, TMP, info, warn

CACHE_DIR = TMP / "llm_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# When set, a cache miss is a hard error instead of a model call. The grader path
# uses this so a missing cache entry fails loudly rather than silently diverging.
CACHE_ONLY = os.environ.get("ANTHILL_CACHE_ONLY", "0") == "1"

_STATS = {"hits": 0, "misses": 0, "calls": 0, "seconds": 0.0}


class CacheMiss(RuntimeError):
    pass


def _key(model: str, prompt: str, system: str, options: dict) -> str:
    h = hashlib.sha256()
    for part in (model, system, prompt, json.dumps(options, sort_keys=True)):
        h.update(part.encode("utf-8", "ignore"))
        h.update(b"\x00")
    return h.hexdigest()


def _cache_path(key: str) -> Path:
    return CACHE_DIR / f"{key}.json"


def load_cache_bundle(path: Path) -> int:
    """Load a committed JSONL bundle into the on-disk cache."""
    if not path.exists():
        return 0
    n = 0
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            _cache_path(rec["key"]).write_text(
                json.dumps(rec), encoding="utf-8"
            )
            n += 1
    return n


def export_cache_bundle(path: Path) -> int:
    """Dump the on-disk cache to a single committable JSONL file."""
    rows = []
    for p in sorted(CACHE_DIR.glob("*.json")):
        try:
            rows.append(json.loads(p.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            continue
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)


def generate(
    prompt: str,
    system: str = "",
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 400,
    seed: int = SEED,
) -> str:
    """Single-turn completion. Cached, deterministic settings by default."""
    model = model or GEN_MODEL
    options = {
        "temperature": temperature,
        "seed": seed,
        "num_predict": max_tokens,
        "top_p": 1.0,
    }
    key = _key(model, prompt, system, options)
    path = _cache_path(key)

    if path.exists():
        _STATS["hits"] += 1
        return json.loads(path.read_text(encoding="utf-8"))["response"]

    _STATS["misses"] += 1
    if CACHE_ONLY:
        raise CacheMiss(
            f"cache miss for {model} (key {key[:12]}) while ANTHILL_CACHE_ONLY=1.\n"
            "The committed cache does not cover this prompt. Either you changed a\n"
            "prompt template, or the bundle was not loaded. Run `make generate` to\n"
            "rebuild it with a live model."
        )

    import requests

    t0 = time.time()
    resp = requests.post(
        f"{OLLAMA_HOST}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": options,
        },
        timeout=300,
    )
    resp.raise_for_status()
    text = resp.json().get("response", "").strip()
    elapsed = time.time() - t0
    _STATS["calls"] += 1
    _STATS["seconds"] += elapsed

    path.write_text(
        json.dumps(
            {
                "key": key,
                "model": model,
                "system": system,
                "prompt": prompt,
                "options": options,
                "response": text,
                "latency_s": round(elapsed, 3),
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return text


_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def generate_json(prompt: str, system: str = "", **kw) -> dict[str, Any]:
    """Completion parsed as JSON, tolerant of the wrapping small models add.

    3B models routinely emit ```json fences, a preamble, or a trailing comma. We
    repair what is cheaply repairable and return {} otherwise; callers treat {}
    as a parse failure and fall back. Parse-failure RATE is itself reported in
    the results - it is a real failure mode of small local models, not something
    to paper over.
    """
    raw = generate(prompt, system=system, **kw)
    for candidate in (raw, _JSON_BLOCK.search(raw).group(0) if _JSON_BLOCK.search(raw) else ""):
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            repaired = re.sub(r",\s*([}\]])", r"\1", candidate)
            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                continue
    return {}


def stats() -> dict:
    return dict(_STATS)


def print_stats() -> None:
    s = _STATS
    total = s["hits"] + s["misses"]
    if not total:
        return
    info(
        f"llm: {total} calls, {s['hits']} cached ({s['hits']/total:.0%}), "
        f"{s['calls']} live, {s['seconds']:.1f}s spent"
    )


def health() -> bool:
    """Is Ollama reachable? Used to give a clear error instead of a stack trace."""
    try:
        import requests

        r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


if __name__ == "__main__":
    print(f"ollama at {OLLAMA_HOST}: {'up' if health() else 'DOWN'}")
    print(f"cache entries: {len(list(CACHE_DIR.glob('*.json')))}")
    if health():
        print("\ntest generation:")
        print(generate("Reply with exactly the word: ready", max_tokens=10))
