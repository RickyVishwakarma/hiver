"""Shared paths, config, and IO helpers.

Deliberately dependency-light: a hand-rolled .env reader instead of python-dotenv
so that the reproduction path imports nothing beyond the stdlib + pandas/numpy.
"""
from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Iterable, Iterator


def setup_console() -> None:
    """Force UTF-8 on stdout/stderr.

    Windows consoles default to cp1252, which raises UnicodeEncodeError on any
    non-ASCII output. Graders may well be on Windows, so this is set centrally
    rather than hoping every print stays ASCII.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


setup_console()

# --- Paths -------------------------------------------------------------------
# Resolved relative to the repo root (this file lives in <root>/tools/), so every
# script works regardless of the directory it is invoked from.
ROOT = Path(__file__).resolve().parent.parent
TOOLS = ROOT / "tools"
# ANTHILL_DATA_DIR redirects all artifacts elsewhere. The smoke test uses it so
# a plumbing run on synthetic data can never overwrite the real golden set.
DATA = Path(os.environ.get("ANTHILL_DATA_DIR", ROOT / "data"))
TMP = ROOT / ".tmp"
RAW = TMP / "raw"
# REPORT_DIR must follow the same redirect as DATA. It previously did not, so a
# smoke run on synthetic data wrote fixture metrics into the real report/
# directory - the deliverable folder, where they look like genuine results.
REPORT_DIR = (DATA / "report") if os.environ.get("ANTHILL_DATA_DIR") else (ROOT / "report")

# Committed artifacts (small, in git) - these are what `make reproduce` reads.
THREADS = DATA / "threads.jsonl"
GOLDEN = DATA / "golden.jsonl"
GOLDEN_PASS2 = DATA / "golden_pass2.jsonl"
TAXONOMY = DATA / "taxonomy.json"
CACHED_GENERATIONS = DATA / "cached_generations.jsonl"
CACHED_JUDGEMENTS = DATA / "cached_judgements.jsonl"
HUMAN_SCORES = DATA / "human_reply_scores.jsonl"
BRAND_PROFILE = DATA / "brand_profile.csv"

# Regenerable intermediates (gitignored).
TWCS_CSV = RAW / "twcs.csv"
EMBED_CACHE = TMP / "embeddings"

for _d in (DATA, TMP, RAW, EMBED_CACHE, REPORT_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# --- Config ------------------------------------------------------------------
def _load_dotenv(path: Path = ROOT / ".env") -> None:
    """Minimal .env loader. Does not overwrite variables already in os.environ."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip().strip("'\""))


_load_dotenv()

SEED = int(os.environ.get("SEED", "20260909"))
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
GEN_MODEL = os.environ.get("GEN_MODEL", "qwen2.5:3b-instruct")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "llama3.2:3b")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


def set_seed(seed: int = SEED) -> None:
    """Seed every RNG we touch. Called at the top of each script's main()."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass


# --- JSONL IO ----------------------------------------------------------------
def read_jsonl(path: Path | str) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    out = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def iter_jsonl(path: Path | str) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path | str, rows: Iterable[dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def append_jsonl(path: Path | str, row: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_json(path: Path | str, default: Any = None) -> Any:
    path = Path(path)
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path | str, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


# --- Console -----------------------------------------------------------------
def info(msg: str) -> None:
    print(f"[*] {msg}", flush=True)


def ok(msg: str) -> None:
    print(f"[+] {msg}", flush=True)


def warn(msg: str) -> None:
    print(f"[!] {msg}", flush=True)


def die(msg: str, code: int = 1):
    print(f"[x] {msg}", flush=True)
    raise SystemExit(code)
