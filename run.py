"""Task runner. Cross-platform, no `make` required.

  python run.py reproduce     <- THE GRADER PATH. Scores committed model outputs.
                                 No GPU, no Ollama, no Kaggle account. ~2 min.
  python run.py all           Full pipeline from the raw Kaggle dump (~60 min).

Individual stages:
  fetch        download the dataset          (needs Kaggle creds or --zip)
  profile      rank brands, choose one
  threads      reconstruct conversations     --brand X
  induce       cluster -> candidate intents
  sample       stratified golden-set draw
  label        hand-label the golden set     (manual)
  generate     run agent + baselines, populate the LLM cache
  judge        LLM-as-judge over all replies
  score        hand-score replies, blind     (manual)
  eval         metrics + bootstrap CIs
  validate     judge-vs-human agreement, bias probes
  freeze       export the LLM cache for committing
  smoke        end-to-end test on a synthetic fixture (no download needed)
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable


def run(*args: str, env: dict | None = None) -> int:
    cmd = [PY, *args]
    print(f"\n\033[36m$ {' '.join(str(c) for c in cmd)}\033[0m", flush=True)
    e = {**os.environ, **(env or {})}
    return subprocess.call(cmd, cwd=ROOT, env=e)


def tool(name: str, *args: str, env: dict | None = None) -> int:
    return run(str(ROOT / "tools" / name), *args, env=env)


def need(rc: int) -> None:
    if rc != 0:
        sys.exit(rc)


def cmd_reproduce(argv: list[str]) -> None:
    """Replay committed outputs. This must never call a model."""
    env = {"ANTHILL_CACHE_ONLY": "1"}
    print("=" * 72)
    print("  REPRODUCING HEADLINE RESULTS FROM COMMITTED OUTPUTS")
    print("  no GPU * no Ollama * no API key * no Kaggle account")
    print("=" * 72)
    need(tool("load_cache.py"))
    need(tool("run_eval.py", "--detail", env=env))
    tool("validate_judge.py", "--all", env=env)


def cmd_all(argv: list[str]) -> None:
    brand = argv[0] if argv else None
    need(tool("fetch_dataset.py"))
    need(tool("profile_brands.py"))
    if not brand:
        print("\nPick a brand from the table above, then run:")
        print("  python run.py all <BrandHandle>")
        return
    need(tool("build_threads.py", "--brand", brand))
    need(tool("induce_intents.py", "--k", "10"))
    print("\nMANUAL STEP: name the clusters in data/taxonomy_draft.json,")
    print("save as data/taxonomy.json, then: python run.py sample")


COMMANDS = {
    "reproduce": cmd_reproduce,
    "all": cmd_all,
    "fetch": lambda a: sys.exit(tool("fetch_dataset.py", *a)),
    "profile": lambda a: sys.exit(tool("profile_brands.py", *a)),
    "threads": lambda a: sys.exit(tool("build_threads.py", *a)),
    "embed": lambda a: sys.exit(tool("embed.py", *a)),
    "induce": lambda a: sys.exit(tool("induce_intents.py", *a)),
    "sample": lambda a: sys.exit(tool("sample_golden.py", *a)),
    "label": lambda a: sys.exit(tool("label_cli.py", *a)),
    "judge": lambda a: sys.exit(tool("judge.py", "--all", *a)),
    "score": lambda a: sys.exit(tool("score_replies.py", *a)),
    "eval": lambda a: sys.exit(tool("run_eval.py", "--detail", *a)),
    "validate": lambda a: sys.exit(tool("validate_judge.py", "--all", *a)),
    "freeze": lambda a: sys.exit(tool("freeze_cache.py", *a)),
    "smoke": lambda a: sys.exit(tool("smoke_test.py", *a)),
}


def cmd_generate(argv: list[str]) -> None:
    need(tool("agent.py", "--golden", *argv))
    need(tool("baselines.py", "--all"))


COMMANDS["generate"] = cmd_generate


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help", "help"):
        print(__doc__)
        return
    name, argv = sys.argv[1], sys.argv[2:]
    fn = COMMANDS.get(name)
    if not fn:
        print(f"unknown command: {name}\n")
        print(__doc__)
        sys.exit(2)
    fn(argv)


if __name__ == "__main__":
    main()
