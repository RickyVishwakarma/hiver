"""Stage E.6 - the same pairs, by hand, blind. ~10 minutes.

This is the yardstick judge_pairwise.py is measured against. It is the pairwise
counterpart of score_replies.py, and it exists because the absolute-score
instrument failed its validation (kappa 0.003) and a failed instrument cannot be
repaired by inspecting it - only by measuring a different one.

BLINDING
  1. system identity hidden - "REPLY A" and "REPLY B" are assigned by a seeded
     coin in pairs.py, so left/right carries no information about which system
     wrote which
  2. the judge's verdict is never shown
  3. pair order is the shuffled order from pairs.py, identical for judge and
     human, so neither sees an easier or differently-ordered task

Why this takes ~10 minutes where the 5-axis session took ~45: one forced choice
per item instead of five 1-5 judgements, and no rubric to hold in memory.

Progress is saved after every pair; safe to quit and resume.

Usage:
  python tools/score_pairs.py
"""
from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone

from common import DATA, die, info, ok, read_jsonl, warn, write_jsonl
from pairs import build_pairs

OUT = DATA / "human_pairwise.jsonl"


def clear():
    os.system("cls" if os.name == "nt" else "clear")


def wrap(text: str, width: int = 70, indent: str = "    ") -> None:
    words, line = (text or "").split(), ""
    for w in words:
        if len(line) + len(w) + 1 > width:
            print(indent + line)
            line = w
        else:
            line = f"{line} {w}".strip()
    if line:
        print(indent + line)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    pairs = build_pairs()
    if not pairs:
        die("no pairs - run agent.py --golden and baselines.py --all first")
    if args.limit:
        pairs = pairs[: args.limit]

    existing = {r["pair_id"]: r for r in read_jsonl(OUT)}
    if existing:
        info(f"resuming: {len(existing)} of {len(pairs)} already judged")
    todo = [p for p in pairs if p["pair_id"] not in existing]
    if not todo:
        ok(f"all {len(pairs)} pairs already judged in {OUT.name}")
        print("\nnext: python tools/validate_judge.py --pairwise\n")
        return

    warn("Which system wrote which reply is hidden, and so is the judge's answer.")
    warn("Pick the one you would rather send to the customer. Ties are allowed but")
    warn("prefer a decision - a set of all ties measures nothing.")
    input("press enter to begin...")

    scores = dict(existing)
    for n, p in enumerate(todo, 1):
        clear()
        print("=" * 76)
        print(f"  BLIND PAIRWISE COMPARISON     {n}/{len(todo)}     (systems hidden)")
        print("=" * 76)
        print("\n  CUSTOMER ASKED:\n")
        wrap(p["customer_message"])

        past = p.get("retrieved") or []
        print("\n  WHAT THIS BRAND SAID TO SIMILAR MESSAGES (grounding evidence):\n")
        if past:
            for q in past[:3]:
                print(f"    * {str(q.get('brand_reply',''))[:140]}")
        else:
            print("    (none retrieved)")

        print("\n  " + "-" * 72)
        print("  REPLY A:\n")
        wrap(p["reply_a"])
        print("\n  " + "-" * 72)
        print("  REPLY B:\n")
        wrap(p["reply_b"])
        print("  " + "-" * 72)
        print("\n    1 = A is better    2 = B is better    t = genuine tie")
        print("    q = save and quit")

        choice = None
        while choice is None:
            c = input("  > ").strip().lower()
            if c == "q":
                break
            if c == "1":
                choice = "A"
            elif c == "2":
                choice = "B"
            elif c == "t":
                choice = "tie"
            else:
                print("  (1, 2, t, or q)")
        if choice is None:
            break

        # Echo the recorded choice. The golden set had to be thrown away once
        # because a labelling CLI advanced silently on a mis-key; every input
        # surface here confirms what it stored.
        winner_system = (
            p["system_a"] if choice == "A" else p["system_b"] if choice == "B" else "tie"
        )
        print("")
        print("  >>> RECORDED: " + ("TIE" if choice == "tie" else "REPLY " + choice) + " <<<")
        scores[p["pair_id"]] = {
            "pair_id": p["pair_id"],
            "golden_id": p["golden_id"],
            "comparison": p["comparison"],
            "verdict": choice,
            "winner_system": winner_system,
            "scored_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        write_jsonl(OUT, [scores[k] for k in sorted(scores)])

    clear()
    ok(f"{len(scores)} human pairwise judgements -> {OUT}")
    if len(scores) < len(pairs):
        info(f"{len(pairs)-len(scores)} remaining - re-run to resume")
    else:
        print("\nnext: python tools/validate_judge.py --pairwise\n")


if __name__ == "__main__":
    main()
