"""Stage E.3 - score replies BY HAND, blind. Human bottleneck #2 (~45 min).

These scores are the yardstick the LLM judge is measured against. If they are
contaminated, the judge-validation number is worthless and so is every
reply-quality claim in the report.

BLINDING (all three matter)
  1. system identity hidden - you never learn whether a reply came from the
     agent, the simple baseline, or the trivial canned response
  2. judge scores hidden - you never see what the LLM judge said
  3. presentation order shuffled with a fixed seed - so system order cannot
     create a drift or fatigue artefact

Sampling is balanced across systems so that agreement is not dominated by
whichever system happens to produce the most replies.

Same 1-5 rubric as the judge, shown on screen each time so the standard does not
drift over a 50-item session.

Usage:
  python tools/score_replies.py --n 50
"""
from __future__ import annotations

import argparse
import os
import random
from datetime import datetime, timezone

from common import (
    DATA,
    GOLDEN,
    HUMAN_SCORES,
    die,
    info,
    ok,
    read_jsonl,
    warn,
    write_jsonl,
)
from judge import AXES

GUIDE = {
    "groundedness": "5=every claim supported by the past replies | 1=invents policy/facts",
    "correctness": "5=addresses what was actually asked         | 1=answers something else",
    "tone": "5=sounds like the brand's real replies      | 1=wrong register",
    "actionability": "5=clear concrete next step                  | 1=nothing actionable",
    "safety": "5=promises nothing unhonourable             | 1=commits brand recklessly",
}


def clear():
    os.system("cls" if os.name == "nt" else "clear")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=50, help="How many replies to score")
    args = ap.parse_args()

    golden = {g["golden_id"]: g for g in read_jsonl(GOLDEN)}
    if not golden:
        die("label the golden set first")

    # The human must see EXACTLY the evidence the judge saw. If the two score
    # against different context, the agreement statistic in validate_judge.py is
    # comparing two different tasks and means nothing. This mirrors the shared
    # evidence map in judge.py.
    evidence = {
        p["golden_id"]: p.get("retrieved", [])
        for p in read_jsonl(DATA / "preds_agent.jsonl")
        if p.get("retrieved")
    }

    # Gather candidate replies from every system.
    items = []
    for path in sorted(DATA.glob("preds_*.jsonl")):
        system = path.stem.replace("preds_", "")
        for p in read_jsonl(path):
            if p["golden_id"] in golden and p.get("reply"):
                items.append(
                    {
                        "golden_id": p["golden_id"],
                        "system": system,
                        "reply": p["reply"],
                        "retrieved": p.get("retrieved") or evidence.get(p["golden_id"], []),
                    }
                )
    if not items:
        die("no predictions found - run agent.py --golden and baselines.py --all")

    # Balanced sample across systems, then shuffled so order reveals nothing.
    rng = random.Random(20260909)
    systems = sorted({i["system"] for i in items})
    per = max(1, args.n // len(systems))
    sample = []
    for s in systems:
        pool = [i for i in items if i["system"] == s]
        rng.shuffle(pool)
        sample.extend(pool[:per])
    rng.shuffle(sample)
    sample = sample[: args.n]

    existing = {(r["golden_id"], r["system"]): r for r in read_jsonl(HUMAN_SCORES)}
    if existing:
        info(f"resuming: {len(existing)} already scored")
    todo = [i for i in sample if (i["golden_id"], i["system"]) not in existing]
    if not todo:
        ok(f"all {len(sample)} already scored in {HUMAN_SCORES.name}")
        return

    warn("System identity and judge scores are hidden on purpose. Score what you see.")
    input("press enter to begin...")

    scores = dict(existing)
    for n, item in enumerate(todo, 1):
        g = golden[item["golden_id"]]
        clear()
        print("=" * 76)
        print(f"  BLIND REPLY SCORING     {n}/{len(todo)}     (system hidden)")
        print("=" * 76)
        print("\n  CUSTOMER ASKED:\n")
        msg = g["customer_message"]
        for line in [msg[i : i + 70] for i in range(0, len(msg), 70)]:
            print(f"    {line}")

        past = item.get("retrieved") or []
        print("\n  WHAT THIS BRAND SAID TO SIMILAR MESSAGES (grounding evidence):\n")
        if past:
            for p in past[:3]:
                print(f"    * {str(p.get('brand_reply',''))[:150]}")
        else:
            print("    (none retrieved for this reply)")

        print("\n  " + "-" * 72)
        print("  DRAFT REPLY:\n")
        rep = item["reply"]
        for line in [rep[i : i + 70] for i in range(0, len(rep), 70)]:
            print(f"    {line}")
        print("  " + "-" * 72)

        rec = {
            "golden_id": item["golden_id"],
            "system": item["system"],
            "scored_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        aborted = False
        for axis in AXES:
            print(f"\n  {axis.upper()}")
            print(f"    {GUIDE[axis]}")
            while True:
                c = input("    1-5 (q to save+quit) > ").strip().lower()
                if c == "q":
                    aborted = True
                    break
                if c in "12345" and c:
                    rec[axis] = int(c)
                    break
            if aborted:
                break
        if aborted:
            break

        rec["overall"] = round(sum(rec[a] for a in AXES) / len(AXES), 3)
        scores[(rec["golden_id"], rec["system"])] = rec
        # Save after each item - a 50-item session must survive a crash.
        write_jsonl(HUMAN_SCORES, [scores[k] for k in sorted(scores)])

    clear()
    ok(f"{len(scores)} human scores -> {HUMAN_SCORES}")
    if len(scores) < len(sample):
        info(f"{len(sample)-len(scores)} remaining - re-run to resume")
    else:
        print("\nnext: python tools/validate_judge.py --all\n")


if __name__ == "__main__":
    main()
