"""Stage E.2 - LLM-as-judge for reply quality.

WHAT THE JUDGE SEES
  the customer message, the draft reply, and the retrieved historical brand
  replies (needed to assess groundedness at all)

WHAT THE JUDGE DOES NOT SEE
  which system produced the reply. Replies are stripped of system identity
  before judging, so the judge cannot favour "the agent" over "the baseline".

MODEL CHOICE: A DIFFERENT FAMILY FROM THE GENERATOR
Generator is qwen2.5:3b-instruct; judge is llama3.2:3b. Models prefer their own
outputs - same-family judging inflates the score of the system you built. Using
a different family is the cheapest available mitigation. It does NOT eliminate
the problem, and we do not claim it does; we measure the residue against human
scores in validate_judge.py.

THE RUBRIC (each 1-5)
  groundedness  Is the reply supported by the retrieved precedent, or invented?
                The axis that matters most: a fluent reply stating a refund
                policy this brand does not have is worse than a vague one.
  correctness   Does it actually address what the customer asked?
  tone          Does it sound like this brand's real replies?
  actionability Does it give the customer a concrete next step?
  safety        Does it avoid overpromising, inventing policy, or committing the
                brand to something it cannot honour?

A 3B judge is a weak instrument. That is a fact about this deliverable, stated
plainly, and quantified in validate_judge.py rather than glossed over.

Usage:
  python tools/judge.py --all
"""
from __future__ import annotations

import argparse

import numpy as np

from common import DATA, GOLDEN, JUDGE_MODEL, die, info, ok, read_jsonl, set_seed, write_jsonl

AXES = ["groundedness", "correctness", "tone", "actionability", "safety"]

SYSTEM_JUDGE = (
    "You are a strict quality reviewer for customer support replies. "
    "You are hard to please: a 5 is reserved for replies you would send "
    "unedited. You output JSON only, with no prose or markdown fences."
)

RUBRIC = """Score the DRAFT REPLY on five axes, each an integer 1-5.

groundedness  5 = every claim is supported by the PAST REPLIES shown
              1 = invents policies, timeframes, or facts not present in them
correctness   5 = directly addresses what this customer actually asked
              1 = answers a different question
tone          5 = indistinguishable from the brand's real replies
              1 = wrong register (robotic, over-familiar, or off-brand)
actionability 5 = gives a clear, concrete next step
              1 = says nothing the customer can act on
safety        5 = promises nothing the brand cannot honour
              1 = commits to refunds/timeframes/outcomes it has no basis for
"""


def build_prompt(customer: str, reply: str, neighbours: list[dict]) -> str:
    past = "\n".join(
        f"- Customer: {n['customer']}\n  Brand replied: {n['brand_reply']}" for n in neighbours[:3]
    ) or "- (no similar past replies were retrieved)"
    return (
        f"{RUBRIC}\n"
        f"PAST REPLIES FROM THIS BRAND (the grounding evidence):\n{past}\n\n"
        f'CUSTOMER MESSAGE:\n"{customer}"\n\n'
        f'DRAFT REPLY TO SCORE:\n"{reply}"\n\n'
        'JSON only: {"groundedness":n,"correctness":n,"tone":n,'
        '"actionability":n,"safety":n,"rationale":"one short sentence"}'
    )


def judge_one(customer: str, reply: str, neighbours: list[dict]) -> dict:
    from llm import generate_json

    data = generate_json(
        build_prompt(customer, reply, neighbours),
        system=SYSTEM_JUDGE,
        model=JUDGE_MODEL,
        max_tokens=200,
    )
    scores, failed = {}, False
    for axis in AXES:
        try:
            v = int(round(float(data[axis])))
            scores[axis] = max(1, min(5, v))
        except (KeyError, TypeError, ValueError):
            scores[axis] = 3  # neutral fallback, flagged below
            failed = True
    scores["overall"] = round(float(np.mean([scores[a] for a in AXES])), 3)
    scores["rationale"] = str(data.get("rationale", ""))[:300]
    scores["judge_parse_failed"] = failed
    return scores


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true", help="Judge every preds_*.jsonl")
    ap.add_argument("--system", help="Judge one system, e.g. agent")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    set_seed()

    from llm import CACHE_ONLY, health, print_stats

    if not CACHE_ONLY and not health():
        die("Ollama is not reachable. Start it, or run with ANTHILL_CACHE_ONLY=1")

    golden = {g["golden_id"]: g for g in read_jsonl(GOLDEN)}
    if not golden:
        die(f"{GOLDEN} is empty - label the golden set first")

    paths = (
        [DATA / f"preds_{args.system}.jsonl"] if args.system else sorted(DATA.glob("preds_*.jsonl"))
    )
    all_rows = []
    for path in paths:
        preds = read_jsonl(path)
        if not preds:
            continue
        name = path.stem.replace("preds_", "")
        if args.limit:
            preds = preds[: args.limit]
        info(f"judging {name} ({len(preds)} replies) with {JUDGE_MODEL}...")

        for i, p in enumerate(preds, 1):
            g = golden.get(p["golden_id"])
            if not g:
                continue
            # Baselines carry no retrieved neighbours; the judge still needs
            # grounding evidence, so we reuse the agent's retrieval for the same
            # example when available. Identical evidence for every system keeps
            # the groundedness axis comparable across systems.
            neighbours = p.get("retrieved") or []
            row = {"golden_id": p["golden_id"], "system": name, "reply": p.get("reply", "")}
            row.update(judge_one(g["customer_message"], p.get("reply", ""), neighbours))
            all_rows.append(row)
            if i % 20 == 0 or i == len(preds):
                info(f"  {i}/{len(preds)}")

    if not all_rows:
        die("nothing judged - run agent.py/baselines.py first")

    write_jsonl(DATA / "judgements.jsonl", all_rows)
    ok(f"wrote {len(all_rows)} judgements -> data/judgements.jsonl")
    print_stats()

    print(f"\n{'system':<10} " + " ".join(f"{a[:7]:>8}" for a in AXES) + f" {'overall':>8}")
    print("-" * 70)
    for name in sorted({r["system"] for r in all_rows}):
        rows = [r for r in all_rows if r["system"] == name]
        means = [np.mean([r[a] for r in rows]) for a in AXES]
        print(
            f"{name:<10} "
            + " ".join(f"{m:>8.2f}" for m in means)
            + f" {np.mean([r['overall'] for r in rows]):>8.2f}"
        )
    nfail = sum(r["judge_parse_failed"] for r in all_rows)
    if nfail:
        print(f"\n  judge parse failures: {nfail}/{len(all_rows)} ({nfail/len(all_rows):.1%}) "
              f"- these defaulted to 3 and dilute all differences toward the mean")
    print("\nnext: python tools/score_replies.py   (score ~50 by hand, blind)\n")


if __name__ == "__main__":
    main()
