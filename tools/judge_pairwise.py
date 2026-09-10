"""Stage E.5 - pairwise LLM judge, built because the absolute-score judge failed.

THE PROBLEM THIS ADDRESSES
judge.py scores each reply 1-5 on five axes. Measured against blind human
scores it reached quadratic-weighted kappa 0.003 - indistinguishable from
guessing - so the report makes no reply-quality claim at all. A 3B model has no
stable notion of what a "4" is; its scores piled up at 3-4 for every system and
carried no ranking information.

Pairwise forced choice removes the anchor problem: the model never has to say
how good a reply is, only which of two concrete replies is better. That is a
strictly easier question, and it is the question the report actually needs
answered.

POSITION BIAS IS MEASURED, NOT ASSUMED
Every pair is judged twice - once as presented, once with the two replies
swapped. An unbiased judge returns the same winner both times. The flip rate is
reported, and pairs where the two orders disagree are recorded as `undecided`
rather than being resolved by coin-flip. This is the position-bias probe the
brief asks for, and it comes free with the design.

Same judge model as judge.py (llama3.2:3b), still a different family from the
generator (qwen2.5:3b-instruct), for the same self-preference reason.

Usage:
  python tools/judge_pairwise.py
"""
from __future__ import annotations

import argparse

from common import (
    DATA,
    JUDGE_MODEL,
    die,
    info,
    ok,
    set_seed,
    warn,
    write_jsonl,
)
from pairs import build_pairs

SYSTEM_JUDGE = (
    "You are a strict reviewer of customer support replies. You compare two "
    "candidate replies and pick the better one. You output JSON only, with no "
    "prose and no markdown fences."
)

RUBRIC = """Two support agents drafted a reply to the same customer. Decide which
reply you would rather send, weighing in this order:

1. groundedness   - claims supported by the PAST REPLIES shown, not invented
2. correctness    - addresses what this customer actually asked
3. actionability  - gives a concrete next step
4. safety         - promises nothing the brand cannot honour
5. tone           - sounds like the brand's real replies

Answer "tie" only if the two are genuinely equal in quality. Prefer a decision.
"""


# Two prompt variants, both measured. `direct` asks straight for a winner.
# `deliberate` forces a one-line assessment of EACH reply before the verdict,
# which is the standard mitigation for position bias: a model that has just
# written something about reply A cannot answer purely from recency. The
# variant is a CLI flag rather than a silent edit so that the failed attempt
# stays in the record alongside the one that shipped.
VARIANTS = {
    "direct": 'JSON only: {"winner":"A" or "B" or "tie","reason":"one short sentence"}',
    "deliberate": (
        "First assess each reply on its own, then decide. Do not let position\n"
        "influence you - A and B were ordered by a coin flip.\n"
        'JSON only: {"assess_a":"one short sentence about REPLY A",'
        '"assess_b":"one short sentence about REPLY B",'
        '"winner":"A" or "B" or "tie","reason":"why the winner is better"}'
    ),
}


def build_prompt(
    customer: str, reply_a: str, reply_b: str, neighbours: list[dict], variant: str = "direct"
) -> str:
    past = "\n".join(
        f"- Customer: {n['customer']}\n  Brand replied: {n['brand_reply']}" for n in neighbours[:3]
    ) or "- (no similar past replies were retrieved)"
    return (
        f"{RUBRIC}\n"
        f"PAST REPLIES FROM THIS BRAND (the grounding evidence):\n{past}\n\n"
        f'CUSTOMER MESSAGE:\n"{customer}"\n\n'
        f'REPLY A:\n"{reply_a}"\n\n'
        f'REPLY B:\n"{reply_b}"\n\n'
        f"{VARIANTS[variant]}"
    )


def judge_pair(
    customer: str, reply_a: str, reply_b: str, neighbours: list[dict], variant: str = "direct"
) -> tuple[str, str, bool]:
    """Returns (winner in {A,B,tie}, reason, parse_failed)."""
    from llm import generate_json

    data = generate_json(
        build_prompt(customer, reply_a, reply_b, neighbours, variant),
        system=SYSTEM_JUDGE,
        model=JUDGE_MODEL,
        max_tokens=150 if variant == "direct" else 260,
    )
    raw = str(data.get("winner", "")).strip().lower()
    if raw in ("a", "reply a"):
        return "A", str(data.get("reason", ""))[:200], False
    if raw in ("b", "reply b"):
        return "B", str(data.get("reason", ""))[:200], False
    if raw.startswith("tie") or raw in ("equal", "both"):
        return "tie", str(data.get("reason", ""))[:200], False
    # Unparseable output is recorded as a tie AND flagged, so the failure rate is
    # visible rather than silently diluting the agreement statistic.
    return "tie", str(data.get("reason", ""))[:200], True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--variant", choices=sorted(VARIANTS), default="direct",
                    help="direct = ask for a winner; deliberate = assess each reply first")
    ap.add_argument("--out", default=None, help="override the output path (for variant comparison)")
    args = ap.parse_args()
    set_seed()

    from llm import CACHE_ONLY, health, print_stats, unload_except

    if not CACHE_ONLY and not health():
        die("Ollama is not reachable. Start it, or run with ANTHILL_CACHE_ONLY=1")
    if not CACHE_ONLY:
        unload_except(JUDGE_MODEL)

    pairs = build_pairs()
    if not pairs:
        die("no pairs could be built - run agent.py --golden and baselines.py --all first")
    if args.limit:
        pairs = pairs[: args.limit]
    info(f"judging {len(pairs)} pairs twice each ({len(pairs)*2} calls) with "
         f"{JUDGE_MODEL}, prompt variant '{args.variant}'...")

    rows, flips, parse_fail = [], 0, 0
    for i, p in enumerate(pairs, 1):
        # as presented
        w1, r1, f1 = judge_pair(
            p["customer_message"], p["reply_a"], p["reply_b"], p["retrieved"], args.variant)
        # swapped: the same two replies, opposite positions
        w2, r2, f2 = judge_pair(
            p["customer_message"], p["reply_b"], p["reply_a"], p["retrieved"], args.variant)

        # Map both verdicts into presentation space (A = whatever was shown first
        # in the original ordering) so they can be compared directly.
        w2_mapped = {"A": "B", "B": "A", "tie": "tie"}[w2]
        consistent = w1 == w2_mapped
        if not consistent:
            flips += 1
        parse_fail += int(f1) + int(f2)

        verdict = w1 if consistent else "undecided"
        winner_system = (
            p["system_a"] if verdict == "A" else p["system_b"] if verdict == "B" else verdict
        )
        rows.append(
            {
                "pair_id": p["pair_id"],
                "golden_id": p["golden_id"],
                "comparison": p["comparison"],
                "system_a": p["system_a"],
                "system_b": p["system_b"],
                "verdict": verdict,
                "winner_system": winner_system,
                "verdict_as_shown": w1,
                "verdict_swapped": w2_mapped,
                "order_consistent": consistent,
                "reason": r1,
                "variant": args.variant,
                "parse_failed": bool(f1 or f2),
            }
        )
        if i % 10 == 0 or i == len(pairs):
            info(f"  {i}/{len(pairs)}")

    out = DATA / (args.out or "judgements_pairwise.jsonl")
    write_jsonl(out, rows)
    ok(f"wrote {len(rows)} pairwise judgements -> {out}")
    print_stats()

    flip_rate = flips / len(rows)
    print(f"\n{'='*72}\n  PAIRWISE JUDGE - INTERNAL CONSISTENCY\n{'='*72}")
    print(f"  position-bias flip rate   {flip_rate:.1%}  ({flips}/{len(rows)} pairs)")
    print("    " + (
        "SEVERE - the judge's answer depends mostly on which reply came first."
        if flip_rate > 0.4
        else "high - a large share of verdicts are order artefacts."
        if flip_rate > 0.25
        else "acceptable for a 3B judge."
    ))
    if parse_fail:
        # A pair where one call failed to parse is recorded tie-vs-something, which
        # can look like agreement between the two orders. Re-state the flip rate on
        # pairs where both calls parsed, so the consistency figure is not flattered
        # by the judge's own malformed output.
        clean = [r for r in rows if not r["parse_failed"]]
        print(f"  unparseable verdicts      {parse_fail}/{len(rows)*2} calls (recorded as tie)")
        if clean:
            cf = sum(not r["order_consistent"] for r in clean) / len(clean)
            print(f"  flip rate, parsed pairs   {cf:.1%}  ({len(clean)}/{len(rows)} pairs)")

    print(f"\n  {'comparison':<26} {'wins':>6} {'losses':>7} {'tie':>5} {'undecided':>10}")
    print("  " + "-" * 60)
    for comp in sorted({r["comparison"] for r in rows}):
        sub = [r for r in rows if r["comparison"] == comp]
        first = comp.split("_vs_")[0]
        wins = sum(r["winner_system"] == first for r in sub)
        ties = sum(r["verdict"] == "tie" for r in sub)
        und = sum(r["verdict"] == "undecided" for r in sub)
        print(f"  {comp:<26} {wins:>6} {len(sub)-wins-ties-und:>7} {ties:>5} {und:>10}")

    print("\nnext: python tools/score_pairs.py   (the same pairs, by hand, blind)\n")


if __name__ == "__main__":
    main()
