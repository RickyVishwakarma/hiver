"""Stage C.1 - draw a STRATIFIED sample for hand-labelling.

WHY NOT RANDOM SAMPLING
Support traffic is heavily head-dominated: two or three intents can be 60% of
volume. A random sample of 200 would give the tail classes 2-4 examples each,
so per-class recall would have an error bar wider than the metric itself and
macro-F1 would be pure noise. Random sampling also under-represents exactly the
cases worth measuring: ambiguous phrasing and escalation-worthy complaints.

STRATA
  1. cluster            floor of --min-per-cluster each, so every intent is
                        measurable, then proportional allocation for the rest
  2. ambiguity          a fixed quota of low-centroid-similarity threads - the
                        boundary cases where classifiers actually fail
  3. escalation signal  a fixed quota of threads matching high-stakes patterns,
                        otherwise escalation recall is estimated from ~5 positives

This deliberately makes the golden set NOT match production distribution. That
is a real trade-off with a real cost: aggregate accuracy on this set is not a
production estimate. We keep the sampling weights so metrics can be reweighted
back, and we say so plainly in the report.

Usage:
  python tools/sample_golden.py --n 200
"""
from __future__ import annotations

import argparse
import random
import re

from common import DATA, GOLDEN, THREADS, die, info, ok, read_jsonl, set_seed, write_jsonl

# High-stakes language. Deliberately over-broad: this only decides what gets
# SAMPLED for labelling, not what gets escalated. Recall matters here, not
# precision - a false hit just means one more interesting example to label.
ESCALATION_SIGNALS = re.compile(
    r"\b(refund|charged|chargeback|billing|fraud|scam|stolen|hacked|breach|"
    r"lawyer|legal|sue|lawsuit|ombudsman|complaint|compensation|unacceptable|"
    r"disgusting|appalling|never again|cancel my|close my account|"
    r"data protection|gdpr|injured|safety|dangerous|emergency|urgent)\b",
    re.IGNORECASE,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=200, help="Golden set size (brief allows 150-250)")
    ap.add_argument("--min-per-cluster", type=int, default=10)
    ap.add_argument("--ambiguous-quota", type=float, default=0.15, help="Fraction reserved for low-similarity cases")
    ap.add_argument("--escalation-quota", type=float, default=0.20, help="Fraction reserved for high-stakes cases")
    ap.add_argument("--pass2-n", type=int, default=40, help="Subset re-labelled later for self-agreement")
    ap.add_argument("--out", default=str(DATA / "golden_unlabelled.jsonl"))
    args = ap.parse_args()
    set_seed()
    rng = random.Random(20260909)

    threads = read_jsonl(THREADS)
    assigns = {a["thread_id"]: a for a in read_jsonl(DATA / "cluster_assignments.jsonl")}
    if not threads:
        die("no threads - run build_threads.py")
    if not assigns:
        die("no cluster assignments - run induce_intents.py")

    # Only sample threads the agent could plausibly be asked to handle:
    # a real customer opening, and not a bystander pile-on.
    pool = [
        t
        for t in threads
        if t["thread_id"] in assigns
        and not t["multi_customer"]
        and len(t["opening_msg"]) >= 20
    ]
    info(f"eligible pool: {len(pool):,} of {len(threads):,} threads")
    for t in pool:
        t["_cluster"] = assigns[t["thread_id"]]["cluster"]
        t["_max_sim"] = assigns[t["thread_id"]]["max_sim"]
        t["_esc_signal"] = bool(ESCALATION_SIGNALS.search(t["opening_msg_raw"]))

    picked: dict[str, dict] = {}

    def take(candidates: list[dict], k: int, stratum: str) -> int:
        rng.shuffle(candidates)
        n = 0
        for t in candidates:
            if n >= k:
                break
            if t["thread_id"] in picked:
                continue
            t = dict(t)
            t["_stratum"] = stratum
            picked[t["thread_id"]] = t
            n += 1
        return n

    # --- quota 1: high-stakes / escalation-likely -----------------------------
    esc_target = int(args.n * args.escalation_quota)
    got = take([t for t in pool if t["_esc_signal"]], esc_target, "escalation_signal")
    info(f"escalation stratum: {got}/{esc_target}")

    # --- quota 2: ambiguous (bottom decile of centroid similarity) ------------
    amb_target = int(args.n * args.ambiguous_quota)
    by_sim = sorted(pool, key=lambda t: t["_max_sim"])
    got = take(by_sim[: max(len(by_sim) // 10, amb_target * 4)], amb_target, "ambiguous")
    info(f"ambiguous stratum: {got}/{amb_target}")

    # --- quota 3: floor per cluster, so every intent is measurable ------------
    clusters = sorted({t["_cluster"] for t in pool})
    for c in clusters:
        have = sum(1 for t in picked.values() if t["_cluster"] == c)
        need = max(0, args.min_per_cluster - have)
        if need:
            take([t for t in pool if t["_cluster"] == c], need, f"cluster_floor_{c}")
    info(f"after per-cluster floor: {len(picked)}")

    # --- remainder: proportional to cluster size ------------------------------
    remaining = args.n - len(picked)
    if remaining > 0:
        sizes = {c: sum(1 for t in pool if t["_cluster"] == c) for c in clusters}
        total = sum(sizes.values())
        for c in clusters:
            k = round(remaining * sizes[c] / total)
            take([t for t in pool if t["_cluster"] == c], k, f"proportional_{c}")
    # Top up if rounding left us short.
    if len(picked) < args.n:
        take(list(pool), args.n - len(picked), "topup")

    sample = list(picked.values())[: args.n]
    rng.shuffle(sample)  # present in random order so labelling is not blocked by stratum

    # Strip everything the annotator must NOT see. The brand's actual reply is
    # withheld: seeing how support answered would anchor the intent label and
    # the escalation call toward the historical decision.
    out_rows = []
    for i, t in enumerate(sample):
        out_rows.append(
            {
                "golden_id": f"g{i:04d}",
                "thread_id": t["thread_id"],
                "customer_message": t["opening_msg_raw"],
                "customer_message_clean": t["opening_msg"],
                "_stratum": t["_stratum"],
                "_cluster": t["_cluster"],
                "_max_sim": t["_max_sim"],
                "_esc_signal": t["_esc_signal"],
                "_n_turns": t["n_turns"],
            }
        )

    write_jsonl(args.out, out_rows)
    ok(f"wrote {len(out_rows)} unlabelled examples -> {args.out}")

    # Pre-select the second-pass subset now, so the choice cannot be influenced
    # by which examples turned out to be easy during labelling.
    pass2 = rng.sample([r["golden_id"] for r in out_rows], min(args.pass2_n, len(out_rows)))
    write_jsonl(DATA / "pass2_ids.jsonl", [{"golden_id": g} for g in sorted(pass2)])
    ok(f"pre-selected {len(pass2)} ids for the self-agreement re-label")

    from collections import Counter

    print("\nstratum breakdown:")
    for k, v in Counter(r["_stratum"].split("_")[0] for r in out_rows).most_common():
        print(f"  {k:<14} {v}")
    print("\ncluster coverage:")
    for c, v in sorted(Counter(r["_cluster"] for r in out_rows).items()):
        print(f"  cluster {c:<3} {v}")
    print(f"\nnext: python tools/label_cli.py\n")


if __name__ == "__main__":
    main()
