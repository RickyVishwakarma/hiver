"""Stage D.0 - build few-shot exemplars for the intent classifier.

WHY THIS EXISTS
The zero-shot 3B classifier collapsed: it never emitted booking_change_refund
or other, and dumped 44% of predictions into two catch-all-sounding labels
(support_unreachable, misc_request). Verbose intent descriptions alone were not
enough to anchor it. Concrete examples are.

NO TEST CONTAMINATION
Exemplars come only from threads whose thread_id is absent from the golden set,
selected by cosine similarity to their cluster centroid. Intent assignment comes
from the taxonomy's cluster->intent mapping, i.e. from unsupervised clustering,
NOT from any human label. So no golden-set label or text reaches the prompt.
The output file lists every exemplar so this claim is auditable.

Usage:
  python tools/build_fewshot.py --per-intent 2
"""
from __future__ import annotations

import argparse

import numpy as np

from common import DATA, GOLDEN, TAXONOMY, die, info, ok, read_json, read_jsonl, set_seed, write_json


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-intent", type=int, default=2)
    ap.add_argument("--max-chars", type=int, default=160)
    args = ap.parse_args()
    set_seed()

    tax = read_json(TAXONOMY) or die("taxonomy.json missing")
    threads = {t["thread_id"]: t for t in read_jsonl(DATA / "threads.jsonl")}
    assigns = read_jsonl(DATA / "cluster_assignments.jsonl")
    golden_ids = {g["thread_id"] for g in read_jsonl(GOLDEN)}
    if not assigns:
        die("cluster_assignments.jsonl missing - run induce_intents.py")

    # Highest-confidence cluster members first, golden-set threads excluded.
    by_cluster: dict[int, list[dict]] = {}
    for a in sorted(assigns, key=lambda x: -x["max_sim"]):
        if a["thread_id"] in golden_ids or a["thread_id"] not in threads:
            continue
        by_cluster.setdefault(a["cluster"], []).append(a)

    out = {"note": "exemplars exclude every golden-set thread; intent comes from "
                   "cluster mapping, not human labels", "examples": []}
    for intent in tax["intents"]:
        name = intent["name"]
        picked = 0
        for c in intent.get("source_clusters", []):
            for a in by_cluster.get(c, []):
                if picked >= args.per_intent:
                    break
                msg = threads[a["thread_id"]]["opening_msg"].strip()
                if not (30 <= len(msg) <= args.max_chars):
                    continue
                out["examples"].append({
                    "intent": name, "cluster": c,
                    "thread_id": a["thread_id"], "sim": a["max_sim"],
                    "message": msg,
                })
                picked += 1
        if picked == 0 and name != "other":
            info(f"no exemplar found for {name} (cluster {intent.get('source_clusters')})")

    leaked = [e for e in out["examples"] if e["thread_id"] in golden_ids]
    if leaked:
        die(f"{len(leaked)} exemplars leak golden-set threads - aborting")

    write_json(DATA / "fewshot.json", out)
    ok(f"wrote {len(out['examples'])} exemplars -> data/fewshot.json (0 golden-set leaks)")
    for e in out["examples"]:
        print(f"  {e['intent']:<24} {e['message'][:74]}")


if __name__ == "__main__":
    main()
