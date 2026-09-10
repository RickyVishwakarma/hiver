"""Stage F.1 - mine the top failure modes with real examples.

Produces the evidence behind the report's failure-analysis section. Every mode
is backed by counts over the golden set plus verbatim examples, so the write-up
cites data rather than impressions.

Usage:
  python tools/failure_analysis.py            # console
  python tools/failure_analysis.py --json     # machine-readable for the report
"""
from __future__ import annotations

import argparse
import collections
import json
import re

from common import DATA, GOLDEN, REPORT_DIR, ok, read_jsonl, set_seed

# Contact details a reply might hand a customer. Fabricating one of these is the
# most damaging thing this system can do: it is fluent, confident, and wrong.
CONTACT = re.compile(
    r"(1-?800-?[A-Z0-9-]{3,}|\b\d{3}[-.]\d{3}[-.]\d{4}\b|delta\.com/\S+|www\.\S+)",
    re.IGNORECASE,
)
COMPETITOR = re.compile(
    r"@(AmericanAir|united|SouthwestAir|JetBlue|VirginAtlantic|British_Airways"
    r"|AlaskaAir|SpiritAirlines)",
    re.IGNORECASE,
)
# A reply that opens with an anonymised numeric handle is addressing whichever
# customer the retrieved precedent was written for, not the current one.
STALE_HANDLE = re.compile(r"^@\d{4,}")


def load(system: str):
    gold = {g["golden_id"]: g for g in read_jsonl(GOLDEN)}
    preds = {p["golden_id"]: p for p in read_jsonl(DATA / f"preds_{system}.jsonl")}
    return gold, preds, sorted(set(gold) & set(preds))


def analyse(system: str, n_examples: int) -> dict:
    gold, preds, ids = load(system)
    out: dict = {"system": system, "n": len(ids)}

    # 1. most frequent intent confusions
    pairs = collections.Counter()
    for g in ids:
        t, p = gold[g]["intent"], preds[g]["intent"]
        if t != p:
            pairs[(t, p)] += 1
    out["intent_confusions"] = [
        {
            "true": t,
            "predicted": p,
            "count": n,
            "examples": [
                gold[g]["customer_message"][:200]
                for g in ids
                if gold[g]["intent"] == t and preds[g]["intent"] == p
            ][:n_examples],
        }
        for (t, p), n in pairs.most_common(5)
    ]

    # 2. over-escalation, attributed to whichever component decided it
    over = [g for g in ids if preds[g]["escalate"] and not gold[g]["escalate"]]
    out["over_escalation"] = {
        "count": len(over),
        "by_source": dict(collections.Counter(preds[g].get("decided_by", "?") for g in over)),
        "examples": [
            {
                "msg": gold[g]["customer_message"][:180],
                "true_intent": gold[g]["intent"],
                "decided_by": preds[g].get("decided_by"),
            }
            for g in over[:n_examples]
        ],
    }

    # 3. missed escalations - the expensive error, so list all of them
    missed = [g for g in ids if gold[g]["escalate"] and not preds[g]["escalate"]]
    out["missed_escalation"] = {
        "count": len(missed),
        "examples": [
            {
                "msg": gold[g]["customer_message"][:180],
                "true_intent": gold[g]["intent"],
                "human_reason": gold[g].get("escalation_reason"),
            }
            for g in missed
        ],
    }

    # 4. is self-reported confidence informative at all?
    right = [preds[g].get("self_reported_confidence", 0) for g in ids
             if gold[g]["intent"] == preds[g]["intent"]]
    wrong = [preds[g].get("self_reported_confidence", 0) for g in ids
             if gold[g]["intent"] != preds[g]["intent"]]
    out["calibration"] = {
        "mean_confidence_when_correct": round(sum(right) / max(len(right), 1), 3),
        "mean_confidence_when_wrong": round(sum(wrong) / max(len(wrong), 1), 3),
        "n_correct": len(right),
        "n_wrong": len(wrong),
    }

    # 5. replies copied verbatim from the retrieved precedent
    degen = []
    for g in ids:
        reply = (preds[g].get("reply") or "").strip()
        if reply and any(reply == (n.get("brand_reply") or "").strip()
                         for n in preds[g].get("retrieved", [])):
            degen.append({"msg": gold[g]["customer_message"][:150], "reply": reply[:150]})
    out["verbatim_copies"] = {"count": len(degen), "examples": degen[:n_examples]}

    # 6. replies addressed to the wrong customer
    stale = [g for g in ids if STALE_HANDLE.match((preds[g].get("reply") or "").strip())]
    out["stale_mentions"] = {
        "count": len(stale),
        "note": "reply opens with another customer's anonymised handle, copied from precedent",
        "examples": [(preds[g].get("reply") or "")[:130] for g in stale[:n_examples]],
    }

    # 7. fabricated contact details
    fabricated, grounded = [], []
    for g in ids:
        m = CONTACT.search(preds[g].get("reply") or "")
        if not m:
            continue
        evidence = " ".join((n.get("brand_reply") or "") for n in preds[g].get("retrieved", []))
        record = {"golden_id": g, "detail": m.group(0), "reply": (preds[g].get("reply") or "")[:150]}
        (grounded if m.group(0).lower() in evidence.lower() else fabricated).append(record)
    out["fabricated_contact_details"] = {
        "fabricated": len(fabricated),
        "grounded": len(grounded),
        "examples": fabricated[:5],
    }

    # 8. messages that are not actually addressed to this brand
    mixed = [g for g in ids if COMPETITOR.search(gold[g]["customer_message"])]
    absent = [g for g in ids if not re.search(r"@delta", gold[g]["customer_message"], re.IGNORECASE)]
    out["not_addressed_to_brand"] = {
        "mentions_competitor": len(mixed),
        "never_mentions_delta": len(absent),
        "examples": [gold[g]["customer_message"][:160] for g in mixed[:3]],
    }
    return out


def report(out: dict) -> None:
    bar = "=" * 74
    print("\n" + bar)
    print("  FAILURE ANALYSIS - {} (n={})".format(out["system"], out["n"]))
    print(bar)

    print("\n1. TOP INTENT CONFUSIONS")
    for c in out["intent_confusions"]:
        print("\n  {}x  true={}  ->  predicted={}".format(c["count"], c["true"], c["predicted"]))
        for e in c["examples"]:
            print("      - " + e[:110])

    o = out["over_escalation"]
    print("\n2. OVER-ESCALATION: {} cases".format(o["count"]))
    for k, v in sorted(o["by_source"].items(), key=lambda x: -x[1]):
        print("      {:<32} {}".format(k, v))

    m = out["missed_escalation"]
    print("\n3. MISSED ESCALATION: {} cases (the expensive error)".format(m["count"]))
    for e in m["examples"]:
        print("      [{} / {}] {}".format(e["true_intent"], e["human_reason"], e["msg"][:95]))

    c = out["calibration"]
    print("\n4. CONFIDENCE CALIBRATION")
    print("      when correct: {}  (n={})".format(c["mean_confidence_when_correct"], c["n_correct"]))
    print("      when wrong:   {}  (n={})".format(c["mean_confidence_when_wrong"], c["n_wrong"]))

    v = out["verbatim_copies"]
    print("\n5. VERBATIM COPIES OF RETRIEVED EVIDENCE: {}".format(v["count"]))
    for e in v["examples"]:
        print("      - " + e["reply"][:110])

    s = out["stale_mentions"]
    print("\n6. REPLIES OPENING WITH ANOTHER CUSTOMER'S HANDLE: {}".format(s["count"]))
    for e in s["examples"]:
        print("      - " + e[:110])

    f = out["fabricated_contact_details"]
    print("\n7. CONTACT DETAILS: {} FABRICATED, {} grounded".format(f["fabricated"], f["grounded"]))
    for e in f["examples"]:
        print("      {}: {}  (absent from retrieved evidence)".format(e["golden_id"], e["detail"]))

    b = out["not_addressed_to_brand"]
    print("\n8. NOT ADDRESSED TO DELTA: {} mention a competitor, {} never mention @Delta".format(
        b["mentions_competitor"], b["never_mentions_delta"]))
    for e in b["examples"]:
        print("      - " + e[:105])
    print()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--system", default="agent")
    ap.add_argument("--examples", type=int, default=3)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    set_seed()

    out = analyse(args.system, args.examples)
    if args.json:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        path = REPORT_DIR / "failure_analysis.json"
        path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
        ok("wrote {}".format(path))
        return
    report(out)


if __name__ == "__main__":
    main()
