"""Stage C.2 - hand-label the golden set. This is the human bottleneck.

DESIGN RULE: THE ANNOTATOR IS KEPT BLIND
This tool never shows (a) any model prediction for the example, or (b) the
brand's actual historical reply. Both would anchor the label toward the thing we
are trying to measure independently, and would quietly inflate every downstream
number. It costs hours; it is the entire reason the golden set is worth having.

Two modes:
  default   label every example -> data/golden.jsonl
  --pass2   re-label a pre-selected subset -> data/golden_pass2.jsonl
            Pass-1 labels are NOT shown. Comparing the two passes gives
            intra-annotator agreement = the human ceiling. If you agree with
            yourself only 80% of the time, no classifier can be expected to
            exceed ~80% agreement with you, and a "76%" model is near-ceiling
            rather than mediocre.

Progress is saved after every example, so it is safe to quit and resume.

Keys:  <number> intent (type the number and press Enter)   b back   s skip   q save+quit
       then: a = auto-handle, e = escalate (+ a reason number)
"""
from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone

from common import (
    DATA,
    GOLDEN,
    GOLDEN_PASS2,
    TAXONOMY,
    die,
    info,
    ok,
    read_json,
    read_jsonl,
    warn,
    write_jsonl,
)

# Structured escalation reasons. Categorical rather than free text so we can
# report WHY escalation was needed and compare the agent's reason to the human's,
# not just the binary decision.
REASONS = [
    ("policy_exception", "Needs a decision only a human can authorise (refund, goodwill, waiver)"),
    ("account_security", "Account compromise, fraud, unauthorised access, payment dispute"),
    ("legal_regulatory", "Legal threat, regulator, data-protection, formal complaint"),
    ("safety_harm", "Physical safety, injury, medical, or distress"),
    ("insufficient_info", "Cannot be answered without private account data"),
    ("high_emotion", "Severe frustration; a canned reply would make it worse"),
    ("ambiguous_request", "Genuinely unclear what the customer wants"),
    ("out_of_scope", "Not a support request this brand handles"),
    ("other", "Something else (free text)"),
]


def clear():
    os.system("cls" if os.name == "nt" else "clear")


def render(ex: dict, idx: int, total: int, intents: list[dict], done: int) -> None:
    clear()
    print("=" * 76)
    print(f"  GOLDEN SET LABELLING     example {idx + 1}/{total}     labelled: {done}")
    print("=" * 76)
    print("\n  CUSTOMER MESSAGE:\n")
    msg = ex["customer_message"]
    for line in [msg[i : i + 70] for i in range(0, len(msg), 70)]:
        print(f"    {line}")
    print("\n" + "-" * 76)
    print("  INTENT:")
    for i, it in enumerate(intents, 1):
        print(f"    {i}. {it['name']:<24} {it.get('description','')[:44]}")
    print("-" * 76)


def ask_escalation(intent: str) -> tuple[bool | None, str, str]:
    # Echo the selection back. Without this a mis-key is invisible: the
    # screen just advances and a wrong label is saved silently. Measured
    # cost of the version without it: 28 of 40 pass-2 labels flipped, many
    # between unrelated intents (flight_disruption <-> praise_compliment).
    print("")
    print("  >>> RECORDED: " + intent.upper() + " <<<")
    print("\n  Should this be AUTO-HANDLED or ESCALATED to a human?")
    print("    a = auto-handle    e = escalate    x = wrong intent, redo")
    while True:
        c = input("  > ").strip().lower()
        if c == "x":
            return None, "", ""
        if c == "a":
            return False, "", ""
        if c == "e":
            break
        print("  (a, e, or x)")
    print("\n  Reason for escalation:")
    for i, (code, desc) in enumerate(REASONS, 1):
        print(f"    {i}. {code:<20} {desc}")
    while True:
        c = input("  > ").strip()
        if c.isdigit() and 1 <= int(c) <= len(REASONS):
            code = REASONS[int(c) - 1][0]
            note = input("  note (optional): ").strip() if code == "other" else ""
            return True, code, note
        print(f"  (1-{len(REASONS)})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pass2", action="store_true", help="Re-label the self-agreement subset")
    ap.add_argument("--infile", default=str(DATA / "golden_unlabelled.jsonl"))
    args = ap.parse_args()

    taxonomy = read_json(TAXONOMY)
    if not taxonomy:
        die(
            f"{TAXONOMY} not found.\n"
            "    Run induce_intents.py, name the clusters in data/taxonomy_draft.json,\n"
            "    then save it as data/taxonomy.json"
        )
    intents = [i for i in taxonomy["intents"] if not i["name"].startswith("TODO_")]
    if len(intents) < 3:
        die("taxonomy.json still has TODO_ placeholders - name the clusters first")

    examples = read_jsonl(args.infile)
    if not examples:
        die(f"no examples at {args.infile} - run sample_golden.py")

    outfile = GOLDEN_PASS2 if args.pass2 else GOLDEN
    if args.pass2:
        keep = {r["golden_id"] for r in read_jsonl(DATA / "pass2_ids.jsonl")}
        examples = [e for e in examples if e["golden_id"] in keep]
        info(f"pass 2: re-labelling {len(examples)} pre-selected examples")
        warn("Your pass-1 labels are hidden on purpose. Label as if seeing these fresh.")
        input("press enter to begin...")

    existing = {r["golden_id"]: r for r in read_jsonl(outfile)}
    if existing:
        info(f"resuming: {len(existing)} already labelled in {outfile.name}")

    todo = [e for e in examples if e["golden_id"] not in existing]
    if not todo:
        ok(f"all {len(examples)} examples already labelled in {outfile.name}")
        return

    labels = dict(existing)
    i = 0
    while i < len(todo):
        ex = todo[i]
        render(ex, len(existing) + i, len(examples), intents, len(labels))
        print("  <number> = intent   b = back   s = skip   q = save+quit")
        c = input("  > ").strip().lower()

        if c == "q":
            break
        if c == "s":
            i += 1
            continue
        if c == "b":
            i = max(0, i - 1)
            continue
        if not (c.isdigit() and 1 <= int(c) <= len(intents)):
            continue

        intent = intents[int(c) - 1]["name"]
        escalate, reason, note = ask_escalation(intent)
        if escalate is None:  # wrong intent - redo this example
            continue
        labels[ex["golden_id"]] = {
            "golden_id": ex["golden_id"],
            "thread_id": ex["thread_id"],
            "customer_message": ex["customer_message"],
            "intent": intent,
            "escalate": escalate,
            "escalation_reason": reason,
            "escalation_note": note,
            "stratum": ex.get("_stratum"),
            "labelled_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "pass": 2 if args.pass2 else 1,
        }
        # Persist every single example: a crash at #180 must not cost the session.
        write_jsonl(outfile, [labels[k] for k in sorted(labels)])
        route = ("ESCALATE (" + reason + ")") if escalate else "auto-handle"
        print("")
        print("  saved: " + intent + " / " + route)
        i += 1

    clear()
    ok(f"{len(labels)} labelled -> {outfile}")
    remaining = len(examples) - len(labels)
    if remaining > 0:
        info(f"{remaining} still to go - re-run this command to resume")
    else:
        if args.pass2:
            print("\nnext: python tools/validate_judge.py --self-agreement\n")
        else:
            print("\nnext: python tools/label_cli.py --pass2   (after a break - a delay makes")
            print("      the self-agreement estimate honest rather than a memory test)\n")


if __name__ == "__main__":
    main()
