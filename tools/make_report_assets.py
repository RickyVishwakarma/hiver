"""Stage F - render report tables straight from the metrics files.

Every number in REPORT.md comes from here. Hand-copying figures out of console
output is how a report ends up disagreeing with its own repo: a metric gets
re-run, one table is updated and another is missed. Generating the markdown means
the report cannot drift from `report/metrics.json`.

Usage:
  python tools/make_report_assets.py            # writes report/tables.md
  python tools/make_report_assets.py --stdout
"""
from __future__ import annotations

import argparse
import json

from common import REPORT_DIR, info, ok, read_json, read_jsonl, warn

METRICS = REPORT_DIR / "metrics.json"
VALIDATION = REPORT_DIR / "judge_validation.json"
ORDER = ["trivial", "simple", "agent"]


def _fmt_ci(pair) -> str:
    if not pair:
        return "-"
    return f"[{pair[0]:.3f}, {pair[1]:.3f}]"


def headline_table(metrics: dict) -> str:
    names = [n for n in ORDER if n in metrics] + [n for n in metrics if n not in ORDER]
    lines = [
        "| System | Intent acc. | 95% CI | Macro-F1 | 95% CI | Esc. recall | Esc. prec. | Missed | Over |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for n in names:
        m = metrics[n]
        lines.append(
            f"| `{n}` | {m['intent_accuracy']:.3f} | {_fmt_ci(m.get('intent_accuracy_ci'))} "
            f"| **{m['intent_macro_f1']:.3f}** | {_fmt_ci(m.get('intent_macro_f1_ci'))} "
            f"| {m['escalation_recall']:.3f} | {m['escalation_precision']:.3f} "
            f"| {m['escalation_fn']} | {m['escalation_fp']} |"
        )
    return "\n".join(lines)


def judge_table(validation: dict) -> str:
    ja = validation.get("judge_agreement") or {}
    if not ja:
        return "_Judge validation has not been run yet._"
    lines = [
        "| Axis | Weighted kappa | Spearman | Judge mean | Human mean | Verdict |",
        "|---|---|---|---|---|---|",
    ]
    for axis, v in ja.items():
        if axis.startswith("_"):
            continue
        k = v["kappa_quadratic"]
        verdict = (
            "noise" if k < 0.2 else "weak" if k < 0.4 else "moderate" if k < 0.6 else "good"
        )
        lines.append(
            f"| {axis} | {k:.3f} | {v['spearman']:.3f} | {v['judge_mean']:.2f} "
            f"| {v['human_mean']:.2f} | {verdict} |"
        )
    gap = ja.get("_min_detectable_gap")
    if gap is not None:
        lines.append("")
        lines.append(
            f"Judge-human residual SD = **{gap:.2f}** rubric points. Reply-quality "
            f"differences smaller than this are not claimed anywhere in this report."
        )
    return "\n".join(lines)


def ceiling_block(validation: dict) -> str:
    hc = validation.get("human_ceiling") or {}
    if not hc:
        return "_Second-pass labelling has not been run yet._"
    # Pace is reported alongside because the two are not independent: a re-label
    # pass run at a few seconds per example measures input speed, not
    # consistency, and the figure is then a lower bound rather than a ceiling.
    pace = hc.get("pass2_seconds_per_label")
    caveat = ""
    if hc.get("is_lower_bound_only"):
        caveat = (
            "\n\n> **This is a lower bound, not a ceiling.** The re-label pass averaged "
            f"{pace:.1f}s per example (against {hc.get('pass1_seconds_per_label', 0):.1f}s on "
            "the first pass), which is not enough time to read a tweet and choose among "
            "9 intents plus an escalation reason. It bounds reliability from below; the "
            "true ceiling is unmeasured."
        )
    return caveat and (_ceiling_body(hc) + caveat) or (_ceiling_body(hc))


def _ceiling_body(hc: dict) -> str:
    return (
        f"- Examples re-labelled: **{hc['n']}**\n"
        f"- Intent: raw agreement **{hc['intent_raw_agreement']:.3f}**, "
        f"Cohen's kappa **{hc['intent_kappa']:.3f}**\n"
        f"- Escalation: raw agreement **{hc['escalation_raw_agreement']:.3f}**, "
        f"Cohen's kappa **{hc['escalation_kappa']:.3f}**\n\n"
        f"Model scores are read against this figure rather than against 100%."
    )


def bias_block(validation: dict) -> str:
    b = validation.get("bias") or {}
    if not b:
        return "_Bias probes have not been run yet._"
    out = [f"- Length bias: Spearman(reply length, judge score) = **{b['length_bias_spearman']:+.3f}**"]
    sp = b.get("self_preference") or {}
    if sp:
        out.append("- Judge-minus-human gap by system:")
        for system, gap in sp.items():
            out.append(f"  - `{system}`: {gap:+.2f}")
    d = b.get("self_preference_delta_vs_others")
    if d is not None:
        out.append(
            f"- Agent scored **{d:+.2f}** more generously than other systems relative to the "
            f"human{' - evidence of self-preference toward generated text' if d > 0.25 else ''}."
        )
    return "\n".join(out)


def escalation_sources() -> str:
    """Rule-vs-LLM split, straight from the agent's decided_by field."""
    from common import DATA

    rows = read_jsonl(DATA / "preds_agent.jsonl")
    esc = [r for r in rows if r.get("escalate")]
    if not esc:
        return "_No agent predictions yet._"
    from collections import Counter

    counts = Counter(r.get("decided_by", "?") for r in esc)
    lines = ["| Decided by | Count | Share of escalations |", "|---|---|---|"]
    for src, n in counts.most_common():
        lines.append(f"| `{src}` | {n} | {n/len(esc):.0%} |")
    lines.append("")
    n_rule = sum(n for s, n in counts.items() if s.startswith("rule:"))
    lines.append(
        f"{n_rule}/{len(esc)} ({n_rule/len(esc):.0%}) of escalations were decided by a "
        f"deterministic rule rather than the LLM."
    )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stdout", action="store_true")
    args = ap.parse_args()

    metrics = read_json(METRICS, {})
    validation = read_json(VALIDATION, {})
    if not metrics:
        warn(f"{METRICS} not found - run: python run.py eval")

    doc = f"""<!-- GENERATED by tools/make_report_assets.py - do not edit by hand -->

## Headline results

{headline_table(metrics) if metrics else "_Evaluation has not been run yet._"}

Macro-F1 is the headline metric: it weights every intent equally, so a system
that ignores the tail cannot hide behind the head. Escalation precision and
recall are never averaged - a missed escalation and an over-escalation differ in
cost by orders of magnitude.

## Annotator self-agreement

{ceiling_block(validation)}

## Judge validation

{judge_table(validation)}

## Judge bias probes

{bias_block(validation)}

## Escalation: rules vs. LLM

{escalation_sources()}
"""
    if args.stdout:
        print(doc)
        return
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "tables.md").write_text(doc, encoding="utf-8")
    ok(f"wrote {REPORT_DIR / 'tables.md'}")
    info("paste into REPORT.md, or reference it directly")


if __name__ == "__main__":
    main()
