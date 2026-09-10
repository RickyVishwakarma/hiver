"""Stage E.4 - is the measuring instrument any good?

The brief asks for "evidence of how well your judge agrees with a human". An
unvalidated LLM judge produces numbers with no known relationship to quality;
reporting them as evidence would be the central dishonesty this assignment is
screening for. This script produces three things:

1. --self-agreement   HUMAN CEILING
   Compare pass-1 and pass-2 labels of the same examples by the same annotator.
   Cohen's kappa here is the ceiling on what any system can achieve against
   these labels. If you agree with yourself kappa=0.75, a classifier scoring
   0.70 is near-ceiling, not mediocre. Without this, model scores have no scale.

2. --judge-agreement  IS THE JUDGE MEASURING ANYTHING?
   Quadratic-weighted Cohen's kappa (correct for ordinal 1-5 scores: disagreeing
   by 1 is penalised far less than by 4) plus Spearman correlation, per axis and
   overall, against the blind human scores.
   Interpretation used in the report: <0.2 the judge is noise and its scores
   must not be reported as evidence; 0.2-0.4 weak, directional only; 0.4-0.6
   moderate, usable for large gaps; >0.6 good for a 3B model.

3. --bias             WHAT IS THE JUDGE ACTUALLY REWARDING?
   length bias    correlation between reply length and judge score. A judge that
                  mostly rewards verbosity will rank a rambling system top.
   self-preference does the judge score the generator's own family higher than
                  the human does? Computed as judge-minus-human gap per system:
                  if the gap is larger for the LLM agent than for the verbatim
                  baseline, the judge is flattering generated text.

Usage:
  python tools/validate_judge.py --all
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from sklearn.metrics import cohen_kappa_score

from common import (
    DATA,
    GOLDEN,
    GOLDEN_PASS2,
    HUMAN_SCORES,
    REPORT_DIR,
    info,
    ok,
    read_jsonl,
    set_seed,
    warn,
)
from judge import AXES


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation without scipy (keeps the repro path dependency-light)."""
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")

    def rank(x):
        order = np.argsort(x)
        r = np.empty(len(x), float)
        r[order] = np.arange(len(x), dtype=float)
        # average ranks for ties
        _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
        for i, c in enumerate(counts):
            if c > 1:
                r[inv == i] = r[inv == i].mean()
        return r

    ra, rb = rank(a), rank(b)
    return float(np.corrcoef(ra, rb)[0, 1])


def interpret(k: float) -> str:
    if np.isnan(k):
        return "undefined (no variance)"
    if k < 0.2:
        return "NOISE - do not report judge scores as evidence"
    if k < 0.4:
        return "weak - directional only"
    if k < 0.6:
        return "moderate - usable for large gaps only"
    return "good (for a 3B judge)"


# --- 1. human ceiling --------------------------------------------------------
def self_agreement() -> dict:
    p1 = {r["golden_id"]: r for r in read_jsonl(GOLDEN)}
    p2 = {r["golden_id"]: r for r in read_jsonl(GOLDEN_PASS2)}
    shared = sorted(set(p1) & set(p2))
    if len(shared) < 10:
        warn(f"only {len(shared)} examples labelled twice - run: python tools/label_cli.py --pass2")
        return {}

    i1 = [p1[g]["intent"] for g in shared]
    i2 = [p2[g]["intent"] for g in shared]
    e1 = [bool(p1[g]["escalate"]) for g in shared]
    e2 = [bool(p2[g]["escalate"]) for g in shared]

    k_intent = float(cohen_kappa_score(i1, i2))
    k_esc = float(cohen_kappa_score(e1, e2))
    agree_intent = float(np.mean([a == b for a, b in zip(i1, i2)]))
    agree_esc = float(np.mean([a == b for a, b in zip(e1, e2)]))

    print(f"\n{'='*72}\n  HUMAN CEILING - you vs. yourself (n={len(shared)})\n{'='*72}")
    print(f"  intent      raw agreement {agree_intent:.3f}   Cohen's kappa {k_intent:.3f}")
    print(f"  escalation  raw agreement {agree_esc:.3f}   Cohen's kappa {k_esc:.3f}")
    print(
        f"\n  READ THIS AS: no system can be expected to exceed ~{agree_intent:.0%} agreement\n"
        f"  with these intent labels, because the labels themselves are only that\n"
        f"  stable. Model scores must be read against this ceiling, not against 100%."
    )
    disagreements = [(g, p1[g]["intent"], p2[g]["intent"]) for g in shared if p1[g]["intent"] != p2[g]["intent"]]
    if disagreements:
        print(f"\n  {len(disagreements)} intent flips (these mark genuinely ambiguous boundaries):")
        for g, a, b in disagreements[:8]:
            print(f"    {g}: {a} -> {b}")
    return {
        "n": len(shared),
        "intent_raw_agreement": agree_intent,
        "intent_kappa": k_intent,
        "escalation_raw_agreement": agree_esc,
        "escalation_kappa": k_esc,
        "intent_flips": [{"golden_id": g, "pass1": a, "pass2": b} for g, a, b in disagreements],
    }


# --- 2. judge vs human -------------------------------------------------------
def judge_agreement() -> dict:
    human = {(r["golden_id"], r["system"]): r for r in read_jsonl(HUMAN_SCORES)}
    judged = {(r["golden_id"], r["system"]): r for r in read_jsonl(DATA / "judgements.jsonl")}
    shared = sorted(set(human) & set(judged))
    if len(shared) < 10:
        warn(f"only {len(shared)} replies scored by both - run: python tools/score_replies.py")
        return {}

    print(f"\n{'='*72}\n  JUDGE vs HUMAN (n={len(shared)} replies)\n{'='*72}")
    print(f"  {'axis':<14} {'kappa_w':>8} {'spearman':>9} {'judge_mu':>9} {'human_mu':>9}  interpretation")
    print("  " + "-" * 82)

    out = {}
    for axis in AXES + ["overall"]:
        h = np.array([human[k][axis] for k in shared], dtype=float)
        j = np.array([judged[k][axis] for k in shared], dtype=float)
        if axis == "overall":
            # overall is a mean of 5 axes -> continuous; round for kappa
            kw = float(cohen_kappa_score(np.round(h).astype(int), np.round(j).astype(int), weights="quadratic"))
        else:
            kw = float(cohen_kappa_score(h.astype(int), j.astype(int), weights="quadratic"))
        rho = _spearman(h, j)
        print(f"  {axis:<14} {kw:>8.3f} {rho:>9.3f} {j.mean():>9.2f} {h.mean():>9.2f}  {interpret(kw)}")
        out[axis] = {"kappa_quadratic": kw, "spearman": rho, "judge_mean": float(j.mean()), "human_mean": float(h.mean())}

    ov = out["overall"]["kappa_quadratic"]
    # Smallest gap the instrument can resolve: roughly the SD of the judge-human
    # residual. Differences below this are not evidence of anything.
    resid = np.array([judged[k]["overall"] - human[k]["overall"] for k in shared])
    mde = float(np.std(resid))
    print(
        f"\n  CONSEQUENCE FOR THE REPORT:\n"
        f"    overall weighted kappa = {ov:.3f} ({interpret(ov)})\n"
        f"    judge-human residual SD = {mde:.2f} rubric points\n"
        f"    => do NOT claim a reply-quality difference smaller than ~{mde:.1f} points.\n"
        f"    Systems closer than that are indistinguishable to this instrument."
    )
    out["_min_detectable_gap"] = mde
    return out


# --- 3. bias probes ----------------------------------------------------------
def bias_probes() -> dict:
    human = {(r["golden_id"], r["system"]): r for r in read_jsonl(HUMAN_SCORES)}
    judged = [r for r in read_jsonl(DATA / "judgements.jsonl")]
    if not judged:
        warn("no judgements yet")
        return {}

    print(f"\n{'='*72}\n  JUDGE BIAS PROBES\n{'='*72}")

    lengths = np.array([len(r.get("reply", "")) for r in judged], dtype=float)
    scores = np.array([r["overall"] for r in judged], dtype=float)
    rho_len = _spearman(lengths, scores)
    print(f"  length bias      spearman(reply_length, judge_score) = {rho_len:+.3f}")
    print(
        "    "
        + (
            "STRONG - the judge substantially rewards verbosity; treat its\n"
            "    ranking as partly a length ranking."
            if abs(rho_len) > 0.4
            else "weak-to-moderate; note it, but it does not invalidate the ranking."
            if abs(rho_len) > 0.2
            else "negligible."
        )
    )

    out = {"length_bias_spearman": rho_len, "self_preference": {}}

    shared = sorted(set(human) & {(r["golden_id"], r["system"]) for r in judged})
    if len(shared) >= 10:
        jmap = {(r["golden_id"], r["system"]): r for r in judged}
        print("\n  self-preference  judge_mean - human_mean, by system:")
        gaps = {}
        for system in sorted({s for _, s in shared}):
            keys = [k for k in shared if k[1] == system]
            if len(keys) < 3:
                continue
            gap = float(np.mean([jmap[k]["overall"] - human[k]["overall"] for k in keys]))
            gaps[system] = gap
            print(f"    {system:<10} {gap:+.2f}  (n={len(keys)})")
        out["self_preference"] = gaps
        if "agent" in gaps and len(gaps) > 1:
            others = np.mean([v for k, v in gaps.items() if k != "agent"])
            delta = gaps["agent"] - others
            print(
                f"\n    agent is scored {delta:+.2f} more generously than other systems,\n"
                f"    relative to the human. "
                + (
                    "This is evidence of self-preference toward generated\n    text; the agent's reply-quality lead is inflated by roughly this much."
                    if delta > 0.25
                    else "No meaningful self-preference detected."
                )
            )
            out["self_preference_delta_vs_others"] = float(delta)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--self-agreement", action="store_true")
    ap.add_argument("--judge-agreement", action="store_true")
    ap.add_argument("--bias", action="store_true")
    args = ap.parse_args()
    set_seed()

    run_all = args.all or not (args.self_agreement or args.judge_agreement or args.bias)
    out = {}
    if run_all or args.self_agreement:
        out["human_ceiling"] = self_agreement()
    if run_all or args.judge_agreement:
        out["judge_agreement"] = judge_agreement()
    if run_all or args.bias:
        out["bias"] = bias_probes()

    path = REPORT_DIR / "judge_validation.json"
    path.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    ok(f"wrote {path}")


if __name__ == "__main__":
    main()
