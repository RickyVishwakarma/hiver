"""Stage E.1 - deterministic metrics with honest uncertainty.

WHAT THIS REPORTS AND WHY

  accuracy        Included, but it is the number most likely to mislead: on an
                  imbalanced set the trivial baseline scores well on it.
  macro-F1        The headline. Averages per-class F1 with equal weight, so a
                  system that ignores the tail cannot hide behind the head.
  per-class P/R/F1  Where the failures actually live.
  escalation      Precision and recall reported SEPARATELY and never averaged
                  into one number, because the two errors have wildly different
                  costs. A missed escalation sends a canned reply to a fraud
                  victim. An over-escalation costs a few minutes of agent time.
                  We optimise recall and report the precision we paid for it.

  bootstrap 95% CI  At n=200 the CI on accuracy is roughly +/-6pp. Without this,
                  a 3-point "improvement" reads as progress when it is noise.
  paired bootstrap  For system-vs-system claims. Resamples the SAME examples for
                  both systems, so it measures the difference directly and is
                  far more sensitive than comparing two overlapping CIs. This is
                  what decides whether "the agent beats the simple baseline" is
                  a supportable statement or wishful thinking.

Usage:
  python tools/run_eval.py                     # every preds_*.jsonl found
  python tools/run_eval.py --compare agent simple
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from common import DATA, GOLDEN, REPORT_DIR, die, info, ok, read_jsonl, set_seed

N_BOOT = 2000
RNG = np.random.default_rng(20260909)


# --- statistics --------------------------------------------------------------
def bootstrap_ci(fn, *arrays, n_boot: int = N_BOOT, alpha: float = 0.05):
    """Percentile bootstrap CI for any metric function of aligned arrays."""
    n = len(arrays[0])
    stats = np.empty(n_boot)
    for b in range(n_boot):
        idx = RNG.integers(0, n, n)
        stats[b] = fn(*[a[idx] for a in arrays])
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def paired_bootstrap(y_true, pred_a, pred_b, fn, n_boot: int = N_BOOT):
    """P(metric_a > metric_b) by resampling the same indices for both systems."""
    n = len(y_true)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = RNG.integers(0, n, n)
        diffs[b] = fn(y_true[idx], pred_a[idx]) - fn(y_true[idx], pred_b[idx])
    return float(diffs.mean()), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def _acc(y, p):
    return float((y == p).mean())


def _macro_f1(y, p):
    return float(f1_score(y, p, average="macro", zero_division=0))


def _prf(y_true_bool, y_pred_bool):
    tp = int((y_true_bool & y_pred_bool).sum())
    fp = int((~y_true_bool & y_pred_bool).sum())
    fn_ = int((y_true_bool & ~y_pred_bool).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn_) if tp + fn_ else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1, tp, fp, fn_


# --- evaluation --------------------------------------------------------------
def evaluate(name: str, golden: list[dict], preds: list[dict]) -> dict:
    by_id = {p["golden_id"]: p for p in preds}
    aligned = [(g, by_id[g["golden_id"]]) for g in golden if g["golden_id"] in by_id]
    if not aligned:
        die(f"{name}: no overlap between golden ids and predictions")
    if len(aligned) < len(golden):
        info(f"{name}: {len(aligned)}/{len(golden)} examples have predictions")

    y_true = np.array([g["intent"] for g, _ in aligned])
    y_pred = np.array([p["intent"] for _, p in aligned])
    e_true = np.array([bool(g["escalate"]) for g, _ in aligned])
    e_pred = np.array([bool(p["escalate"]) for _, p in aligned])

    acc = _acc(y_true, y_pred)
    mf1 = _macro_f1(y_true, y_pred)
    acc_lo, acc_hi = bootstrap_ci(_acc, y_true, y_pred)
    f1_lo, f1_hi = bootstrap_ci(_macro_f1, y_true, y_pred)
    prec, rec, f1e, tp, fp, fn_ = _prf(e_true, e_pred)
    rec_lo, rec_hi = bootstrap_ci(lambda a, b: _prf(a, b)[1], e_true, e_pred)

    return {
        "system": name,
        "n": len(aligned),
        "intent_accuracy": acc,
        "intent_accuracy_ci": [acc_lo, acc_hi],
        "intent_macro_f1": mf1,
        "intent_macro_f1_ci": [f1_lo, f1_hi],
        "escalation_precision": prec,
        "escalation_recall": rec,
        "escalation_recall_ci": [rec_lo, rec_hi],
        "escalation_f1": f1e,
        "escalation_tp": tp,
        "escalation_fp": fp,
        "escalation_fn": fn_,
        "escalation_rate": float(e_pred.mean()),
        "_y_true": y_true,
        "_y_pred": y_pred,
        "_e_true": e_true,
        "_e_pred": e_pred,
    }


def print_report(res: dict, detail: bool) -> None:
    print(f"\n{'='*72}\n  {res['system'].upper()}  (n={res['n']})\n{'='*72}")
    a, (alo, ahi) = res["intent_accuracy"], res["intent_accuracy_ci"]
    m, (mlo, mhi) = res["intent_macro_f1"], res["intent_macro_f1_ci"]
    print(f"  intent accuracy    {a:.3f}   95% CI [{alo:.3f}, {ahi:.3f}]")
    print(f"  intent macro-F1    {m:.3f}   95% CI [{mlo:.3f}, {mhi:.3f}]   <- headline")
    rlo, rhi = res["escalation_recall_ci"]
    print(
        f"\n  escalation recall  {res['escalation_recall']:.3f}   95% CI [{rlo:.3f}, {rhi:.3f}]"
        f"   <- the one that matters"
    )
    print(f"  escalation prec.   {res['escalation_precision']:.3f}")
    print(
        f"  missed escalations {res['escalation_fn']}   (costly)      "
        f"over-escalations {res['escalation_fp']}   (cheap)"
    )
    print(f"  escalation rate    {res['escalation_rate']:.1%}")

    if detail:
        print("\n  per-class:")
        print(
            classification_report(
                res["_y_true"], res["_y_pred"], zero_division=0, digits=3
            )
        )
        labels = sorted(set(res["_y_true"]) | set(res["_y_pred"]))
        cm = confusion_matrix(res["_y_true"], res["_y_pred"], labels=labels)
        width = max(len(l) for l in labels) + 1
        print("  confusion (rows=true, cols=pred):")
        print(" " * (width + 2) + " ".join(f"{l[:6]:>6}" for l in labels))
        for l, row in zip(labels, cm):
            print(f"  {l:<{width}} " + " ".join(f"{v:>6}" for v in row))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--detail", action="store_true", help="Per-class report + confusion matrix")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"), help="Paired bootstrap of A vs B")
    ap.add_argument("--out", default=str(REPORT_DIR / "metrics.json"))
    args = ap.parse_args()
    set_seed()

    golden = read_jsonl(GOLDEN)
    if not golden:
        die(f"{GOLDEN} is empty - label the golden set first")

    results = {}
    for path in sorted(DATA.glob("preds_*.jsonl")):
        name = path.stem.replace("preds_", "")
        preds = read_jsonl(path)
        if preds:
            results[name] = evaluate(name, golden, preds)

    if not results:
        die("no preds_*.jsonl found - run agent.py --golden and baselines.py --all")

    order = [n for n in ("trivial", "simple", "agent") if n in results]
    order += [n for n in results if n not in order]
    for name in order:
        print_report(results[name], args.detail)

    # --- summary table -------------------------------------------------------
    print(f"\n{'='*72}\n  SUMMARY\n{'='*72}")
    print(f"  {'system':<10} {'acc':>7} {'macroF1':>9} {'esc.rec':>9} {'esc.prec':>9} {'missed':>7}")
    print("  " + "-" * 60)
    for name in order:
        r = results[name]
        print(
            f"  {name:<10} {r['intent_accuracy']:>7.3f} {r['intent_macro_f1']:>9.3f} "
            f"{r['escalation_recall']:>9.3f} {r['escalation_precision']:>9.3f} {r['escalation_fn']:>7}"
        )

    # --- paired comparison ---------------------------------------------------
    pairs = [tuple(args.compare)] if args.compare else [("agent", "simple"), ("simple", "trivial")]
    for a, b in pairs:
        if a not in results or b not in results:
            continue
        ra, rb = results[a], results[b]
        mean, lo, hi = paired_bootstrap(ra["_y_true"], ra["_y_pred"], rb["_y_pred"], _macro_f1)
        verdict = (
            "SIGNIFICANT (CI excludes 0)"
            if lo > 0 or hi < 0
            else "NOT significant - CI includes 0, this difference is not supported by n=%d" % ra["n"]
        )
        print(f"\n  paired bootstrap, macro-F1: {a} - {b}")
        print(f"    mean diff {mean:+.3f}   95% CI [{lo:+.3f}, {hi:+.3f}]")
        print(f"    {verdict}")

    serialisable = {
        k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in results.items()
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(serialisable, indent=2), encoding="utf-8")
    ok(f"wrote {args.out}")


if __name__ == "__main__":
    main()
