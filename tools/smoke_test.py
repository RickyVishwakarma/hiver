"""End-to-end plumbing test on synthetic data. No Kaggle download required.

Runs every stage of the pipeline against a generated fixture and asserts each
one produced sane output. Its job is to prove the wiring works - NOT to produce
any number that appears in the report.

Isolation: everything is written to .tmp/smoke/ via ANTHILL_DATA_DIR, so this
can never overwrite the real golden set or the committed cache.

The intent labels used here are derived mechanically from the fixture's own
generator groups. That is legitimate for a plumbing test and would be worthless
as evaluation data - which is exactly why the real golden set is hand-labelled.

Usage:
  python tools/smoke_test.py            # full run (needs Ollama for LLM stages)
  python tools/smoke_test.py --no-llm   # skip agent/judge, test data stages only
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from common import setup_console

setup_console()

ROOT = Path(__file__).resolve().parent.parent
SMOKE = ROOT / ".tmp" / "smoke"
PY = sys.executable

# Keyword -> intent, mirroring make_fixture.py's groups.
FIXTURE_INTENTS = [
    ("account_security", ["hacked", "compromised", "unauthorised", "fraud", "accessed"]),
    ("billing_dispute", ["charged", "refund", "billed", "pounds", "subscription", "renewal"]),
    ("login_access", ["log into", "password", "locked out", "login"]),
    ("device_power", ["charge", "battery", "drains", "power"]),
    ("order_tracking", ["order", "package", "tracking", "delivered"]),
    ("complaint_escalation", ["appalling", "unacceptable", "lawyer", "manager", "legal", "worst"]),
    ("account_admin", ["change my", "update the", "cancel auto", "invoices", "how do i"]),
]
ESCALATE_INTENTS = {"account_security", "complaint_escalation", "billing_dispute"}

PASS, FAIL = "\033[32m  PASS\033[0m", "\033[31m  FAIL\033[0m"
_results: list[tuple[str, bool, str]] = []


def check(name: str, cond: bool, detail: str = "") -> bool:
    _results.append((name, bool(cond), detail))
    print(f"{PASS if cond else FAIL}  {name}" + (f"  ({detail})" if detail else ""))
    return bool(cond)


def sh(script: str, *args: str) -> tuple[int, str]:
    env = {**os.environ, "ANTHILL_DATA_DIR": str(SMOKE)}
    p = subprocess.run(
        [PY, str(ROOT / "tools" / script), *args],
        cwd=ROOT, env=env, capture_output=True, text=True,
    )
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def jsonl(name: str) -> list[dict]:
    path = SMOKE / name
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _real_metrics_digest() -> str:
    """Hash of the committed report/metrics.json, or '' if absent.

    Used to prove a smoke run never writes fixture numbers into the real
    deliverable directory - which it once did, before REPORT_DIR followed
    ANTHILL_DATA_DIR.
    """
    import hashlib

    path = ROOT / "report" / "metrics.json"
    if not path.exists():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rules_short_circuit() -> bool:
    """Does a deterministic rule decide before any model is consulted?

    The expensive error in routing is a MISSED escalation, which is why the
    rules run first and short-circuit. This asserts that ordering on a message
    that must trip one, independently of what the fixture sample contains.
    """
    env = {**os.environ, "ANTHILL_DATA_DIR": str(SMOKE), "ANTHILL_CACHE_ONLY": "1"}
    code = (
        "import sys; sys.path.insert(0, r'%s');"
        "from agent import Agent;"
        "e, r, d = Agent.route(None, 'my account has been hacked', 'other', 0.99, False);"
        "assert d.startswith('rule:'), d; assert e; print('OK')"
    ) % str(ROOT / "tools")
    p = subprocess.run([PY, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True)
    return p.returncode == 0 and "OK" in (p.stdout or "")


def stage(title: str) -> None:
    print(f"\n\033[36m-- {title} {'-' * max(0, 56 - len(title))}\033[0m")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-llm", action="store_true", help="Skip stages needing Ollama")
    ap.add_argument("--n-llm", type=int, default=6, help="Examples for the LLM stages")
    args = ap.parse_args()

    SMOKE.mkdir(parents=True, exist_ok=True)
    real_metrics_before = _real_metrics_digest()
    print("=" * 64)
    print("  SMOKE TEST - synthetic fixture, isolated in .tmp/smoke/")
    print("=" * 64)

    # --- data stages ---------------------------------------------------------
    stage("fixture + profile")
    rc, out = sh("make_fixture.py", "--threads", "900")
    check("fixture generated", rc == 0)
    rc, out = sh("profile_brands.py", "--top", "3")
    check("brand profile ran", rc == 0)
    check("fixture brand ranked first", "FixtureSupport" in out.split("\n")[-8] if rc == 0 else False)

    stage("thread reconstruction")
    rc, out = sh("build_threads.py", "--brand", "FixtureSupport")
    threads = jsonl("threads.jsonl")
    check("threads built", rc == 0 and len(threads) > 500, f"{len(threads)} threads")
    check("has resolved threads", sum(t["resolved_proxy"] for t in threads) > 100)
    check("captured unanswered messages", any(t["n_brand_replies"] == 0 for t in threads))
    check("detected DM punts", any(t["punted_to_dm"] for t in threads))
    check("turns are ordered customer-first", all(
        t["turns"][0]["role"] == "customer" for t in threads[:50]))

    stage("intent induction")
    rc, out = sh("induce_intents.py", "--k", "7", "--exemplars", "2")
    assigns = jsonl("cluster_assignments.jsonl")
    check("clustering ran", rc == 0 and len(assigns) == len(threads), f"{len(assigns)} assigned")
    check("clusters are non-degenerate", len({a["cluster"] for a in assigns}) == 7)

    # Build a taxonomy without the manual naming step (fixture only).
    (SMOKE / "taxonomy.json").write_text(json.dumps({
        "brand": "FixtureSupport", "k": len(FIXTURE_INTENTS),
        "intents": [{"cluster_id": i, "name": n, "description": f"fixture intent: {n}",
                     "share": 0.0, "terms": kws, "examples": []}
                    for i, (n, kws) in enumerate(FIXTURE_INTENTS)]
        + [{"cluster_id": -1, "name": "other", "description": "fallback",
            "share": 0.0, "terms": [], "examples": []}],
    }, indent=2), encoding="utf-8")
    check("taxonomy written", (SMOKE / "taxonomy.json").exists())

    stage("golden set sampling")
    rc, out = sh("sample_golden.py", "--n", "80", "--min-per-cluster", "5", "--pass2-n", "20")
    unlab = jsonl("golden_unlabelled.jsonl")
    check("sampler ran", rc == 0 and len(unlab) == 80, f"{len(unlab)} sampled")
    check("all clusters represented", len({u["_cluster"] for u in unlab}) >= 6)
    check("escalation stratum non-empty", any(u["_esc_signal"] for u in unlab))
    check("ambiguous stratum present", any(u["_stratum"] == "ambiguous" for u in unlab))
    check("annotator cannot see the reply",
          all("first_brand_reply" not in u and "turns" not in u for u in unlab))

    # Mechanical labels (fixture only - never a substitute for hand-labelling).
    labelled = []
    for u in unlab:
        msg = u["customer_message"].lower()
        intent = next((n for n, kws in FIXTURE_INTENTS if any(k in msg for k in kws)), "other")
        labelled.append({
            "golden_id": u["golden_id"], "thread_id": u["thread_id"],
            "customer_message": u["customer_message"], "intent": intent,
            "escalate": intent in ESCALATE_INTENTS,
            "escalation_reason": "policy_exception" if intent in ESCALATE_INTENTS else "",
            "escalation_note": "", "stratum": u["_stratum"],
            "labelled_at": "fixture", "pass": 1,
        })
    (SMOKE / "golden.jsonl").write_text(
        "\n".join(json.dumps(r) for r in labelled) + "\n", encoding="utf-8")
    n_int = len({r["intent"] for r in labelled})
    check("synthetic labels created", n_int >= 5, f"{n_int} intents, "
          f"{sum(r['escalate'] for r in labelled)} escalations")

    stage("baselines")
    rc, out = sh("baselines.py", "--all")
    triv, simp = jsonl("preds_trivial.jsonl"), jsonl("preds_simple.jsonl")
    check("trivial baseline ran", rc == 0 and len(triv) == len(labelled), out.strip()[-90:] if rc else "")
    check("simple baseline ran", len(simp) == len(labelled))
    check("trivial predicts one class", len({p["intent"] for p in triv}) == 1)
    check("trivial never escalates", not any(p["escalate"] for p in triv))
    check("simple copies real replies", all(p["reply"] for p in simp))

    # --- LLM stages ----------------------------------------------------------
    if args.no_llm:
        print("\n  (skipping LLM stages: --no-llm)")
    else:
        sys.path.insert(0, str(ROOT / "tools"))
        from llm import health

        if not health():
            print("\n  \033[33mSKIP\033[0m  Ollama unreachable - LLM stages not tested")
        else:
            stage(f"agent (n={args.n_llm})")
            rc, out = sh("agent.py", "--golden", "--limit", str(args.n_llm))
            preds = jsonl("preds_agent.jsonl")
            check("agent ran", rc == 0 and len(preds) == args.n_llm, out.strip()[-120:] if rc else "")
            if preds:
                check("agent produced replies", all(p["reply"] for p in preds))
                check("intents are in-taxonomy",
                      all(p["intent"] in {n for n, _ in FIXTURE_INTENTS} | {"other"} for p in preds))
                check("retrieval returned neighbours", all(len(p["retrieved"]) > 0 for p in preds))
                check("escalations carry a reason",
                      all(p["escalation_reason"] and p["decided_by"] for p in preds if p["escalate"]))
                # Asserted directly rather than by hoping one of the sampled
                # fixture examples happens to trip a rule. It did not: the first
                # six golden examples all routed via the LLM, so this check
                # failed for reasons that had nothing to do with the invariant.
                # route() is called under ANTHILL_CACHE_ONLY=1 with an empty
                # cache dir, so if the rule did NOT short-circuit, the LLM call
                # would raise CacheMiss and the check would fail loudly.
                check("rules fire before the LLM", _rules_short_circuit())

            stage("judge")
            rc, out = sh("judge.py", "--system", "agent", "--limit", str(args.n_llm))
            judged = jsonl("judgements.jsonl")
            check("judge ran", rc == 0 and len(judged) > 0, out.strip()[-120:] if rc else "")
            if judged:
                check("scores are within 1-5", all(
                    1 <= r[a] <= 5 for r in judged
                    for a in ("groundedness", "correctness", "tone", "actionability", "safety")))

    stage("evaluation")
    rc, out = sh("run_eval.py", "--detail")
    check("eval ran", rc == 0, out.strip()[-120:] if rc else "")
    # Regression guard: a smoke run must leave the REAL deliverable dir alone.
    # This previously failed - REPORT_DIR ignored ANTHILL_DATA_DIR, so fixture
    # metrics landed in report/ looking like genuine results.
    check("smoke run wrote metrics to the isolated dir",
          (SMOKE / "report" / "metrics.json").exists())
    # The real report/metrics.json is a committed deliverable, so "must not
    # exist" is the wrong invariant. What must hold is that a smoke run leaves
    # it byte-for-byte unchanged: fixture numbers must never reach it.
    check("smoke run left real report/metrics.json untouched",
          _real_metrics_digest() == real_metrics_before)
    if rc == 0:
        check("bootstrap CIs computed", "95% CI" in out)
        check("paired bootstrap ran", "paired bootstrap" in out)
        check("trivial exposed by macro-F1", "trivial" in out)

    # --- summary -------------------------------------------------------------
    passed = sum(1 for _, ok_, _ in _results if ok_)
    total = len(_results)
    print("\n" + "=" * 64)
    print(f"  {passed}/{total} checks passed")
    print("=" * 64)
    if passed < total:
        print("\n  failures:")
        for name, ok_, detail in _results:
            if not ok_:
                print(f"    - {name} {detail}")
        sys.exit(1)
    print("  pipeline is wired correctly end to end.\n")


if __name__ == "__main__":
    main()
