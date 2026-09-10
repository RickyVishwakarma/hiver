"""Stage D.2 - two baselines the LLM agent must beat to justify its cost.

BASELINE 1 - "trivial" (the floor)
  intent:     always predict the majority class
  reply:      one fixed canned response for everything
  escalation: never escalate
Exists to expose metric inflation. On an imbalanced set this scores a
surprisingly high ACCURACY while having macro-F1 near zero and escalation recall
of exactly zero. If a fancy system only modestly beats this on accuracy, the
accuracy number was never measuring what we thought.

BASELINE 2 - "simple" (the real bar)
  intent:     TF-IDF + logistic regression, trained on the golden labels
  reply:      copy the brand's historical reply from the nearest neighbour,
              verbatim - no generation at all
  escalation: keyword rules only (the same RULES the agent uses, minus the LLM)
This is the honest competitor. It is ~1000x cheaper and needs no GPU. The
verbatim-copy replier is a strong baseline precisely because a real past reply
from this brand is guaranteed on-brand and factually grounded - it just may
answer the wrong question.

The simple classifier is evaluated with STRATIFIED 5-FOLD CROSS-VALIDATION on
the golden set, because it has no other training data. Reporting its training
accuracy would be a lie, and giving it a train split the LLM never saw would
compare two different quantities. Cross-validated out-of-fold predictions are
the fair comparison: every prediction comes from a model that did not see that
example.

Usage:
  python tools/baselines.py --all
"""
from __future__ import annotations

import argparse
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline

from common import DATA, GOLDEN, THREADS, die, info, ok, read_jsonl, set_seed, write_jsonl

CANNED = (
    "Sorry to hear about this! Please DM us with a few more details and "
    "we'll be happy to look into it for you."
)


def trivial(golden: list[dict]) -> list[dict]:
    majority = Counter(g["intent"] for g in golden).most_common(1)[0][0]
    info(f"trivial baseline: majority class = '{majority}'")
    return [
        {
            "golden_id": g["golden_id"],
            "system": "trivial",
            "intent": majority,
            "reply": CANNED,
            "escalate": False,
            "escalation_reason": "",
            "decided_by": "constant",
            "retrieval_score": 0.0,
        }
        for g in golden
    ]


def simple(golden: list[dict]) -> list[dict]:
    from agent import RULES

    texts = [g["customer_message"] for g in golden]
    y = np.array([g["intent"] for g in golden])

    # --- intent: cross-validated, so no example is scored by a model that saw it
    counts = Counter(y)
    n_splits = min(5, min(counts.values()))
    if n_splits < 2:
        die(f"a class has only {min(counts.values())} example(s); cannot cross-validate")
    if n_splits < 5:
        info(f"using {n_splits}-fold (smallest class has {min(counts.values())} examples)")

    preds = np.empty(len(y), dtype=object)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    for train_idx, test_idx in skf.split(texts, y):
        pipe = make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True, stop_words="english"),
            LogisticRegression(max_iter=2000, class_weight="balanced", C=2.0),
        )
        pipe.fit([texts[i] for i in train_idx], y[train_idx])
        preds[test_idx] = pipe.predict([texts[i] for i in test_idx])

    # --- reply: verbatim nearest historical reply --------------------------
    from embed import encode, load_or_encode

    threads = read_jsonl(THREADS)
    pool = [t for t in threads if t.get("resolved_proxy") and t.get("first_brand_reply")]
    pool_vecs = load_or_encode([t["opening_msg"] for t in pool], tag="pool_openings")
    q = encode(texts)
    sims = q @ pool_vecs.T
    nearest = sims.argmax(axis=1)

    rows = []
    for i, g in enumerate(golden):
        msg = g["customer_message"]
        esc, reason = False, ""
        for code, pattern in RULES:
            if pattern.search(msg):
                esc, reason = True, code
                break
        rows.append(
            {
                "golden_id": g["golden_id"],
                "system": "simple",
                "intent": str(preds[i]),
                "reply": pool[nearest[i]]["first_brand_reply"],
                "escalate": esc,
                "escalation_reason": reason,
                "decided_by": f"rule:{reason}" if esc else "rule:none_matched",
                "retrieval_score": round(float(sims[i, nearest[i]]), 4),
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--trivial", action="store_true")
    ap.add_argument("--simple", action="store_true")
    args = ap.parse_args()
    set_seed()

    golden = read_jsonl(GOLDEN)
    if not golden:
        die(f"{GOLDEN} is empty - label the golden set first")
    info(f"golden set: {len(golden)} examples, {len(set(g['intent'] for g in golden))} intents")

    if args.all or args.trivial:
        rows = trivial(golden)
        write_jsonl(DATA / "preds_trivial.jsonl", rows)
        ok(f"wrote {len(rows)} -> preds_trivial.jsonl")

    if args.all or args.simple:
        rows = simple(golden)
        write_jsonl(DATA / "preds_simple.jsonl", rows)
        ok(f"wrote {len(rows)} -> preds_simple.jsonl")

    if not (args.all or args.trivial or args.simple):
        die("pass --all, --trivial or --simple")


if __name__ == "__main__":
    main()
