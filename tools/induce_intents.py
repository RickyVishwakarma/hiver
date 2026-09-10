"""Stage B.3 - induce a candidate intent taxonomy from the data.

The assignment says intents must be DEFINED FROM THE DATA, so there is no label
set to import. This script does the machine half; a human does the naming half.

  machine: embed opening messages -> cluster -> characterise each cluster with
           distinctive terms + exemplars nearest the centroid
  human:   read the printout, name 8-12 intents, merge/split/drop, write
           data/taxonomy.json

WHY k IS SMALL (8-12 + an explicit `other`)
A 40-cluster taxonomy looks impressive and is unlabellable: annotators cannot
hold it in their head, boundaries blur, inter-annotator agreement collapses, and
macro-F1 becomes noise dominated by 5-example classes. We would rather have 10
classes we can label consistently than 40 we cannot. An explicit `other` bucket
is mandatory - without one, annotators force bad fits into the nearest class and
silently corrupt the golden set.

Cluster naming uses c-TF-IDF (term frequency within a cluster, weighted against
the corpus) which surfaces what makes a cluster DISTINCTIVE rather than just
frequent - otherwise every cluster's top words are "the", "my", "please".

Usage:
  python tools/induce_intents.py --sweep            # pick k
  python tools/induce_intents.py --k 10 --exemplars 12
"""
from __future__ import annotations

import argparse

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

from common import DATA, THREADS, info, ok, read_jsonl, set_seed, write_json, write_jsonl

SWEEP_SAMPLE = 6000  # silhouette is O(n^2); sample it


def cluster_terms(texts: list[str], labels: np.ndarray, k: int, top_n: int = 10) -> dict[int, list[str]]:
    """c-TF-IDF: treat each cluster as one document, find its distinctive terms."""
    docs = []
    for c in range(k):
        docs.append(" ".join(t for t, l in zip(texts, labels) if l == c))
    vec = TfidfVectorizer(
        max_features=8000,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True,
    )
    X = vec.fit_transform(docs)
    vocab = np.array(vec.get_feature_names_out())
    out = {}
    for c in range(k):
        row = X[c].toarray().ravel()
        out[c] = vocab[np.argsort(-row)[:top_n]].tolist()
    return out


def sweep(vecs: np.ndarray, lo: int, hi: int) -> None:
    rng = np.random.default_rng(0)
    idx = rng.choice(len(vecs), size=min(SWEEP_SAMPLE, len(vecs)), replace=False)
    sample = vecs[idx]
    print("\n  k   inertia      silhouette")
    print("  " + "-" * 34)
    for k in range(lo, hi + 1):
        km = KMeans(n_clusters=k, n_init=5, random_state=0).fit(sample)
        sil = silhouette_score(sample, km.labels_, metric="cosine")
        print(f"  {k:<3} {km.inertia_:>10.1f}   {sil:.4f}")
    print(
        "\n  Silhouette on short noisy tweets is always low (~0.03-0.10). Do not\n"
        "  chase the maximum - pick a k whose clusters you can actually NAME.\n"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--threads", default=str(THREADS))
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--sweep", action="store_true", help="Try a range of k and stop")
    ap.add_argument("--sweep-range", default="6,16")
    ap.add_argument("--exemplars", type=int, default=10)
    ap.add_argument("--max-threads", type=int, default=40000, help="Cap clustering input")
    args = ap.parse_args()
    set_seed()

    from embed import load_or_encode  # imported late: needs torch

    rows = read_jsonl(args.threads)
    if not rows:
        raise SystemExit(f"no threads at {args.threads} - run build_threads.py first")

    # Cluster on the customer's OPENING message: that is what the live agent
    # sees. Clustering whole threads would leak the brand's answer into the
    # representation and make the taxonomy easier than the real task.
    rows = rows[: args.max_threads]
    texts = [r["opening_msg"] for r in rows]
    vecs = load_or_encode(texts, tag="opening_msg")

    if args.sweep:
        lo, hi = (int(x) for x in args.sweep_range.split(","))
        sweep(vecs, lo, hi)
        return

    info(f"clustering {len(vecs):,} openings into k={args.k}...")
    km = KMeans(n_clusters=args.k, n_init=10, random_state=0).fit(vecs)
    labels = km.labels_
    terms = cluster_terms(texts, labels, args.k)

    # Distance to centroid, for picking representative examples.
    sims = vecs @ km.cluster_centers_.T
    scaffold = {"brand": rows[0].get("brand", "?"), "k": args.k, "intents": []}

    print("\n" + "=" * 78)
    print(f"CANDIDATE CLUSTERS (k={args.k}) - name these yourself")
    print("=" * 78)
    order = np.argsort(-np.bincount(labels, minlength=args.k))
    for c in order:
        members = np.where(labels == c)[0]
        share = len(members) / len(labels)
        ranked = members[np.argsort(-sims[members, c])]
        print(f"\n[cluster {c}]  n={len(members):,}  ({share:.1%})")
        print(f"  distinctive terms: {', '.join(terms[c])}")
        print("  exemplars nearest centroid:")
        for i in ranked[: args.exemplars]:
            print(f"    - {texts[i][:150]}")
        scaffold["intents"].append(
            {
                "cluster_id": int(c),
                "name": f"TODO_name_cluster_{c}",
                "description": "TODO one line: what customer need does this represent?",
                "share": round(float(share), 4),
                "terms": terms[c],
                "examples": [texts[i][:200] for i in ranked[:5]],
            }
        )

    scaffold["intents"].append(
        {
            "cluster_id": -1,
            "name": "other",
            "description": "Anything that does not fit a defined intent. Mandatory bucket - "
            "prevents annotators from forcing bad fits into the nearest class.",
            "share": 0.0,
            "terms": [],
            "examples": [],
        }
    )

    out = DATA / "taxonomy_draft.json"
    write_json(out, scaffold)
    ok(f"wrote draft scaffold -> {out}")

    # Persist assignments so the sampler can stratify without re-clustering.
    # max_sim is the confidence that a thread belongs to its cluster at all -
    # low values are the ambiguous boundary cases the golden set must include.
    assign = [
        {
            "thread_id": r["thread_id"],
            "cluster": int(labels[i]),
            "max_sim": round(float(sims[i, labels[i]]), 4),
        }
        for i, r in enumerate(rows)
    ]
    write_jsonl(DATA / "cluster_assignments.jsonl", assign)
    ok(f"wrote cluster assignments -> {DATA / 'cluster_assignments.jsonl'} ({len(assign):,})")
    print(
        "\nNEXT (manual, ~30 min):\n"
        f"  1. Read the clusters above.\n"
        f"  2. Edit {out}: replace every TODO_name with a real intent name, merge\n"
        "     duplicates, split anything doing two jobs, drop noise clusters.\n"
        "  3. Save it as data/taxonomy.json\n"
        "  4. Record what you merged/split/dropped - that goes in DECISIONS.md.\n"
    )


if __name__ == "__main__":
    main()
