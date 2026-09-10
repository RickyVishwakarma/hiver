# Workflow 03 — Induce the intent taxonomy

## Objective

Define 8–12 intents *from the data*, plus a mandatory `other` bucket.

## Procedure

```bash
python run.py induce -- --sweep          # inspect k
python run.py induce --k 10 --exemplars 12
```

The machine clusters and characterises; **a human names**. Clustering on the
customer's *opening message only* — clustering whole threads would leak the
brand's answer into the representation and make the taxonomy easier than the
live task, where only the opening message exists.

Cluster naming uses **c-TF-IDF**: each cluster is treated as one document and
scored against the corpus, surfacing what makes it *distinctive* rather than
merely frequent. Without this, every cluster's top terms are "the", "my", "please".

## Choosing k

Silhouette on short noisy tweets is always low (~0.03–0.10). **Do not chase the
maximum.** Pick a k whose clusters you can actually name and label consistently.

A 40-cluster taxonomy looks sophisticated and is unlabellable: annotators cannot
hold it in working memory, boundaries blur, agreement collapses, and macro-F1
becomes noise dominated by 5-example classes. Ten classes labelled consistently
beat forty labelled arbitrarily.

## The manual step (~30 min)

Read the printed clusters, then edit `data/taxonomy_draft.json` → save as
`data/taxonomy.json`:

- Replace every `TODO_name_cluster_N` with a real intent name.
- **Merge** clusters that route to the same resolution. Splitting on *phrasing*
  rather than *resolution path* is the most common error — e.g. the fixture run
  produced separate clusters for "charged twice" and "took £X twice", which are
  one billing intent.
- **Split** any cluster doing two jobs.
- **Drop** noise clusters; their members fall to `other`.
- Keep the `other` entry. Without it annotators force bad fits into the nearest
  class and corrupt the set invisibly.

Name intents after **what the support team would do**, not what the customer
mentions. Intents exist to route work.

## Record for `DECISIONS.md`

Final k and why; what was merged, split, dropped, and the reasoning; any cluster
that looked coherent to the algorithm but not to you.

## Outputs

- `data/taxonomy.json` — the label set (committed)
- `data/cluster_assignments.jsonl` — thread → cluster + `max_sim`, used by the
  sampler to stratify and to find the ambiguous boundary cases

## Note on Banking77

Not used. It is retail-banking intent data; the chosen brand is not a bank, and
its 77 labels describe a different resolution structure. Importing them would
produce a taxonomy that fits neither dataset. Recorded under "what I chose not to
build" rather than bolted on for the sake of using an optional resource.
