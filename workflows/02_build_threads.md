# Workflow 02 — Reconstruct conversations

## Objective

Turn flat tweet rows into conversations, without loading 2.8M rows into 8GB RAM.

## Procedure

```bash
python run.py threads --brand <Brand>
```

## How it works

The CSV is flat: each row carries `in_response_to_tweet_id` (parent) and
`response_tweet_id` (comma-separated children). `tools/build_threads.py` walks
those pointers by **iterative transitive closure**:

1. Stream the file, keep every tweet authored by the brand, note the ids they reference.
2. Re-scan, pull records for referenced ids not yet seen, collect *their* references.
3. Repeat to a fixed point (3–4 rounds typical). Ids that never resolve are
   deleted tweets — stop chasing them.
4. **Extra pass:** collect inbound `@brand` mentions that were never answered.

Step 4 exists because steps 1–3 are seeded from the brand's own tweets and can
therefore only reach messages support *answered*. Evaluating on those alone is a
selection bias with a predictable direction: ignored messages are
disproportionately off-topic, abusive, duplicated, or unanswerable — exactly the
hard routing cases. Disable with `--no-unanswered` only to demonstrate the bias.

## Edge cases

| Case | Handling |
|---|---|
| Fan-out (several children) | Continue with a participant already in the thread; bystanders dropped |
| Orphan parent (deleted tweet) | Child becomes a thread root |
| No brand reply | **Kept**, `resolved_proxy=False` — labelled, but excluded from grounding |
| Lone brand tweet | Dropped (not a conversation) |
| Multi-customer pile-on | Flagged `multi_customer`, excluded from sampling |
| Cycles in reply pointers | Guarded by a `seen_ids` set during the walk |

## `resolved_proxy` is a heuristic, not truth

Defined as: the brand replied publicly **and** did not punt to DMs. It does *not*
mean the customer's problem was fixed — Twitter never tells us that. A separate
`customer_thanked` flag (gratitude appearing *after* a brand reply, so "thanks in
advance" in an opening message does not count) is a weak positive signal, not
confirmation.

Everything downstream that says "resolved" means "looks resolved by this
heuristic". This is one of the report's headline caveats.

## Sanity checks

Compare against the profile from workflow 01 — `punted_to_dm` here should track
`dm_punt_rate` there. Then read five raw threads by hand and confirm the
reconstruction matches. Automated closure is easy to get subtly wrong in ways no
aggregate reveals.

## Output

`data/threads.jsonl` — one object per thread with `opening_msg`, ordered `turns`,
`first_brand_reply`, and the flags above. Printed stats show turn distribution,
punt rate, and grounding-pool size.
