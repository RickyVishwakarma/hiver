"""Stage B.1 - reconstruct conversations for one brand from tweet-level rows.

The raw CSV is flat: each row is a tweet with `in_response_to_tweet_id` (parent)
and `response_tweet_id` (comma-separated children). Conversations must be rebuilt
by walking those pointers.

ALGORITHM (iterative transitive closure, streaming)
  Round 1: scan the file, keep every tweet authored by the brand. Record the
           parent/child ids they reference.
  Round k: re-scan, pulling in records for ids we want but have not seen yet, and
           collect *their* references. Repeat until the frontier is empty.
Typically converges in 3-4 rounds because support threads are short. This is
correct rather than depth-limited, and never holds the full 2.8M-row frame.

EDGE CASES HANDLED
  - fan-out: one tweet with several children (brand replies twice, or a customer
    and a bystander both reply) -> we keep the customer/brand chain, drop others
  - orphans: a parent id that does not exist in the dump (deleted tweet) -> the
    child becomes a thread root
  - threads with no brand reply -> KEPT and flagged (resolved_proxy=False), so
    the golden set can include messages support never answered; excluded from
    the grounding pool because they contain no resolution to learn from
  - multi-customer threads (bystanders chiming in) -> flagged, excluded by default

RESOLUTION IS A PROXY, NOT GROUND TRUTH
`resolved_proxy` is a heuristic. Twitter does not tell us whether the customer's
problem was actually fixed, and threads that end with "DM us" are resolved
somewhere we cannot see. Everything downstream that says "resolved" means
"looks resolved by this heuristic". This limitation is carried into the report.

Usage:
  python tools/build_threads.py --brand AppleSupport
  python tools/build_threads.py --brand Delta --max-threads 20000
"""
from __future__ import annotations

import argparse
import re
from collections import defaultdict

import pandas as pd

from common import DATA, THREADS, TWCS_CSV, die, info, ok, set_seed, write_jsonl
from profile_brands import DM_PATTERNS

CHUNK = 200_000
USECOLS = [
    "tweet_id",
    "author_id",
    "inbound",
    "created_at",
    "text",
    "response_tweet_id",
    "in_response_to_tweet_id",
]
MAX_ROUNDS = 6

# Customer gratitude at the end of a thread is our positive-resolution signal.
THANKS = re.compile(
    r"\b(thanks|thank you|thx|ty|cheers|appreciate it|sorted|fixed|worked|"
    r"resolved|perfect|great help|awesome)\b",
    re.IGNORECASE,
)
# Handle mentions are noise for intent modelling; strip at read time, keep raw too.
MENTION = re.compile(r"@\w+")
URL = re.compile(r"https?://\S+")


def _split_children(val: str) -> list[str]:
    if not val:
        return []
    return [v.strip() for v in val.split(",") if v.strip()]


def clean_text(text: str) -> str:
    """Normalised form used for embedding/classification. Raw text is preserved
    separately - we never want to evaluate on text the model never saw."""
    text = URL.sub(" <url> ", text)
    text = MENTION.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def _chunk_reader():
    """Stream the CSV in fixed-size chunks. Never materialises the full frame."""
    return pd.read_csv(
        TWCS_CSV,
        usecols=USECOLS,
        dtype=str,
        chunksize=CHUNK,
        on_bad_lines="skip",
        engine="c",
    )


def _scan(wanted: set[str] | None, brand: str, collect_brand: bool):
    """One streaming pass. Yields rows that are either authored by `brand`
    (when collect_brand) or whose tweet_id is in `wanted`."""
    for chunk in _chunk_reader():
        chunk = chunk.fillna("")
        if collect_brand:
            mask = chunk["author_id"] == brand
        else:
            mask = chunk["tweet_id"].isin(wanted)
        for rec in chunk[mask].to_dict("records"):
            yield rec


def gather_unanswered(brand: str, tweets: dict[str, dict], cap: int) -> int:
    """Collect inbound tweets that mention the brand but were never replied to.

    WHY THIS PASS EXISTS
    The closure above is seeded from the brand's own tweets, so it can only ever
    reach messages support ANSWERED. Evaluating on those alone is a selection
    bias with a direction we can predict: ignored messages are disproportionately
    off-topic, abusive, duplicated, or unanswerable, i.e. exactly the hard
    routing cases. Without this pass the escalation metrics would be measured on
    a friendlier distribution than production.
    """
    mention = re.compile(rf"@{re.escape(brand)}\b", re.IGNORECASE)
    added = 0
    for chunk in _chunk_reader():
        chunk = chunk.fillna("")
        inbound = chunk[chunk["inbound"].str.lower() == "true"]
        for rec in inbound.to_dict("records"):
            if added >= cap:
                return added
            if rec["tweet_id"] in tweets or not mention.search(rec["text"]):
                continue
            # No child in our set == nobody from the brand ever answered it.
            if any(c in tweets for c in _split_children(rec["response_tweet_id"])):
                continue
            tweets[rec["tweet_id"]] = rec
            added += 1
    return added


def gather(brand: str, include_unanswered: bool = True, unanswered_cap: int = 20000) -> dict[str, dict]:
    """Collect every tweet reachable from the brand's tweets."""
    tweets: dict[str, dict] = {}
    frontier: set[str] = set()

    info(f"round 1: collecting tweets authored by @{brand}...")
    for rec in _scan(None, brand, collect_brand=True):
        tweets[rec["tweet_id"]] = rec
    if not tweets:
        die(f"no tweets found for author_id '{brand}' - check spelling against data/brand_profile.csv")
    info(f"round 1: {len(tweets):,} brand tweets")

    def refs(rec: dict) -> set[str]:
        out = set(_split_children(rec["response_tweet_id"]))
        if rec["in_response_to_tweet_id"]:
            out.add(rec["in_response_to_tweet_id"])
        return out

    for rec in tweets.values():
        frontier |= refs(rec)
    frontier -= tweets.keys()

    for rnd in range(2, MAX_ROUNDS + 1):
        if not frontier:
            break
        info(f"round {rnd}: fetching {len(frontier):,} referenced tweets...")
        new_refs: set[str] = set()
        found = 0
        for rec in _scan(frontier, brand, collect_brand=False):
            tweets[rec["tweet_id"]] = rec
            new_refs |= refs(rec)
            found += 1
        info(f"round {rnd}: found {found:,} (of {len(frontier):,} requested)")
        # Ids still missing are deleted/absent tweets - stop chasing them.
        frontier = new_refs - tweets.keys()

    ok(f"gathered {len(tweets):,} tweets across the brand's conversations")

    if include_unanswered:
        info("scanning for inbound messages the brand never answered...")
        n = gather_unanswered(brand, tweets, unanswered_cap)
        ok(f"added {n:,} unanswered inbound messages (total {len(tweets):,})")
    return tweets


def assemble(tweets: dict[str, dict], brand: str) -> list[dict]:
    """Walk parent->child links into linear threads."""
    children = defaultdict(list)
    for tid, rec in tweets.items():
        parent = rec["in_response_to_tweet_id"]
        if parent:
            children[parent].append(tid)

    # A root is a tweet whose parent is absent from our collection.
    roots = [
        tid
        for tid, rec in tweets.items()
        if not rec["in_response_to_tweet_id"] or rec["in_response_to_tweet_id"] not in tweets
    ]
    info(f"{len(roots):,} thread roots")

    threads = []
    for root in roots:
        chain, cur = [], root
        seen_ids = set()
        while cur and cur in tweets and cur not in seen_ids:
            seen_ids.add(cur)
            chain.append(tweets[cur])
            kids = children.get(cur, [])
            if not kids:
                break
            # Fan-out: prefer continuing with a participant already in the thread
            # (the customer or the brand) rather than a bystander.
            participants = {r["author_id"] for r in chain}
            preferred = [k for k in kids if tweets[k]["author_id"] in participants]
            cur = sorted(preferred or kids)[0]

        # A lone CUSTOMER tweet is kept: the brand never answered it, but a live
        # agent would still be handed that message, so it is a legitimate
        # classification/escalation example. Dropping these would bias the golden
        # set toward messages support chose to answer. It is excluded from the
        # grounding pool below via resolved_proxy, since it contains no reply.
        # A lone BRAND tweet is not a conversation and is dropped.
        if len(chain) < 2 and chain[0]["author_id"] == brand:
            continue

        turns = []
        for rec in chain:
            is_brand = rec["author_id"] == brand
            turns.append(
                {
                    "tweet_id": rec["tweet_id"],
                    "role": "brand" if is_brand else "customer",
                    "author_id": rec["author_id"],
                    "created_at": rec["created_at"],
                    "text_raw": rec["text"],
                    "text": clean_text(rec["text"]),
                }
            )

        customer_turns = [t for t in turns if t["role"] == "customer"]
        brand_turns = [t for t in turns if t["role"] == "brand"]
        if not customer_turns:
            continue

        customer_ids = {t["author_id"] for t in customer_turns}
        brand_text = " ".join(t["text_raw"] for t in brand_turns)
        punted = bool(DM_PATTERNS.search(brand_text))
        last_customer = customer_turns[-1]
        # Gratitude counts only if it comes AFTER a brand reply, otherwise
        # "thanks in advance" in the opening message would score as resolved.
        thanked = bool(
            brand_turns
            and THANKS.search(last_customer["text_raw"])
            and last_customer["tweet_id"] != customer_turns[0]["tweet_id"]
        )

        threads.append(
            {
                "thread_id": chain[0]["tweet_id"],
                "brand": brand,
                "opening_msg": customer_turns[0]["text"],
                "opening_msg_raw": customer_turns[0]["text_raw"],
                "turns": turns,
                "n_turns": len(turns),
                "n_brand_replies": len(brand_turns),
                "first_brand_reply": brand_turns[0]["text_raw"] if brand_turns else None,
                "multi_customer": len(customer_ids) > 1,
                "punted_to_dm": punted,
                "customer_thanked": thanked,
                # Groundable = there IS a visible brand answer that stayed on-platform.
                "resolved_proxy": bool(brand_turns) and not punted,
                "created_at": chain[0]["created_at"],
            }
        )
    return threads


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--brand", required=True, help="Brand author_id, e.g. AppleSupport")
    ap.add_argument("--max-threads", type=int, default=None, help="Cap output size")
    ap.add_argument("--out", default=str(THREADS))
    ap.add_argument(
        "--no-unanswered",
        action="store_true",
        help="Skip the unanswered-mentions pass (biases the set toward answered messages)",
    )
    ap.add_argument("--unanswered-cap", type=int, default=20000)
    args = ap.parse_args()
    set_seed()

    if not TWCS_CSV.exists():
        die(f"{TWCS_CSV} not found. Run: python tools/fetch_dataset.py")

    tweets = gather(
        args.brand,
        include_unanswered=not args.no_unanswered,
        unanswered_cap=args.unanswered_cap,
    )
    threads = assemble(tweets, args.brand)
    if not threads:
        die("no threads assembled - inspect the gathered tweets")

    threads.sort(key=lambda t: t["thread_id"])
    if args.max_threads:
        threads = threads[: args.max_threads]

    n = write_jsonl(args.out, threads)
    ok(f"wrote {n:,} threads -> {args.out}")

    df = pd.DataFrame(
        [
            {k: t[k] for k in ("n_turns", "n_brand_replies", "punted_to_dm", "customer_thanked", "resolved_proxy", "multi_customer")}
            for t in threads
        ]
    )
    print("\n" + "=" * 72)
    print(f"THREAD STATS for @{args.brand}")
    print("=" * 72)
    print(f"  threads                 {len(df):,}")
    print(f"  median turns            {df['n_turns'].median():.0f}")
    print(f"  with >=1 brand reply    {(df['n_brand_replies'] > 0).mean():.1%}")
    print(f"  punted to DM            {df['punted_to_dm'].mean():.1%}   <- resolution invisible")
    print(f"  customer thanked after  {df['customer_thanked'].mean():.1%}   <- weak positive signal")
    print(f"  resolved_proxy (usable) {df['resolved_proxy'].mean():.1%}   <- grounding pool")
    print(f"  multi-customer threads  {df['multi_customer'].mean():.1%}")
    print("\nnext: python tools/embed.py && python tools/induce_intents.py\n")


if __name__ == "__main__":
    main()
