"""Stage A.2 - profile every brand so the brand choice is made on evidence.

Two streaming passes over twcs.csv (never loads the full frame; 8GB RAM safe):

  Pass 1  For each brand (= non-numeric author_id on outbound tweets):
            - n_replies            how much support traffic they actually handle
            - dm_punt_rate         % of replies that push the user to DMs
            - median_reply_chars   substantive answers vs. one-liners
            - ascii_frac           crude English proxy
            - n_answers_to_customer  replies that link back to a customer tweet

  Pass 2  Inbound mentions per brand handle (@Brand in customer tweet text),
          giving a reply-rate denominator.

WHY dm_punt_rate IS THE DECIDING COLUMN
Brands that answer "please DM us" resolve the issue off-platform. Those
resolutions are NOT in this dataset. Since the assignment requires replies
"grounded in how that brand has historically resolved similar issues", a brand
with a high punt rate has no observable resolutions to ground in - the ceiling
on the whole system is set here, before any modelling.

Usage:
  python tools/profile_brands.py                # full file, both passes
  python tools/profile_brands.py --limit 500000 # quick look while developing
"""
from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from common import BRAND_PROFILE, TWCS_CSV, die, info, ok, set_seed

CHUNK = 200_000
USECOLS = [
    "tweet_id",
    "author_id",
    "inbound",
    "text",
    "response_tweet_id",
    "in_response_to_tweet_id",
]

# "Contact us privately" phrasing. Kept explicit and readable rather than one
# clever regex, because we have to justify every classification decision.
DM_PATTERNS = re.compile(
    r"\b(dm|dms|d\.m\.|direct message|private message|inbox us|"
    r"send us a (?:dm|private|direct)|pm us|message us privately)\b",
    re.IGNORECASE,
)


def _is_brand(author_id: str) -> bool:
    """Customers are anonymised to numeric ids; brands keep their handle."""
    return not author_id.isdigit()


def _ascii_frac(s: str) -> float:
    if not s:
        return 1.0
    return sum(1 for ch in s if ord(ch) < 128) / len(s)


def _chunks(limit: int | None):
    reader = pd.read_csv(
        TWCS_CSV,
        usecols=USECOLS,
        dtype=str,
        chunksize=CHUNK,
        on_bad_lines="skip",
        engine="c",
    )
    seen = 0
    for chunk in reader:
        yield chunk
        seen += len(chunk)
        if limit and seen >= limit:
            return


def pass1(limit: int | None) -> pd.DataFrame:
    replies = Counter()
    dm_punts = Counter()
    answers_to_customer = Counter()
    reply_chars: dict[str, list[int]] = defaultdict(list)
    ascii_sum: dict[str, float] = defaultdict(float)

    rows = 0
    for chunk in _chunks(limit):
        chunk = chunk.fillna("")
        # inbound is "False" for brand replies (string, since we read everything as str)
        outbound = chunk[chunk["inbound"].str.lower() == "false"]
        for author_id, text, in_reply_to in zip(
            outbound["author_id"], outbound["text"], outbound["in_response_to_tweet_id"]
        ):
            if not _is_brand(author_id):
                continue
            replies[author_id] += 1
            if DM_PATTERNS.search(text):
                dm_punts[author_id] += 1
            if in_reply_to:
                answers_to_customer[author_id] += 1
            # Reservoir-free: keep lengths only for the first 5k replies per brand
            # so memory stays flat regardless of brand size.
            if len(reply_chars[author_id]) < 5000:
                reply_chars[author_id].append(len(text))
            ascii_sum[author_id] += _ascii_frac(text)
        rows += len(chunk)
        if rows % 1_000_000 == 0:
            info(f"pass 1: {rows:,} rows scanned...")

    info(f"pass 1 complete: {rows:,} rows, {len(replies)} brands found")

    recs = []
    for brand, n in replies.items():
        lengths = sorted(reply_chars[brand])
        median = lengths[len(lengths) // 2] if lengths else 0
        recs.append(
            {
                "brand": brand,
                "n_replies": n,
                "dm_punt_rate": round(dm_punts[brand] / n, 4),
                "median_reply_chars": median,
                "ascii_frac": round(ascii_sum[brand] / n, 4),
                "n_answers_to_customer": answers_to_customer[brand],
            }
        )
    return pd.DataFrame(recs)


def pass2(handles: list[str], limit: int | None) -> Counter:
    """Count inbound customer tweets mentioning each brand handle."""
    # One compiled alternation over all handles is far faster than N searches.
    # Sorted longest-first so e.g. @AmazonHelp wins over a hypothetical @Amazon.
    ordered = sorted(handles, key=len, reverse=True)
    pattern = re.compile(
        r"@(" + "|".join(re.escape(h) for h in ordered) + r")\b", re.IGNORECASE
    )
    lookup = {h.lower(): h for h in handles}
    mentions = Counter()

    rows = 0
    for chunk in _chunks(limit):
        chunk = chunk.fillna("")
        inbound = chunk[chunk["inbound"].str.lower() == "true"]
        for text in inbound["text"]:
            for m in set(pattern.findall(text)):
                canonical = lookup.get(m.lower())
                if canonical:
                    mentions[canonical] += 1
        rows += len(chunk)
        if rows % 1_000_000 == 0:
            info(f"pass 2: {rows:,} rows scanned...")
    info(f"pass 2 complete: {rows:,} rows")
    return mentions


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=None, help="Stop after N rows (dev only)")
    ap.add_argument("--top", type=int, default=25, help="How many brands to print")
    args = ap.parse_args()
    set_seed()

    if not TWCS_CSV.exists():
        die(f"{TWCS_CSV} not found. Run: python tools/fetch_dataset.py")

    df = pass1(args.limit)
    if df.empty:
        die("no brands found - check the inbound column encoding")

    # Only bother counting mentions for brands with real volume.
    handles = df.nlargest(60, "n_replies")["brand"].tolist()
    mentions = pass2(handles, args.limit)
    df["n_inbound_mentions"] = df["brand"].map(mentions).fillna(0).astype(int)
    # Only the top-60 handles get a mention count (pass 2), so brands outside
    # that set have 0 mentions and an undefined reply rate. Use np.nan rather
    # than pd.NA: pandas 3.0 refuses to cast NAType to float, which crashed the
    # first real run. NaN divides and formats cleanly.
    denom = df["n_inbound_mentions"].astype(float).replace(0.0, np.nan)
    df["reply_rate"] = (df["n_replies"] / denom).round(3)

    # groundable_replies = replies that answer a customer AND don't punt to DMs.
    # This is the pool a retrieval-grounded drafter can actually learn from.
    df["groundable_replies"] = (
        df["n_answers_to_customer"] * (1 - df["dm_punt_rate"])
    ).astype(int)

    df = df.sort_values("groundable_replies", ascending=False).reset_index(drop=True)
    df.to_csv(BRAND_PROFILE, index=False)
    ok(f"wrote {BRAND_PROFILE}")

    cols = [
        "brand",
        "n_replies",
        "n_inbound_mentions",
        "reply_rate",
        "dm_punt_rate",
        "median_reply_chars",
        "ascii_frac",
        "groundable_replies",
    ]
    print("\n" + "=" * 100)
    print("BRAND PROFILE - ranked by groundable_replies (the pool we can ground drafts in)")
    print("=" * 100)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(df[cols].head(args.top).to_string(index=False))
    print(
        "\nRead this as: high groundable_replies + low dm_punt_rate + high ascii_frac\n"
        "= a brand whose resolutions are actually visible in the data.\n"
    )


if __name__ == "__main__":
    main()
