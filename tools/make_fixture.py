"""Generate a synthetic twcs.csv so the pipeline can be tested without the 500MB
Kaggle download.

The fixture deliberately reproduces the messy structure of the real dump - that
is the point. If the pipeline only works on clean data it does not work.
  - reply chains via in_response_to_tweet_id / response_tweet_id
  - fan-out (two children on one parent, one of them a bystander)
  - orphans (a parent id that does not exist in the file)
  - DM punts, so punted_to_dm is exercised
  - threads with no brand reply
  - several distinct intent groups, so clustering has something to find
  - escalation-signal language, so those strata are non-empty

This is NOT training or evaluation data. It exists only to prove the plumbing
works end to end. Nothing produced from it appears in the report.
"""
from __future__ import annotations

import argparse
import csv
import random
from datetime import datetime, timedelta

from common import TWCS_CSV, ok, set_seed

BRAND = "FixtureSupport"

# (customer templates, brand reply templates) per synthetic intent group
GROUPS = [
    (
        ["my {device} wont charge since the {v} update", "battery drains so fast after {v}",
         "{device} battery dying in 2 hours since update", "power issues after installing {v}"],
        ["Sorry to hear that! Try a forced restart, then check Settings > Battery for any app using excess power. Let us know how you get on.",
         "That's not expected. Please update to the latest version and reset settings, then tell us if it persists."],
    ),
    (
        ["where is my order {n}", "my package hasnt arrived, order {n}", "order {n} says delivered but nothing here",
         "tracking hasnt updated for order {n} in days"],
        ["We're sorry for the delay. Tracking can lag 24-48h after dispatch. If it hasn't moved by tomorrow we'll open an investigation.",
         "Apologies for that. Please confirm the delivery postcode and we'll chase the courier for you."],
    ),
    (
        ["i was charged twice for my subscription", "you took {n} pounds twice this month",
         "double charged again, i want a refund", "billed twice and no one is helping"],
        ["That shouldn't happen. Duplicate authorisations usually drop off within 3-5 working days, but we'll check the account for you.",
         "Sorry about this. We can look into the duplicate charge - our billing team will need to verify a few details."],
    ),
    (
        ["cant log into my account", "password reset email never arrives", "locked out of my account again",
         "login keeps failing with correct password"],
        ["Let's get you back in. Check your spam folder for the reset email, and make sure the address matches the one on the account.",
         "Sorry for the trouble. Try clearing your browser cache, then request a new reset link - the old one expires after 30 minutes."],
    ),
    (
        ["my account has been hacked someone changed my email", "unauthorised transactions on my account",
         "someone accessed my account and ordered stuff", "i think my account was compromised, fraud"],
        ["We take this seriously. Please change your password immediately and enable two-factor authentication.",
         "That's concerning. We'll need to secure the account right away - our security team will review the recent activity."],
    ),
    (
        ["your service is appalling i want to speak to a manager", "this is unacceptable, im contacting my lawyer",
         "worst company ever, filing a formal complaint", "i will take legal action if this isnt resolved"],
        ["We're sorry you've had this experience and we'd like to put it right. We're escalating this to our complaints team.",
         "That's not the standard we aim for. A senior member of the team will review your case directly."],
    ),
    (
        ["how do i change my delivery address", "can i update the email on my account",
         "how do i cancel auto renewal", "where do i find my invoices"],
        ["You can update that under Account > Settings > Details. Changes apply to future orders only.",
         "Head to Account > Billing to manage that. Let us know if the option isn't showing for you."],
    ),
]

DM_PUNTS = [
    "Sorry to hear that! Please DM us your account details and we'll take a look.",
    "We'd like to help - could you send us a DM with your order number?",
]
DEVICES = ["phone", "laptop", "tablet", "watch"]
VERSIONS = ["v14.2", "the latest update", "v15", "the new release"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--threads", type=int, default=1200)
    ap.add_argument("--out", default=str(TWCS_CSV))
    args = ap.parse_args()
    set_seed()
    rng = random.Random(20260909)

    rows: list[dict] = []
    tid = 1000
    t0 = datetime(2024, 3, 1, 9, 0, 0)

    def stamp(offset_min: int) -> str:
        return (t0 + timedelta(minutes=offset_min)).strftime("%a %b %d %H:%M:%S +0000 %Y")

    for i in range(args.threads):
        group = rng.choice(GROUPS)
        cust_text = rng.choice(group[0]).format(
            device=rng.choice(DEVICES), v=rng.choice(VERSIONS), n=rng.randint(1000, 99999)
        )
        customer_id = str(rng.randint(100000, 999999))

        tid += 1
        cust_tid = tid
        # 8% of threads get no brand reply at all
        no_reply = rng.random() < 0.08
        # 18% are punted to DMs
        punt = rng.random() < 0.18
        # 6% reference a parent that does not exist in the file (orphan)
        orphan_parent = str(tid - 5000) if rng.random() < 0.06 else ""

        if no_reply:
            rows.append({
                "tweet_id": cust_tid, "author_id": customer_id, "inbound": "True",
                "created_at": stamp(i * 7), "text": f"@{BRAND} {cust_text}",
                "response_tweet_id": "", "in_response_to_tweet_id": orphan_parent,
            })
            continue

        tid += 1
        brand_tid = tid
        brand_text = rng.choice(DM_PUNTS) if punt else rng.choice(group[1])

        children = [str(brand_tid)]
        # 12% fan-out: a bystander also replies to the customer
        bystander_tid = None
        if rng.random() < 0.12:
            tid += 1
            bystander_tid = tid
            children.append(str(bystander_tid))

        rows.append({
            "tweet_id": cust_tid, "author_id": customer_id, "inbound": "True",
            "created_at": stamp(i * 7), "text": f"@{BRAND} {cust_text}",
            "response_tweet_id": ",".join(children), "in_response_to_tweet_id": orphan_parent,
        })
        rows.append({
            "tweet_id": brand_tid, "author_id": BRAND, "inbound": "False",
            "created_at": stamp(i * 7 + 3), "text": f"@{customer_id} {brand_text}",
            "response_tweet_id": "", "in_response_to_tweet_id": str(cust_tid),
        })
        if bystander_tid:
            rows.append({
                "tweet_id": bystander_tid, "author_id": str(rng.randint(100000, 999999)),
                "inbound": "True", "created_at": stamp(i * 7 + 4),
                "text": f"@{BRAND} same thing happens to me all the time",
                "response_tweet_id": "", "in_response_to_tweet_id": str(cust_tid),
            })

        # 35% get a customer follow-up, sometimes grateful (resolution signal)
        if rng.random() < 0.35:
            tid += 1
            follow = tid
            grateful = rng.random() < 0.55
            text = rng.choice(["thanks that worked perfectly", "thank you, sorted now", "cheers, fixed it"]) \
                if grateful else rng.choice(["that didnt help at all", "still not working", "no thats not the issue"])
            rows[-1 if not bystander_tid else -2]["response_tweet_id"] = str(follow)
            rows.append({
                "tweet_id": follow, "author_id": customer_id, "inbound": "True",
                "created_at": stamp(i * 7 + 20), "text": f"@{BRAND} {text}",
                "response_tweet_id": "", "in_response_to_tweet_id": str(brand_tid),
            })

    # Noise from other brands, so brand filtering is actually exercised.
    for other in ("OtherCo", "ThirdBrand"):
        for _ in range(150):
            tid += 1
            cid = str(rng.randint(100000, 999999))
            rows.append({
                "tweet_id": tid, "author_id": cid, "inbound": "True",
                "created_at": stamp(0), "text": f"@{other} something unrelated is broken",
                "response_tweet_id": str(tid + 1), "in_response_to_tweet_id": "",
            })
            tid += 1
            rows.append({
                "tweet_id": tid, "author_id": other, "inbound": "False",
                "created_at": stamp(2), "text": f"@{cid} Sorry about that, please DM us.",
                "response_tweet_id": "", "in_response_to_tweet_id": str(tid - 1),
            })

    rng.shuffle(rows)  # the real file is not ordered by thread
    out = args.out
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "tweet_id", "author_id", "inbound", "created_at", "text",
            "response_tweet_id", "in_response_to_tweet_id"])
        w.writeheader()
        w.writerows(rows)

    ok(f"wrote {len(rows):,} synthetic tweets -> {out}")
    print(f"    brand handle: {BRAND}")


if __name__ == "__main__":
    main()
