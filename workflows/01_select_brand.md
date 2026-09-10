# Workflow 01 — Select the brand

## Objective

Choose one brand on evidence, and record the numbers behind the choice.

## Why it is not just "pick the biggest"

The brief requires replies *grounded in how the brand has historically resolved
similar issues*. A brand that answers "please DM us" resolves the issue somewhere
this dataset cannot see. Its visible replies are routing messages, not
resolutions. Volume is therefore the wrong primary criterion — **groundable
replies** is the right one.

## Procedure

```bash
python run.py profile          # add --limit 500000 for a fast dev pass
```

Two streaming passes produce `data/brand_profile.csv`:

| Column | Meaning |
|---|---|
| `n_replies` | outbound tweets by the brand — support activity |
| `n_inbound_mentions` | inbound `@brand` tweets — demand |
| `reply_rate` | replies ÷ mentions — how much they actually answer |
| `dm_punt_rate` | share of replies pushing to DMs — **invisible resolutions** |
| `median_reply_chars` | substantive answers vs. one-liners |
| `ascii_frac` | crude English proxy |
| `groundable_replies` | `n_answers_to_customer × (1 − dm_punt_rate)` — **rank on this** |

## Decision criteria, in priority order

1. **`groundable_replies` ≥ ~5,000** — enough precedent for retrieval to find a
   real neighbour rather than a loose topical match.
2. **`dm_punt_rate` low** — below ~0.35 preferred. High punt rate caps the whole
   system's ceiling before any modelling starts.
3. **`ascii_frac` high** — we are not evaluating multilingual handling, and
   silently dropping non-English traffic is a bias we would rather avoid.
4. **`median_reply_chars` moderate** — very short medians mean templated
   acknowledgements with no resolution content.
5. **Intent diversity** — a brand whose traffic is 90% one complaint makes for a
   degenerate taxonomy. Confirm in workflow 03; revisit if clusters collapse.

## Validation

The fixture run is a sanity check that the metric discriminates: two synthetic
brands that punt 100% of the time score `groundable_replies = 0`, while the main
fixture brand (18% punt rate) scores 909. If a real profile shows every brand
scoring similarly, the DM regex is probably not matching that brand's phrasing —
inspect actual replies before trusting the column.

## Output

Record in `DECISIONS.md`: the brand chosen, its row from the table, and the
runner-up with the reason it lost. If the chosen brand has a high punt rate,
state the ceiling that imposes rather than discovering it later in the results.
