# Workflow 00 — Pipeline overview

## Objective

Turn ~2.8M raw support tweets into a trustworthy single-brand support agent, and
produce the evidence needed to decide whether to trust it.

## The nine stages

| # | Stage | Tool | Manual? | Output |
|---|---|---|---|---|
| 1 | Acquire | `fetch_dataset.py` | – | `.tmp/raw/twcs.csv` |
| 2 | Choose brand | `profile_brands.py` | decide | `data/brand_profile.csv` |
| 3 | Reconstruct threads | `build_threads.py` | – | `data/threads.jsonl` |
| 4 | Induce intents | `induce_intents.py` | **name clusters** | `data/taxonomy.json` |
| 5 | Sample golden set | `sample_golden.py` | – | `data/golden_unlabelled.jsonl` |
| 6 | Label | `label_cli.py` | **~3 h** | `data/golden.jsonl` |
| 7 | Run agent + baselines | `agent.py`, `baselines.py` | – | `data/preds_*.jsonl` |
| 8 | Judge + human scores | `judge.py`, `score_replies.py` | **~45 min** | `data/judgements.jsonl` |
| 9 | Evaluate + validate | `run_eval.py`, `validate_judge.py` | – | `report/*.json` |

Three manual steps. They are the deliverable's backbone, not friction to
optimise away: automating any of them would mean measuring the system against
itself. See workflow 04.

## Two execution paths

**Reproduction (graders):** `python run.py reproduce` replays the committed
`data/llm_cache_bundle.jsonl` with `ANTHILL_CACHE_ONLY=1`, so a cache miss is a
hard error rather than a silent live call. No GPU, no Ollama, no keys. ~2 min.

**Full regeneration:** `python run.py all <Brand>` plus the manual steps. ~60 min
of compute on a 4GB laptop GPU, plus ~4 h of human labelling.

## Verify without any download

```bash
python run.py smoke
```

Runs every stage against a synthetic fixture that reproduces the real dump's
structural mess, asserting 34 properties. Isolated via `ANTHILL_DATA_DIR` so it
cannot touch real artifacts.

## Operating notes learned while building

- **Memory.** The full CSV must never be loaded. All ingest streams in 200k-row
  chunks; 8GB RAM cannot hold the frame plus a working copy.
- **Thread reconstruction needs iteration, not a fixed depth.** Follow reply
  pointers to a fixed point (3–4 rounds typical). Ids that stay unresolved are
  deleted tweets — stop chasing them.
- **Brand-seeded collection has a blind spot.** Walking out from the brand's own
  tweets can only reach messages support *answered*. A separate pass collects
  unanswered `@brand` mentions, or evaluation runs on a friendlier distribution
  than production.
- **Console encoding.** Windows consoles are cp1252 and raise on non-ASCII;
  `common.setup_console()` forces UTF-8 everywhere.
- **Small models mangle JSON.** `llm.generate_json` repairs fenced/trailing-comma
  output and returns `{}` on failure. Parse-failure *rate* is reported as a real
  finding rather than hidden — it is a genuine property of 3B models.
- **Ollama holds one model at a time** on 4GB VRAM. Generator and judge run in
  separate passes; do not interleave them.

## Where the real risk lives

Not in the agent — it is a conventional classify/retrieve/route design. The risk
is that the *evidence* is weaker than it looks:

- the judge is a 3B model, so its agreement with a human must be measured before
  any reply-quality claim is made (workflow 07);
- the golden set has one annotator, so consistency ≠ correctness;
- `resolved_proxy` is a heuristic — Twitter never tells us whether the customer's
  problem was actually fixed, and DM-punted threads resolve where we cannot see.

Each is quantified where possible and stated plainly where not.
