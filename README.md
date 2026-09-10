# Twitter Support Agent — intent, grounded replies, and escalation

An AI customer-support agent for a single brand, built from the
[Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
dataset, together with the evaluation needed to decide whether to trust it.

The system is the smaller half of this repo. The larger half is the evidence:
a hand-labelled golden set, a validated LLM judge, two baselines, and an explicit
account of where the headline number misleads.

---

## Reproduce the headline results (measured: 56 seconds)

```bash
pip install -r requirements.txt
python run.py reproduce
```

That is the whole grader path. Verified from a clean `git clone`: **56 seconds**,
producing bit-identical macro-F1 values and confidence intervals. It needs **no
GPU, no Ollama, no API key, and no Kaggle account** — it replays committed model outputs and recomputes every metric,
confidence interval, and judge-validation statistic from scratch.

Why replay rather than regenerate: the models here are local 3B models on a 4GB
laptop GPU. Producing ~600 generations plus judgements takes 45-60 minutes, which
cannot fit in a 15-minute budget. Every model call is content-addressed by
`sha256(model + prompt + options)` and committed in `data/llm_cache_bundle.jsonl`,
so replay is exact rather than approximate. `ANTHILL_CACHE_ONLY=1` makes a cache
miss a hard error, so the reproduction path can never silently drift by calling a
model.

### Verify the pipeline itself, with no download

```bash
python run.py smoke
```

Generates a synthetic dataset that reproduces the real dump's structural mess
(reply-pointer chains, fan-out, orphaned parents, DM punts, unanswered mentions),
runs all nine stages against it, and asserts 34 properties. Isolated in
`.tmp/smoke/` — it cannot touch the real golden set.

---

## Regenerate everything from raw data

```bash
pip install -r requirements-full.txt
cp .env.example .env          # add Kaggle credentials
python run.py fetch           # or: python tools/fetch_dataset.py --zip <archive.zip>
python run.py profile         # rank brands on evidence; pick one
python run.py threads --brand <Brand>
python run.py induce --k 10
#   MANUAL: name the clusters in data/taxonomy_draft.json -> data/taxonomy.json
python run.py sample --n 200
#   MANUAL: python run.py label            (~2-3 h)
#   MANUAL: python run.py label --pass2    (~30 min, after a break)
python run.py generate        # agent + baselines
python run.py judge
#   MANUAL: python run.py score            (~45 min, blind)
python run.py eval
python run.py validate
python run.py freeze          # export the cache for committing
```

The three manual steps are the point, not an oversight — see *Golden set* below.

---

## Architecture

Three layers, following the repo's WAT convention: `workflows/` are the SOPs,
`tools/` are deterministic scripts, and the agent coordinates.

```
tools/
  fetch_dataset.py    download (Kaggle API or a manually downloaded zip)
  profile_brands.py   rank brands: volume, reply rate, DM-punt rate, groundability
  build_threads.py    tweet rows -> conversations (streaming transitive closure)
  embed.py            MiniLM mean-pooling encoder, cached
  induce_intents.py   cluster + c-TF-IDF characterisation -> candidate taxonomy
  sample_golden.py    stratified draw for labelling
  label_cli.py        blind labelling UI (+ --pass2 for self-agreement)
  agent.py            classify -> retrieve+draft -> route
  baselines.py        trivial and simple baselines
  judge.py            LLM-as-judge, 5-axis rubric, different model family
  score_replies.py    blind human scoring UI
  run_eval.py         metrics, bootstrap CIs, paired bootstrap
  validate_judge.py   judge-vs-human agreement, human ceiling, bias probes
  smoke_test.py       end-to-end test on a synthetic fixture
```

### The agent

1. **Classify** — LLM picks an intent from the induced taxonomy. It also
   self-reports a confidence, which is recorded but *not* trusted for routing:
   3B models say 0.9 for everything.
2. **Draft** — retrieve the *k* nearest historically-resolved threads for this
   brand and put the real customer message and real brand reply in the prompt.
   That is what "grounded in how the brand resolved similar issues" means here:
   the model is shown precedent rather than asked to invent policy.
3. **Route** — hybrid. Deterministic rules fire first and short-circuit (safety,
   account compromise, legal, refund/policy); the LLM adjudicates only what
   survives. A missed escalation is the expensive error and a regex is far more
   reliable than a 3B model at catching *"my account was hacked"*. Every decision
   records a `decided_by` field, so routing is auditable rather than vibes.

Retrieval is brute-force cosine over normalised MiniLM vectors — sub-millisecond
at this corpus size, and one fewer dependency to defend in a live review.

---

## Evaluation

**Golden set.** 200 examples, hand-labelled, stratified rather than random.
Random sampling would give tail intents 2-4 examples each and make macro-F1
noise; it would also under-sample the escalation-worthy cases that matter most.
Strata: per-cluster floor, an ambiguity quota (lowest-similarity decile), and a
high-stakes quota. The annotator never sees a model prediction or the brand's
actual reply — anchoring would inflate every downstream number.

**Human ceiling.** A pre-selected 40-example subset is re-labelled after a delay.
Self-agreement bounds what any system can score against these labels. A
classifier at 0.72 against a ceiling of 0.78 is near-ceiling, not mediocre — and
without this number there is no scale to read model scores against.

**Judge.** Five axes (groundedness, correctness, tone, actionability, safety),
scored by `llama3.2:3b` while the generator is `qwen2.5:3b-instruct` —
deliberately different families, because models flatter their own outputs.
Replies are stripped of system identity before judging.

**Judge validation.** ~50 replies scored by hand, blind to both system identity
and judge output, then compared via quadratic-weighted Cohen's κ and Spearman.
The residual SD sets a floor on what the instrument can resolve; differences
smaller than that are not claimed. Length-bias and self-preference probes are
reported alongside.

**Baselines.** *Trivial*: majority class, one canned reply, never escalate —
exists to expose metric inflation. *Simple*: TF-IDF + logistic regression
(cross-validated, so no example is scored by a model that saw it) plus verbatim
nearest-neighbour replies and keyword-rule escalation. The simple baseline is a
genuinely strong competitor: a real past reply from this brand is guaranteed
on-brand and factually grounded; it just may answer the wrong question.

All metrics carry bootstrap 95% CIs, and system-vs-system claims use a paired
bootstrap on the same resampled examples.

---

## Results

| System | Intent acc. | Macro-F1 | Esc. recall | Esc. precision | Missed |
|---|---|---|---|---|---|
| `trivial` | 0.200 | 0.037 | 0.000 | 0.000 | 34 |
| `simple` | 0.285 | 0.205 | 0.353 | 0.545 | 22 |
| `agent_zeroshot` | 0.260 | 0.230 | 0.824 | 0.203 | 6 |
| `agent` | **0.375** | **0.346** | **0.824** | 0.201 | **6** |

The agent beats the simple baseline by +0.138 macro-F1, 95% CI [+0.065, +0.209],
and misses 6 escalations against 22 — but it escalates 70% of all traffic to do
it, against a true rate of 17%.

**Two results worth reading the report for:**

- **Intent accuracy is 0.375; my own labels self-agree at 0.350.** The metric
  cannot separate model error from label noise. That is the report's first
  caveat rather than a footnote.
- **The LLM judge scores kappa = 0.003 against blind human scores** (Spearman
  −0.261). It measures nothing, so **no reply-quality claim is made anywhere** —
  the conclusions rest on intent and escalation, which do not depend on it.

See [REPORT.md](REPORT.md) for failure analysis, the mandatory
*"What is misleading about my headline number?"* section, and next steps.
Non-obvious choices are logged in [DECISIONS.md](DECISIONS.md).

## Reproducibility notes

- Seeds fixed throughout (`SEED=20260909`); `run.py reproduce` is deterministic.
- Local models only. Generation used `qwen2.5:3b-instruct`, judging
  `llama3.2:3b`, embeddings `all-MiniLM-L6-v2` — all on a 4GB laptop GPU.
- The dataset is subsampled to one brand, as the brief expects.

## Credits

- Dataset: Stuart Axelbrooke, *Customer Support on Twitter*, Kaggle (CC BY-NC-SA 4.0).
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2` (Apache 2.0), used via
  `transformers` with mean pooling rather than the `sentence-transformers` package.
- Models served locally with [Ollama](https://ollama.com).
- c-TF-IDF cluster characterisation follows the idea popularised by
  [BERTopic](https://github.com/MaartenGr/BERTopic); implemented here directly in
  ~15 lines of scikit-learn rather than taking the dependency.
- Quadratic-weighted κ via `sklearn.metrics.cohen_kappa_score`.
