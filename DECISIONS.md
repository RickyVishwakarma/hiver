# Decision log

Non-obvious choices and the reasoning behind them. Entries marked **[pending]**
are filled once the pipeline has run on the real dataset.

---

### 1. Ranked brands by `groundable_replies`, not volume
The brief requires replies grounded in *how the brand resolved* issues. A brand
answering "please DM us" resolves off-platform, so its visible replies are
routing messages, not resolutions. Defined
`groundable_replies = answers_to_customer × (1 − dm_punt_rate)` and ranked on
that. Validated on the fixture: two synthetic brands that always punt score 0,
while the 18%-punt brand scores 909. **[pending: chosen brand + its row]**

### 2. Kept messages the brand never answered
Thread reconstruction walks outward from the brand's own tweets, so it can only
reach messages support *answered*. A separate pass collects unanswered `@brand`
mentions. Without it, evaluation runs on a friendlier distribution than
production — ignored messages are disproportionately off-topic, abusive, or
unanswerable, i.e. the hard routing cases. In the fixture this recovered 6.9% of
threads that the brand-seeded closure missed entirely.

### 3. `resolved_proxy` is a heuristic and is named as one
Defined as "brand replied publicly AND did not punt to DMs". Twitter never
reveals whether the problem was fixed. A `customer_thanked` flag (gratitude
appearing *after* a brand reply, so "thanks in advance" doesn't count) is a weak
positive signal, not confirmation. Everything downstream saying "resolved" means
"looks resolved by this heuristic" — a headline caveat, not a footnote.

### 4. Clustered opening messages only, not whole threads
Clustering full threads leaks the brand's answer into the representation and
makes the taxonomy easier than the live task, where only the opening message
exists. Costs cluster coherence; buys a taxonomy that matches deployment.

### 5. Capped the taxonomy at ~10 intents + a mandatory `other`
A 40-cluster taxonomy is unlabellable: annotators can't hold it in working
memory, boundaries blur, agreement collapses, and macro-F1 becomes noise over
5-example classes. Ten labelled consistently beat forty labelled arbitrarily.
`other` is mandatory — without it annotators force bad fits into the nearest
class and corrupt the set invisibly. **[pending: what was merged/split/dropped]**

### 6. Did not use Banking77
It is retail-banking intent data with 77 labels describing a different resolution
structure; the chosen brand is not a bank. Importing it would produce a taxonomy
fitting neither dataset. Listed under "what I chose not to build" rather than
bolted on to use an optional resource.

### 7. Stratified the golden set instead of sampling randomly
Random sampling over-weights the head: tail intents would get 2–4 examples and
their recall would have an error bar wider than the metric. Strata: 20%
escalation-signal, 15% ambiguous (lowest-similarity decile), ≥10 per cluster
floor, remainder proportional. **Accepted cost:** the set no longer matches
production distribution, so aggregate accuracy is not a production estimate.
Sampling weights are retained so metrics can be reweighted.

### 8. Never showed the annotator a model prediction or the brand's real reply
Pre-filling labels would have saved hours. It would also anchor the annotator
into agreeing with the system and inflate every downstream number, converting the
golden set into a measurement of the system against itself. The brand's actual
reply is withheld for the same reason: it anchors both the intent label and the
escalation call toward the decision support happened to make.

### 9. Pre-selected the re-label subset *before* labelling started
The 40 second-pass ids are drawn by `sample_golden.py`, not chosen afterwards.
Picking them later would let the choice be influenced by which examples turned
out easy, biasing the self-agreement estimate upward.

### 10. Measured a human ceiling, and read model scores against it
Same annotator re-labels after a delay; Cohen's κ bounds what any system can
score against these labels. A classifier at 0.72 against a ceiling of 0.78 is
near-ceiling, not mediocre. Without this there is no scale. Limitation stated:
one annotator means this measures *consistency*, not *correctness*.

### 11. Hybrid escalation — rules first, LLM second
Deterministic rules (safety, account security, legal, refund/policy) fire first
and short-circuit; the LLM adjudicates the remainder. The expensive error is a
missed escalation, and a regex is far more reliable than a 3B model at catching
"my account was hacked". Every decision records `decided_by`, so routing is
auditable and the rule-vs-LLM split is reportable.

### 12. Recorded LLM self-reported confidence but do not route on it
3B models report ~0.9 for nearly everything. Routing confidence comes from
retrieval score instead — "this brand has never handled anything like this" is a
genuine escalation reason and is independent of the model's opinion of itself.
Self-confidence is kept so its (probable) miscalibration can be reported.

### 13. Generator and judge from different model families
`qwen2.5:3b-instruct` generates, `llama3.2:3b` judges. Models prefer their own
outputs; same-family judging inflates the score of the system we built. This
blunts self-preference rather than eliminating it — the residue is measured
against human scores, and a per-system judge-minus-human gap is reported.

### 14. Committed the LLM cache so reproduction needs no model
~600 local 3B calls take 45–60 min, which cannot meet the brief's 15-minute
reproduction requirement. Every call is content-addressed by
`sha256(model + prompt + options)` and committed. `ANTHILL_CACHE_ONLY=1` makes a
cache miss a *hard error*, so the grader path can never silently drift by calling
a model. Replay is exact rather than approximate — which also sidesteps the fact
that llama.cpp is not bit-identical across GPU/CPU backends.

### 15. Cross-validated the simple baseline rather than giving it a train split
It has no training data other than the golden set. Reporting its training
accuracy would be a lie; giving it a split the LLM never saw would compare two
different quantities. 5-fold out-of-fold predictions mean every prediction comes
from a model that did not see that example — the only fair comparison available.

### 16. Made the "simple" baseline genuinely strong
It copies the nearest historical brand reply **verbatim**, so it is guaranteed
on-brand and factually grounded — it just may answer the wrong question. A weak
strawman baseline would make the agent look good and tell us nothing. If this
baseline lands within noise of the agent, that is reported prominently.

### 17. Report escalation precision and recall separately, never averaged
Missed escalation (fraud victim gets a canned reply) and over-escalation (a human
reads a routine question) differ in cost by orders of magnitude. Averaging them
into one F1 destroys the only information that matters for the routing decision.
Raw FN/FP counts are printed, because at n=200 "recall 0.87" hides whether that
means 2 misses or 9.

### 18. Used a paired bootstrap for system comparisons
Resampling the same examples for both systems measures the difference directly
and is far more sensitive than comparing overlapping CIs. When the CI includes
zero the report states the difference is unsupported at this sample size —
regardless of which system we would prefer to win.

### 19. Dropped `sentence-transformers` for ~20 lines of `transformers`
Mean pooling + L2 norm reproduces `all-MiniLM-L6-v2`'s intended usage exactly.
Avoids a version conflict with the installed transformers 5.x, removes a
dependency, and leaves code we can explain line-by-line in a live review — which
the brief warns will happen. Validated: related pairs 0.46/0.57 vs. 0.20 unrelated.

### 20. No vector database
Brute-force cosine over normalised vectors is sub-millisecond at this corpus
size. FAISS would add a dependency, a build step, and an index-parameter
conversation, for no measurable gain at this scale.

### 21. Built a synthetic-fixture smoke test instead of waiting on the download
`make_fixture.py` reproduces the real dump's structural mess (reply chains,
fan-out, orphaned parents, DM punts, unanswered mentions); `smoke_test.py` runs
all nine stages against it and asserts 34 properties. It caught two real bugs
before any real data existed: threads with no brand reply being silently dropped,
and the brand-seeded blind spot in decision #2. Isolated via `ANTHILL_DATA_DIR`
so it can never overwrite the real golden set.

### 22. Forced UTF-8 on stdout centrally
Windows consoles default to cp1252 and raise `UnicodeEncodeError` on any
non-ASCII output. Graders may be on Windows, so this is set once in
`common.setup_console()` rather than hoping every print stays ASCII.

### 23. Chose a Python task runner over a Makefile
`make` is not present on this Windows machine and cannot be assumed on a
grader's. `python run.py reproduce` works identically on every platform with no
extra tooling.

---

**[pending]** — retrieval-score escalation threshold: currently 0.45 as a
starting value. To be set from the observed score distribution (a low percentile)
and recorded here with its percentile. Explicitly *not* tuned against golden-set
escalation labels, which would fit the threshold to the test set.
