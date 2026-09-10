# Decision log

Non-obvious choices and the reasoning behind them, in the order they arose.

---

### 1. Ranked brands by `groundable_replies`, not volume
The brief requires replies grounded in *how the brand resolved* issues. A brand
answering "please DM us" resolves off-platform, so its visible replies are
routing messages, not resolutions. Defined
`groundable_replies = answers_to_customer × (1 − dm_punt_rate)` and ranked on
that. Validated on the fixture: two synthetic brands that always punt score 0,
while the 18%-punt brand scores 909. Chosen: **Delta** - 42,253 replies,
45,291 inbound mentions, 16.5% punt rate, 99.8% ASCII, 35,251 groundable.

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
class and corrupt the set invisibly.
Applied to Delta at k=10: **merged** clusters 9+3 into `service_complaint`
(c3 mixed praise with complaints); **dropped** `misc_request` after labelling
(see #30); **kept** cluster 7 (photo/promo chatter) as `social_chatter` rather
than discarding it, because roughly a third of Delta's inbound is not a support
request at all and the agent must be able to triage that.

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

### 10. Measured annotator self-agreement to give model scores a scale
Same annotator re-labels a pre-selected subset; Cohen's kappa bounds what any
system can score against these labels. Two limitations, both stated rather than
hidden: one annotator measures *consistency*, not *correctness*; and the pass
actually run averaged 6.3s per label, so what we obtained is a lower bound, not
a true ceiling. See #27 - this entry's original framing did not survive contact
with the measurement, and is left visible rather than quietly rewritten.

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

### 24. Chose Delta over AmazonHelp despite a 5x smaller grounding pool
AmazonHelp had 168,186 groundable replies to Delta's 35,251 and a lower punt
rate (0.65% vs 16.5%). Rejected anyway: its ASCII fraction was the lowest of any
top brand (0.955, i.e. substantial German and other non-English traffic), its
replies are largely generic ("describe the issue") or chit-chat, and its volume
is dominated by "where is my order", giving a degenerate head-heavy taxonomy.
Delta has genuinely diverse intents and abundant natural escalation cases.
Rejected British_Airways separately: its replies are split across "1/2","2/2"
continuation tweets, so `first_brand_reply` would capture half a sentence.

### 25. Sampled reply text before committing to a brand
The profile ranked brands on punt rate and volume, but neither reveals whether a
reply carries resolution *content*. Reading actual replies changed the decision:
it exposed AmazonHelp's multilingual/generic problem and BA's fragmentation,
neither of which appears in any aggregate column. Cheap check, changed the
outcome.

### 26. The first golden set failed its own reliability check, and was rebuilt
Pass-2 self-agreement came back at 0.30 (kappa 0.204), with escalation at
kappa 0.091. Diagnosis showed it was not taxonomy ambiguity: only 7 of 28 flips
stayed inside the fuzzy non-actionable cluster, the rest crossed unrelated
boundaries, and individual cases were plainly wrong ("Trying to send a complaint"
labelled praise_compliment). Root cause was a tool defect - the CLI never echoed
the selected intent, so a mis-key saved a wrong label silently. Rebuilt the CLI
and re-labelled all 200. Evidence that v2 is better calibrated: v1 put 40
examples in flight_disruption (menu option 1) against a cluster allocation of
~20; v2 put 18.

### 27. Report self-agreement as a lower bound, not a ceiling
The re-label pass averaged 6.3 seconds per example (against 25.9s on pass 1),
which is not enough time to read a tweet and choose among 10 intents plus an
escalation reason. `validate_judge` now prints seconds-per-label beside the
agreement figure and refuses to call it a ceiling below 10s/example. The number
is therefore reported as a LOWER BOUND on annotator reliability. An unqualified
"human ceiling = 35%" would not survive anyone checking the timestamps, which
are in the committed data.

### 28. Fixed the classifier with few-shot exemplars drawn from outside the golden set
Zero-shot, the 3B model never emitted booking_change_refund or other and put 44%
of predictions into two catch-all-sounding labels (support_unreachable 52/200,
misc_request 36/200). Verified as a model limitation, not a label-snapping bug,
by inspecting raw output: "double charged for some tickets" -> misc_request.
Exemplars are selected only from threads absent from the golden set, and their
intent comes from the cluster->intent mapping rather than any human label, so
neither test text nor test labels reach the prompt. The builder asserts zero
leaks and writes every exemplar to data/fewshot.json so the claim is auditable.

### 29. Kept the zero-shot run as a separate reported system
Re-running after a prompt fix and reporting only the improved number would hide
the size of the fix and invite the suspicion that the prompt was tuned against
the test set. Both runs are evaluated side by side as `agent_zeroshot` and
`agent`.

### 30. Dropped misc_request after seeing the data, and said when it was flagged
It drew 3 labels, forced cross-validation down to 3 folds, and was over-predicted
12x its true rate; its own cluster exemplars turned out to be one complaint and
one compliment. Folded into `other`. Flagged as provisional when the taxonomy was
built, so this is a pre-registered call rather than a post-hoc convenience.

### 31. Self-reported LLM confidence averaged 0.961 against 0.270 accuracy
Recorded but never used for routing, which decision #12 predicted before the data
existed. Routing confidence comes from retrieval score instead. The gap is
reported as a calibration finding.

---

### 32. The retrieval-score escalation threshold never fired, and was left alone
Set a priori at 0.45. Measured distribution over the golden set: min 0.460, p5
0.587, median 0.703. So the `weak_retrieval` rule never triggered once - Delta's
19,534-thread grounding pool is dense enough that every message finds a
neighbour. It is reported as dead code rather than silently retuned, because
picking a new threshold from this distribution would be fitting it to the test
set - the exact error the rule exists to avoid. The correct fix, listed under
next steps, is to calibrate it on held-out non-golden threads and report the
full precision-recall trade-off curve so a deployer picks their own operating
point.

### 33. Escalation over-fires from the LLM adjudicator, not the rules
The agent escalated 69% of messages against a 17% true rate: recall 0.824 (only
6 missed) at precision 0.203. Since the retrieval rule never fired, the excess
comes from the keyword rules plus a trigger-happy LLM triage prompt. The
`decided_by` field was built for exactly this question and makes the split
attributable rather than a guess.

