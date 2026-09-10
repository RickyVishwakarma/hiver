# Decision log

Sixteen non-obvious choices, in the order they came up.

---

### 1. Ranked brands by "groundable replies", not volume

The brief wants replies grounded in *how the brand resolved* issues. A brand that
answers "please DM us" resolves off-platform, so its visible replies are routing
messages, not resolutions. I defined
`groundable_replies = answers_to_customer x (1 - dm_punt_rate)` and ranked on that.

It changed the answer. AppleSupport punts **52.5%** of replies to DMs — half its
resolutions simply are not in the dataset. AmazonHelp has 4.8x Delta's pool and
the lowest punt rate, but its ASCII fraction is the lowest of any large brand
(German and other languages), its replies are mostly generic ("describe the
issue") or chit-chat, and its traffic is dominated by "where is my order".

I also read actual replies before committing, which no aggregate column would
have told me: British_Airways splits replies across "1/2" and "2/2" tweets, so
`first_brand_reply` would capture half a sentence. Chose **Delta**: 35,251
groundable replies, 16.5% punt rate, 99.8% ASCII.

### 2. Kept messages the brand never answered

Thread reconstruction walks outward from the brand's own tweets, so it can only
reach messages support *answered*. A separate pass collects unanswered `@Delta`
mentions. Without it, evaluation runs on a friendlier distribution than
production: ignored messages are disproportionately off-topic, abusive, or
unanswerable — exactly the hard routing cases. Recovered 1,073 threads.

### 3. "Resolved" is a heuristic, and is named as one everywhere

`resolved_proxy` means "Delta replied publicly and did not punt to DMs". Twitter
never says whether the problem was fixed. For the 21.6% punted to DMs, the real
resolution happened where this dataset cannot see. Every downstream claim about
grounding in "how the brand resolved similar issues" is really grounding in *how
the brand responded*.

### 4. Capped the taxonomy at ~10 intents, with a mandatory `other`

A 40-cluster taxonomy looks impressive and is unlabellable: annotators cannot
hold it in working memory, boundaries blur, agreement collapses, and macro-F1
becomes noise over 5-example classes. `other` is mandatory — without it
annotators force bad fits into the nearest class and corrupt the set invisibly.

Clustering runs on the customer's opening message only. Clustering whole threads
would leak the brand's answer into the representation and make the taxonomy
easier than the live task. Merged clusters 9+3 into `service_complaint`; later
dropped `misc_request` (3 labels, forced 3-fold CV, over-predicted 12x its true
rate) — which I had flagged as provisional when the taxonomy was built, so it was
a pre-registered call rather than post-hoc convenience.

### 5. Stratified the golden set instead of sampling randomly

Random sampling over-weights the head: tail intents would get 2-4 examples each
and their recall would carry an error bar wider than the metric. Quotas: 20%
escalation-signal, 15% ambiguous (lowest-similarity decile), a floor of 10 per
cluster, remainder proportional.

**Accepted cost:** the set no longer matches production distribution, so
aggregate accuracy is *not* a production estimate. Sampling weights are retained
so it could be reweighted; I did not do that reweighting.

### 6. Never showed the annotator a model prediction or the brand's real reply

Pre-filling labels would have saved hours, anchored me into agreeing with the
system, and turned the golden set into a measurement of the system against
itself. The brand's actual reply is withheld for the same reason: it anchors both
the intent label and the escalation call toward whatever support happened to do.

### 7. Pre-selected the re-label subset before labelling started

The 40 second-pass ids are drawn by `sample_golden.py`, not chosen afterwards.
Picking them later would let the choice be influenced by which examples turned
out easy, biasing self-agreement upward.

### 8. Threw away the first golden set when it failed its own reliability check

Self-agreement came back at 0.30 (kappa 0.204). The flips were crossing unrelated
intents — `flight_disruption` to `praise_compliment`, twice — and individual cases
were plainly wrong ("Trying to send a complaint" labelled `praise_compliment`).
That is mis-keying, not ambiguity.

Root cause was my own tool: the CLI never echoed the selected intent, so pressing
the wrong number saved a wrong label silently. I rebuilt it to print
`>>> RECORDED: <INTENT> <<<` and offer a redo key, then re-labelled all 200. The
v1 labels are archived rather than deleted.

### 9. Re-labelled a second time when the first re-label was too fast to trust

The first re-label pass averaged **6.3 seconds per example** against 25.9s on the
original pass - not enough time to read a tweet and choose among 9 intents plus
an escalation reason. `validate_judge.py` now prints seconds-per-label beside the
agreement figure and refuses to call it a ceiling below 10s/example, so the tool
itself blocks the overclaim.

I redid it at 12.7s. Intent agreement rose from 0.350 to **0.425** (kappa 0.326),
but the slower pass also exposed something the fast one had hidden: my escalation
decisions moved from 5/40 to 21/40 on the same examples, kappa **0.036**, which is
chance. The fast pass had reported escalation agreement of 0.775, which looked
reassuring and was an artefact of both passes defaulting to "auto" (its kappa was
negative).

The worse-looking number is the one I kept, and the report now says the agent's
strongest result - 6 missed escalations against the baseline's 22 - is measured
against a threshold that shifts when I re-apply it. Both passes are archived.

### 10. Hybrid escalation: deterministic rules first, LLM second

Rules for safety, account security, legal and refund fire first and
short-circuit; the LLM adjudicates the rest. The expensive error is a *missed*
escalation, and a regex is far more reliable than a 3B model at catching "my
account was hacked". Every decision records `decided_by`, which is how I could
later attribute 100 of 111 over-escalations to the LLM rather than the rules.

Related: the model's self-reported confidence is recorded but never routed on.
I predicted it would be uncalibrated before seeing data; it came back at 0.974
when correct and 0.975 when wrong. Routing confidence uses retrieval score, which
is independent of the model's opinion of itself.

### 11. Generator and judge from different model families

`qwen2.5:3b-instruct` generates, `llama3.2:3b` judges. Models prefer their own
outputs, and same-family judging inflates the score of the system you built. This
blunts self-preference rather than eliminating it, so I measured the residue
against human scores instead of assuming it away.

### 12. Made the "simple" baseline genuinely strong, and cross-validated it

It copies the nearest historical Delta reply **verbatim**, so it is guaranteed
on-brand and factually grounded — it just may answer the wrong question. A weak
strawman would have made the agent look good and taught me nothing.

Its classifier has no training data beyond the golden set, so it is 5-fold
cross-validated: every prediction comes from a model that did not see that
example. Reporting its training accuracy would have been a lie; giving it a split
the LLM never saw would have compared two different quantities.

### 13. Never averaged escalation precision and recall, and used a paired bootstrap

A missed escalation (stranded passenger gets a canned reply) and an
over-escalation (a human reads a routine question) differ in cost by orders of
magnitude. Collapsing them into one F1 destroys the only information that matters
for routing. Raw counts are printed too, because at n=200 "recall 0.82" hides
whether that means 2 misses or 9.

System comparisons resample the *same* examples for both systems, which measures
the difference directly and is far more sensitive than eyeballing overlapping
CIs. When the interval includes zero the report says the difference is
unsupported — regardless of which system I would have preferred to win.

### 14. Fixed the classifier with exemplars drawn from outside the golden set

Zero-shot, the model never emitted `booking_change_refund` or `other` at all and
put 44% of predictions into two catch-all labels. Exemplars come only from
threads absent from the golden set, with intents from *clustering* rather than
human labels, so neither test text nor test labels reach the prompt. The builder
asserts zero leaks and writes every exemplar to `data/fewshot.json` so the claim
is auditable.

I kept the zero-shot run as a separately reported system. Re-running after a
prompt fix and publishing only the improved number would hide the size of the fix
and invite the suspicion that the prompt was tuned against the test set.

### 15. Committed the model cache so reproduction needs no model at all

~1,500 local 3B calls take about an hour, which cannot meet the brief's
15-minute reproduction requirement. Every call is content-addressed by
`sha256(model + prompt + options)` and committed. `ANTHILL_CACHE_ONLY=1` makes a
cache miss a *hard error*, so the grader path can never silently drift by calling
a model. Replay is exact rather than approximate — which also sidesteps the fact
that llama.cpp is not bit-identical across GPU/CPU backends. Measured: **70
seconds** from a clean clone, with bit-identical numbers (verified by
re-running in a fresh clone and diffing the regenerated metrics against the
committed ones: no change).

### 16. Rebuilt the judge as a forced choice, then reported that too as a failure

The 5-axis judge scored kappa 0.003, so the report could make no reply-quality
claim. The hypothesis worth testing was that *absolute 1-5 scoring* was the
problem rather than the model: a 3B has no stable notion of what a "4" is, but
choosing between two concrete replies needs no anchor.

Three things made the rebuild worth doing even though it failed:

- **Each pair is judged in both orders.** Position bias falls out of the design
  instead of needing a separate probe, and pairs where the two orders disagree
  are recorded `undecided` rather than resolved by coin-flip. The `direct`
  prompt picked the reply shown *second* 37 times out of 40.
- **Both prompt variants are kept and both are scored** against the same human
  choices. The `deliberate` variant halves the flip rate to 52.5% and is the
  better instrument; publishing only it would have been tuning the judge against
  its own validation set.
- **The human yardstick got the same probe.** My decisive verdicts split 25/10
  toward whichever reply was shown first (p = 0.017), so the agreement figure is
  a lower bound rather than a clean measurement. A validation tool that only
  ever audits the machine is not a validation tool.

Cost: 160 model calls and 17 minutes of labelling, for a result that added
nothing to the headline table. It is in the report because "we measured the
instrument twice and it does not work" is a finding, and because the alternative
- quietly keeping an unvalidated judge - is the failure this assignment screens
for.

---

*Smaller implementation calls: mean-pooled `transformers` instead of
`sentence-transformers` (one less dependency, no conflict with transformers 5.x);
brute-force cosine instead of a vector DB (sub-millisecond at 19.5k vectors); a
Python task runner instead of a Makefile (`make` is not on Windows); and a
synthetic-fixture smoke test so the pipeline could be verified before the 500MB
download finished — it caught two real bugs before any real data existed.*
