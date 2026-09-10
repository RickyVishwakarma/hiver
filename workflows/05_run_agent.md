# Workflow 05 — Run the agent and the baselines

## Objective

Produce predictions for every golden example from three systems, and populate the
LLM cache that makes the results reproducible offline.

## Procedure

```bash
python run.py generate     # agent + both baselines
python run.py freeze       # export the cache for committing
```

Single ad-hoc message, useful when debugging a failure mode:

```bash
python tools/agent.py --message "my account was hacked and someone ordered stuff"
```

## The three heads

**Classify.** LLM picks from `taxonomy.json`. Output is snapped to the nearest
valid label, because small models paraphrase names. A self-reported confidence is
recorded but **not used for routing** — 3B models report ~0.9 for everything.
Whether that self-confidence correlates with correctness is worth reporting as a
calibration finding.

**Draft.** Retrieve top-k (default 4) nearest resolved threads and put the real
customer message and real brand reply in the prompt. Grounding means showing
precedent, not asking the model to invent policy. The system prompt forbids
inventing refund amounts, timeframes, and account details.

Retrieval is brute-force cosine over normalised MiniLM vectors. No vector DB:
sub-millisecond at this corpus size, and one fewer dependency to explain live.

**Route.** Hybrid, in this order:

1. Deterministic rules — safety, account security, legal, policy/refund. First
   match wins and becomes the stated reason.
2. Classifier parse failure → escalate (`ambiguous_request`).
3. `retrieval_score < 0.45` → escalate (`insufficient_info`).
4. Otherwise the LLM adjudicates.

Rules run first because **the expensive error is a missed escalation**, and a
regex is far more reliable than a 3B model at catching *"my account was hacked"*.
The LLM adds coverage for cases no rule anticipated. Every decision records
`decided_by` (`rule:<code>` or `llm`), so routing is auditable and the rule/LLM
split can be reported.

## Setting the retrieval threshold

0.45 is a starting value, not a tuned one. Set it from the actual distribution:

```bash
python -c "import sys;sys.path.insert(0,'tools');from common import read_jsonl;import numpy as np;
s=[r['retrieval_score'] for r in read_jsonl('data/preds_agent.jsonl')];
print(np.percentile(s,[5,10,25,50,75]))"
```

Choose a low percentile so it fires on genuine novelty rather than routinely.
Record the chosen value and its percentile in `DECISIONS.md`. **Do not tune it
against golden-set escalation labels** — that fits the threshold to the test set
and inflates escalation recall.

## Baselines

*Trivial*: majority class, one canned reply, never escalate. Exists to expose
metric inflation — it scores respectably on accuracy while macro-F1 is near zero
and escalation recall is exactly zero.

*Simple*: TF-IDF + logistic regression, **5-fold cross-validated** so no example
is scored by a model that saw it (it has no other training data; reporting
training accuracy would be a lie). Replies are the nearest historical reply
copied **verbatim**. Escalation is the same keyword rules, minus the LLM.

The verbatim replier is deliberately strong: a real past reply from this brand is
guaranteed on-brand and factually grounded. It just may answer the wrong
question — which is precisely what the groundedness and correctness axes separate.

## Cost of running

~3 LLM calls per example (classify, triage, draft) × 200 examples on a 3B model
at ~4GB VRAM ≈ 45–60 min. Ollama holds one model at a time — do not interleave
generator and judge passes.

## Outputs

`data/preds_agent.jsonl`, `preds_simple.jsonl`, `preds_trivial.jsonl`, and
`data/llm_cache_bundle.jsonl` (committed — this is the reproduction path).
