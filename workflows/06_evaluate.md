# Workflow 06 — Evaluate

## Objective

Report what the systems do, with uncertainty attached, in a way that cannot
flatter the system we built.

## Procedure

```bash
python run.py eval        # all systems, per-class detail, paired comparisons
```

## Metrics and why each is there

| Metric | Role |
|---|---|
| **accuracy** | Reported, but the number most likely to mislead — the trivial baseline scores well on it |
| **macro-F1** | **Headline.** Equal weight per class, so ignoring the tail cannot hide behind the head |
| **per-class P/R/F1** | Where failures actually live |
| **confusion matrix** | Which intents collapse into which — drives the failure analysis |
| **escalation precision / recall** | **Never averaged together** |
| **escalation rate** | Sanity check: a system escalating 80% is not automating anything |
| **parse-failure rate** | A real property of small local models, not noise to hide |

### On escalation

The two errors have wildly different costs, so combining them into one F1 would
destroy the information that matters:

- **Missed escalation (FN)** — a fraud victim gets a canned reply. Expensive:
  trust, complaints, possibly regulatory.
- **Over-escalation (FP)** — a human reads a routine question. Cheap: minutes.

We optimise **recall** and report the precision we paid for it. The eval prints
raw FN and FP counts, not just rates, because at n=200 "recall 0.87" hides
whether that means 2 misses or 9.

## Uncertainty

**Bootstrap 95% CIs** on every headline metric (2000 resamples, fixed seed). At
n=200 the CI on accuracy is roughly ±6pp. Without this a 3-point "improvement"
reads as progress when it is noise.

**Paired bootstrap** for system-vs-system claims: resamples the *same* examples
for both systems and measures the difference directly. Far more sensitive than
eyeballing two overlapping CIs, and it is what decides whether "the agent beats
the simple baseline" is supportable or wishful. If the CI includes zero, the
report says the difference is **not supported at this sample size** — regardless
of which system we would prefer to win.

## Reading the results honestly

- Compare macro-F1 against the **human ceiling** from workflow 04, not against
  100%. A classifier at 0.72 against a ceiling of 0.78 is near-ceiling.
- If the simple baseline is within noise of the agent on intent, **say so
  prominently.** It is the most credibility-building result available: it shows
  the evaluation was run to find out rather than to confirm.
- Reply-quality claims are gated on workflow 07. If judge κ is weak, lean the
  headline on intent and escalation, which do not depend on the judge at all.

## Outputs

`report/metrics.json` — every statistic, consumed by the report.
