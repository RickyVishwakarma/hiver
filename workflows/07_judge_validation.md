# Workflow 07 — Validate the judge before believing it

## Objective

Establish whether the LLM judge measures reply quality at all, and what size of
difference it can actually resolve.

## Why this exists

An LLM-as-judge score is a number with no inherent meaning. Reporting
"reply quality 3.8/5" without knowing whether the judge agrees with a human is
the exact failure this assignment is screening for. The judge here is a 3B model
on a laptop — a weak instrument. That is fine and reportable; an *unmeasured*
instrument is not.

## Procedure

### 1. Judge every system's replies

```bash
python run.py judge
```

- Judge model is `llama3.2:3b`; the generator is `qwen2.5:3b-instruct`.
  **Different families on purpose** — models prefer their own outputs, and
  same-family judging inflates the score of the system you built. This blunts
  self-preference; it does not eliminate it, and we do not claim it does.
- Replies are stripped of system identity before judging.
- Every system's replies are judged against the *same* retrieved evidence, so
  the groundedness axis is comparable across systems.
- Temperature 0, fixed seed, cached by prompt hash.

### 2. Score ~50 replies by hand, blind

```bash
python run.py score
```

Three blinds, all of which matter: system identity hidden, judge scores hidden,
presentation order shuffled with a fixed seed. Sampling is balanced across
systems so agreement is not dominated by whichever system produced most replies.
The rubric is reprinted on every screen so the standard does not drift across a
50-item session.

### 3. Compute agreement and biases

```bash
python run.py validate
```

Produces three blocks:

**Human ceiling** — pass-1 vs pass-2 self-agreement (see workflow 04). The scale
against which model scores are read.

**Judge vs human** — quadratic-weighted Cohen's κ (correct for ordinal 1–5
scores: disagreeing by 1 is penalised far less than by 4) plus Spearman, per axis
and overall.

| κ_w | Verdict | What we may claim |
|---|---|---|
| < 0.2 | noise | Nothing. Do not report judge scores as evidence. |
| 0.2–0.4 | weak | Direction only, never a ranking. |
| 0.4–0.6 | moderate | Large gaps only. |
| > 0.6 | good for 3B | Rankings, with the CI attached. |

**Minimum detectable gap** — the SD of the judge-minus-human residual. Reply-
quality differences smaller than this are not claimed anywhere in the report.

**Bias probes**
- *Length bias*: Spearman(reply length, judge score). Above ~0.4, the judge's
  ranking is partly a length ranking and must be described as such.
- *Self-preference*: judge-minus-human gap per system. If the gap is larger for
  the LLM agent than for the verbatim-copy baseline, the judge is flattering
  generated text and the agent's lead is inflated by roughly that amount.

## Outputs

`report/judge_validation.json` — every statistic above, consumed by the report.

## How results feed the report

The overall κ and the minimum detectable gap go directly into
*"What is misleading about my headline number?"*. If κ lands in the weak band,
the correct move is to **downgrade the reply-quality claim to directional** and
lean the headline on the deterministic intent and escalation metrics, which do
not depend on the judge at all. Reporting a weak κ honestly is a stronger
deliverable than a confident number with no validation behind it.

## Known limitations

- ~50 double-scored replies gives κ its own wide CI.
- Same person labelled the golden set, wrote the rubric, and scored the replies —
  a systematic misreading of "groundedness" would be invisible.
- Blinding hides system identity, but the verbatim-copy baseline is often
  recognisable from style alone; that residual leak cannot be removed.
