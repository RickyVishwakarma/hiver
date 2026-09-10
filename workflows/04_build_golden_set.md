# Workflow 04 — Build the golden evaluation set

**Doubles as the assignment's required note on how the golden set was sampled and labelled.**

## Objective

Produce 150–250 hand-labelled examples that can be trusted as ground truth for
intent classification and escalation routing, plus a measurement of how reliable
those labels actually are.

## Why this is the highest-stakes stage

Every number in the report is measured against these labels. If they are
anchored, inconsistent, or unrepresentative, the entire evaluation is decorative.
Two failure modes destroy a golden set, and both are invisible afterwards:

1. **Anchoring** — the annotator sees a model prediction and agrees with it. The
   set then measures agreement-with-itself, and scores come out high for the
   wrong reason.
2. **Head-only sampling** — the set matches the traffic distribution, so tail
   intents get 2–4 examples and their per-class recall has an error bar wider
   than the metric.

## Inputs

- `data/threads.jsonl` (from workflow 02)
- `data/cluster_assignments.jsonl` and `data/taxonomy.json` (from workflow 03)

## Procedure

### 1. Draw the sample

```bash
python run.py sample --n 200
```

`tools/sample_golden.py` draws a **stratified** sample. Eligible pool excludes
multi-customer pile-ons and messages under 20 characters. Strata, in the order
they are filled:

| Stratum | Share | Why |
|---|---|---|
| escalation signal | 20% | Otherwise escalation recall rests on ~5 positives and its CI is uninformative |
| ambiguous | 15% | Bottom decile of centroid similarity — the boundary cases where classifiers actually fail |
| per-cluster floor | ≥10 each | Guarantees every intent is measurable, so macro-F1 means something |
| proportional | remainder | Keeps some resemblance to real traffic |

The 40-example second-pass subset is **pre-selected here**, before any labelling,
so the choice cannot be influenced by which examples turned out to be easy.

**Accepted cost:** this set deliberately does not match production distribution.
Aggregate accuracy on it is therefore *not* a production estimate. Sampling
weights are retained so metrics can be reweighted, and the report says so.

### 2. Label, blind

```bash
python run.py label
```

The CLI shows **only the customer message and the intent options**. It never
shows a model prediction, and never shows the brand's actual historical reply —
seeing how support really answered would anchor both the intent label and the
escalation call toward the decision support happened to make.

Per example, record:
- **intent** — one label from `taxonomy.json`, or `other`
- **escalate** — auto-handle vs. escalate
- **escalation reason** — a structured code, not free text, so reasons can be
  aggregated and compared against the agent's stated reason

Progress saves after every example; quitting and resuming is safe.

### 3. Labelling rules (apply consistently; this is the protocol)

- Label the customer's **primary** need. A message with two asks gets the one it
  leads with; if genuinely co-equal, use `other` and note it.
- Judge on the **opening message alone**. That is what the live agent receives.
  Do not use hindsight from later turns.
- **Escalate** when a competent bot answering from precedent would be wrong or
  insufficient: it needs authorisation (refund, goodwill), private account data,
  involves account compromise or money disputes, shows severe distress, carries
  legal or safety weight, or is genuinely unclear.
- **Do not** escalate merely because the message is rude or long.
- When torn between two intents, pick the one whose *resolution path* differs.
  Intents exist to route work, not to describe topics.
- Use `other` freely. Forcing bad fits into the nearest class silently corrupts
  the set — that is what the bucket is for.

### 4. Second pass, after a break

```bash
python run.py label --pass2
```

Re-label the pre-selected 40 **after at least a few hours** — same-session
re-labelling measures memory, not consistency. Pass-1 labels are hidden.

### 5. Measure the ceiling

```bash
python tools/validate_judge.py --self-agreement
```

Reports Cohen's κ and raw agreement between the two passes. **This is the human
ceiling.** No system can be expected to exceed it against these labels, and every
model score in the report is read against it rather than against 100%.

## Outputs

| File | Contents |
|---|---|
| `data/golden.jsonl` | the labelled set — committed |
| `data/golden_pass2.jsonl` | second-pass labels — committed |
| `data/pass2_ids.jsonl` | pre-selected subset ids |
| `report/judge_validation.json` | includes the `human_ceiling` block |

## Edge cases and what to do

| Situation | Action |
|---|---|
| Message is not English | Label `other`, note it. Do not guess. |
| Message is pure spam/bot | Label `other`, auto-handle. |
| Message has no brand reply in the data | Still label it — the agent would see it live. `resolved_proxy=False` keeps it out of the grounding pool. |
| Two intents genuinely co-equal | `other` + note. Recurrence means the taxonomy needs a split — record it. |
| A cluster turns out to be incoherent while labelling | Do **not** silently relabel earlier examples. Finish, then record the problem in `DECISIONS.md`. Retrofitting labels mid-pass makes the set inconsistent in an undetectable way. |

## Known limitations (carried into the report)

- **Single annotator.** Self-agreement measures consistency, not correctness. A
  systematic misunderstanding is invisible to it, and the same person wrote the
  prompts — so annotator and system share blind spots in a correlated way.
- n = 200 gives roughly ±6pp on accuracy; small differences are not resolvable.
- One brand, one time period.
