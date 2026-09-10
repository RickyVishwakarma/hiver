"""Stage D.1 - the support agent: classify -> retrieve+draft -> route.

THREE HEADS
  1. classify   LLM picks an intent from our taxonomy. It also self-reports a
                confidence, which we record but DO NOT trust: small models are
                badly calibrated and say 0.9 for everything. The confidence we
                actually route on is retrieval-based (see below).

  2. draft      Retrieve the k most similar RESOLVED historical threads for this
                brand and put the real customer message + real brand reply in
                the prompt. This is what "grounded in how that brand has
                historically resolved similar issues" means operationally: the
                model is shown precedent, not asked to invent policy.

  3. route      HYBRID escalation. Deterministic rules fire first and short-
                circuit; the LLM only adjudicates what survives. Rationale: the
                expensive error is a MISSED escalation (an angry fraud victim
                gets a canned reply), and a regex is far more reliable than a 3B
                model at catching "my account was hacked". Rules give recall;
                the LLM adds coverage for cases no rule anticipated. Every
                decision carries a machine-readable source so escalations are
                auditable rather than vibes.

RETRIEVAL CONFIDENCE
`retrieval_score` = mean cosine similarity of the top-k neighbours. Low score
means "this brand has never handled anything like this", which is a genuine
reason to escalate and is independent of the LLM's opinion of itself.

No vector DB: brute-force numpy over a few tens of thousands of vectors is
sub-millisecond and is one fewer dependency to explain in a live review.

Usage:
  python tools/agent.py --golden            # run over the golden set
  python tools/agent.py --message "..."     # single ad-hoc message
"""
from __future__ import annotations

import argparse
import re
import time

import numpy as np

from common import (
    CACHED_GENERATIONS,
    DATA,
    GOLDEN,
    TAXONOMY,
    THREADS,
    die,
    info,
    ok,
    read_json,
    read_jsonl,
    set_seed,
    write_jsonl,
)

TOP_K = 4

# --- Deterministic escalation rules -----------------------------------------
# Ordered by severity: the first match wins and becomes the stated reason.
# These are intentionally high-precision phrasings, not broad topic words: this
# decides real routing, unlike the deliberately over-broad sampler patterns.
RULES: list[tuple[str, re.Pattern]] = [
    (
        "safety_harm",
        re.compile(r"\b(injur\w+|hospital|ambulance|assault\w*|threaten\w*|"
                   r"suicid\w*|unsafe|caught fire|burn(?:ed|t|ing)?\b|emergency)\b", re.I),
    ),
    (
        "account_security",
        re.compile(r"\b(hack\w+|compromis\w+|unauthoris\w+|unauthoriz\w+|"
                   r"someone (?:else )?(?:has|used|accessed)|fraud\w*|scam\w*|"
                   r"stolen|identity theft|phish\w*)\b", re.I),
    ),
    (
        "legal_regulatory",
        re.compile(r"\b(lawyer|solicitor|legal action|sue |suing|lawsuit|"
                   r"small claims|ombudsman|regulator|gdpr|data protection|"
                   r"formal complaint|trading standards)\b", re.I),
    ),
    (
        "policy_exception",
        re.compile(r"\b(refund|reimburs\w+|chargeback|compensat\w+|"
                   r"cancel my (?:account|subscription|order)|close my account|"
                   r"double charged|charged twice|billed twice)\b", re.I),
    ),
]

SYSTEM_CLASSIFY = (
    "You are an intent classifier for a customer support team. "
    "You reply with JSON only. No prose, no markdown fences."
)

SYSTEM_DRAFT = (
    "You are a customer support agent replying on Twitter. "
    "Write like the brand's past replies: same tone, same length, same level of "
    "specificity. Stay under 280 characters. Never invent policies, refund "
    "amounts, timeframes, or account details that are not in the examples."
)


class Agent:
    def __init__(self, top_k: int = TOP_K):
        self.taxonomy = read_json(TAXONOMY) or die(f"{TAXONOMY} missing - name your clusters first")
        self.intents = [i for i in self.taxonomy["intents"] if not i["name"].startswith("TODO_")]
        self.names = [i["name"] for i in self.intents]
        self.top_k = top_k

        threads = read_jsonl(THREADS)
        # Grounding pool: only threads where the brand actually answered in public.
        # DM-punted threads contain no resolution to learn from.
        self.pool = [
            t for t in threads
            if t.get("resolved_proxy") and t.get("first_brand_reply") and not t.get("multi_customer")
        ]
        if not self.pool:
            die("grounding pool is empty - check resolved_proxy in build_threads.py")
        info(f"grounding pool: {len(self.pool):,} resolved threads")

        from embed import load_or_encode

        self.pool_vecs = load_or_encode([t["opening_msg"] for t in self.pool], tag="pool_openings")

    # --- head 2 helper -------------------------------------------------------
    def retrieve(self, message: str) -> tuple[list[dict], float]:
        from embed import encode

        q = encode([message])[0]
        sims = self.pool_vecs @ q  # both L2-normalised -> cosine
        k = min(self.top_k, len(sims))
        if k < len(sims):
            idx = np.argpartition(-sims, k - 1)[:k]  # kth must be < len(sims)
        else:
            idx = np.arange(len(sims))
        idx = idx[np.argsort(-sims[idx])]
        neighbours = [
            {
                "thread_id": self.pool[i]["thread_id"],
                "customer": self.pool[i]["opening_msg_raw"],
                "brand_reply": self.pool[i]["first_brand_reply"],
                "sim": round(float(sims[i]), 4),
            }
            for i in idx
        ]
        return neighbours, round(float(np.mean([n["sim"] for n in neighbours])), 4)

    # --- head 1 --------------------------------------------------------------
    def classify(self, message: str) -> tuple[str, float, bool]:
        from llm import generate_json

        options = "\n".join(f"- {i['name']}: {i.get('description','')}" for i in self.intents)
        prompt = (
            f"Classify this customer support message into exactly one intent.\n\n"
            f"INTENTS:\n{options}\n\n"
            f'MESSAGE:\n"{message}"\n\n'
            'Respond with JSON only: {"intent": "<one name from the list>", '
            '"confidence": <0.0-1.0>}'
        )
        data = generate_json(prompt, system=SYSTEM_CLASSIFY, max_tokens=80)
        raw = str(data.get("intent", "")).strip()
        parse_failed = not raw
        # Snap to the closest valid label; a small model will paraphrase names.
        intent = next((n for n in self.names if n.lower() == raw.lower()), None)
        if intent is None:
            intent = next((n for n in self.names if n.lower() in raw.lower() or raw.lower() in n.lower()), None)
        if intent is None:
            intent, parse_failed = "other", True
        try:
            conf = float(data.get("confidence", 0.0))
        except (TypeError, ValueError):
            conf = 0.0
        return intent, max(0.0, min(1.0, conf)), parse_failed

    # --- head 2 --------------------------------------------------------------
    def draft(self, message: str, intent: str, neighbours: list[dict]) -> str:
        from llm import generate

        examples = "\n\n".join(
            f"Customer: {n['customer']}\nBrand replied: {n['brand_reply']}" for n in neighbours
        )
        prompt = (
            f"Here is how this brand has replied to similar messages before.\n\n"
            f"{examples}\n\n"
            f"---\nNow reply to this new message (intent: {intent}).\n"
            f'Customer: "{message}"\n\n'
            f"Write only the reply text, under 280 characters."
        )
        reply = generate(prompt, system=SYSTEM_DRAFT, max_tokens=160).strip()
        # Models like to wrap replies in quotes or prefix "Brand:".
        reply = re.sub(r'^(?:brand(?:\s+replied)?|reply|response)\s*:\s*', "", reply, flags=re.I)
        return reply.strip().strip('"').strip()

    # --- head 3 --------------------------------------------------------------
    def route(
        self, message: str, intent: str, retrieval_score: float, parse_failed: bool
    ) -> tuple[bool, str, str]:
        """Returns (escalate, reason_code, decided_by)."""
        for code, pattern in RULES:
            if pattern.search(message):
                return True, code, f"rule:{code}"

        if parse_failed:
            return True, "ambiguous_request", "rule:classifier_parse_failure"

        # Threshold chosen from the retrieval-score distribution, not guessed:
        # see workflows/05_run_agent.md. Below this the brand has no comparable
        # precedent, so any draft would be unsupported invention.
        if retrieval_score < 0.45:
            return True, "insufficient_info", "rule:weak_retrieval"

        from llm import generate_json

        prompt = (
            "You are triaging a customer support message. Decide whether a "
            "support bot can safely answer it, or whether a human must handle it.\n\n"
            "Escalate if it needs a decision only a human can authorise, involves "
            "money/account access disputes, shows severe distress, or is unclear.\n"
            "Auto-handle routine questions the bot can answer from precedent.\n\n"
            f'MESSAGE:\n"{message}"\n\n'
            'JSON only: {"escalate": true|false, "reason": "<short reason>"}'
        )
        data = generate_json(prompt, system=SYSTEM_CLASSIFY, max_tokens=100)
        if data.get("escalate") is True:
            return True, "high_emotion", "llm"
        if data.get("escalate") is False:
            return False, "", "llm"
        return True, "ambiguous_request", "rule:triage_parse_failure"

    # --- orchestration -------------------------------------------------------
    def handle(self, message: str) -> dict:
        t0 = time.time()
        intent, self_conf, parse_failed = self.classify(message)
        neighbours, score = self.retrieve(message)
        escalate, reason, decided_by = self.route(message, intent, score, parse_failed)
        reply = self.draft(message, intent, neighbours)
        return {
            "intent": intent,
            "self_reported_confidence": self_conf,
            "retrieval_score": score,
            "reply": reply,
            "escalate": escalate,
            "escalation_reason": reason,
            "decided_by": decided_by,
            "classifier_parse_failed": parse_failed,
            "retrieved": neighbours,
            "latency_s": round(time.time() - t0, 3),
        }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--golden", action="store_true", help="Run over the labelled golden set")
    ap.add_argument("--message", help="Handle a single message and print the result")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=str(DATA / "preds_agent.jsonl"))
    args = ap.parse_args()
    set_seed()

    agent = Agent()

    if args.message:
        import json

        print(json.dumps(agent.handle(args.message), indent=2, ensure_ascii=False))
        return

    if not args.golden:
        die("pass --golden or --message")

    golden = read_jsonl(GOLDEN)
    if not golden:
        die(f"{GOLDEN} is empty - label the golden set first (tools/label_cli.py)")
    if args.limit:
        golden = golden[: args.limit]

    rows = []
    for i, ex in enumerate(golden, 1):
        res = agent.handle(ex["customer_message"])
        res.update({"golden_id": ex["golden_id"], "system": "agent"})
        rows.append(res)
        if i % 10 == 0 or i == len(golden):
            info(f"  {i}/{len(golden)} handled")

    write_jsonl(args.out, rows)
    ok(f"wrote {len(rows)} predictions -> {args.out}")

    from llm import print_stats

    print_stats()
    esc = sum(r["escalate"] for r in rows)
    print(f"\n  escalated: {esc}/{len(rows)} ({esc/len(rows):.0%})")
    print(f"  parse failures: {sum(r['classifier_parse_failed'] for r in rows)}")
    print(f"  mean retrieval score: {np.mean([r['retrieval_score'] for r in rows]):.3f}\n")


if __name__ == "__main__":
    main()
