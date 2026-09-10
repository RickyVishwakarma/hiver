"""Shared construction of the pairwise-comparison set.

WHY THIS IS ITS OWN MODULE
The judge and the human must compare the *same* pairs, in the *same* left/right
order, or the agreement statistic is comparing two different tasks. Building the
set twice - once in judge_pairwise.py, once in score_pairs.py - is exactly the
kind of drift that produces a number nobody can trust. One seeded function, two
callers.

WHY PAIRWISE AT ALL
The 5-axis absolute-score judge scored quadratic-weighted kappa 0.003 against
blind human scores: it measures nothing. Absolute 1-5 quality judgements are
hard for a 3B model - it has no stable internal anchor for what a "4" is, and
its scores collapsed toward 3-4 for everything. A forced choice between two
concrete replies removes the need for an anchor. This module exists to test
whether that reframing rescues the instrument. It might not; the point is to
measure it either way.

WHICH COMPARISONS
Weighted toward the comparisons the report would actually want to make. Pairs
against `trivial` (one canned reply for every message) are easy, and including
many of them would inflate agreement without evidence that the judge can resolve
anything interesting - so they are a minority of the set and validate_judge.py
reports agreement with and without them.
"""
from __future__ import annotations

import random

from common import DATA, GOLDEN, SEED, read_jsonl

# (system_a, system_b, how_many). Deliberately excludes agent_narrow/agent_strict:
# those differ from `agent` only in triage posture and emit byte-identical reply
# text, so pairing them would be comparing a reply against itself.
COMPARISONS = [
    ("agent", "simple", 15),
    ("agent", "agent_zeroshot", 15),
    ("agent", "trivial", 10),
]


def _norm(text: str) -> str:
    return " ".join((text or "").split()).lower()


def build_pairs() -> list[dict]:
    """Deterministic pair set. Same output for every caller, every run."""
    golden = {g["golden_id"]: g for g in read_jsonl(GOLDEN)}

    replies: dict[str, dict[str, str]] = {}
    for path in sorted(DATA.glob("preds_*.jsonl")):
        system = path.stem.replace("preds_", "")
        replies[system] = {
            p["golden_id"]: p.get("reply", "")
            for p in read_jsonl(path)
            if p["golden_id"] in golden and p.get("reply", "").strip()
        }

    # The human and the judge must both see the evidence groundedness is judged
    # against; mirrors the shared-evidence map in judge.py.
    evidence = {
        p["golden_id"]: p.get("retrieved", [])
        for p in read_jsonl(DATA / "preds_agent.jsonl")
        if p.get("retrieved")
    }

    rng = random.Random(SEED)
    pairs: list[dict] = []
    used: set[tuple[str, str, str]] = set()

    for sys_a, sys_b, n in COMPARISONS:
        if sys_a not in replies or sys_b not in replies:
            continue
        # Identical replies carry no signal and would be scored "tie" by
        # construction, padding agreement for free.
        pool = [
            gid
            for gid in sorted(set(replies[sys_a]) & set(replies[sys_b]))
            if _norm(replies[sys_a][gid]) != _norm(replies[sys_b][gid])
        ]
        rng.shuffle(pool)
        for gid in pool:
            if len([p for p in pairs if p["comparison"] == f"{sys_a}_vs_{sys_b}"]) >= n:
                break
            key = (gid, sys_a, sys_b)
            if key in used:
                continue
            used.add(key)
            # Seeded coin decides presentation order, so "A" is not a system.
            flip = rng.random() < 0.5
            left, right = (sys_b, sys_a) if flip else (sys_a, sys_b)
            pairs.append(
                {
                    "golden_id": gid,
                    "comparison": f"{sys_a}_vs_{sys_b}",
                    "customer_message": golden[gid]["customer_message"],
                    "system_a": left,
                    "system_b": right,
                    "reply_a": replies[left][gid],
                    "reply_b": replies[right][gid],
                    "retrieved": evidence.get(gid, []),
                }
            )

    rng.shuffle(pairs)
    for i, p in enumerate(pairs, 1):
        p["pair_id"] = f"p{i:03d}"
    return pairs
