"""Generic lens: the kit's naive keyword baseline (harness sanity check) and a hybrid + rerank default."""
from __future__ import annotations

import re

from core.pack import build_pack
from core.types import EvidencePack, Passage

VARIANTS = {
    "naive": {"mode": "naive"},
    "hybrid_rr": {"mode": "hybrid", "pool": 300, "rerank_top": 100, "budget": 12000, "window": 1},
}


def _naive(q, hx, k: int = 4, budget: int = 12000) -> EvidencePack:
    words = set(re.findall(r"\w{4,}", q.question.lower()))
    scored = []
    for s in hx.sessions:
        for t in s["turns"]:
            ov = len(words & set(re.findall(r"\w{4,}", t["content"].lower())))
            if ov:
                scored.append((ov, s, t))
    scored.sort(key=lambda x: -x[0])
    passages, turns = [], set()
    for _, s, t in scored[:k]:
        passages.append(Passage(id=s["session_id"], date=s["date_str"], text=f"{t['role']}: {t['content']}"[: budget // k]))
        turns.add(t["turn_id"])
    return EvidencePack(q.question_id, "naive", "generic", passages, included_turns=turns)


def build(q, hx, _variant: str = "", mode: str = "hybrid", pool: int = 300, rerank_top: int = 100,
          budget: int = 12000, window: int = 1, **_) -> EvidencePack:
    if mode == "naive":
        return _naive(q, hx)
    cand = hx.hybrid(q.question, k=rerank_top, pool=pool)
    ranked = hx.rerank(q.question, [u for u, _ in cand])
    passages, turns, _chars = build_pack(q, hx, ranked, budget=budget, window=window)
    top = ranked[0][1] if ranked else -20.0
    abstain_p = 1 / (1 + pow(2.718281828, top + 2))  # crude: low best rerank logit -> likely unanswerable
    return EvidencePack(q.question_id, _variant, "generic", passages, included_turns=turns, abstain_p=abstain_p,
                        debug={"ranking": [u for u, _ in ranked], "top_logit": top})
