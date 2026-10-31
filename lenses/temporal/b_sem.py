"""Temporal lens B, round 1 ("Semantyka"): b_evcal's per-event calendar for questions that name their events, plus a
separate whole-question route for questions that do not ("what did I do two weeks ago", "order of the six museums",
"which airline on Valentine's day").

Event route (parsed kinds apart/after/before/between/since_when/ago_when/ago/first/order-with-events): unchanged
b_evcal.rank_event (date-bearing pool U hybrid, cross-encoder + subject/contrast/year features), dates resolved in
code, gap in the hint, top-1 session per event cited.

Whole-question route (no events parsed): hybrid(question) U anchor-window hybrid -> cross-encoder on the question
-> + time prior toward the question's anchor ("two weeks ago") + small user-turn prior -> pack by score,
cite every session whose best unit is within `cite_gap` logits of the top one (capped by the number the question
names: "the three trips" -> 3).
"""
from __future__ import annotations

import math
import re
from datetime import timedelta

from core.types import EvidencePack
from lenses.temporal import b_evcal as E

_BASE = dict(E.VARIANTS["evcal"])
VARIANTS = {
    # r1 step 1: whole-question route for unparsed questions
    "sem": {**_BASE, "route_other": True, "k_other": 60, "budget_other": 10000, "cite_gap": 3.0, "w_user": 1.0,
            "cite_max": 3},
    # same, computed gap only when the per-event margin is clear
    # same routes, but bi-encoder cosine (bge-small, cached unit vectors) instead of the cross-encoder: one embed call
    # per query instead of a CE pass over 40-80 units; pseudo-logit = dense_scale * (cos - dense_mid)
    "sem_dense": {**_BASE, "route_other": True, "k_other": 60, "budget_other": 10000, "cite_gap": 3.0, "w_user": 1.0,
                  "cite_max": 3, "scorer": "dense", "dense_scale": 25.0, "dense_mid": 0.6},
    # r1 step 2: whole-question route over USER turns only (all dev gold turns are user statements "... today"),
    # no neighbours, so the budget holds more sessions (order-of-N / "last time" questions need all of them)
    "sem_user": {**_BASE, "conf_min": 0.6, "route_other": True, "k_other": 60, "budget_other": 10000,
                 "cite_gap": 3.0, "w_user": 0.0, "cite_max": 3, "other_roles": "user", "window_other": 0},
    # r1 step 3: event route also packs every USER unit of the cited sessions that shares an event token (dev "first"
    # questions had the right sessions but missed the second gold turn of a session: ATR 0.5)
    "sem_turns": {**_BASE, "conf_min": 0.6, "route_other": True, "k_other": 60, "budget_other": 10000,
                  "cite_gap": 3.0, "w_user": 0.0, "cite_max": 3, "other_roles": "user", "window_other": 0,
                  "sess_user_fill": True},
    # r1 step 4: same as sem_turns with a 9k event-route budget (sem_turns lost ALL to the extra units at 7k)
    "sem_turns9": {**_BASE, "conf_min": 0.6, "route_other": True, "k_other": 60, "budget_other": 10000,
                   "cite_gap": 3.0, "w_user": 0.0, "cite_max": 3, "other_roles": "user", "window_other": 0,
                   "sess_user_fill": True, "budget": 9000},
    "sem_gated": {**_BASE, "conf_min": 0.6, "route_other": True, "k_other": 60, "budget_other": 10000,
                  "cite_gap": 3.0, "w_user": 1.0, "cite_max": 3},
}

_NUMW = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten".split())}
_COUNT = re.compile(r"\b(?:the|these|those|my)\s+(two|three|four|five|six|seven|eight|nine|ten|\d)\b", re.I)


def expected_items(text: str) -> int | None:
    """'the order of the three trips' / 'the six museums' / 'among Emma, Rachel and Alex' -> 3 / 6 / 3."""
    m = _COUNT.search(text)
    if m:
        w = m.group(1).lower()
        return int(w) if w.isdigit() else _NUMW.get(w)
    m = re.search(r"\bamong\s+([^?]+)", text, re.I)
    if m:
        return len(re.split(r",\s*(?:and\s+)?|\s+and\s+", m.group(1).strip()))
    return None


def rank_question(hx, q, p: dict) -> list[dict]:
    qtext = q.question
    anchor = E.question_anchor(qtext, q.date)
    roles = p.get("other_roles")
    pool = {i for i, _ in hx.hybrid(qtext, k=p["k_other"], roles=roles)}
    if anchor is not None:
        lo, hi = anchor[0] - timedelta(days=3 * anchor[1] + 2), anchor[0] + timedelta(days=anchor[1] + 2)
        win = [u.uid for u in hx.units if lo <= u.date.date() <= hi]
        if win:
            pool |= {i for i, _ in hx.hybrid(qtext, k=20, uids=win, roles=roles)}
    pool = [i for i in pool if hx.units[i].date <= q.date + timedelta(days=1)]
    if not pool:
        return []
    rows = []
    for uid, ce in hx.rerank(qtext, pool):
        u = hx.units[uid]
        f = ce + p["w_user"] * (u.role == "user")
        if anchor is not None:
            f += 2.0 * math.exp(-abs((u.date.date() - anchor[0]).days) / max(2, anchor[1]))
        rows.append({"uid": uid, "score": f, "ce": ce, "session_id": u.session_id})
    rows.sort(key=lambda r: -r["score"])
    return rows


def cite_sessions(rows: list[dict], gap: float, cap: int) -> list[str]:
    best: dict[str, float] = {}
    for r in rows:
        best.setdefault(r["session_id"], r["score"])
    if not best:
        return []
    top = max(best.values())
    sids = [s for s, v in sorted(best.items(), key=lambda x: -x[1]) if v >= top - gap]
    return sids[:max(1, cap)]


def build_other(q, hx, p: dict) -> EvidencePack:
    from core.pack import build_pack

    rows = rank_question(hx, q, p)
    order_uids = [r["uid"] for r in rows]
    passages, included, _ = build_pack(q, hx, order_uids, budget=p["budget_other"],
                                       window=p.get("window_other", p["window"]),
                                       order="chrono")
    n = expected_items(q.question)
    cite = cite_sessions(rows, p["cite_gap"], n if n else p["cite_max"])
    in_pack = {x.id for x in passages}
    cite = [s for s in cite if s in in_pack] or None
    best = rows[0]["ce"] if rows else -12.0
    abstain_p = round(E._sigmoid(-(best + 2.0) / 2.0), 3)
    return EvidencePack(question_id=q.question_id, variant=p.get("_variant", ""), route="temporal", passages=passages,
                        hint="", cite_plan=cite, abstain_p=abstain_p, included_turns=included, computed=None,
                        debug={"ranking": order_uids, "spec": {"kind": "other"}, "route": "question",
                               "n_expected": n})


class DenseHX:
    """Wraps a HistoryIndex so that .rerank() returns scaled bi-encoder cosines (no cross-encoder call)."""

    def __init__(self, hx, scale: float, mid: float):
        self._hx, self._scale, self._mid = hx, scale, mid

    def __getattr__(self, name):
        return getattr(self._hx, name)

    def rerank(self, query: str, uids: list[int]) -> list[tuple[int, float]]:
        uids = list(dict.fromkeys(uids))
        if not uids:
            return []
        return [(u, self._scale * (s - self._mid)) for u, s in self._hx.dense(query, k=len(uids), uids=uids)]


def build_event(q, hx, p: dict) -> EvidencePack:
    from core.pack import build_pack

    pk = E.build(q, hx, **p)
    if not p.get("sess_user_fill") or not pk.cite_plan:
        return pk
    evs = pk.debug.get("spec", {}).get("events") or []
    tok = set().union(*(E.content_tokens(e) for e in evs)) if evs else set()
    cited = set(pk.cite_plan)
    extra = [u.uid for u in hx.units if u.session_id in cited and u.role == "user" and (E.content_tokens(u.text) & tok)]
    ranking = list(pk.debug.get("ranking") or [])
    n_top = p["alts"] * max(1, len(evs))
    order = list(dict.fromkeys(ranking[:n_top] + extra + ranking[n_top:]))
    passages, included, _ = build_pack(q, hx, order, budget=p["budget"], window=p["window"], order="chrono",
                                       hint=pk.hint)
    pk.passages, pk.included_turns = passages, included
    pk.debug["ranking"] = order
    return pk


def build(q, hx, **params) -> EvidencePack:
    p = {**VARIANTS["sem"], **params}
    if p.get("scorer") == "dense":
        hx = DenseHX(hx, p["dense_scale"], p["dense_mid"])
    spec = E.parse_question(q.question)
    if p.get("route_other") and not spec["events"]:
        return build_other(q, hx, p)
    return build_event(q, hx, p)
