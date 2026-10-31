"""Temporal lens B, round 2: a copy of the integrator's merged_r1 (gap questions -> lens A ftb_bm25, everything else ->
b_sem sem_turns9) with two B-side changes, each switchable per variant:

1. Gap fallback (`fb`): when lens A emits no computed value on a gap question, run b_sem's event route (cross-encoder
   event picks, gated by the per-event margin) and, if it computes a gap, put B's dated picks + gap into the hint and
   `computed`. A's passages stay (B's event sessions are pinned in if they are missing). Brief r2 item (3).
2. Session-centric whole-question route (`sess`): for questions with no parsed events ("what did I do two weeks ago",
   "order of the three trips"), rank sessions by their best user-unit score (CE on the question + time prior to the
   anchor), keep only the top K sessions (K from "the three ...", else `sess_k`), score EVERY user unit of those
   sessions (not just the hybrid pool) and pack them by score. Fewer passages (Sall), and the gold turn of a kept
   session is no longer lost to a sibling unit (ATR). Brief r2 items (1) and (4).
   `w_today`: extra score for a user unit that dates its own event ("today", "yesterday", "just", "this morning") when
   the question carries a relative anchor ("X ago", "last Saturday").
"""
from __future__ import annotations

import math
import re
from datetime import timedelta

from core.types import EvidencePack
from core.loader import load_lens

_GAP = re.compile(r"how many (days|weeks|months)\b.*\b(apart|after|before|between|passed)|days? (apart|after|before)", re.I)
_A = load_lens("lenses/temporal/a_calendar.py")
_B = load_lens("lenses/temporal/b_sem.py")
from lenses.temporal import b_evcal as E  # noqa: E402

_TODAY = re.compile(r"\b(today|yesterday|this morning|this afternoon|tonight|last night|just|earlier this week)\b", re.I)
_TIME = re.compile(r"\b(?:(?:a|one|two|three|four|five|six|seven|eight|nine|ten|\d+|a couple of|a few)\s+"
                   r"(?:days?|weeks?|months?|years?)\s+ago"
                   r"|(?:last|past|this)\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|week(?:end)?|month)"
                   r"|on the (?:monday|tuesday|wednesday|thursday|friday|saturday|sunday))\b", re.I)

_SEM = {"a_variant": "ftb_bm25", "b_variant": "sem_turns9"}
VARIANTS = {
    "base": {**_SEM},                                                   # == merged_r1 gapA_semB
    "fb": {**_SEM, "fb": True},                                         # + B fallback computed on gap questions
    "sess5": {**_SEM, "sess": True, "sess_k": 5, "sess_gap": 99.0, "w_today": 0.0, "budget_sess": 10000},
    "sess5t": {**_SEM, "sess": True, "sess_k": 5, "sess_gap": 99.0, "w_today": 1.5, "budget_sess": 10000},
    "ev1_mix_g": {**_SEM, "fb": True, "fb_guard": True, "cite_wide": True, "cite_gap8": 8.0, "cite_top": 4,
                  "ev_fill": True, "fill_cap": 3, "ev_extra": 1, "oq_k": 99, "oq_mix": True},
    # whole-question route: cite the top `cite_top` sessions (gap 8 = nearly always the top N; gold has ~3 sessions)
    "fb_c4": {**_SEM, "fb": True, "cite_wide": True, "cite_gap8": 8.0, "cite_top": 4},
    "fb_c3": {**_SEM, "fb": True, "cite_wide": True, "cite_gap8": 8.0, "cite_top": 3},
    # + event route: the cited sessions' user units that share event tokens enter right after the picks, ranked by
    # overlap, at most `fill_cap` per session, without neighbours (r1 sem_turns added them in uid order with +/-1
    # neighbours, so the budget ran out before the second gold turn of "first" questions)
    "fb_c4_ev": {**_SEM, "fb": True, "cite_wide": True, "cite_gap8": 8.0, "cite_top": 4, "ev_fill": True,
                 "fill_cap": 3},
    # + event route packs at most len(cite_plan) + `ev_extra` sessions (passages close to the cite set -> Sall)
    "fb_c4_ev1": {**_SEM, "fb": True, "cite_wide": True, "cite_gap8": 8.0, "cite_top": 4, "ev_fill": True,
                  "fill_cap": 3, "ev_extra": 1},
    "fb_c4_ev2": {**_SEM, "fb": True, "cite_wide": True, "cite_gap8": 8.0, "cite_top": 4, "ev_fill": True,
                  "fill_cap": 3, "ev_extra": 2},
    # + whole-question route packs at most `oq_k` sessions (r1: 13.8 passages for ~3 gold sessions)
    "ev1_k8": {**_SEM, "fb": True, "cite_wide": True, "cite_gap8": 8.0, "cite_top": 4, "ev_fill": True,
               "fill_cap": 3, "ev_extra": 1, "oq_k": 8},
    "ev1_k6": {**_SEM, "fb": True, "cite_wide": True, "cite_gap8": 8.0, "cite_top": 4, "ev_fill": True,
               "fill_cap": 3, "ev_extra": 1, "oq_k": 6},
    # + whole-question route packs units from two rankings interleaved: with the anchor time prior and without it
    # (the gold set of "what did I do two weeks ago" also holds same-category sessions at other dates)
    "ev1_mix": {**_SEM, "fb": True, "cite_wide": True, "cite_gap8": 8.0, "cite_top": 4, "ev_fill": True,
                "fill_cap": 3, "ev_extra": 1, "oq_k": 99, "oq_mix": True},
    "ev1_mix_k8": {**_SEM, "fb": True, "cite_wide": True, "cite_gap8": 8.0, "cite_top": 4, "ev_fill": True,
                   "fill_cap": 3, "ev_extra": 1, "oq_k": 8, "oq_mix": True},
}


# ---------------------------------------------------------------------------------------------------------------------
# whole-question route, session-centric
# ---------------------------------------------------------------------------------------------------------------------
def _score_units(hx, q, qtext, uids, anchor, w_today, w_prior=2.0):
    rows = []
    for uid, ce in hx.rerank(qtext, list(uids)):
        u = hx.units[uid]
        f = ce
        if anchor is not None:
            f += w_prior * math.exp(-abs((u.date.date() - anchor[0]).days) / max(2, anchor[1]))
            if w_today and _TODAY.search(u.text):
                f += w_today
        rows.append({"uid": uid, "score": f, "ce": ce, "session_id": u.session_id})
    rows.sort(key=lambda r: -r["score"])
    return rows


def build_sess(q, hx, p: dict) -> EvidencePack:
    from core.pack import build_pack

    sp = {**_B.VARIANTS[p["b_variant"]]}
    qtext = q.question
    anchor = E.question_anchor(qtext, q.date)
    rows = _B.rank_question(hx, q, sp)                          # hybrid(user) U anchor window -> CE + time prior
    if p.get("w_today") and anchor is not None:
        for r in rows:
            if _TODAY.search(hx.units[r["uid"]].text):
                r["score"] += p["w_today"]
        rows.sort(key=lambda r: -r["score"])
    sess_best: dict[str, float] = {}
    for r in rows:
        sess_best.setdefault(r["session_id"], r["score"])
    n = _B.expected_items(qtext)
    k = max(p["sess_k"], (n or 0) + 1)
    top = max(sess_best.values()) if sess_best else 0.0
    keep = [s for s, v in sorted(sess_best.items(), key=lambda x: -x[1]) if v >= top - p["sess_gap"]][:k]
    # every user unit of the kept sessions, scored on the question
    cutoff = q.date + timedelta(days=1)
    extra = [u.uid for u in hx.units if u.session_id in set(keep) and u.role == "user" and u.date <= cutoff]
    srows = _score_units(hx, q, qtext, extra, anchor, p.get("w_today", 0.0)) if extra else []
    rank_s = {s: i for i, s in enumerate(keep)}
    # order: best unit of each kept session first (by session rank), then the rest by score
    firsts, seen = [], set()
    for r in srows:
        if r["session_id"] not in seen:
            seen.add(r["session_id"])
            firsts.append(r)
    firsts.sort(key=lambda r: rank_s.get(r["session_id"], 99))
    order = list(dict.fromkeys([r["uid"] for r in firsts] + [r["uid"] for r in srows]))
    passages, included, _ = build_pack(q, hx, order, budget=p["budget_sess"], window=0, order="chrono",
                                       max_sessions=k)
    cite = _B.cite_sessions(rows, sp["cite_gap"], n if n else sp["cite_max"])
    in_pack = {x.id for x in passages}
    cite = [s for s in cite if s in in_pack] or None
    best = rows[0]["ce"] if rows else -12.0
    abstain_p = round(E._sigmoid(-(best + 2.0) / 2.0), 3)
    return EvidencePack(question_id=q.question_id, variant=p.get("_variant", ""), route="temporal", passages=passages,
                        hint="", cite_plan=cite, abstain_p=abstain_p, included_turns=included, computed=None,
                        debug={"ranking": order + [r["uid"] for r in rows if r["uid"] not in set(order)],
                               "spec": {"kind": "other"}, "route": "question_sess", "n_expected": n, "keep": keep})


# ---------------------------------------------------------------------------------------------------------------------
# gap fallback: B's event picks when A has no computed value
# ---------------------------------------------------------------------------------------------------------------------
def _fallback_sane(bpk, picks) -> bool:
    """General sanity checks before B's computed gap is used: every picked sentence shares a content word with its
    event and is not about someone else, and 'after' / 'before' questions get dates in the stated order."""
    spec = (bpk.debug or {}).get("spec") or {}
    if len(picks) < 2:
        return True
    for x in picks:
        if x.get("subj") == "other":
            return False
        if not (E.content_tokens(x.get("sent", "")) & E.content_tokens(x["event"])):
            return False
    d0, d1 = str(picks[0]["date"]), str(picks[1]["date"])
    if spec.get("kind") in ("after", "since_when") and d1 < d0:
        return False
    if spec.get("kind") == "before" and d1 > d0:
        return False
    return True


def gap_fallback(q, hx, pack, p: dict):
    from core.pack import build_pack

    bv = p["b_variant"]
    bpk = _B.build(q, hx, _variant=bv, **_B.VARIANTS[bv])
    if bpk.computed is None:
        return pack, None
    picks = (bpk.debug or {}).get("picks") or []
    if p.get("fb_guard") and not _fallback_sane(bpk, picks):
        return pack, None
    pin = [x["uid"] for x in picks if "uid" in x]
    have = {x.id for x in pack.passages}
    need = [u for u in pin if hx.units[u].session_id not in have]
    if need:   # rebuild A's pack with B's event units pinned in front of A's own ranking
        ranking = list((pack.debug or {}).get("ranking") or [])
        passages, included, _ = build_pack(q, hx, ranking, budget=p.get("fb_budget", 9000), window=1, order="chrono",
                                           hint=bpk.hint, pinned=need)
        pack.passages, pack.included_turns = passages, included
    pack.hint = bpk.hint
    pack.computed = bpk.computed
    cites = list(dict.fromkeys([hx.units[u].session_id for u in pin]))
    pack.cite_plan = [s for s in cites if s in {x.id for x in pack.passages}] or pack.cite_plan
    return pack, bpk.computed


def build_other_k(q, hx, p: dict) -> EvidencePack:
    """b_sem whole-question route, but at most max(oq_k, n_expected + 1) sessions in the pack."""
    from core.pack import build_pack

    sp = {**_B.VARIANTS[p["b_variant"]]}
    rows = _B.rank_question(hx, q, sp)
    order_uids = [r["uid"] for r in rows]
    if p.get("oq_mix"):
        plain = [r["uid"] for r in sorted(rows, key=lambda r: -r["ce"])]
        order_uids = list(dict.fromkeys(x for pair in zip(order_uids, plain) for x in pair))
    n = _B.expected_items(q.question)
    k = max(p["oq_k"], (n or 0) + 1)
    passages, included, _ = build_pack(q, hx, order_uids, budget=sp["budget_other"], window=sp.get("window_other", 0),
                                       order="chrono", max_sessions=k)
    cite = _B.cite_sessions(rows, sp["cite_gap"], n if n else sp["cite_max"])
    in_pack = {x.id for x in passages}
    cite = [s for s in cite if s in in_pack] or None
    best = rows[0]["ce"] if rows else -12.0
    abstain_p = round(E._sigmoid(-(best + 2.0) / 2.0), 3)
    return EvidencePack(question_id=q.question_id, variant=p.get("_variant", ""), route="temporal", passages=passages,
                        hint="", cite_plan=cite, abstain_p=abstain_p, included_turns=included, computed=None,
                        debug={"ranking": order_uids, "spec": {"kind": "other"}, "route": "question_k",
                               "n_expected": n})


def build_event2(q, hx, p: dict) -> EvidencePack:
    """b_sem event route with a better session fill (see VARIANTS fb_c4_ev)."""
    from core.pack import build_pack

    sp = {**_B.VARIANTS[p["b_variant"]]}
    pk = E.build(q, hx, **sp)
    if not pk.cite_plan:
        return pk
    evs = pk.debug.get("spec", {}).get("events") or []
    tok = set().union(*(E.content_tokens(e) for e in evs)) if evs else set()
    cited = list(pk.cite_plan)
    ranking = list(pk.debug.get("ranking") or [])
    n_top = sp["alts"] * max(1, len(evs))
    w = sp["window"]
    head = []
    for uid in ranking[:n_top]:
        head += [uid] + [j for j in hx.neighbors(uid, w, w) if j != uid]
    per = {s: [] for s in cited}
    for u in hx.units:
        if u.session_id in per and u.role == "user":
            ov = len(E.content_tokens(u.text) & tok)
            if ov:
                per[u.session_id].append((-ov, u.uid))
    extra = []
    for j in range(p.get("fill_cap", 3)):          # interleave sessions: best-overlap unit of each cited session first
        for s in cited:
            lst = sorted(per[s])
            if j < len(lst):
                extra.append(lst[j][1])
    order = list(dict.fromkeys(head + extra + ranking[n_top:]))
    ms = len(cited) + p["ev_extra"] if p.get("ev_extra") is not None else None
    passages, included, _ = build_pack(q, hx, order, budget=sp["budget"], window=0, order="chrono", hint=pk.hint,
                                       max_sessions=ms)
    pk.passages, pk.included_turns = passages, included
    pk.debug["ranking"] = order
    return pk


def build(q, hx, _variant: str = "", **params):
    p = {**_SEM, **params}
    use_a = bool(_GAP.search(q.question))
    if use_a:
        av = p["a_variant"]
        pack = _A.build(q, hx, _variant=av, **_A.VARIANTS[av])
        src = f"a_calendar:{av}"
        if p.get("fb") and pack.computed is None:
            pack, comp = gap_fallback(q, hx, pack, p)
            if comp is not None:
                src += "+b_fallback"
    else:
        bv = p["b_variant"]
        spec = E.parse_question(q.question)
        if p.get("sess") and not spec["events"]:
            pack = build_sess(q, hx, p)
            src = "b_merge:sess"
        elif p.get("oq_k") and not spec["events"]:
            pack = build_other_k(q, hx, p)
            src = "b_merge:other_k"
        elif p.get("ev_fill") and spec["events"]:
            pack = build_event2(q, hx, p)
            src = "b_merge:event2"
        else:
            pack = _B.build(q, hx, _variant=bv, **_B.VARIANTS[bv])
            src = f"b_sem:{bv}"
    if p.get("cite_wide") and not use_a and not E.parse_question(q.question)["events"] and not p.get("sess"):
        rows = _B.rank_question(hx, q, {**_B.VARIANTS[p["b_variant"]]})       # cached CE: same rows as the pack
        n = _B.expected_items(q.question)
        cite = _B.cite_sessions(rows, p["cite_gap8"], n if n else p["cite_top"])
        in_pack = {x.id for x in pack.passages}
        pack.cite_plan = [s for s in cite if s in in_pack] or pack.cite_plan
    pack.variant = _variant
    pack.debug = {**(pack.debug or {}), "merged_from": src}
    return pack
