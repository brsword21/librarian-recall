"""Lens B ("Semantyka") for multi-session, ROUND 2: citation-robust packing.

Problem (round 1, deep8_10k): the pack carries ~8 sessions (test: ~12) for 2-3 gold sessions. If Reactor cites every
passage we send, CF1 falls from 0.90 to 0.47 (S 86.6 -> Sall 73.9). This lens keeps the round-1 retrieval
(multi-query hybrid -> CE on the top 90 -> intra-session CE of the top-15 sessions' user units) and then
**prunes the pack to the sessions it would cite**:

1. session score = best CE logit of the session's units (rmax), optionally max'd with a sentence-level CE score
   (hx.rerank_texts over the user sentences of the top sessions: a one-sentence fact inside a 700-char unit is diluted
   at unit level; 'presented a poster' for 'projects led').
2. keep = the session-score prefix of length clamp(#sessions within `keep_d` logits of the top, keep_min, keep_max),
   plus each facet's top session ('car cover and detailing spray': every facet must be represented).
   Time-window prior on test-scale histories (span >= 45 days): sessions outside the question window lose `win_pen`
   logits before the cut.
3. depth: up to `per_sess` units per kept session (session-first, then depth), sentence-trimmed snippets, `budget`.
4. cite_plan = the kept sessions in the pack (pack == cite set, so S == Sall) or the round-1 'rel' rule inside them.

Helpers (question analysis, hint, abstention, selection, trimming, rendering) are imported from b_sweep.py (round 1).
"""
from __future__ import annotations

import math
import re
from collections import defaultdict

from core.types import EvidencePack

from lenses.multi import b_sweep as R1

BASE = dict(R1.VARIANTS["deep8_10k"])
BASE.update(keep_d=8.0, keep_min=2, keep_max=8, sent=False, sent_top=10, sent_off=0.0, win_pen=3.0,
            cite_mode="keep", facet_keep=True, latest_line=True)


def _v(**kw):
    d = dict(BASE)
    d.update(kw)
    return d


VARIANTS = {
    # pruning by unit-level CE (rmax) only
    "k8m2": _v(keep_d=8.0, keep_min=2, keep_max=8),
    "k10m2": _v(keep_d=10.0, keep_min=2, keep_max=8),
    "k7m3": _v(keep_d=7.0, keep_min=3, keep_max=8),
    # + sentence-level CE in the session score
    "s8m2": _v(keep_d=8.0, keep_min=2, keep_max=8, sent=True),
    "s7m3": _v(keep_d=7.0, keep_min=3, keep_max=8, sent=True),
    "s10m2": _v(keep_d=10.0, keep_min=2, keep_max=8, sent=True),
    # cite subset inside the kept sessions (S > Sall)
    "k8m2_rel": _v(keep_d=8.0, keep_min=2, keep_max=8, cite_mode="rel"),
    # p(gold session) model (5 features, logistic, fit on dev only: scratch_b/keep_fit.py); keep p >= tk
    "p10": _v(keep="model", tk=0.10, keep_min=2, keep_max=8, facet_keep=False),
    "p08": _v(keep="model", tk=0.08, keep_min=2, keep_max=8, facet_keep=False),
    "p15": _v(keep="model", tk=0.15, keep_min=2, keep_max=8, facet_keep=False),
    "p10f": _v(keep="model", tk=0.10, keep_min=2, keep_max=8, facet_keep=True),
    "p10c35": _v(keep="model", tk=0.10, keep_min=2, keep_max=8, facet_keep=False, cite_mode="p", tc=0.35),
    "p08c35": _v(keep="model", tk=0.08, keep_min=2, keep_max=8, facet_keep=False, cite_mode="p", tc=0.35),
    "p06": _v(keep="model", tk=0.06, keep_min=2, keep_max=8, facet_keep=False),
    # + lens A's conjunct / pair / title absence check for abstain_p (a_tally.abstain_score; credit: multi-A)
    "p08c35_amax": _v(keep="model", tk=0.08, keep_min=2, keep_max=8, facet_keep=False, cite_mode="p", tc=0.35,
                      abst="max"),
    "p08c35_aor": _v(keep="model", tk=0.08, keep_min=2, keep_max=8, facet_keep=False, cite_mode="p", tc=0.35,
                     abst="or"),
}

# logistic p(session is gold) over [gap to top CE, rank (<=10), question-term coverage of the session's user text,
# units in the top-40 ranking (<=10), log1p(rank of the session's first unit)]; fit on dev multi (121 answerable q)
P_COEF = (-0.3061, -0.3145, 0.961, 0.3231, -1.3884)
P_ICPT = 3.1546


def p_gold(f: dict) -> float:
    x = (f["gap"], min(f["r"], 10), f["cov"], min(f["n40"], 10), math.log1p(f["first"]))
    z = P_ICPT + sum(c * v for c, v in zip(P_COEF, x))
    return 1.0 / (1.0 + math.exp(-z))


# ------------------------------------------------------------------------------------------------------------------

def _sent_scores(q, hx, sids: list[str], max_sents: int = 450) -> tuple[dict[str, float], dict[int, float]]:
    """Sentence-level CE: best sentence logit per session and per unit, over the user units of `sids`."""
    texts, owners = [], []
    by_sess: dict[str, list[int]] = defaultdict(list)
    for u in hx.units:
        if u.session_id in sids and u.role == "user":
            by_sess[u.session_id].append(u.uid)
    for sid in sids:
        for uid in by_sess.get(sid, []):
            for s in R1.split_sents(hx.units[uid].text):
                if 20 <= len(s) <= 400:
                    texts.append(s)
                    owners.append(uid)
    texts, owners = texts[:max_sents], owners[:max_sents]
    sc = hx.rerank_texts(q.question, texts) if texts else []
    smax: dict[str, float] = {}
    umax: dict[int, float] = {}
    for uid, v in zip(owners, sc):
        sid = hx.units[uid].session_id
        smax[sid] = max(smax.get(sid, -99.0), v)
        umax[uid] = max(umax.get(uid, -99.0), v)
    return smax, umax


def _features(hx, a, order, score, rmax, smax, ranked, facet_top, n_top: int = 25) -> dict:
    """Per-session features for the top sessions (offline fit of the keep rule: scratch_b/keep_fit.py)."""
    first: dict[str, int] = {}
    n40: dict[str, int] = defaultdict(int)
    for r, u in enumerate(ranked):
        sid = hx.units[u].session_id
        first.setdefault(sid, r)
        if r < 40:
            n40[sid] += 1
    keys = {R1._stem5(w) for w in a["terms"]}
    ftop = {hx.units[us[0]].session_id for us in facet_top.values() if us}
    top = score[order[0]] if order else -99.0
    within4 = sum(1 for s in order if score[s] >= top - 4)
    out = {}
    for i, sid in enumerate(order[:n_top]):
        toks = set()
        for t in hx.sess_by_id[sid]["turns"]:
            if t["role"] == "user":
                toks |= {R1._stem5(w) for w in re.findall(r"[a-z0-9][a-z0-9'\-]*", t["content"].lower())}
        cov = len(keys & toks) / max(1, len(keys))
        nxt = score[order[i + 1]] if i + 1 < len(order) else score[sid] - 5
        out[sid] = dict(r=i, gap=round(top - score[sid], 3), gn=round(score[sid] - nxt, 3), rmax=round(rmax.get(sid, -99), 3),
                        smax=round(smax.get(sid, -99), 3), n40=n40.get(sid, 0), first=first.get(sid, 999),
                        cov=round(cov, 3), ft=int(sid in ftop), top=round(top, 3), w4=within4)
    return out


def _in_window(d, win) -> bool:
    if not win:
        return True
    return (win["start"] - d).days <= 3 and (d - win["end"]).days <= 10


def build(q, hx, **params) -> EvidencePack:
    P = dict(BASE)
    P.update(params)
    a = R1.analyze(q.question, q.date)
    terms = a["terms"] + [w for f in a["facets"] for w in R1.content_terms(f)]
    ranked, final, rr, lab, facet_top, rest, win_active = R1._score_units(q, hx, a, P)

    # ---- abstention (round 1) -----------------------------------------------------------------------------------
    miss = R1.missing_terms(hx, a["terms"])
    miss_frac = len(miss) / max(1, len([w for w in a["terms"] if w not in R1._GENERIC and not w.isdigit()]))
    top_rr = max(rr.values()) if rr else -10.0
    head = [u for u in ranked[:10] if R1.focus_sentences(hx.units[u].text, a["terms"])]
    all_other = bool(head) and not a["others_ok"] and all(lab[u] == "other" for u in head[:6])
    abstain_p = R1.abstain_prob(top_rr, miss_frac, all_other)
    hint = R1.build_hint(a, q.date, abstain_p)
    if P.get("latest_line") and a["kind"] in ("count", "sum") and not a["latest"] and not a.get("rate"):
        hint += " If a later session restates the same item with a new number, the latest statement wins."

    # ---- session scores -----------------------------------------------------------------------------------------
    rmax: dict[str, float] = {}
    for u, sc in rr.items():
        unit = hx.units[u]
        if P["asst"] != "none" or unit.role == "user":
            rmax[unit.session_id] = max(rmax.get(unit.session_id, -99.0), sc)
    score = dict(rmax)
    smax: dict[str, float] = {}
    if P.get("sent"):
        top_s = sorted(rmax, key=lambda s: -rmax[s])[:P["sent_top"]]
        smax, umax = _sent_scores(q, hx, set(top_s))
        for sid, v in smax.items():
            score[sid] = max(score.get(sid, -99.0), v - P["sent_off"])      # max-fusion
    if win_active and a["window"]:
        for sid in score:
            if not _in_window(hx.sess_by_id[sid]["date"].date(), a["window"]):
                score[sid] -= P["win_pen"]
    order = sorted(score, key=lambda s: -score[s])
    top = score[order[0]] if order else -99.0
    feat = _features(hx, a, order, score, rmax, smax, ranked, facet_top)
    pg = {sid: p_gold(f) for sid, f in feat.items()}
    if P.get("keep") == "model":
        keep = [s for s in order if pg.get(s, 0.0) >= P["tk"]][:P["keep_max"]]
        if len(keep) < P["keep_min"]:
            keep = order[:P["keep_min"]]
        n = len(keep)
    else:
        n = sum(1 for s in order if score[s] >= top - P["keep_d"])
        n = max(P["keep_min"], min(P["keep_max"], n))
        keep = order[:n]
    if P.get("facet_keep"):
        for us in facet_top.values():
            for u in us[:1]:
                sid = hx.units[u].session_id
                if sid not in keep:
                    keep.append(sid)
    keep_set = set(keep)

    # ---- selection inside the kept sessions ---------------------------------------------------------------------
    ranked_k = [u for u in ranked if hx.units[u].session_id in keep_set]
    ft_k = {f: [u for u in us if hx.units[u].session_id in keep_set] for f, us in facet_top.items()}
    Pk = dict(P)
    Pk["max_sess"] = len(keep)
    Pk["deep_top"] = len(keep)
    picks = R1._select(hx, ranked_k, ft_k, Pk)
    passages, included, chars, chosen = R1._pack_trimmed(q, hx, picks, hint, Pk, terms)
    in_pack = {p.id for p in passages}

    if P["cite_mode"] == "rel":
        o = [s for s in order if s in in_pack]
        t = score[o[0]] if o else -99.0
        cite = [s for s in o if score[s] >= t - P.get("cite_d", 4.0)][:P.get("cite_max", 6)]
        if len(cite) < 2:
            cite = o[:2]
    elif P["cite_mode"] == "p":
        o = [s for s in order if s in in_pack]
        cite = [s for s in o if pg.get(s, 0.0) >= P["tc"]]
        if len(cite) < 2:
            cite = o[:2]
    else:
        cite = [s for s in order if s in in_pack]

    sess_feat = {sid: dict(f, p=round(pg[sid], 4)) for sid, f in feat.items()}
    ap_a, why_a = None, None
    if P.get("abst"):
        from lenses.multi import a_tally as A
        qa = A.analyze(q.question, q.date)
        use_win = A.window_applies(hx, qa["window"])
        n_in = sum(1 for sid in in_pack if qa["window"] and A._in_window(hx.sess_by_id[sid]["date"], qa["window"]))
        ap_a, why_a = A.abstain_score(hx, qa, use_win, n_in)
        if P["abst"] == "max":
            abstain_p = max(abstain_p, ap_a)
        else:
            abstain_p = 1.0 - (1.0 - abstain_p) * (1.0 - max(0.0, ap_a - 0.05))
    return EvidencePack(question_id=q.question_id, variant=params.get("_variant", ""), route="multi",
                        passages=passages, hint=hint, cite_plan=cite or None, abstain_p=abstain_p,
                        included_turns=included, computed=None,
                        debug={"ranking": ranked + rest, "keep": keep, "n_keep": n, "top": top,
                               "window_active": win_active, "top_rr": top_rr, "missing_terms": miss,
                               "chars": chars, "sess2": sess_feat, "abst_a": [ap_a, why_a]})
