"""Lens A ("Struktura") for multi-session, ROUND 3: sentence-level instance rescue on top of the frozen b_tight.

v1 multi = b_tight p08c35_aor (pack == likely-gold sessions, ~3.5 per question). Its remaining PA losses are
category -> instance questions: the gold session states the instance in ONE first-person sentence buried in an
off-topic unit ("I viewed a 1-bedroom condo on February 10th" for "how many properties did I view", "my hearing aids"
for "health-related devices"). Unit-level CE puts these sessions 7-14 logits below the top, so the p(gold) model drops
them. Sentence-level bge cosine over ALL own declarative user sentences of the history ranks them 2nd-5th
(scratch_a/r3_sentrank.py), because a short sentence is not diluted by the rest of the unit.

This lens = b_tight.build (copied, unchanged up to the keep set) plus:
1. rescue (brief item 1, 'pinned instance sentences'): sessions not kept whose best own sentence ranks in the top
   len(keep)+r_extra sessions by sentence cosine (and within r_delta of the top), the sentence not declined / not
   someone else's fact, up to r_max sessions; the unit holding the sentence enters right after the first pass.
2. pin: every packed unit keeps its best own sentence when trimmed (the 240-char trim keeps question-term sentences,
   which drops the instance sentence when it shares no term: 'hearing aids' vs 'health-related devices').
cite: rescued sessions are cited only with r_cite (they are low-p by construction).
"""
from __future__ import annotations

import math
import re
from collections import defaultdict

from core.types import EvidencePack, Passage

from lenses.multi import b_sweep as R1
from lenses.multi import b_tight as T

BASE = dict(T.VARIANTS["p08c35_aor"])
BASE.update(resc=False, r_extra=2, r_delta=0.15, r_max=2, r_cite=False, pin=False, r_pos=0, r_min=None, r_fact=False,
            hint_pass=False, r_good=False, r_units=1, r_factb=0.0,
            abst_s=None, r_lex=False)


def _v(**kw):
    d = dict(BASE)
    d.update(kw)
    return d


VARIANTS = {
    "ref": _v(),                                                   # == b_tight p08c35_aor (sanity)
    "pin": _v(pin=True),                                           # instance sentence survives trimming
    "r2": _v(resc=True, r_extra=2, r_max=2),                       # rescue up to 2, not cited
    "r2c": _v(resc=True, r_extra=2, r_max=2, r_cite=True),         # rescue up to 2, cited
    "r2p": _v(resc=True, r_extra=2, r_max=2, pin=True),
    "r1": _v(resc=True, r_extra=1, r_max=1),
    # precision gates: the rescued sentence must beat the weakest kept session's best sentence (r_min = offset)
    "r2m": _v(resc=True, r_extra=2, r_max=2, r_min=0.0),
    "r2f": _v(resc=True, r_extra=2, r_max=2, r_fact=True),               # sentence states a dated / quantified fact
    "r2mf": _v(resc=True, r_extra=2, r_max=2, r_min=0.0, r_fact=True),
    # session strength from GOOD sentences only (a 'considering condos' opener no longer vetoes 'I viewed a condo'),
    # and the 2 best good units of a rescued session (gold turns are often 2 per session)
    "g2": _v(resc=True, r_extra=2, r_max=2, r_good=True, r_units=2),
    "g2m": _v(resc=True, r_extra=2, r_max=2, r_good=True, r_units=2, r_min=0.0),
    "g2mf": _v(resc=True, r_extra=2, r_max=2, r_good=True, r_units=2, r_min=0.0, r_fact=True),
    "g2mb": _v(resc=True, r_extra=2, r_max=2, r_good=True, r_units=2, r_min=0.0, r_factb=0.05),
    "g2mb_h": _v(resc=True, r_extra=2, r_max=2, r_good=True, r_units=2, r_min=0.0, r_factb=0.05, hint_pass=True),
    # abstention: the best GOOD own sentence's cosine to the question (unanswerable multi questions lack the instance
    # sentence: dev unanswerable median 0.635 vs answerable 0.708); noisy-OR with b_tight's abstain_p. (c, k) fit on dev
    "ab": _v(abst_s=(0.63, 20.0)),
    "g2m_ab": _v(resc=True, r_extra=2, r_max=2, r_good=True, r_units=2, r_min=0.0, abst_s=(0.63, 20.0)),
    # strict rescue: the sentence names the asked category AND states an event / quantity (test: 'I just finished
    # reading a book about the history of São Paulo', 'I picked up one Phalaenopsis orchid at the market yesterday')
    "lf": _v(resc=True, r_extra=2, r_max=2, r_good=True, r_units=2, r_fact=True, r_lex=True),
    "lfm": _v(resc=True, r_extra=2, r_max=2, r_good=True, r_units=2, r_fact=True, r_lex=True, r_min=0.0),
    "lf1": _v(resc=True, r_extra=2, r_max=1, r_good=True, r_units=1, r_fact=True, r_lex=True),
    "g1m": _v(resc=True, r_extra=2, r_max=1, r_good=True, r_units=2, r_min=0.0),
}

_Q_END = re.compile(r"\?\s*$")


def _own_sents(hx):
    """(sentence, uid) for user sentences that state something about me: first person, declarative, 20-400 chars."""
    key = "_a_r3_own"
    c = getattr(hx, key, None)
    if c is not None:
        return c
    out = []
    for u in hx.units:
        if u.role != "user":
            continue
        for s in R1.split_sents(u.text):
            if 20 <= len(s) <= 400 and R1._FIRST.search(s) and not _Q_END.search(s):
                out.append((s, u.uid))
    try:
        setattr(hx, key, out)
    except Exception:
        pass
    return out


def _lex_hit(s: str, terms: list[str]) -> bool:
    """The sentence names a question / facet term (prefix-5 stem): the instance is OF the asked category."""
    keys = {R1._stem5(w) for w in terms if w not in R1._GENERIC and not w.isdigit()}
    toks = {R1._stem5(w) for w in re.findall(r"[a-z0-9][a-z0-9'\-]*", s.lower())}
    return bool(keys & toks)


def _bad_sent(s: str) -> bool:
    return bool(R1._NEG.search(s) or R1._REPORT.search(s) or R1._THIRD_SUBJ.search(s) or R1._RECAP.search(s))


def _sent_dense(q, hx, fact_b: float = 0.0):
    own = _own_sents(hx)
    sc = hx.dense_texts(q.question, [s for s, _ in own]) if own else []
    if fact_b:
        # a stated event / quantity ('I viewed a 1-bedroom condo on February 10th') beats an intent with the same
        # topic ('I'm looking at options for condo') at sentence level
        sc = [v + (fact_b if R1._FACT.search(s) else 0.0) for (s, _), v in zip(own, sc)]
    best_unit: dict[int, tuple[float, str]] = {}
    for (s, uid), v in zip(own, sc):
        if v > best_unit.get(uid, (-9.0, ""))[0]:
            best_unit[uid] = (v, s)
    # best sentence per session (text + uid)
    sess_best: dict[str, tuple[float, int, str]] = {}
    for uid, (v, s) in best_unit.items():
        sid = hx.units[uid].session_id
        if v > sess_best.get(sid, (-9.0, -1, ""))[0]:
            sess_best[sid] = (v, uid, s)
    # per session: units ranked by their best GOOD sentence (not declined / planned / someone else's / recap)
    good_unit: dict[int, tuple[float, str]] = {}
    for (s, uid), v in zip(own, sc):
        if not _bad_sent(s) and v > good_unit.get(uid, (-9.0, ""))[0]:
            good_unit[uid] = (v, s)
    sess_good: dict[str, list] = defaultdict(list)
    for uid, (v, s) in good_unit.items():
        sess_good[hx.units[uid].session_id].append((v, uid, s))
    for sid in sess_good:
        sess_good[sid].sort(key=lambda t: -t[0])
    return sess_best, best_unit, sess_good


def _trim_pin(text: str, terms: list[str], max_chars: int, pin: str | None) -> str:
    t, _ = R1.trim_text(text, terms, max_chars)
    if not pin or pin in t or len(text) <= max_chars:
        return t
    ss = R1.split_sents(text)
    if pin not in ss:
        return t
    j = ss.index(pin)
    idx = R1.focus_sentences(text, terms)
    keep = {j}
    for i in idx:
        if sum(len(ss[k]) + 1 for k in keep | {i}) <= max_chars + len(pin):
            keep.add(i)
    out, prev = [], None
    for k in sorted(keep):
        if prev is not None and k != prev + 1:
            out.append("…")
        out.append(ss[k])
        prev = k
    s = " ".join(out)
    if min(keep) > 0:
        s = "… " + s
    if max(keep) < len(ss) - 1:
        s += " …"
    return s


def _render_session(hx, uids, terms, trim, pins):
    parts, turns, prev = [], set(), None
    for i in sorted(set(uids)):
        u = hx.units[i]
        text = _trim_pin(u.text, terms, trim, pins.get(i)) if trim else u.text
        if prev is not None and i != prev + 1:
            parts.append("[...]")
        if u.role == "user" and u.ctx and u.chunk_idx == 0 and len(u.text) < 160 and (prev is None or i != prev + 1):
            parts.append(f"assistant: … {u.ctx}")
        parts.append(f"{u.role}: {text}")
        turns.add(u.turn_id)
        prev = i
    return "\n".join(parts), turns


def _pack(q, hx, picks, hint, P, terms, pins):
    from core.pack import message_chars
    chosen: dict[str, list[int]] = {}

    def passages(sel):
        sids = sorted(sel, key=lambda s: hx.sess_by_id[s]["sess_idx"])
        return [Passage(id=s, date=hx.sess_by_id[s]["date_str"],
                        text=_render_session(hx, sel[s], terms, P["trim"], pins)[0]) for s in sids]

    chars = message_chars(q, [], hint)
    for u in picks:
        sid = hx.units[u].session_id
        trial = {k: list(v) for k, v in chosen.items()}
        trial.setdefault(sid, []).append(u)
        c = message_chars(q, passages(trial), hint)
        if c > P["budget"]:
            continue
        chosen, chars = trial, c
        if P["budget"] - chars < 120:
            break
    included = set()
    for s, us in chosen.items():
        included |= _render_session(hx, us, terms, P["trim"], pins)[1]
    return passages(chosen), included, chars, chosen


def build(q, hx, **params) -> EvidencePack:
    P = dict(BASE)
    P.update(params)
    a = R1.analyze(q.question, q.date)
    terms = a["terms"] + [w for f in a["facets"] for w in R1.content_terms(f)]
    ranked, final, rr, lab, facet_top, rest, win_active = R1._score_units(q, hx, a, P)

    # ---- abstention + hint: b_tight unchanged ---------------------------------------------------------------------
    miss = R1.missing_terms(hx, a["terms"])
    miss_frac = len(miss) / max(1, len([w for w in a["terms"] if w not in R1._GENERIC and not w.isdigit()]))
    top_rr = max(rr.values()) if rr else -10.0
    head = [u for u in ranked[:10] if R1.focus_sentences(hx.units[u].text, a["terms"])]
    all_other = bool(head) and not a["others_ok"] and all(lab[u] == "other" for u in head[:6])
    abstain_p = R1.abstain_prob(top_rr, miss_frac, all_other)
    hint = R1.build_hint(a, q.date, abstain_p)
    if P.get("hint_pass"):
        # Reactor calibration (e1, road-trip hours): a past drive mentioned in passing inside a 'planning a new trip'
        # message was dropped as 'planned' -> false INSUFFICIENT_EVIDENCE. Keep the decoy rule, exempt past mentions.
        hint = hint.replace("tried or decided against.", "tried or decided against (a past event I mention in passing "
                            "while planning something new still counts).")
    if P.get("latest_line") and a["kind"] in ("count", "sum") and not a["latest"] and not a.get("rate"):
        hint += " If a later session restates the same item with a new number, the latest statement wins."

    # ---- session scores + keep (b_tight model keep) ---------------------------------------------------------------
    rmax: dict[str, float] = {}
    for u, sc in rr.items():
        unit = hx.units[u]
        if P["asst"] != "none" or unit.role == "user":
            rmax[unit.session_id] = max(rmax.get(unit.session_id, -99.0), sc)
    score = dict(rmax)
    if win_active and a["window"]:
        for sid in score:
            if not T._in_window(hx.sess_by_id[sid]["date"].date(), a["window"]):
                score[sid] -= P["win_pen"]
    order = sorted(score, key=lambda s: -score[s])
    top = score[order[0]] if order else -99.0
    feat = T._features(hx, a, order, score, rmax, {}, ranked, facet_top)
    pg = {sid: T.p_gold(f) for sid, f in feat.items()}
    keep = [s for s in order if pg.get(s, 0.0) >= P["tk"]][:P["keep_max"]]
    if len(keep) < P["keep_min"]:
        keep = order[:P["keep_min"]]
    n = len(keep)

    # ---- sentence-level rescue -------------------------------------------------------------------------------------
    pins: dict[int, str] = {}
    rescued: list[str] = []
    resc_units: list[int] = []
    dbg_r = {}
    s_top_good = None
    if P.get("resc") or P.get("pin") or P.get("abst_s"):
        sess_best, best_unit, sess_good = _sent_dense(q, hx, P.get("r_factb", 0.0))
        s_top_good = max((lst[0][0] for lst in sess_good.values() if lst), default=0.0)
        if P.get("r_good"):
            sess_best = {sid: lst[0] for sid, lst in sess_good.items()}
        if P.get("pin"):
            pins = {uid: s for uid, (v, s) in best_unit.items()}
        if P.get("resc") and sess_best:
            s_order = sorted(sess_best, key=lambda s: -sess_best[s][0])
            s_top = sess_best[s_order[0]][0]
            N = len(keep) + P["r_extra"]
            kept_v = [sess_best[s][0] for s in keep if s in sess_best]
            floor = (min(kept_v) + P["r_min"]) if (P.get("r_min") is not None and kept_v) else -9.0
            for sid in s_order[:N]:
                if sid in keep or len(rescued) >= P["r_max"]:
                    continue
                v, uid, s = sess_best[sid]
                if v < s_top - P["r_delta"] or v < floor or _bad_sent(s):
                    continue
                if P.get("r_fact") and not R1._FACT.search(s):
                    continue
                if P.get("r_lex") and not _lex_hit(s, terms):
                    continue
                if win_active and a["window"] and not T._in_window(hx.sess_by_id[sid]["date"].date(), a["window"]):
                    continue
                rescued.append(sid)
                for v2, u2, s2 in (sess_good.get(sid) or [(v, uid, s)])[:P.get("r_units", 1)]:
                    if u2 not in resc_units:
                        resc_units.append(u2)
                        pins[u2] = s2
                if uid not in resc_units:
                    resc_units.append(uid)
                    pins[uid] = s
            dbg_r = {"s_top": round(s_top, 4), "s_rank": [(s, round(sess_best[s][0], 4)) for s in s_order[:N]]}
    keep_set = set(keep) | set(rescued)

    # ---- selection inside the kept sessions ---------------------------------------------------------------------
    ranked_k = [u for u in ranked if hx.units[u].session_id in keep_set and hx.units[u].session_id not in rescued]
    ft_k = {f: [u for u in us if hx.units[u].session_id in set(keep)] for f, us in facet_top.items()}
    Pk = dict(P)
    Pk["max_sess"] = len(keep)
    Pk["deep_top"] = len(keep)
    picks = R1._select(hx, ranked_k, ft_k, Pk)
    if resc_units:
        k0 = min(len(picks), len(keep))
        picks = picks[:k0] + resc_units + picks[k0:]
    passages, included, chars, chosen = _pack(q, hx, picks, hint, Pk, terms, pins)
    in_pack = {p.id for p in passages}

    o = [s for s in order if s in in_pack]
    cite = [s for s in o if pg.get(s, 0.0) >= P["tc"]]
    if len(cite) < 2:
        cite = o[:2]
    if P.get("r_cite"):
        cite += [s for s in rescued if s in in_pack and s not in cite]

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
    p_s = None
    if P.get("abst_s") and s_top_good is not None:
        c, k = P["abst_s"]
        p_s = 1.0 / (1.0 + math.exp(-k * (c - s_top_good)))
        abstain_p = 1.0 - (1.0 - abstain_p) * (1.0 - p_s)
    dbg_r["s_top_good"] = s_top_good
    dbg_r["p_s"] = p_s
    return EvidencePack(question_id=q.question_id, variant=params.get("_variant", ""), route="multi",
                        passages=passages, hint=hint, cite_plan=cite or None, abstain_p=abstain_p,
                        included_turns=included, computed=None,
                        debug={"ranking": ranked + rest, "keep": keep, "rescued": rescued, "n_keep": n, "top": top,
                               "window_active": win_active, "top_rr": top_rr, "missing_terms": miss,
                               "chars": chars, "sess2": sess_feat, "abst_a": [ap_a, why_a], "resc": dbg_r})
