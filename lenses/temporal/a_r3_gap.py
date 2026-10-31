"""Temporal lens A, round 3 ("Struktura"): the frozen v1 temporal route (merged_r2 evA_mixB = a_gap ct_evgba on gap /
named-event questions + b_merge ev1_mix_g elsewhere) plus two structural additions that target what the test audit
showed (test temporal = 35 gap questions + 1 "how many years did I serve" question):

  * span    - "how many years did I serve / work / spend as X": the answer is end age - start age (or end year - start
              year) from two first-person statements ("I joined the fire brigade when I was 23", "I retired from the
              fire department the day I turned 63"). v1 sent B's whole-question pack with neither statement. The two
              statements are found by an age / year pattern + start / end verb scan over user sentences, filtered to
              the user's own statements (subject_kind) and ranked by cross-encoder relevance to the question.
  * own     - own-absence check for gap / ago packs without a confident computed value: when an event of the
              question has no first-person user statement that covers its content words (only "Camille bought
              trail-running shoes today", only "I still need to sort out my travel vaccines", no "loaf" at all), the
              pack is cut to the sessions of the closest mentions (budget own_budget), the hint lists those mentions
              as what they are (someone else's / not done yet) instead of "date notes", and abstain_p >= own_ap.
Imports the frozen lenses (a_gap, a_calendar, merged_r2); does not modify them.
"""
from __future__ import annotations

import re

from core.types import EvidencePack
from core.loader import load_lens
from lenses.temporal import a_calendar as A
from lenses.temporal import a_gap as G

_M = load_lens("lenses/temporal/merged_r2.py")

VARIANTS = {
    "v1": {},
    "span": {"span": True},
    "own": {"own": True},
    "span_own": {"span": True, "own": True},
    # tight gap packs without the runner-up session (passages = cite set)
    "so_nor": {"span": True, "own": True, "a_over": {"max_runner": 0}},
    # order questions (wide packs): two more own mentions (restatements) per event + the opening user turn
    "so_ordm": {"span": True, "own": True, "a_over": {"more": 2, "first_turn": True}},
}

# --------------------------------------------------------------------------------------------------------------------
# span: "how many years did I serve as a firefighter?"
# --------------------------------------------------------------------------------------------------------------------
_SPAN = re.compile(r"\bhow (?:many years|long) did I (serve|work|spend|stay|teach|coach|volunteer|play|live)\b\s*(.*)",
                   re.I)
_AGE = re.compile(r"\b(?:when I was|at the age of|at age|aged|I turned|I was|turned)\s+(\d{1,2})\b"
                  r"|\bat (\d{2})\b(?!\s*(?:am|pm|a\.m|p\.m|:|%|percent|minutes|mph|degrees|km|miles))", re.I)
_YEAR = re.compile(r"\b(?:in|since|back in)\s+((?:19|20)\d\d)\b")
_START = re.compile(r"\b(?:joined|join|started|start|began|begin|became|become|enlisted|signed up|got hired|was hired|"
                    r"first (?:worked|started)|entered)\b", re.I)
_END = re.compile(r"\b(?:retired|retire|left|quit|stopped|resigned|ended|finished|hung up)\b", re.I)


def _user_sentences(hx):
    for u in hx.units:
        if u.role != "user":
            continue
        for _, s in A.sentences(u.text):
            yield u, s


def build_span(q, hx, params: dict) -> EvidencePack | None:
    from core.pack import build_pack
    m = _SPAN.search(q.question)
    if not m:
        return None
    qlow = q.question.lower()
    cands = []          # (uid, sentence, 'start'|'end', value, unit)
    for u, s in _user_sentences(hx):
        if not (_START.search(s) or _END.search(s)):
            continue
        ages = [int(a or b) for a, b in _AGE.findall(s)]
        years = [int(y) for y in _YEAR.findall(s)]
        if not ages and not years:
            continue
        if A.subject_kind(s, "user", qlow) != "self":
            continue
        kind = "end" if _END.search(s) and not _START.search(s) else "start" if _START.search(s) and \
            not _END.search(s) else None
        if kind is None:
            continue
        val, unit = (ages[0], "age") if ages else (years[0], "year")
        if unit == "age" and not (10 <= val <= 99):
            continue
        cands.append((u.uid, s, kind, val, unit))
    if not cands:
        return None
    try:
        ce = hx.rerank_texts(q.question, [c[1] for c in cands])
    except Exception:
        return None
    scored = sorted(zip(ce, cands), key=lambda x: -x[0])
    thr = params.get("span_ce", -4.0)
    st = next(((c, x) for c, x in scored if x[2] == "start" and c >= thr), None)
    en = next(((c, x) for c, x in scored if x[2] == "end" and c >= thr and (st is None or x[4] == st[1][4])), None)
    if st is None or en is None:
        return None
    (c1, (u1, s1, _, v1, un)), (c2, (u2, s2, _, v2, _)) = st, en
    diff = v2 - v1
    if diff <= 0 or diff > 70:
        return None
    sid1, sid2 = hx.units[u1].session_id, hx.units[u2].session_id
    lab = "age" if un == "age" else "year"
    hint = (f"Notes found by code in the passages: \"{A._snip(s1, 160)}\" (session {hx.units[u1].date_str[:10]}) -> "
            f"start at {lab} {v1}; \"{A._snip(s2, 160)}\" (session {hx.units[u2].date_str[:10]}) -> end at {lab} {v2}. "
            f"Difference: {diff} years. Statements about other people are decoys. Answer with just the number.")
    passages, included, _ = build_pack(q, hx, [u1, u2], budget=params.get("span_budget", 6000), window=1,
                                       order="chrono", hint=hint)
    pids = {p.id for p in passages if p.id}
    cite = [s for s in dict.fromkeys([sid1, sid2]) if s in pids] or None
    return EvidencePack(question_id=q.question_id, variant=params.get("_variant", ""), route="temporal",
                        passages=passages, hint=hint, cite_plan=cite, abstain_p=0.05, included_turns=included,
                        computed=str(diff),
                        debug={"ranking": [u1, u2] + [x[0] for _, x in scored if x[0] not in (u1, u2)],
                               "span": [(round(c1, 2), s1, v1), (round(c2, 2), s2, v2)], "merged_from": "a_r3:span"})


# --------------------------------------------------------------------------------------------------------------------
# own: does the user ever state the event as their own (done) action?
# --------------------------------------------------------------------------------------------------------------------
_UNDONE = re.compile(r"\b(?:still need to|need to|have to|haven't|have not|hasn't|not yet|yet to|planning to|plan to|"
                     r"going to|gonna|want to|would like to|hoping to|thinking of|thinking about|can't|cannot|won't|"
                     r"should I|should i|might|maybe|considering)\b", re.I)
_EV_VERB = re.compile(r"^(?:\w+ing|do|did|get|got|go|went|make|made|have|had|take|took|bake|buy|bought|start|"
                      r"finish|attend|visit|see|saw|meet|met|join)\b\s*", re.I)


def _undone(s: str, terms: dict) -> bool:
    """A plan / negation marker shortly (<= 35 chars) before the first event word: "I still need to sort out my travel
    vaccines", "I can't take on a sourdough starter"."""
    s = s.replace("\u2019", "'")
    pos = [p for t, p in A.toks(s) if t in terms]
    if not pos:
        return False
    first = min(pos)
    for m in _UNDONE.finditer(s):
        if m.end() <= first and first - m.end() <= 35:
            return True
    for a, b in A.negated_spans(s):
        if a <= first <= b:
            return True
    return False


def own_coverage(q, hx, ev: str) -> tuple[float, float, str | None, str | None]:
    """(best coverage by an own, done user statement, best coverage by any user statement, own sentence, other
    sentence). Coverage = IDF share of the event's content stems (leading verb dropped) found in one sentence."""
    c = A._hxc(hx)
    core = _EV_VERB.sub("", re.sub(r"\([^)]*\)", " ", ev).strip())
    terms = {}
    for t, _ in A.toks(core):
        terms[t] = max(terms.get(t, 0.0), c.idf(t))
    tot = sum(terms.values())
    if not terms or tot <= 0:
        return 1.0, 1.0, None, None
    qlow = q.question.lower()
    best_own, best_any, s_own, s_any = 0.0, 0.0, None, None
    for u, s in _user_sentences(hx):
        st = {t for t, _ in A.toks(s)}
        cov = sum(w for t, w in terms.items() if t in st) / tot
        if cov <= best_own and cov <= best_any:
            continue
        if cov > best_any:
            best_any, s_any = cov, s
        if cov > best_own and A.subject_kind(s, "user", qlow) == "self" and not _undone(s, terms):
            best_own, s_own = cov, s
    return best_own, best_any, s_own, s_any


def _own_check(q, hx, events: list[str], thr: float):
    rows = []
    for ev in events[:2]:
        own, anyc, s_own, s_any = own_coverage(q, hx, ev)
        rows.append((ev, own, anyc, s_own, s_any))
    # an event is missing when no own, done statement covers it AND either a full-coverage statement exists that is
    # not the user's own done action (someone else's / a plan) or nothing covers it at all
    missing = [r for r in rows if r[1] < thr and (r[2] >= 0.9 or r[2] < 0.4)]
    return rows, missing


def build_own(q, hx, pack: EvidencePack, params: dict) -> EvidencePack:
    """Cut a no-computed gap / ago pack to the closest mentions when an event has no own statement."""
    from core.pack import build_pack
    d = pack.debug or {}
    parse = d.get("parse") or {}
    if parse.get("kind") not in ("gap", "ago") or pack.computed is not None:
        return pack
    events = parse.get("events") or []
    if not events:
        return pack
    rows, missing = _own_check(q, hx, events, params.get("own_thr", 0.75))
    d["own"] = [(ev, round(o, 2), round(a, 2)) for ev, o, a, _, _ in rows]
    if not missing:
        return pack
    # pack: the sessions holding the closest sentence of each event (any speaker), ranked by v1's own ranking
    want = []
    for ev, o, a, s_own, s_any in rows:
        for s, c in ((s_own, o), (s_any, a)):
            if s and c >= 0.5:
                want.append(s)
    ranked = []
    for u, s in _user_sentences(hx):
        if s in want and u.uid not in ranked:
            ranked.append(u.uid)
    lines = []
    miss_ev = {m[0] for m in missing}
    for ev, o, a, s_own, s_any in rows:
        if ev not in miss_ev:
            continue
        if s_any and a >= 0.5:
            lines.append(f"- \"{ev}\": no statement that you did this was found; the closest mention is "
                         f"\"{A._snip(s_any, 140)}\"")
        else:
            lines.append(f"- \"{ev}\": not mentioned in the history.")
    hint = "Notes found by code:\n" + "\n".join(lines) + \
        "\nMentions of other people's events or of plans are not your events."
    passages, included, _ = build_pack(q, hx, ranked, budget=params.get("own_budget", 4000), window=1,
                                       order="chrono", hint=hint)
    out = EvidencePack(question_id=pack.question_id, variant=pack.variant, route=pack.route, passages=passages,
                       hint=hint, cite_plan=None, abstain_p=max(pack.abstain_p or 0.0, params.get("own_ap", 0.8)),
                       included_turns=included, computed=None,
                       debug={**d, "ranking": ranked + [u for u in d.get("ranking", []) if u not in ranked],
                              "own_cut": [m[0] for m in missing]})
    return out


def build(q, hx, **params) -> EvidencePack:
    if params.get("span"):
        p = build_span(q, hx, params)
        if p is not None:
            return p
    ov = params.get("a_over")
    ap = G.VARIANTS["ct_evgba"]
    if ov and G._use_a(q, ap):
        pack = G.build(q, hx, _variant="ct_evgba", **{**ap, **ov})
        pack.debug = {**(pack.debug or {}), "merged_r2_from": "a_gap:ct_evgba+" + ",".join(ov)}
    else:
        pack = _M.build(q, hx, _variant="evA_mixB", **_M.VARIANTS["evA_mixB"])
    pack.variant = params.get("_variant", "")
    if params.get("own") and str((pack.debug or {}).get("merged_r2_from", "")).startswith("a_gap"):
        pack = build_own(q, hx, pack, params)
    return pack
