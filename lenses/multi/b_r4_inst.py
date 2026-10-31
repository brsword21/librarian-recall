"""Lens B ("Semantyka") for multi-session, ROUND 4: verb-class instance rescue on top of the frozen v2 multi route.

Test defect (v2 packs, no gold, visible by reading): count / sum packs miss an obvious own instance that sits as one
short sentence in an off-topic session, so CE ranks the session below care chatter and decoys:
  - "How many orchids have I added ...": "Oh, and I bought two Cattleya orchids for the balcony this week" is absent,
    the pack carries repotting / fertilizer / "my daughter gave me two seedlings in 2021".
  - "How much is left in my bakery fund ...": the opening deposit ("opened a separate savings account for the bakery
    and deposited 25,000 pesos") is absent, the pack carries the sister's 30,000-peso fund (a decoy).
  - "combined attendance at the March and June book club meetings": March ("7 people came") is absent.
  - "How many books have I read since March?": both own reads are absent; the pack carries Carlos's "three books".

Fix (general, no per-question rule): after the frozen v2 route (lenses/multi/merged_r2.py tight_role9 with the
lenses/ssa/a_r3_role.py detector; nothing in it changes), look at user sentences OUTSIDE the packed sessions that
  1. are own and affirmed (first person; no report "X told me he", no kin subject, no plan / negation cue),
  2. carry a quantity (digit or number word) or money amount,
  3. match the question's event class (acquire / read / attend / money / plant / eat / make, from the question verb);
     questions with no class need the head noun,
  4. match the question's head noun lexically, or (class match only) reach a bge similarity floor on the question,
  5. sit inside the question window when the window prior is active,
rank them by cross-encoder on the question, take at most r_max sessions above r_thr, and add each as a one-unit
passage (the holding user unit, untrimmed) in date order, cited.
"""
from __future__ import annotations

import re

from core.types import EvidencePack, Passage
from core.pack import message_chars

from lenses.multi import b_sweep as R1
from lenses.multi import b_r3_full as F
from lenses.multi import b_tight as T
from lenses.multi import merged_r2 as M

_INNER = dict(M._BASE, role_module="lenses/ssa/a_r3_role.py")   # == configs/best.json multi route (v2)


def _v(**kw):
    d = dict(_INNER)
    d.update(kw)
    return d


VARIANTS = {
    "v2": _v(resc=False),
    "vc2": _v(resc=True, r_max=2, r_thr=-7.0, d_thr=0.62, d_ce=-3.0),
    "vc1": _v(resc=True, r_max=1, r_thr=-7.0, d_thr=0.62, d_ce=-3.0),
    "vc2_h": _v(resc=True, r_max=2, r_thr=-7.0, d_thr=9.0, d_ce=-3.0),     # head noun only (no dense path)
    "vc2_t4": _v(resc=True, r_max=2, r_thr=-4.0, d_thr=0.62, d_ce=-3.0),
    "vc3": _v(resc=True, r_max=3, r_thr=-7.0, d_thr=0.62, d_ce=-3.0),
    "vc3_t11": _v(resc=True, r_max=3, r_thr=-11.0, d_thr=0.62, d_ce=-3.0, r_gap=5.0),
}

# question event class -> sentence evidence of that class
_QCLASS = [
    ("acq", re.compile(r"\b(add|added|adding|buy|bought|purchas\w*|acquir\w*|collect\w*|adopt\w*|got|get|own)\b", re.I),
     re.compile(r"\b(bought|buy|got|added|purchased|picked up|ordered|adopted|acquired|received|brought home|"
                r"gave me|gifted|rescued|found)\b", re.I)),
    ("read", re.compile(r"\bread\b", re.I),
     re.compile(r"\b(read|finished reading|finished (?:that|the|my|a|an)|listened to)\b", re.I)),
    ("attend", re.compile(r"\b(attendance|attended|attendees|came|showed up|people)\b", re.I),
     re.compile(r"\b(came|attended|showed up|turned up|people|members|guests|attendees)\b", re.I)),
    ("money", re.compile(r"\b(fund|savings|saved|save|account|left in|budget|spent|spend|cost|paid|earn\w*|"
                         r"raised|donat\w*)\b", re.I),
     re.compile(r"\b(deposit\w*|added|put|moved|transferr\w*|spent|paid|used|withdr\w*|took out|earned|saved|"
                r"raised|donated|bought|splurged)\b.*([$€£]\s?\d|\d[\d,.]*\s?(?:k\b|pesos|dollars|euros|shillings|"
                r"kes|usd|pounds|rupees|yen|reais|bucks))", re.I)),
    ("plant", re.compile(r"\bplant\w*\b", re.I), re.compile(r"\bplant\w*\b", re.I)),
    ("eat", re.compile(r"\b(eat|ate|eaten)\b", re.I), re.compile(r"\b(ate|eat|had|ordered|tried)\b", re.I)),
    ("make", re.compile(r"\b(make|made|bake|baked|cook\w*|brew\w*)\b", re.I),
     re.compile(r"\b(made|baked|cooked|brewed|prepared|whipped up)\b", re.I)),
]
_MONEYQ = re.compile(r"\bhow much\b|\btotal\b|\bfund\b", re.I)
_TIMEW = {"january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
          "november", "december", "since", "start", "summer", "winter", "spring", "autumn", "fall", "month", "year",
          "week", "today", "total", "combined", "left", "told", "mentioned", "transactions", "different", "times",
          "end", "much", "many", "after", "before", "during", "going", "two", "three"}
_QTY = re.compile(r"\b(\d[\d,.]*|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|dozen|couple|"
                  r"twenty|thirty|forty|fifty|hundred)\b", re.I)
_PLANISH = re.compile(r"\b(will|going to|gonna|'ll|hope|hoping|plan|planning|maybe|should|could|would|can(?!')|"
                      r"have to|has to|had to|got to|gotta|still have \w+ to|still have \w+ \w+ to)\b", re.I)
_FIRST = re.compile(r"\bI\b|\bI'(?:ve|d|m|ll)\b|\b(?:[Mm]y|[Ww]e|[Oo]ur|[Mm]e)\b")
_I = re.compile(r"\bI\b|\bI'(?:ve|d|m|ll)\b")
_OTHER_OWN = re.compile(r"\b(his|her|their) own\b", re.I)
_KIN_ANY = re.compile(r"\b(?:my|our)\s+(?:[\w'-]+\s+){0,2}?(?:" + R1._KIN + r")s?\b", re.I)
_ROUTINE = re.compile(r"\b(each|every|per|a) (day|week|month|morning|night|evening)\b|\bweekly\b|\bdaily\b", re.I)
# a few category -> instance nouns for the commonest count category (books); everything else needs the head noun
_FAMILY = {"book": {"novel", "memoi", "biogr", "thril", "paper", "audio", "myste"}}


def q_classes(question: str) -> list[tuple[str, re.Pattern]]:
    out = []
    for name, qre, sre in _QCLASS:
        if qre.search(question):
            out.append((name, sre))
    return out


def q_heads(question: str) -> set[str]:
    hs = F.head_terms(question)
    if not hs:
        hs = [w for w in R1.content_terms(question) if w not in R1._GENERIC and w.lower() not in _TIMEW
              and not w.isdigit()]
    out = {F._st(w) for w in hs if w.lower() not in _TIMEW}
    for h in list(out):
        out |= _FAMILY.get(h, set())
    return out


def candidates(q, hx, a, excl: set[str], win_active: bool):
    classes = q_classes(q.question)
    heads = q_heads(q.question)
    if not classes or _ROUTINE.search(q.question):
        return [], classes, heads
    out = []   # (uid, sentence, head_hit, class_names)
    for u in hx.units:
        if u.role != "user" or u.session_id in excl:
            continue
        if win_active and a["window"] and not T._in_window(u.date.date(), a["window"]):
            continue
        for se in F.split_sents(u.text):
            if len(se) < 20 or len(se) > 400 or se.rstrip().endswith("?"):
                continue
            if not _FIRST.search(se):
                continue
            if R1._NEG.search(se) or R1._REPORT.search(se) or R1._THIRD_SUBJ.search(se) or R1._RECAP.search(se):
                continue
            if _PLANISH.search(se) or _OTHER_OWN.search(se):
                continue
            if _KIN_ANY.search(se) and not _I.search(se):
                continue
            toks = {F._st(w) for w in re.findall(r"[a-z0-9][a-z0-9'\-]*", se.lower())}
            head_hit = bool(heads & toks)
            cls = [n for n, sre in classes if sre.search(se)]
            if not cls:
                continue
            if not head_hit and not _QTY.search(se):
                continue
            if head_hit and not (_QTY.search(se) or R1._FACT.search(se)):
                continue
            out.append((u.uid, se, head_hit, cls))
    return out, classes, heads


def rescue(q, hx, a, P, excl: set[str], win_active: bool) -> dict:
    cands, classes, heads = candidates(q, hx, a, excl, win_active)
    if not cands:
        return {}
    cands = cands[:300]
    texts = [c[1] for c in cands]
    ce = hx.rerank_texts(q.question, texts)
    need_dense = [i for i, c in enumerate(cands) if not c[2]]
    dn = {}
    if need_dense:
        ds = hx.dense_texts(q.question, [texts[i] for i in need_dense])
        dn = {i: float(v) for i, v in zip(need_dense, ds)}
    best: dict[str, tuple] = {}
    order = sorted(range(len(cands)), key=lambda i: -ce[i])
    for i in order:
        uid, se, head_hit, cls = cands[i]
        if ce[i] < P["r_thr"]:
            break
        if not head_hit and (dn.get(i, 0.0) < P["d_thr"] or ce[i] < P.get("d_ce", -3.0)):
            continue
        sid = hx.units[uid].session_id
        if sid in best:
            continue
        if best and ce[i] < max(v[2] for v in best.values()) - P.get("r_gap", 4.0):
            break
        best[sid] = (uid, se, round(float(ce[i]), 3), round(dn.get(i, -1.0), 3), cls, head_hit)
        if len(best) >= P["r_max"]:
            break
    return best


def build(q, hx, _variant: str = "", resc: bool = False, r_max: int = 2, r_thr: float = -7.0, d_thr: float = 0.62,
          d_ce: float = -3.0, r_gap: float = 4.0, **params) -> EvidencePack:
    pack = M.build(q, hx, _variant=_variant, **params)
    if not resc or pack.route != "multi" or not pack.passages:
        return pack
    a = R1.analyze(q.question, q.date)
    win_active = bool((pack.debug or {}).get("window_active"))
    in_pack = {p.id for p in pack.passages if p.id}
    best = rescue(q, hx, a, dict(r_max=r_max, r_thr=r_thr, d_thr=d_thr, d_ce=d_ce, r_gap=r_gap), in_pack, win_active)
    pack.debug = {**(pack.debug or {}), "r4_rescue": {k: list(v[1:]) for k, v in best.items()}}
    if not best:
        return pack
    passages = list(pack.passages)
    cite = list(pack.cite_plan or [p.id for p in passages if p.id])
    included = set(pack.included_turns or set())
    for sid, (uid, se, *_r) in best.items():
        u = hx.units[uid]
        s = hx.sess_by_id[sid]
        new = Passage(id=sid, date=s["date_str"], text="user: " + u.text)
        trial = passages + [new]
        if message_chars(q, trial, pack.hint) > 12000:
            continue
        passages = trial
        cite.append(sid)
        included.add(u.turn_id)
    date_of = {sid: hx.sess_by_id[sid]["date"] for sid in {p.id for p in passages if p.id}}
    passages.sort(key=lambda p: date_of.get(p.id) if p.id else max(date_of.values()))
    pack.passages = passages
    pack.cite_plan = cite
    pack.included_turns = included
    return pack
