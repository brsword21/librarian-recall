"""Lens A ("Struktura") for multi-session, ROUND 4: verb-class instance rescue + within-session depth.

Base = a_r3_pin variant 'ref' (== frozen b_tight p08c35_aor, the v2 multi lens). Two optional layers:

1. vc (brief item 2, 'rescue only with a verb-class gate'): the question's action verb is mapped to a verb class
   ("planted" -> plant/sow, "added/bought" -> buy/add/adopt/pick up, "read" -> read/finish). A session that b_tight did
   not keep is rescued only if one of its user sentences
     - is first person, declarative, not negated / planned / someone else's / a recap (a_r3_pin._bad_sent),
     - uses a PAST form of a verb of that class ("I sowed a row of carrot seeds" for "have I planted"),
     - names a question category term (prefix-5 stem: 'vegetable', 'orchid', 'book'),
     - falls inside the question's time window (when b_tight treats the window as active),
     - and ranks in the top vc_top own sentences by bge cosine (statement-form check).
   The unit holding that sentence enters after the first pass with the sentence pinned in the trim.
2. depth (brief item 1): every kept session also gets its best GOOD own sentence's unit (second gold turn inside a
   session already in the pack). Cite set unchanged.

Round-4 recommendation: vlbhc6 = vc + plural-folded category hit (lex2, 'books' ~ 'book', adjectives such as 'new'
are not category words) + small hyponym table (novel ~ book, table ~ furniture, carrot ~ vegetable) + balance class
('how much is left in my bakery fund': money-flow verbs, the fund's modifier noun must appear) + rescued sessions cited,
vc_top 60, vc_max 3. Skipped on rate / latest questions ('classes each week': a one-off workshop is a decoy).
Gold precision of the gate (keep exclusion off): dev 17/19, dev200 5/6. Route wrapper: lenses/multi/a_r4_route.py.
"""
from __future__ import annotations

import re

from core.types import EvidencePack

from lenses.multi import a_r3_pin as P3
from lenses.multi import b_sweep as R1
from lenses.multi import b_tight as T

BASE = dict(P3.BASE)
BASE.update(vc_bal=False, vc=False, vc_max=2, vc_top=12, vc_cite=False, vc_lex=True, vc_lex2=False, vc_wide=0, depth=False,
            depth_n=1, depth_min=None)


def _v(**kw):
    d = dict(BASE)
    d.update(kw)
    return d


VARIANTS = {
    "ref": _v(),
    "vc": _v(vc=True),
    "vcc": _v(vc=True, vc_cite=True),
    "vc20": _v(vc=True, vc_top=20),
    "vl": _v(vc=True, vc_lex2=True),                                  # plural-folded category hit
    "vlc": _v(vc=True, vc_lex2=True, vc_cite=True),
    "vlw": _v(vc=True, vc_lex2=True, vc_top=60, vc_wide=60),           # no category word: verb + fact, top 60
    "vlwc": _v(vc=True, vc_lex2=True, vc_top=60, vc_wide=60, vc_cite=True),
    "vl3": _v(vc=True, vc_lex2=True, vc_max=3),
    "vlb": _v(vc=True, vc_lex2=True, vc_bal=True),                       # + balance questions (money-flow verbs)
    "vlbc": _v(vc=True, vc_lex2=True, vc_bal=True, vc_cite=True),
    "vlbh": _v(vc=True, vc_lex2=True, vc_bal=True, vc_hypo=True),          # + category hyponyms (novel ~ book)
    "vlbhc": _v(vc=True, vc_lex2=True, vc_bal=True, vc_hypo=True, vc_cite=True),
    "vlbhc3": _v(vc=True, vc_lex2=True, vc_bal=True, vc_hypo=True, vc_cite=True, vc_max=3, vc_top=30),
    "vlbhc6": _v(vc=True, vc_lex2=True, vc_bal=True, vc_hypo=True, vc_cite=True, vc_max=3, vc_top=60),
    "dp": _v(depth=True),
    "vc_dp": _v(vc=True, depth=True),
    "vcc_dp": _v(vc=True, vc_cite=True, depth=True),
}

# verb class: question verb (any form) -> past forms that state an instance of it
_CLASSES = [
    (r"plant(?:ed|ing)?|sow(?:ed|n)?|grow(?:n|ing)?|grew", r"planted|sowed|sown|transplanted|put in"),
    (r"buy|bought|purchas\w*|add(?:ed)?|adopt\w*|get|got|acquir\w*|pick(?:ed)? up",
     r"bought|purchased|added|adopted|got (?:a|an|one|two|three|four|five|six|\d+|another|some|my first|my new)|"
     r"picked up|ordered|brought home|acquired|took home"),
    (r"read(?:ing)?|finish(?:ed)?", r"read|finished|devoured|got through|started reading|re-read"),
    (r"attend(?:ed)?|went|go(?:ne)?|visit(?:ed)?", r"attended|went to|visited|joined|took part|participated"),
    (r"eat|ate|eaten|have|had", r"ate|had|made|cooked|ordered|grabbed"),
    (r"make|made|bake[ds]?|cook(?:ed)?", r"made|baked|cooked|prepared|whipped up|produced|cooked up"),
    (r"harvest(?:ed)?|collect(?:ed)?", r"harvested|collected|extracted"),
    (r"replac\w*|swap\w*|chang\w*", r"replaced|swapped|changed|installed|put in"),
    (r"spen[dt]|pa(?:y|id)", r"spent|paid|bought|cost|shelled out|dropped"),
    (r"repair\w*|fix(?:ed)?", r"repaired|fixed|restored|rebuilt"),
    (r"complet\w*|run|ran|do|did", r"completed|ran|did|finished"),
    (r"donat\w*|g[ia]ve", r"donated|gave|contributed"),
]
_QVERB = re.compile(r"\b(?:did|have|has|had|do|am|was|were|will)\s+(?:I|we)\s+(?:\w+\s+){0,2}?(\w+(?:\s+up)?)\b", re.I)
_PAST = re.compile(r"\b(?:I|we)(?:'ve| have| had)?\s+(?:\w+\s+){0,2}?(\w+(?:\s+(?:up|in|to|home|through))?)\b", re.I)


def _verb_class(question: str):
    """Past-form regex for the question's action verb class, or None."""
    for m in _QVERB.finditer(question):
        w = m.group(1).lower()
        for qv, past in _CLASSES:
            if re.fullmatch(qv, w) or re.fullmatch(qv, w.split()[0]):
                return re.compile(r"\b(?:I|we)\b(?:'ve| have| had| just| also| finally| recently| already| actually|"
                                  r" then| even|,)*\s+(?:" + past + r")\b", re.I), w
    return None, None


def _stem(w: str) -> str:
    w = w.lower()
    if len(w) > 4 and w.endswith("ies"):
        w = w[:-3] + "y"
    elif len(w) > 3 and w.endswith("es") and w[-3] in "sxz":
        w = w[:-2]
    elif len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        w = w[:-1]
    return w[:5]


# category noun -> member nouns (a small hyponym table for the usual count categories; general, not per question)
_HYPO = {
    "book": "novel biography memoir autobiography thriller paperback hardcover audiobook ebook",
    "novel": "book thriller",
    "furni": "table chair sofa couch bookshelf bookcase desk mattress bed dresser cabinet wardrobe nightstand ottoman "
             "stool armchair",
    "veget": "tomato carrot zucchini silverbeet lettuce kale garlic onion bean pea potato pepper cucumber spinach "
             "broccoli cabbage beet radish squash pumpkin eggplant chard",
    "cloth": "shirt dress jacket jean shoe sweater coat skirt trouser pant scarf boot",
    "instr": "guitar piano ukulele violin drum keyboard flute saxophone cello",
    "pet": "dog cat puppy kitten rabbit hamster parrot",
}


_ADJ = set("new old own first last other more recent current favorite favourite total since start started".split())


def _lex2(s: str, terms: list[str], verb: str, hypo: bool = False) -> bool:
    """Category hit with plural folding ('books' ~ 'book'); the question's own verb is not a category term."""
    vk = {_stem(x) for x in verb.split()}
    keys = {_stem(w) for w in terms if w not in R1._GENERIC and w not in _ADJ and not w.isdigit()} - vk
    if hypo:
        for k in list(keys):
            keys |= {_stem(x) for x in _HYPO.get(k, "").split()}
    toks = {_stem(w) for w in re.findall(r"[a-z0-9][a-z0-9'\-]*", s.lower())}
    return bool(keys & toks)


_BAL_Q = re.compile(r"\b(?:left|remaining|balance)\b.*?\bmy\s+(?:([a-z]+)\s+)?(?:fund|savings|account|budget|piggy bank)\b"
                    r"|\bmy\s+(?:([a-z]+)\s+)?(?:fund|savings|account|budget)\b.*?\b(?:left|remaining|balance)\b", re.I)
_BAL_V = re.compile(r"\b(?:I|we)\b(?:'ve| have| had| just| also| finally| recently| already| actually| then|,)*\s+"
                    r"(?:deposited|added|put|saved|moved|transferred|used|spent|withdrew|took out|paid|opened|"
                    r"started|set aside|pulled)\b", re.I)


def _balance_class(question: str):
    """'How much is left in my bakery fund': money-flow verbs, the fund's modifier noun must appear ('bakery')."""
    m = _BAL_Q.search(question)
    if not m:
        return None, None
    mod = (m.group(1) or m.group(2) or "").lower()
    return _BAL_V, mod


def _vc_rescue(q, hx, a, terms, keep, win_active, P):
    vre, w = _verb_class(q.question)
    must = None
    if P.get("vc_bal"):
        bre, mod = _balance_class(q.question)
        if bre is not None and mod:
            vre, w, must = bre, "balance", _stem(mod)
    if vre is not None and must is None and (a.get("rate") or a.get("latest")):
        # current routine / latest value ('how many classes do I attend each week'): an extra past instance
        # ('a one-off workshop') is a decoy, not a missing member
        return [], [], {"vc_verb": w, "vc_skip": "rate/latest"}
    if vre is None:
        return [], [], {"vc_verb": None}
    own = P3._own_sents(hx)
    sc = hx.dense_texts(q.question, [s for s, _ in own]) if own else []
    order = sorted(range(len(own)), key=lambda i: -sc[i])
    rank = {i: r for r, i in enumerate(order)}
    keep_set = set(keep)
    cands = []
    for i, (s, uid) in enumerate(own):
        sid = hx.units[uid].session_id
        if sid in keep_set or rank[i] >= P["vc_top"]:
            continue
        if P3._bad_sent(s) or not vre.search(s):
            continue
        if must is not None:
            if must not in {_stem(t) for t in re.findall(r"[a-z]+", s.lower())} or not R1._FACT.search(s):
                continue
        if P.get("vc_lex", True):
            hit = _lex2(s, terms, w, P.get("vc_hypo", False)) if P.get("vc_lex2") else P3._lex_hit(s, terms)
            if not hit and not (P.get("vc_wide") and rank[i] < P["vc_wide"] and R1._FACT.search(s)):
                continue
        if win_active and a["window"] and not T._in_window(hx.sess_by_id[sid]["date"].date(), a["window"]):
            continue
        if hx.sess_by_id[sid]["date"] > q.date:
            continue
        cands.append((sc[i], sid, uid, s))
    cands.sort(key=lambda t: -t[0])
    rescued, units, pins = [], [], {}
    for v, sid, uid, s in cands:
        if sid in rescued:
            continue
        if len(rescued) >= P["vc_max"]:
            break
        rescued.append(sid)
        units.append(uid)
        pins[uid] = s
    return rescued, units, {"vc_verb": w, "vc_pins": pins,
                            "vc_cands": [(sid, round(v, 3), s[:120]) for v, sid, uid, s in cands[:5]]}


def build(q, hx, **params) -> EvidencePack:
    P = dict(BASE)
    P.update(params)
    a = R1.analyze(q.question, q.date)
    terms = a["terms"] + [w for f in a["facets"] for w in R1.content_terms(f)]
    ranked, final, rr, lab, facet_top, rest, win_active = R1._score_units(q, hx, a, P)

    miss = R1.missing_terms(hx, a["terms"])
    miss_frac = len(miss) / max(1, len([w for w in a["terms"] if w not in R1._GENERIC and not w.isdigit()]))
    top_rr = max(rr.values()) if rr else -10.0
    head = [u for u in ranked[:10] if R1.focus_sentences(hx.units[u].text, a["terms"])]
    all_other = bool(head) and not a["others_ok"] and all(lab[u] == "other" for u in head[:6])
    abstain_p = R1.abstain_prob(top_rr, miss_frac, all_other)
    hint = R1.build_hint(a, q.date, abstain_p)
    if P.get("latest_line") and a["kind"] in ("count", "sum") and not a["latest"] and not a.get("rate"):
        hint += " If a later session restates the same item with a new number, the latest statement wins."

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

    pins: dict[int, str] = {}
    rescued, resc_units, dbg = [], [], {}
    if P.get("vc"):
        rescued, resc_units, dbg = _vc_rescue(q, hx, a, terms, keep, win_active, P)
        pins.update(dbg.pop("vc_pins", {}) or {})
    depth_units: list[int] = []
    if P.get("depth"):
        _, _, sess_good = P3._sent_dense(q, hx)
        for sid in keep:
            for v, uid, s in (sess_good.get(sid) or [])[:P["depth_n"]]:
                if P.get("depth_min") is not None and v < P["depth_min"]:
                    continue
                depth_units.append(uid)
                pins[uid] = s
    keep_set = set(keep) | set(rescued)

    ranked_k = [u for u in ranked if hx.units[u].session_id in keep_set and hx.units[u].session_id not in rescued]
    ft_k = {f: [u for u in us if hx.units[u].session_id in set(keep)] for f, us in facet_top.items()}
    Pk = dict(P)
    Pk["max_sess"] = len(keep)
    Pk["deep_top"] = len(keep)
    picks = R1._select(hx, ranked_k, ft_k, Pk)
    extra = [u for u in resc_units + depth_units if u not in picks]
    if extra:
        k0 = min(len(picks), len(keep))
        picks = picks[:k0] + extra + picks[k0:]
    passages, included, chars, chosen = P3._pack(q, hx, picks, hint, Pk, terms, pins)
    in_pack = {p.id for p in passages}

    o = [s for s in order if s in in_pack]
    cite = [s for s in o if pg.get(s, 0.0) >= P["tc"]]
    if len(cite) < 2:
        cite = o[:2]
    if P.get("vc_cite"):
        cite += [s for s in rescued if s in in_pack and s not in cite]

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
                        debug={"ranking": ranked + rest, "keep": keep, "rescued": rescued, "n_keep": n, "top": top,
                               "window_active": win_active, "top_rr": top_rr, "missing_terms": miss,
                               "chars": chars, "abst_a": [ap_a, why_a], "vc": dbg,
                               "depth_units": depth_units})
