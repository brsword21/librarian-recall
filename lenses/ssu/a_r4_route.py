"""SSU-A round 4: post-fixes on top of the FROZEN v2 ssu route (lenses/ssu/a_r3_route.py kg3crga with the
configs/best_v2.json params). Imports the frozen files unchanged; every fix is a gated post-process of the v2 pack.

Variants (one hypothesis each, cumulative):
  v2      = the frozen v2 ssu route (byte-identical packs).
  h       + ownership-hint noun fix (round-4 brief item 2). The hint's anchor comes from a_owner.parse_question, which
            falls back to the last two content words when its ASCII-only NP regex fails ('café' -> 'before work',
            'bicycle I ride to work' -> 'ride work'), keeps participles ('renovation company working') and takes a unit
            for 'by how many ohms' ('ohms'). Only the hint text is rewritten (retrieval keeps the old parse):
            a unicode NP extractor ('name of the X', 'what <attr> is the X', 'by how many <unit> did I <verb> the X')
            gives the noun; if the old anchor is degenerate and no noun is found, the ownership sentence is dropped.
  hp      + residence hop (item 1, category -> instance): 'name of my neighborhood (in <City>)' questions; the answer
            sentence ('I live in Graça') carries neither 'neighborhood' nor the city. Scan the user's first-person
            residence declaratives (I live in / we moved to / my flat is in + Capitalised place), drop places named in
            the question, rank by bge cosine (hx.dense_texts) and add the best one's session to the pack and the cite
            plan when the pack has no residence declarative yet.
  hpd     + before/after delta (item 4): 'By how many <unit> did I <reduce|increase|...> the X in my Y?' -> affirmed
            own statements carrying a <unit> value that share an X word and the question's proper-noun anchor (R-77);
            the two best (cross-encoder) with different values, in date order, give before -> after; the pack is
            rebuilt from those two sessions, computed = |after - before| and the hint states it.
dev / dev200 have 0 questions matching the hp / hpd gates (scratch scan), so only `h` can move gold numbers (chars).
"""
from __future__ import annotations

import re

from core.pack import build_pack
from core.types import Passage
from core.loader import load_lens

_FROZEN = "lenses/ssu/a_r3_route.py"
_V2 = {"ssu_lens": "lenses/ssu/merged_r3.py", "ssu_variant": "ku11fn4s_p5", "role_module": "lenses/ssa/a_r3_role.py",
       "ku_lens": "lenses/ku/a_r3.py", "ku_variant": "r3w"}
_loaded: dict = {}


def _lens(path: str):
    if path not in _loaded:
        _loaded[path] = load_lens(path)
    return _loaded[path]


def _base_params() -> dict:
    return dict(_lens(_FROZEN).VARIANTS["kg3crga"], **_V2)


VARIANTS = {
    "v2": {"fix_hint": False, "hop": False, "delta": False},
    "h": {"fix_hint": True, "hop": False, "delta": False},
    "hp": {"fix_hint": True, "hop": True, "delta": False},
    "hpd": {"fix_hint": True, "hop": True, "delta": True},
}

# ---- 1. ownership-hint noun ---------------------------------------------------------------------------------
_L = r"A-Za-zÀ-ɏ"
_WORD = rf"[{_L}0-9][{_L}0-9'’-]*"
_STOPW = set("""i i'm i've i'd i'll we where that which who you for from in at on with before after now is was were are
did do does every each per these this to when while since during lately recently currently right so by and or has had
usually always still my our the a an""".split())
_ATTRW = {"name", "brand", "breed", "color", "colour", "type", "kind", "size", "model", "style", "title", "make",
          "value", "flavor", "genre"}
_ING_NOUNS = {"building", "wedding", "clothing", "painting", "morning", "evening", "meeting", "ring", "string", "thing",
              "spring", "ceiling", "pudding", "stuffing", "frosting", "icing", "parking", "housing", "training",
              "bedding", "booking", "recording", "setting", "sibling", "duckling", "dumpling", "earring", "king",
              "wing", "swing", "landing", "filling", "sewing", "knitting", "camping", "hiking", "running", "lighting"}
_UNITS = re.compile(r"^(?:ohms?|volts?|watts?|minutes?|hours?|days?|weeks?|months?|years?|miles?|km|pounds?|lbs?|kg|"
                    r"dollars?|percent|inches|feet|cups?|times|pages|laps|reps|seconds?)$", re.I)
_DEGEN = {"work", "home", "before", "after", "today", "tonight", "morning", "end", "use", "using", "up"}


def _np(text: str) -> str:
    """Leading noun phrase: up to 5 words, stops at a stop word, a participle (-ing, not a known noun) or punctuation."""
    out = []
    for w in re.findall(rf"{_WORD}|[?.!,;]", text)[:6]:
        lw = w.lower()
        if not re.match(rf"[{_L}0-9]", w) or lw in _STOPW:
            break
        if lw.endswith("ing") and lw not in _ING_NOUNS and out:
            break
        out.append(w)
        if len(out) == 5:
            break
    return " ".join(out)


_HINT_PATS = [
    re.compile(r"\bby how (?:many|much) \w+ did (?:i|we) \w+ (?:the |my |our )?(.+)", re.I),
    re.compile(r"\b(?:name|brand|title|breed|style|type|kind|model|make) of (?:the |my |that |this |our )(.+)", re.I),
    re.compile(r"\b(?:what|which) (?:name|brand|breed|colou?r|type|kind|model|make|style) (?:is|was|are|were) "
               r"(?:the |my |our )(.+)", re.I),
]


def hint_noun(question: str) -> str:
    for rx in _HINT_PATS:
        m = rx.search(question)
        if m:
            np_ = _np(m.group(1))
            if np_ and np_.split()[-1].lower() not in _DEGEN:
                return np_
    return ""


def degenerate(anchor: str, src: str) -> bool:
    words = anchor.split()
    if not words:
        return True
    last = words[-1]
    # head is a participle ('company working', 'been staying'), a function word ('service have'), a non-noun the
    # fallback picks up ('ride work', 'before work') or a unit ('ohms'); other fallback anchors ('ikea bookshelf',
    # 'japan') read fine and are kept
    if last.endswith("ing") and last not in _ING_NOUNS:
        return True
    return last in _DEGEN or last in _STOPW or bool(_UNITS.match(last))


_OWN1 = re.compile(r"^Answer from what the user said about their own (.+?)\. Mentions of other people's (.+?) "
                   r"\(friends, relatives, neighbors, coworkers\), plans, or things the user did not actually do are "
                   r"not the answer\.\s*")


def fix_hint(q, hint: str) -> tuple[str, dict]:
    from lenses.ssu.a_owner import parse_question
    m = _OWN1.match(hint or "")
    if not m:
        return hint, {"why": "no_own_sentence"}
    qp = parse_question(q.question)
    if not degenerate(qp.anchor, qp.source):
        return hint, {"why": "anchor_ok"}
    noun = hint_noun(q.question)
    rest = hint[m.end():]
    if not noun:
        return rest, {"why": "dropped", "old": qp.anchor}
    head = noun.split()[-1]
    new = (f"Answer from what the user said about their own {noun}. Mentions of other people's {head} (friends, "
           f"relatives, neighbors, coworkers), plans, or things the user did not actually do are not the answer. ")
    return (new + rest).strip(), {"why": "fixed", "old": qp.anchor, "new": noun}


# ---- 2. residence hop -----------------------------------------------------------------------------------------
_PLACE_Q = re.compile(r"\b(?:my|our) (?:\w+ ){0,2}(neighbou?rhood|district|suburb|borough|quarter)\b|"
                      r"\b(neighbou?rhood|district|suburb|borough|quarter) (?:is|do|did) (?:my|our|i|we)\b", re.I)
_CAP = rf"[A-ZÀ-Þ][{_L}'’-]+(?: [A-ZÀ-Þ][{_L}'’-]+){{0,2}}"
_LIVE = re.compile(rf"\b(?:I|we)(?:'m| am|'re| are|'ve| have)? (?:now |just |recently |finally )?"
                   rf"(?:live|lived|living|moved|moving|settled|based)(?: here)? (?:in|to|into) (?:the )?({_CAP})|"
                   rf"\bmy (?:new )?(?:flat|apartment|place|house|home|street) (?:is )?in (?:the )?({_CAP})|"
                   rf"\b(?:I|we) (?:just )?(?:rent|rented|bought) (?:a |an |our |my )?(?:\w+ )?(?:flat|apartment|place|"
                   rf"house) in (?:the )?({_CAP})")


_STATED = re.compile(rf"\b(?:live|living|moved|moving|apartment|flat|place|house|home|lease|neighbou?rhood)\b[^.!?]{{0,40}}?"
                     rf"\b(?:in|to) (?:the )?({_CAP})")


def residence_hop(q, hx, pack, min_cos: float = 0.35, min_persona: float = 0.62) -> tuple[bool, dict]:
    if not _PLACE_Q.search(q.question):
        return False, {"why": "no_gate"}
    from lenses.ssu.a_owner import sentences
    qcaps = {w.lower() for w in re.findall(rf"[A-Z][{_L}-]+", q.question[1:])}
    cands = []
    for u in hx.units:
        if u.role != "user":
            continue
        for s in sentences(u.text):
            m = _LIVE.search(s)
            if not m:
                continue
            place = next(g for g in m.groups() if g)
            if place.lower() in qcaps or place.split()[0].lower() in ("i", "the"):
                continue
            cands.append((u.uid, s, place))
    if not cands:
        return False, {"why": "no_candidates"}
    # pack already holds a residence declarative or states a place for the user's home -> leave it
    in_pack = {p.id for p in pack.passages}
    if any(hx.units[uid].session_id in in_pack for uid, _, _ in cands):
        return False, {"why": "already_in_pack", "n": len(cands)}
    for p_ in pack.passages:
        for line in p_.text.split("\n"):
            if line.startswith("user:"):
                for m in _STATED.finditer(line):
                    if m.group(1).lower() not in qcaps:
                        return False, {"why": "pack_states_place", "place": m.group(1)}
    cos = hx.dense_texts(q.question, [s for _, s, _ in cands])
    best = max(range(len(cands)), key=lambda i: cos[i])
    info = {"n": len(cands), "best": cands[best][1][:120], "cos": round(float(cos[best]), 3)}
    if cos[best] < min_cos:
        info["why"] = "weak"
        return False, info
    # persona coherence: when the question names a place (Lisbon), the candidate unit must read like the user's
    # sessions that mention it (filler sessions from other personas carry 'I just moved to Austin from NYC')
    if qcaps:
        city_units = [u for u in hx.units if u.role == "user" and any(c in u.text.lower() for c in qcaps)]
        if city_units:
            cu = hx.units[cands[best][0]]
            pc = hx.dense_texts(cu.text, [u.text for u in city_units if u.session_id != cu.session_id][:200])
            info["persona_cos"] = round(float(max(pc)), 3) if len(pc) else None
            if not len(pc) or max(pc) < min_persona:
                info["why"] = "persona"
                return False, info
    info["why"] = "fire"
    info["uid"] = cands[best][0]
    return True, info


def add_units(q, hx, pack, uids: list[int], cite: list[str], replace: bool = False, budget: int = 12000):
    """Render `uids` (+-1 neighbour) as passages and merge them into the pack chronologically."""
    hint = pack.hint
    ps, turns, _ = build_pack(q, hx, uids, budget=budget, window=1, order="chrono", hint=hint)
    new_ids = {p.id for p in ps}
    keep = [] if replace else [p for p in pack.passages if p.id not in new_ids]
    allp = keep + ps
    sdate = {s["session_id"]: s["date"] for s in hx.sessions}
    allp.sort(key=lambda p: (sdate.get(p.id) is None, sdate.get(p.id)))
    pack.passages = allp
    pack.included_turns = (set() if replace else set(pack.included_turns or [])) | set(turns)
    pack.cite_plan = cite
    return pack


# ---- 3. before / after delta ----------------------------------------------------------------------------------
_DELTA_Q = re.compile(r"\bby how (?:many|much) ([\w-]+) did (?:i|we) (reduce|increase|lower|raise|cut|drop|change|"
                      r"decrease|boost|extend|shorten|bump|up|trim)\w* (?:the |my |our )?(.+)", re.I)
_VAL = re.compile(r"(\d+(?:\.\d+)?)\s*(k|m)?\s*-?\s*(ohms?|Ω)", re.I)


def _unit_rx(unit: str):
    u = unit.lower().rstrip("s")
    if u == "ohm":
        return _VAL
    return re.compile(rf"(\d+(?:\.\d+)?)\s*()-?\s*({re.escape(u)}s?)\b", re.I)


def before_after(q, hx) -> tuple[bool, dict]:
    m = _DELTA_Q.search(q.question)
    if not m:
        return False, {"why": "no_gate"}
    from lenses.ssu.a_owner import classify, parse_question, sentences
    unit, verb, obj = m.group(1), m.group(2).lower(), m.group(3)
    obj_np = _np(obj)
    obj_words = {w.lower() for w in obj_np.split() if len(w) > 2}
    anchors = {w for w in re.findall(r"\b[A-Z][\w-]*\d[\w-]*\b|\b[A-Z]{2,}[\w-]*", q.question)}
    vrx = _unit_rx(unit)
    qp = parse_question(q.question)
    cands = []
    for u in hx.units:
        if u.role != "user":
            continue
        for s in sentences(u.text):
            vals = [(float(a) * {"k": 1e3, "m": 1e6}.get((b or "").lower(), 1)) for a, b, _ in vrx.findall(s)]
            if len(set(vals)) != 1:
                continue
            sl = s.lower()
            if anchors and not all(a.lower() in sl for a in anchors):
                continue
            if not any(re.search(rf"\b{re.escape(w)}", sl) for w in obj_words):
                continue
            sc = classify(s, qp)
            if sc.owner == "mismatch" or sc.polarity in ("neg", "plan") or re.search(
                    r"\b(briefly|tried|tested|didn't keep|removed it|parts bin|junked|ordered|planning)\b", sl):
                continue
            cands.append((u.uid, s, vals[0]))
    info = {"n": len(cands), "unit": unit, "obj": obj_np}
    if len(cands) < 2:
        info["why"] = "few"
        return False, info
    rr = hx.rerank_texts(q.question, [s for _, s, _ in cands])
    order = sorted(range(len(cands)), key=lambda i: -rr[i])
    first = order[0]
    second = next((i for i in order[1:] if cands[i][2] != cands[first][2]
                   and hx.units[cands[i][0]].session_id != hx.units[cands[first][0]].session_id), None)
    if second is None:
        info["why"] = "no_pair"
        return False, info
    a, b = sorted((first, second), key=lambda i: cands[i][0])
    before, after = cands[a][2], cands[b][2]
    diff = abs(after - before)
    direction = "reduction" if after < before else "increase"
    if (verb in ("reduce", "lower", "cut", "drop", "decrease", "shorten", "trim") and after > before) or \
            (verb in ("increase", "raise", "boost", "extend", "bump", "up") and after < before):
        info["why"] = "direction_mismatch"
        return False, info

    def fmt(v):
        return f"{v / 1e3:g}k" if v >= 1e3 and unit.lower().startswith("ohm") else f"{v:g}"
    ua, ub = hx.units[cands[a][0]], hx.units[cands[b][0]]
    info.update(why="fire", uids=[ua.uid, ub.uid], before=before, after=after, diff=diff,
                hint=(f"From the user's own statements: {fmt(before)} {unit} on {ua.date_str[:10]}, then {fmt(after)} "
                      f"{unit} on {ub.date_str[:10]}, a {direction} of {fmt(diff)} {unit} ({diff:,.0f} {unit})."))
    return True, info


# ---- route lens ---------------------------------------------------------------------------------------------
def build(q, hx, _variant: str = "", fix_hint: bool = False, hop: bool = False, delta: bool = False, **_):
    pack = _lens(_FROZEN).build(q, hx, _variant="kg3crga", **_base_params())
    info: dict = {}
    if pack.route != "ssu":
        pack.variant = _variant
        return pack
    if delta:
        try:
            hit, d = before_after(q, hx)
            info["delta"] = {k: v for k, v in d.items() if k != "hint"}
            if hit:
                sids = list(dict.fromkeys(hx.units[u].session_id for u in d["uids"]))
                pack.hint = d["hint"]
                pack = add_units(q, hx, pack, d["uids"], cite=sids, replace=True)
                pack.computed = f"{d['diff']:g}"
                pack.abstain_p = min(pack.abstain_p or 0.0, 0.05)
        except Exception as exc:
            info["delta_error"] = repr(exc)[:200]
    if fix_hint and info.get("delta", {}).get("why") != "fire":
        new, f = fix_hint_fn(q, pack.hint)
        info["hint_fix"] = f
        pack.hint = new
    if hop and info.get("delta", {}).get("why") != "fire":
        try:
            hit, h = residence_hop(q, hx, pack)
            info["hop"] = {k: v for k, v in h.items()}
            if hit:
                sid = hx.units[h["uid"]].session_id
                pack = add_units(q, hx, pack, [h["uid"]], cite=[sid])
                pack.abstain_p = min(pack.abstain_p or 0.0, 0.1)
        except Exception as exc:
            info["hop_error"] = repr(exc)[:200]
    pack.variant = _variant
    pack.debug = {**(pack.debug or {}), "r4": info}
    return pack


fix_hint_fn = fix_hint
