"""Temporal lens B ("Semantyka"): per-event retrieval over a date-bearing candidate pool, cross-encoder rerank plus
cheap semantic features (subject = me / someone else, contrast cue, event-token overlap, year sanity), date resolution
in code, day gap passed to Reactor as a hint.

Pipeline (see b_design.md):
  1. parse_question(): question -> kind (apart/after/before/since_when/ago/first/order/howlong/other), unit, events.
  2. per event: pool = units with an explicit date  U  hybrid top-k for the event phrase (U time-window units when the
     question carries an anchor like "two weeks ago"); cross-encoder rerank; feature re-score.
  3. resolve each chosen unit's event date (explicit date nearest the event words > relative expression vs the session
     date > session date); compute the gap in the question's unit.
  4. pack = top `alts` units per event (+/- window) then whole-question hybrid fill, chrono order; hint = dates + gap.

Self-test (pure python, no models):  .venv/bin/python -m lenses.temporal.b_evcal
"""
from __future__ import annotations

import math
import re
from datetime import date, datetime, timedelta

from core.dates import extract_dates
from core.types import EvidencePack

VARIANTS = {
    # ranked by expected value on TEST (see b_design.md)
    "evcal": {"k_hyb": 30, "alts": 2, "budget": 7000, "window": 1, "fill": True, "conf_min": 0.0,
              "w_lex": 3.0, "w_date": 1.0, "w_self": 1.0, "w_other": 4.0, "w_neg": 5.0, "w_year": 3.0,
              "time_prior": True, "cite": "top1"},
    "evcal_gated": {"k_hyb": 30, "alts": 2, "budget": 7000, "window": 1, "fill": True, "conf_min": 0.6,
                    "w_lex": 3.0, "w_date": 1.0, "w_self": 1.0, "w_other": 4.0, "w_neg": 5.0, "w_year": 3.0,
                    "time_prior": True, "cite": "top1"},
    "evcal_wide": {"k_hyb": 50, "alts": 3, "budget": 12000, "window": 1, "fill": True, "conf_min": 0.0,
                   "w_lex": 3.0, "w_date": 1.0, "w_self": 1.0, "w_other": 4.0, "w_neg": 5.0, "w_year": 3.0,
                   "time_prior": True, "cite": "top1"},
    "ce_only": {"k_hyb": 30, "alts": 2, "budget": 7000, "window": 1, "fill": True, "conf_min": 0.0,
                "w_lex": 0.0, "w_date": 0.0, "w_self": 0.0, "w_other": 0.0, "w_neg": 0.0, "w_year": 0.0,
                "time_prior": False, "cite": "top1"},
}

# ---------------------------------------------------------------------------------------------------------------------
# question parsing
# ---------------------------------------------------------------------------------------------------------------------
_U = r"(days|weeks|months|years)"
_AUX = r"(?:did|was|were|is|are|do|does|had|have|has|will)"
_LEAD = r"(?:the day (?:when |that )?|the time (?:when )?|when )?"
Q_PATTERNS = [
    ("after", re.compile(rf"how many {_U} after (.+?) did (.+?)[?.!]*$", re.I)),     # prefer ' did ' as the split
    ("before", re.compile(rf"how many {_U} before (.+?) did (.+?)[?.!]*$", re.I)),
    ("apart", re.compile(rf"how many {_U} apart (?:were|was|are|is) (.+?) and (.+?)[?.!]*$", re.I)),
    ("after", re.compile(rf"how many {_U} after (.+?) {_AUX} (.+?)[?.!]*$", re.I)),
    ("before", re.compile(rf"how many {_U} before (.+?) {_AUX} (.+?)[?.!]*$", re.I)),
    ("between", re.compile(rf"how (?:many {_U}|long) (?:had |have |has )?(?:passed |elapsed |gone by |were there |was it )?"
                           rf"between {_LEAD}(.+?) and {_LEAD}(.+?)[?.!]*$", re.I)),
    ("since_when", re.compile(rf"how many {_U} (?:had |have )?(?:passed |elapsed )?since (.+?),? when (.+?)[?.!]*$", re.I)),
    ("ago_when", re.compile(rf"how many {_U} ago did i (.+?),? when (.+?)[?.!]*$", re.I)),
    ("howlong_when", re.compile(rf"how (?:long|many {_U}) (?:had|have) i been (.+?),? when (.+?)[?.!]*$", re.I)),
    ("ago", re.compile(rf"how many {_U} (?:ago (?:did|was|were) (?:i )?|have passed since |has it been since |since )(.+?)[?.!]*$", re.I)),
    ("spent", re.compile(rf"how many {_U} did (?:i|it take me to) (?:spend on |spend |take to )?(.+?)[?.!]*$", re.I)),
]
_FIRST = re.compile(r"\b(?:first|earlier|most recently|last)\b,?\s*(.+?)\s+or\s+(.+?)[?.!]*$", re.I)
_ORDER = re.compile(r"\border\b", re.I)
_THIS_YEAR = re.compile(r"\s*\b(this year|this time)\b", re.I)


def _clean_event(s: str) -> str:
    s = re.sub(r"(?<![\w])['\"\u2018\u201c]|['\"\u2019\u201d](?![\w])", "", s.strip()).strip()
    s = re.sub(r"^(?:the day|the time|when)\s+(?:when\s+|that\s+)?", "", s, flags=re.I)
    s = _THIS_YEAR.sub("", s)
    s = re.sub(r"^(?:she|he|they|it|we)\s+(have|had|do|did)\s+", "", s, flags=re.I)  # "did she have her first vet .."
    return s.strip(" ,")


def parse_question(text: str) -> dict:
    """-> {kind, unit, events: [str], sign: +1 = later - earlier as ordered in `events`, this_year: bool}."""
    q = " ".join(text.strip().split())
    q = q.split("?")[0] + "?" if "?" in q else q          # "...the annual car inspection? I lost track."
    out = {"kind": "other", "unit": "days", "events": [], "this_year": bool(_THIS_YEAR.search(q))}
    for kind, pat in Q_PATTERNS:
        m = pat.search(q)
        if not m:
            continue
        g = [x for x in m.groups()]
        unit = (g[0] or "days").lower() if kind != "between" or g[0] else "days"
        evs = [x for x in g[1:] if x]
        out.update(kind=kind, unit=unit, events=[_clean_event(e) for e in evs])
        return out
    quoted = re.findall(r"['‘\"“]([^'’\"”]{6,}?)['’\"”](?=[,.:;?\s]|$)", q)
    if _ORDER.search(q) or re.search(r"from first to last|earliest to latest", q, re.I):
        if len(quoted) >= 2:
            out.update(kind="order", events=[_clean_event(x) for x in quoted])
            return out
        if ":" in q:
            tail = q.split(":", 1)[1]
            parts = [p for p in re.split(r",\s*(?:and\s+)?|\s+and\s+(?=the day|the|my|i )", tail) if p.strip()]
            if len(parts) >= 2:
                out.update(kind="order", events=[_clean_event(p) for p in parts])
                return out
        out["kind"] = "order"
        return out
    m = _FIRST.search(q)
    if m and re.search(r"\b(which|who|what)\b", q, re.I):
        out.update(kind="first", events=[_clean_event(m.group(1)), _clean_event(m.group(2))])
        return out
    return out


def question_anchor(text: str, qdate: datetime) -> tuple[date, int] | None:
    """Relative time anchor in the question ('two weeks ago', 'last Saturday', 'a month ago') -> (date, tolerance days)."""
    for h in extract_dates(text, qdate):
        if h.kind in ("ago", "weekday", "simple", "period") or h.explicit_year or h.kind in ("mdy", "dmy", "iso"):
            span = (qdate.date() - h.date).days
            tol = 2 if h.kind in ("weekday", "simple") else max(3, round(abs(span) * 0.25))
            if h.kind in ("mdy", "dmy", "iso"):
                tol = 1
            return h.date, tol
    m = re.search(r"\b(past|last)\s+(weekend)\b", text, re.I)
    if m:
        return qdate.date() - timedelta(days=(qdate.weekday() - 5) % 7 or 7), 2
    return None


# ---------------------------------------------------------------------------------------------------------------------
# sentence features
# ---------------------------------------------------------------------------------------------------------------------
_STOP = set("""a an the my our your his her their its i me we you he she they it of to in on at for from with by and or
but is was were are be been being did do does had have has will would that this these those which who whom what when
where how many much day days time first last after before between apart since ago up out about into over just also
again then than so very really get got go went attended attend do did""".split())
_WORD = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
PERSON = (r"friend|coworker|co-worker|colleague|roommate|flatmate|housemate|neighbou?r|boss|manager|sister|brother|"
          r"cousin|mom|mum|mother|dad|father|son|daughter|wife|husband|partner|girlfriend|boyfriend|fianc[eé]e?|aunt|"
          r"uncle|niece|nephew|grandma|grandmother|grandpa|grandfather|classmate|teammate|mentor|teacher|student|kid|"
          r"child|teenager|teen|team ?mate|supervisor|client")
_OTHER = re.compile(rf"^(?:my|your|our)\s+(?:(?:best|close|old|older|younger|little|big|new|good|eldest|youngest)\s+)?"
                    rf"({PERSON})s?\b(?:\s+([A-Z][a-z]+))?", re.I)
_NAME_SUBJ = re.compile(r"^([A-Z][a-z]{2,})(?:'s\b|\s+(?:also\s+|just\s+|finally\s+)?[a-z]+(?:ed|s)?\b)")
_NOT_NAMES = set("""I The My Your Our Also Oh By Actually Honestly Random Speaking Meanwhile Anyway So Well Yes No Sure
Thanks Then And But On In At For After Before Last This That Today Yesterday Small Quick One Just Since While When Also
You Your We They She He It There Here What Speaking Remember Reminder Update Random
Monday Tuesday Wednesday Thursday Friday Saturday Sunday January February March April May June July August September
October November December Interesting Great Good Side Life Calendar Note Plus""".split())
_PREFIX = re.compile(r"^.*?(?::\s+|—\s*|\bthat reminds me(?: of something)?[:,]?\s+|\bby the way[:,]?\s+|"
                     r"\bon a (?:completely )?different note[:,]?\s+|\b(?:oh|also|actually|honestly|plus|and)[:,]?\s+)",
                     re.I)
_SELF_USER = re.compile(r"^(?:i|i'm|i've|i'd|i'll|we|we've|we're|me and|my wife and i|my husband and i)\b", re.I)
_SELF_ASSIST = re.compile(r"^(?:you|you've|you're|you'd)\b", re.I)
_CONTRAST = re.compile(r"\b(?:not|isn't|wasn't|aren't|weren't|rather than|instead of|separate from|unlike|different from|"
                       r"other than|apart from|as opposed to)\s+(?:the\s+|my\s+|your\s+|a\s+|an\s+|her\s+|his\s+)?((?:[\w'-]+\s*){1,4})",
                       re.I)
_VARIANT = re.compile(r"\b(?:a different|another|a second|second set|the other)\b", re.I)
_SENT = re.compile(r"(?<=[.!?])\s+|\n+")


def stem(w: str) -> str:
    w = w.lower().replace("’", "'")
    if w.endswith("'s"):
        w = w[:-2]
    for suf in ("ing", "ed", "es", "s"):
        if len(w) > 4 and w.endswith(suf):
            w = w[: -len(suf)]
            break
    return w[:-1] if len(w) > 4 and w.endswith("e") else w


def content_tokens(text: str) -> set[str]:
    return {stem(w) for w in _WORD.findall(text.lower().replace("’", "'")) if w not in _STOP and len(w) > 1}


def question_persons(event: str) -> set[str]:
    """Person nouns and capitalised names inside the event phrase: these are NOT decoy subjects."""
    out = {m.group(0).lower() for m in re.finditer(rf"\b(?:{PERSON})s?\b", event, re.I)}
    out |= {w for w in re.findall(r"\b[A-Z][a-z]{2,}\b", event)}
    return {stem(x) if x.islower() else x for x in out}


def subject_class(sentence: str, role: str, qpersons: set[str]) -> str:
    """'self' | 'other' | 'none' for the clause that carries the fact."""
    s = sentence.strip().strip("*-•> ").strip()
    clauses = []
    cut = max(s.rfind(": ", 0, int(len(s) * 0.7)), s.rfind("—", 0, int(len(s) * 0.7)))
    if cut > 0:
        clauses.append(s[cut + 1:].strip(" —:"))
    pos, words = 0, 0
    for _ in range(3):                                     # "Also, speaking of Canada logistics, Camille booked ..."
        j = s.find(", ", pos)
        if j < 0:
            break
        words += len(s[pos:j].split())
        if words > 8:
            break
        clauses.insert(0 if cut <= 0 else 1, s[j + 2:])
        pos = j + 2
    m = _PREFIX.match(s)
    if m and m.end() < len(s) - 10:
        clauses.append(s[m.end():])
    clauses.append(s)
    lead = re.compile(r"^(?:(?:and|so|but|oh|also|plus|then|well|anyway|actually|honestly|btw)\b[,:]?\s+)+", re.I)
    for clause in clauses:
        clause = lead.sub("", clause)
        mo = _OTHER.match(clause)
        if mo:
            noun, name = stem(mo.group(1).lower()), mo.group(2)
            if noun in qpersons or (name and name in qpersons):
                return "self"
            return "other"
        mn = _NAME_SUBJ.match(clause)
        if mn and mn.group(1) not in _NOT_NAMES:
            return "self" if mn.group(1) in qpersons else "other"
        if role == "user" and _SELF_USER.match(clause):
            return "self"
        if role == "assistant" and _SELF_ASSIST.match(clause):
            return "self"
        if re.match(r"^(?:my|our)\b", clause, re.I) and role == "user":
            return "self"
        if re.match(r"^your\b", clause, re.I) and role == "assistant":
            return "self"
    return "none"


def contrast_hit(sentence: str, ev_tokens: set[str]) -> bool:
    """'..., not the orthodontist appointment' / 'separate from the play' negating the asked event."""
    for m in _CONTRAST.finditer(sentence):
        if content_tokens(m.group(1)) & ev_tokens:
            return True
    return False


def best_sentence(text: str, ev_tokens: set[str]) -> tuple[str, float]:
    best, score = text, -1.0
    for s in _SENT.split(text):
        if not s.strip():
            continue
        tok = content_tokens(s)
        ov = len(tok & ev_tokens) / max(1, len(ev_tokens))
        if extract_dates(s, None, relative=False):
            ov += 0.01
        if ov > score:
            best, score = s, ov
    return best, max(0.0, score)


def resolve_date(sentence: str, unit_text: str, sess_date: datetime, ev_tokens: set[str]) -> tuple[date, str]:
    """Event date for a unit: explicit date nearest the event words > relative expression > the session date."""
    for txt in (sentence, unit_text):
        hits = extract_dates(txt, sess_date)
        exp = [h for h in hits if h.kind in ("iso", "mdy", "dmy", "slash")]
        rel = [h for h in hits if h.kind not in ("iso", "mdy", "dmy", "slash")]
        for group, tag in ((exp, "explicit"), (rel, "relative")):
            if not group:
                continue
            if len(group) == 1:
                return group[0].date, tag
            low = txt.lower()
            ends = {id(h): max([0] + [g.end for g in group if g.end <= h.start]) for h in group}

            def near(h):   # the clause before the date ("... passport arrived on <date>") plus a short tail
                win = low[max(ends[id(h)], h.start - 90): h.end + 25]
                return len(content_tokens(win) & ev_tokens)
            return max(group, key=lambda h: (near(h), -h.start)).date, tag
    return sess_date.date(), "session"


def convert(days: int, unit: str) -> int:
    return {"days": days, "weeks": round(days / 7), "months": round(days / 30.44),
            "years": round(days / 365.25)}.get(unit, days)


def compute_answer(spec: dict, dates: list[date], qdate: datetime) -> int | None:
    k, unit = spec["kind"], spec["unit"]
    if k == "ago" and dates:
        return convert((qdate.date() - dates[0]).days, unit)
    if len(dates) < 2:
        return None
    a, b = dates[0], dates[1]
    if k in ("apart", "between", "ago_when", "since_when"):
        return convert(abs((b - a).days), unit)
    if k == "after":
        return convert(abs((b - a).days), unit)
    if k == "before":
        return convert(abs((a - b).days), unit)
    return None


def fmt(d: date) -> str:
    return d.strftime("%Y-%m-%d (%a)")


def build_hint(spec: dict, picks: list[dict], answer: int | None, confident: bool) -> str:
    if not picks:
        return ""
    lines = []
    for i, p in enumerate(picks, 1):
        src = {"explicit": "date written in the text", "relative": "relative date resolved against the session date",
               "session": "session date"}[p["how"]]
        lines.append(f"({i}) \"{p['event']}\" -> {fmt(p['date'])} [{src}; session {p['sess_date']}]")
    hint = "Script-extracted event dates (ignore events of other people, other years or different events): " + "; ".join(lines) + "."
    if spec["kind"] in ("first", "order") and all(p.get("date") for p in picks):
        order = sorted(picks, key=lambda p: p["date"])
        hint += " Chronological order: " + " < ".join(f"\"{p['event']}\"" for p in order) + "."
    if answer is not None and confident:
        hint += f" Computed difference: {answer} {spec['unit']}."
    return hint


def other_hint(events: list[str], ranked: list[list[dict]]) -> str:
    bad = [ev for ev, rows in zip(events, ranked) if only_other(rows)]
    if not bad:
        return ""
    return (" Note: for " + "; ".join(f'"{e}"' for e in bad) + " the history only shows someone else (not the user) "
            "doing it; if the user never did it, answer INSUFFICIENT_EVIDENCE.")


# ---------------------------------------------------------------------------------------------------------------------
# retrieval
# ---------------------------------------------------------------------------------------------------------------------
_DATED_CACHE: dict[tuple, list[int]] = {}
_DATED_RE = re.compile(r"\b(?:19|20)\d\d-\d\d-\d\d\b|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2}(?:st|nd|rd|th)?\b"
                       r"|\b\d{1,2}(?:st|nd|rd|th)?\s+(?:of\s+)?(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b"
                       r"|\b\d{1,2}/\d{1,2}/\d{4}\b", re.I)


def dated_units(hx) -> list[int]:
    key = (hx.split, hx.history_id)
    if key not in _DATED_CACHE:
        _DATED_CACHE[key] = [u.uid for u in hx.units if _DATED_RE.search(u.text) and extract_dates(u.text, u.date, relative=False)]
    return _DATED_CACHE[key]


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def rank_event(hx, q, event: str, spec: dict, params: dict, anchor) -> list[dict]:
    ev_tok = content_tokens(event)
    qp = question_persons(event)
    pool = set(dated_units(hx))
    pool |= {i for i, _ in hx.hybrid(event, k=params["k_hyb"])}
    if anchor is not None and params.get("time_prior"):
        lo, hi = anchor[0] - timedelta(days=3 * anchor[1] + 2), anchor[0] + timedelta(days=anchor[1] + 2)
        win = [u.uid for u in hx.units if lo <= u.date.date() <= hi]
        if win:
            pool |= {i for i, _ in hx.hybrid(event, k=20, uids=win)}
    pool = [i for i in pool if hx.units[i].date <= q.date + timedelta(days=1)]
    if not pool:
        return []
    ce = dict(hx.rerank(event, pool))
    rows = []
    for uid in pool:
        u = hx.units[uid]
        sent, ov = best_sentence(u.text, ev_tok)
        d, how = resolve_date(sent, u.text, u.date, ev_tok)
        subj = subject_class(sent, u.role, qp)
        f = ce[uid] + params["w_lex"] * ov
        f += params["w_date"] * (1.0 if how == "explicit" else 0.3 if how == "relative" else 0.0)
        f += params["w_self"] * (subj == "self") - params["w_other"] * (subj == "other")
        if contrast_hit(sent, ev_tok):
            f -= params["w_neg"]
        if _VARIANT.search(sent):
            f -= 0.3 * params["w_neg"]
        if spec.get("this_year") and how == "explicit" and d.year != q.date.year:
            f -= params["w_year"]
        elif how == "explicit" and (u.date.date() - d).days > 200:
            f -= 0.5 * params["w_year"]          # "Our anniversary was June 21, 2022" (stale year)
        if anchor is not None and params.get("time_prior"):
            f += 2.0 * math.exp(-abs((d - anchor[0]).days) / max(2, anchor[1]))
        rows.append({"uid": uid, "score": f, "ce": ce[uid], "date": d, "how": how, "subj": subj, "sent": sent.strip()[:200],
                     "session_id": u.session_id, "sess_date": u.date_str[:10]})
    rows.sort(key=lambda r: -r["score"])
    return rows


def abstain_prob(hx, events: list[str], ranked: list[list[dict]]) -> float:
    """Soft signals: a distinctive question word absent from the whole history; weak best cross-encoder score."""
    if not hasattr(hx, "_b_vocab"):
        hx._b_vocab = set(w for u in hx.units for w in _WORD.findall(u.text.lower()))
    missing = 0
    for ev in events:
        for w in re.findall(r"\b[A-Z][a-zA-Z]{2,}\b|'[^']{3,}'", ev):
            toks = _WORD.findall(w.lower())
            if toks and not all(t in hx._b_vocab for t in toks):
                missing += 1
    p_lex = 0.6 if missing else 0.0
    best = min((r[0]["ce"] if r else -12.0) for r in ranked) if ranked else -12.0
    p_ce = _sigmoid(-(best + 2.0) / 2.0)      # ms-marco logits: > 2 clear match, < -6 noise
    p_other = 0.75 if any(only_other(r) for r in ranked) else 0.0
    return round(1 - (1 - p_lex) * (1 - 0.7 * p_ce) * (1 - p_other), 3)


def only_other(rows: list[dict], top: int = 10) -> bool:
    """Test near-miss trap: the event is only ever done by someone else ('Camille got a sourdough starter today')."""
    rows = rows[:top]
    oth = [r for r in rows if r["subj"] == "other"]
    own = [r for r in rows if r["subj"] != "other"]
    if not oth:
        return False
    best_other = max(r["ce"] for r in oth)
    best_own = max((r["ce"] for r in own), default=-12.0)
    return best_other - best_own > 3.0 and best_other > 0.0


def build(q, hx, **params) -> EvidencePack:
    from core.pack import build_pack   # imported here so the self-test runs without the kit on sys.path

    p = {**VARIANTS["evcal"], **params}
    spec = parse_question(q.question)
    anchor = question_anchor(q.question, q.date) if spec["kind"] in ("other", "first", "order") else None
    events = spec["events"] or [re.sub(r"\b(how many|what|which|who|when|did i|do i)\b", "", q.question, flags=re.I).strip()]

    ranked = [rank_event(hx, q, ev, spec, p, anchor) for ev in events]
    picks, conf = [], 1.0
    for ev, rows in zip(events, ranked):
        if not rows:
            conf = 0.0
            continue
        top = rows[0]
        rival = next((r for r in rows[1:] if r["date"] != top["date"]), None)
        margin = top["score"] - rival["score"] if rival else 6.0
        conf = min(conf, _sigmoid(margin / 2.0))
        picks.append({"event": ev, **top})

    answer = None
    if spec["events"] and len(picks) == len(events):
        answer = compute_answer(spec, [x["date"] for x in picks], q.date)
    trap = any(only_other(r) for r in ranked)            # only someone else did it -> no gap, abstain hint
    confident = answer is not None and conf >= p["conf_min"] and not trap
    hint = build_hint(spec, picks, answer, confident) if spec["events"] else ""
    if spec["events"]:
        hint += other_hint(events, ranked)

    order_uids: list[int] = []
    for j in range(p["alts"]):
        for rows in ranked:
            if j < len(rows):
                order_uids.append(rows[j]["uid"])
    if p.get("fill"):
        for uid, _ in hx.hybrid(q.question, k=40):
            order_uids.append(uid)
    order_uids = list(dict.fromkeys(order_uids))
    passages, included, chars = build_pack(q, hx, order_uids, budget=p["budget"], window=p["window"], order="chrono",
                                           hint=hint)
    cite = list(dict.fromkeys(x["session_id"] for x in picks)) if picks else None
    computed = None
    if confident and spec["kind"] in ("apart", "after", "before", "between", "since_when", "ago_when", "ago"):
        computed = str(answer)
    elif p.get("computed_first") and spec["kind"] == "first" and len(picks) == 2 and picks[0]["date"] != picks[1]["date"]:
        computed = min(picks, key=lambda x: x["date"])["event"]
    return EvidencePack(question_id=q.question_id, variant=p.get("_variant", ""), route="temporal", passages=passages,
                        hint=hint, cite_plan=cite, abstain_p=abstain_prob(hx, events, ranked),
                        included_turns=included, computed=computed,
                        debug={"ranking": order_uids, "spec": spec, "conf": round(conf, 3),
                               "picks": [{k: (str(v) if isinstance(v, date) else v) for k, v in x.items()} for x in picks]})


# ---------------------------------------------------------------------------------------------------------------------
# self-test on sample data (no models, no kit)
# ---------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    import json

    cases = [
        ("How many days apart were my son's school play and my daughter's orthodontist appointment?", "apart"),
        ("How many days after I entered my alfajor recipe in the city baking contest were the winners announced?", "after"),
        ("How many days after I adopted Juniper did she have her first vet appointment?", "after"),
        ("How many days after my wife's birthday was our anniversary this year?", "after"),
        ("How many days passed between the day I started watering my herb garden and the day I harvested my first batch of fresh herbs?", "between"),
        ("How many days had passed since I bought my Adidas running shoes when I realized one of the shoelaces on my old Converse sneakers had broken?", "since_when"),
        ("How many days ago did I attend a baking class at a local culinary school when I made my friend's birthday cake?", "ago_when"),
        ("How many weeks ago did I meet up with my aunt and receive the crystal chandelier?", "ago"),
        ("How many days before the 'Rack Fest' did I participate in the 'Turbocharged Tuesdays' event?", "before"),
        ("Which event happened first, my cousin's wedding or Michael's engagement party?", "first"),
        ("What is the order of the three events: 'I signed up for the rewards program at ShopRite', 'I used a Buy One Get One Free coupon on Luvs diapers at Walmart', and 'I redeemed $12 cashback for a $10 Amazon gift card from Ibotta'?", "order"),
        ("How long had I been bird watching when I attended the bird watching workshop?", "howlong_when"),
        ("What gardening-related activity did I do two weeks ago?", "other"),
    ]
    bad = 0
    for text, want in cases:
        s = parse_question(text)
        ok = s["kind"] == want
        bad += not ok
        print("OK " if ok else "BAD", s["kind"], s["unit"], s["events"], "this_year" if s["this_year"] else "")
    ref = datetime(2023, 10, 6)
    print("anchor", question_anchor("What gardening-related activity did I do two weeks ago?", ref))

    sents = [
        ("user", "Oh, one more thing: My son's school play is on 2023-06-10, so I need to shift a client call that Friday.", "my son's school play", "self"),
        ("user", "My son's school concert is on 2023-06-09, but that is separate from the play.", "my son's school play", "self"),
        ("assistant", "While we're looking at your evening schedule, one quick note: Your daughter's regular dentist checkup is on 2023-07-01, not the orthodontist appointment.", "my daughter's orthodontist appointment", "self"),
        ("assistant", "By the way, I heard something work-related: Your colleague Sam attended the hospital leadership conference on 22 May 2023..", "the hospital leadership conference I attended", "other"),
        ("user", "My friend Camille also repotted her monstera on 2023-03-12, but hers is still drooping.", "my monstera repotting", "other"),
        ("assistant", "Speaking of canoeing, Camille went on a canoe trip in the Gorges du Tarn on 2023-07-19, and said the water was warmer than she expected.", "I left for my Ardèche canoe trip", "other"),
        ("assistant", "That reminds me of something: You left for your Ardèche canoe trip on 2023-07-12, and I remember you said the river was freezing.", "I left for my Ardèche canoe trip", "self"),
        ("user", "Camille's last day of school before summer was also on 2023-07-07, but she had to stay for a staff meeting.", "my last day of school before summer", "other"),
        ("user", "It's funny how time goes—My younger son Daniel is moving to Brisbane for work.", "my trip to Brisbane to help Daniel settle in", "self"),
        ("assistant", "Also, Your roommate bought a Korg Minilogue on August 25, 2023, after trying yours.", "I buy my first hardware synth", "other"),
        ("user", "Juniper’s first vet appointment was on 2023-04-21 and she’s cleared for easy runs.", "her first vet appointment", "none"),
    ]
    for role, s, ev, want in sents:
        got = subject_class(s, role, question_persons(ev))
        neg = contrast_hit(s, content_tokens(ev))
        d, how = resolve_date(s, s, datetime(2023, 6, 1), content_tokens(ev))
        bad += got != want
        print("OK " if got == want else "BAD", f"subj={got:5} want={want:5} neg={neg!s:5} date={d} {how} | {s[:80]}")
    multi = "One useful reference point: Camille sent off her passport renewal on 2023-03-06, and her new passport arrived on 2023-04-03."
    print("nearest date for 'renewed passport arrive':", resolve_date(multi, multi, datetime(2023, 4, 3), content_tokens("my renewed passport arrive")))
    print("relative:", resolve_date("I took a baking class at a local culinary school yesterday.", "", datetime(2022, 3, 21), content_tokens("baking class")))
    print("compute apart:", compute_answer({"kind": "apart", "unit": "days"}, [date(2023, 6, 10), date(2023, 6, 24)], ref))
    print("compute ago weeks:", compute_answer({"kind": "ago", "unit": "weeks"}, [date(2023, 3, 4)], datetime(2023, 4, 1)))
    print("hint:", build_hint({"kind": "apart", "unit": "days"}, [
        {"event": "my son's school play", "date": date(2023, 6, 10), "how": "explicit", "sess_date": "2023/05/28"},
        {"event": "my daughter's orthodontist appointment", "date": date(2023, 6, 24), "how": "explicit", "sess_date": "2023/06/01"}], 14, True))
    # all test temporal questions parse?
    n = ok = 0
    for line in open("../data/test/questions.jsonl"):
        r = json.loads(line)
        if re.search(r"how many days (apart|after)", r["question"], re.I):
            n += 1
            s = parse_question(r["question"])
            ok += s["kind"] in ("apart", "after") and len(s["events"]) == 2
    print(f"test 'days apart/after' parsed: {ok}/{n}")
    print("FAILURES:", bad)
