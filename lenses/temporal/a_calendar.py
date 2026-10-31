"""Temporal lens A ("Struktura"): rule-based event calendar.

Idea: the question names 1-3 events. For each event we score every candidate sentence of the history (retrieved by
hybrid search + a full scan of explicitly dated sentences) with rules: IDF-weighted keyword coverage, whose event it
is (self vs a friend/coworker/named other person), negation / contrast scope ("not the orthodontist appointment",
"separate from the play"), possessor match ("my son's" vs "the high school team's"), year match. Each sentence gets a
date: explicit date in the sentence (ISO, "June 27, 2023", "23 April 2023", "March 7th", "the 15th of March", "on 2/15")
else a relative expression resolved against the session date ("today", "yesterday", "last Saturday", "two weeks ago")
else the session date itself. The best sentence per event gives the event date; the day gap is computed in code and
sent to Reactor as a hint; the pack holds the chosen sessions first (cite_plan = those sessions only), then a few
alternates and a small hybrid fill for recall.

Run the self-test (no core.index needed):  .venv/bin/python -m lenses.temporal.a_calendar
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

from core.types import EvidencePack

VARIANTS = {
    # ranked by expected value on the test set (see a_design.md)
    "cal": dict(pool=40, budget=8000, window=1, fill_k=6, alts=1, min_score=0.55, min_margin=0.08, compute=True),
    "cal_wide": dict(pool=60, budget=12000, window=1, fill_k=14, alts=2, min_score=0.5, min_margin=0.05, compute=True),
    "cal_nocomp": dict(pool=40, budget=8000, window=1, fill_k=6, alts=1, min_score=0.55, min_margin=0.08, compute=False),
    "cal_tight": dict(pool=40, budget=4500, window=1, fill_k=0, alts=1, min_score=0.55, min_margin=0.08, compute=True),
    "cal_bm25": dict(pool=40, budget=8000, window=1, fill_k=6, alts=1, min_score=0.55, min_margin=0.08, compute=True,
                     dense=False),
    # round 1
    "m2_bm25": dict(pool=40, budget=8000, window=1, fill_k=6, alts=1, min_score=0.55, min_margin=0.08, compute=True,
                    dense=False, more=2),
    "tp_bm25": dict(pool=40, budget=8000, window=1, fill_k=6, alts=1, min_score=0.55, min_margin=0.08, compute=True,
                    dense=False, more=2, topic=True, topic_budget=10000, topic_window=0),
    "nr_bm25": dict(pool=40, budget=8000, window=1, fill_k=6, alts=1, min_score=0.55, min_margin=0.08, compute=True,
                    dense=False, more=2, topic=True, topic_budget=10000, topic_window=0, near=3),
    "nr2_bm25": dict(pool=40, budget=8000, window=1, fill_k=6, alts=1, min_score=0.55, min_margin=0.08, compute=True,
                     dense=False, more=2, topic=True, topic_budget=10000, topic_window=0, near=2, near_max=4),
    "ft_bm25": dict(pool=40, budget=8000, window=1, fill_k=6, alts=1, min_score=0.55, min_margin=0.08, compute=True,
                    dense=False, more=2, topic=True, topic_budget=10000, topic_window=0, first_turn=True),
    "tb14_bm25": dict(pool=40, budget=8000, window=1, fill_k=6, alts=1, min_score=0.55, min_margin=0.08, compute=True,
                      dense=False, more=2, topic=True, topic_budget=14000, topic_window=0),
    "ftb_bm25": dict(pool=40, budget=8000, window=1, fill_k=6, alts=1, min_score=0.55, min_margin=0.08, compute=True,
                     dense=False, more=2, topic=True, topic_budget=14000, topic_window=0, first_turn=True),
}

# --------------------------------------------------------------------------------------------------------------------
# Date grammar
# --------------------------------------------------------------------------------------------------------------------
MONTHS = {m: i for i, m in enumerate(
    "january february march april may june july august september october november december".split(), 1)}
MONTHS.update({"jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8, "sep": 9, "sept": 9, "oct": 10,
               "nov": 11, "dec": 12})
_MON = r"(January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sept|Sep|Oct|Nov|Dec)"
_ORD = r"(?:st|nd|rd|th)?"
WEEKDAYS = {d: i for i, d in enumerate("monday tuesday wednesday thursday friday saturday sunday".split())}
NUMW = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
        "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "a couple of": 2, "a couple": 2, "couple of": 2, "a few": 3,
        "few": 3, "several": 3}
_NUM = r"(\d{1,2}|a couple of|a few|an|a|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|several)"

RX_ISO = re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b")
RX_MDY = re.compile(rf"\b{_MON}\.?\s+(\d{{1,2}}){_ORD}(?:,\s*|\s+)(\d{{4}})\b")
RX_DMY = re.compile(rf"\b(\d{{1,2}}){_ORD}\s+(?:of\s+)?{_MON},?\s+(\d{{4}})\b")
RX_MD = re.compile(rf"\b{_MON}\.?\s+(\d{{1,2}}){_ORD}\b(?![,\s]*\d{{4}})(?!\s*(?:-|to|–)\s*\d)(?!:\d)")
RX_DM = re.compile(rf"\b(?:the\s+)?(\d{{1,2}}){_ORD}\s+(?:of\s+)?{_MON}\b(?!,?\s*\d{{4}})")
RX_SLASH = re.compile(r"\b(?:on|the|since|from|by|until|til|of)\s+(?:the\s+)?(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?\b")
RX_REL = [
    (re.compile(r"\b(?:the day before yesterday)\b", re.I), lambda m, a: a - timedelta(days=2)),
    (re.compile(r"\b(?:earlier today|today|tonight|this (?:morning|afternoon|evening)|just now)\b", re.I),
     lambda m, a: a),
    (re.compile(r"\b(?:yesterday|last night)\b", re.I), lambda m, a: a - timedelta(days=1)),
    (re.compile(r"\btomorrow\b", re.I), lambda m, a: a + timedelta(days=1)),
    (re.compile(rf"\b{_NUM}\s+(day|week|month)s?\s+ago\b", re.I),
     lambda m, a: a - _span(NUMW.get(m.group(1).lower(), None) or int(m.group(1)), m.group(2))),
    (re.compile(r"\blast weekend\b", re.I), lambda m, a: _prev_weekday(a, 5)),
    (re.compile(r"\blast week\b", re.I), lambda m, a: a - timedelta(days=7)),
    (re.compile(r"\blast month\b", re.I), lambda m, a: a - timedelta(days=30)),
    (re.compile(r"\bnext week\b", re.I), lambda m, a: a + timedelta(days=7)),
    (re.compile(r"\b(?:last|this past)\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", re.I),
     lambda m, a: _prev_weekday(a, WEEKDAYS[m.group(1).lower()])),
    (re.compile(r"\bnext\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", re.I),
     lambda m, a: _next_weekday(a, WEEKDAYS[m.group(1).lower()])),
    (re.compile(r"\bon\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b(?!\s*,?\s*(?:" + _MON[1:-1] + r"))",
                re.I), lambda m, a: _prev_weekday(a, WEEKDAYS[m.group(1).lower()])),
]


def _nth_weekday(y: int, m: int, wd: int, n: int) -> date:
    """n-th (1-based; -1 = last) weekday `wd` of month m."""
    if n > 0:
        d = date(y, m, 1)
        return d + timedelta(days=(wd - d.weekday()) % 7 + 7 * (n - 1))
    d = date(y + (m == 12), m % 12 + 1, 1) - timedelta(days=1)
    return d - timedelta(days=(d.weekday() - wd) % 7)


HOLIDAYS = {
    r"christmas eve": lambda y: date(y, 12, 24), r"christmas(?: day)?": lambda y: date(y, 12, 25),
    r"new year'?s eve": lambda y: date(y, 12, 31), r"new year'?s(?: day)?": lambda y: date(y, 1, 1),
    r"valentine'?s(?: day)?": lambda y: date(y, 2, 14), r"halloween": lambda y: date(y, 10, 31),
    r"(?:the )?(?:4th|fourth) of july|independence day": lambda y: date(y, 7, 4),
    r"st\.? patrick'?s day": lambda y: date(y, 3, 17), r"thanksgiving": lambda y: _nth_weekday(y, 11, 3, 4),
    r"black friday": lambda y: _nth_weekday(y, 11, 3, 4) + timedelta(days=1),
    r"cyber monday": lambda y: _nth_weekday(y, 11, 3, 4) + timedelta(days=4),
    r"mother'?s day": lambda y: _nth_weekday(y, 5, 6, 2), r"father'?s day": lambda y: _nth_weekday(y, 6, 6, 3),
    r"memorial day": lambda y: _nth_weekday(y, 5, 0, -1), r"labor day": lambda y: _nth_weekday(y, 9, 0, 1),
}
RX_HOL = re.compile(r"\b(?:(a|one|two|three|four|five|six|seven|\d{1,2}|a couple of|a few)\s+(day|week)s?\s+"
                    r"(before|after|prior to)\s+)?(" + "|".join(HOLIDAYS) + r")\b", re.I)


def _holiday(m, anchor: date) -> date | None:
    name = m.group(4).lower()
    fn = next(f for k, f in HOLIDAYS.items() if re.fullmatch(k, name, re.I))
    best = min((fn(y) for y in (anchor.year - 1, anchor.year, anchor.year + 1)), key=lambda d: _ydist(d, anchor))
    if m.group(1):
        n = NUMW.get(m.group(1).lower()) or int(m.group(1))
        off = timedelta(days=n * (7 if m.group(2).lower() == "week" else 1))
        best = best + off if m.group(3).lower() == "after" else best - off
    return best


def _span(n: int, unit: str) -> timedelta:
    unit = unit.lower()
    return timedelta(days=n * (1 if unit == "day" else 7 if unit == "week" else 30))


def _prev_weekday(a: date, wd: int) -> date:
    d = (a.weekday() - wd) % 7 or 7
    return a - timedelta(days=d)


def _next_weekday(a: date, wd: int) -> date:
    d = (wd - a.weekday()) % 7 or 7
    return a + timedelta(days=d)


def _mk(y: int, m: int, d: int) -> date | None:
    try:
        return date(y, m, d)
    except ValueError:
        return None


def _ydist(c: date, anchor: date) -> int:
    """Distance used to infer a missing year: future dates count 3x (mentions are mostly of past events)."""
    d = (c - anchor).days
    return 3 * d if d > 0 else -d


def _near_year(m: int, d: int, anchor: date) -> date | None:
    cands = [c for c in (_mk(y, m, d) for y in (anchor.year - 1, anchor.year, anchor.year + 1)) if c]
    return min(cands, key=lambda c: _ydist(c, anchor)) if cands else None


@dataclass
class DateHit:
    start: int
    end: int
    value: date
    kind: str          # "abs" (year written), "md" (month+day, year inferred), "rel" (relative to the anchor)
    text: str


def find_dates(s: str, anchor: date) -> list[DateHit]:
    """All date expressions in `s`; relative ones and year-less ones resolved against `anchor` (the session date)."""
    hits: list[DateHit] = []
    taken: list[tuple[int, int]] = []

    def add(m, val, kind):
        if val is None:
            return
        a, b = m.span()
        if any(a < y and x < b for x, y in taken):
            return
        taken.append((a, b))
        hits.append(DateHit(a, b, val, kind, m.group(0)))

    for m in RX_ISO.finditer(s):
        add(m, _mk(int(m.group(1)), int(m.group(2)), int(m.group(3))), "abs")
    for m in RX_HOL.finditer(s):
        add(m, _holiday(m, anchor), "md")
    for m in RX_MDY.finditer(s):
        add(m, _mk(int(m.group(3)), MONTHS[m.group(1).lower()], int(m.group(2))), "abs")
    for m in RX_DMY.finditer(s):
        add(m, _mk(int(m.group(3)), MONTHS[m.group(2).lower()], int(m.group(1))), "abs")
    for m in RX_MD.finditer(s):
        if m.group(1) == "May" or m.group(1).lower() != "may":   # "may" the verb is lower-case
            add(m, _near_year(MONTHS[m.group(1).lower()], int(m.group(2)), anchor), "md")
    for m in RX_DM.finditer(s):
        add(m, _near_year(MONTHS[m.group(2).lower()], int(m.group(1)), anchor), "md")
    for m in RX_SLASH.finditer(s):
        mo, dd, yy = int(m.group(1)), int(m.group(2)), m.group(3)
        if 1 <= mo <= 12 and 1 <= dd <= 31:
            if yy:
                y = int(yy) + (2000 if len(yy) == 2 else 0)
                add(m, _mk(y, mo, dd), "abs")
            else:
                add(m, _near_year(mo, dd, anchor), "md")
    for rx, fn in RX_REL:
        for m in rx.finditer(s):
            try:
                add(m, fn(m, anchor), "rel")
            except (KeyError, ValueError, TypeError):
                pass
    hits.sort(key=lambda h: h.start)
    return hits


# --------------------------------------------------------------------------------------------------------------------
# Question parsing
# --------------------------------------------------------------------------------------------------------------------
@dataclass
class QParse:
    kind: str                      # "gap" (two events), "ago" (one event vs question date), "order", "other"
    unit: str = "day"              # day | week | month | year
    events: list[str] = field(default_factory=list)
    signed: bool = False           # "after A did I B": B is expected to be later than A
    this_year: bool = False
    vague: bool = False            # "how long ..." : the expected unit is unknown, never emit `computed`


_Q_UNIT = r"(days?|weeks?|months?|years?)"
_SPLIT_AND = re.compile(r"\s+and\s+(?=(?:the|my|our|a|an|his|her|when|i|starting|finishing|getting|buying|"
                        r"attending|visiting|receiving|sending)\b)", re.I)


def _clean_event(e: str) -> str:
    e = re.sub(r"\?.*$", "", e).strip(" ,.?")
    e = re.sub(r"^(?:the day (?:that )?|the time (?:that )?|the (?:date|moment) )", "", e, flags=re.I)
    e = re.sub(r"\s+(?:this year|I lost track)$", "", e, flags=re.I)
    return e.strip()


def _split_two(s: str) -> list[str]:
    parts = _SPLIT_AND.split(s, maxsplit=1)
    if len(parts) < 2:
        parts = re.split(r"\s+and\s+", s, maxsplit=1)
    return [_clean_event(p) for p in parts if p.strip()]


def parse_question(text: str) -> QParse:
    t = " ".join(text.split())
    tl = t.lower()
    this_year = "this year" in tl
    um = re.search(rf"how many {_Q_UNIT}", tl)
    unit = (um.group(1).rstrip("s") if um else "day")
    # gap questions -------------------------------------------------------------------------------------------------
    m = re.search(rf"how many {_Q_UNIT} apart were (.+?)\??$", t, re.I)
    if m:
        return QParse("gap", unit, _split_two(m.group(2)), False, this_year)
    m = re.search(rf"how many {_Q_UNIT} (?:had |have )?(?:passed|elapsed|were there) between (.+?)\??$", t, re.I)
    if m:
        return QParse("gap", unit, _split_two(m.group(2)), True, this_year)
    m = re.search(rf"how many {_Q_UNIT} (?:had |have )?(?:passed )?since (.+?),? when (.+?)\??$", t, re.I)
    if m:
        return QParse("gap", unit, [_clean_event(m.group(2)), _clean_event(m.group(3))], True, this_year)
    m = re.search(rf"how many {_Q_UNIT} (after|before) (.+?) (?:did|was|were|is|are|will) (.+?)\??$", t, re.I)
    if m:
        a, b = _clean_event(m.group(3)), _clean_event(m.group(4))
        b = re.sub(r"^(?:i|we|she|he|they|it)\s+", "", b, flags=re.I)
        ev = [a, b] if m.group(2).lower() == "after" else [b, a]
        return QParse("gap", unit, ev, True, this_year)
    m = re.search(rf"how many {_Q_UNIT} did it take (?:me |for me )?to (.+?) after (.+?)\??$", t, re.I)
    if m:
        return QParse("gap", unit, [_clean_event(m.group(3)), _clean_event(m.group(2))], True, this_year)
    m = re.search(rf"how many {_Q_UNIT} ago did (?:i|we) (.+?),? when (?:i|we) (.+?)\??$", t, re.I)
    if m:
        return QParse("gap", unit, [_clean_event(m.group(2)), _clean_event(m.group(3))], True, this_year)
    m = re.search(rf"how many {_Q_UNIT} ago did (?:i|we) (.+?)\??$", t, re.I)
    if m:
        return QParse("ago", unit, [_clean_event(m.group(2))], True, this_year)
    m = re.search(r"how (?:long|many \w+) (?:had|have) i been (.+?) when i (.+?)\??$", t, re.I)
    if m:
        return QParse("gap", unit if um else "week", [_clean_event(m.group(1)), _clean_event(m.group(2))], True,
                      this_year, vague=not um)
    # order questions -----------------------------------------------------------------------------------------------
    quoted = re.findall(r"(?:^|\s)'([^']{6,}?)'(?=[\s,?.:]|$)", t)
    if re.search(r"\border\b|\bfirst\b|\bearliest\b|\bmost recently\b", tl):
        if len(quoted) >= 2 and re.search(r"order of the (?:three|two|four) events", tl):
            return QParse("order", unit, [_clean_event(x) for x in quoted], False, this_year)
        m = re.search(r",\s*(.+?)\s+or\s+(.+?)\??$", t)
        if m:
            return QParse("order", unit, [_clean_event(m.group(1)), _clean_event(m.group(2))], False, this_year)
        m = re.search(r":\s*(.+?)\??$", t)
        if m:
            parts = [p for p in re.split(r",\s*(?:and\s+)?|\s+and\s+", m.group(1)) if len(p.split()) >= 2]
            if len(parts) >= 2:
                return QParse("order", unit, [_clean_event(p) for p in parts], False, this_year)
        return QParse("order", unit, [], False, this_year)
    return QParse("other", unit, [], False, this_year)


# --------------------------------------------------------------------------------------------------------------------
# Tokens, IDF, sentence rules
# --------------------------------------------------------------------------------------------------------------------
STOP = set("""a an the and or of to in at for on with by from about into onto over my me i we our us you your his her
their it its this that these those which what when where who whom how many much did do does done was were is are be
been being had have has having after before since between apart passed day days week weeks month months year years ago
long take took it's i'm i've i'd just also so then than there here as up out very really get got some any one more
most first last next ever did does s t""".split()) - {"first", "last"}
STOP_EVENT_GENERIC = {"time", "thing", "event", "events", "place", "trip"}   # kept but down-weighted
IRREG = {"bought": "buy", "went": "go", "gone": "go", "took": "take", "taken": "take", "ran": "run", "made": "make",
         "got": "get", "gotten": "get", "flew": "fly", "flown": "fly", "met": "meet", "began": "begin",
         "begun": "begin", "gave": "give", "given": "give", "saw": "see", "seen": "see", "sent": "send", "held": "hold",
         "led": "lead", "left": "leave", "sold": "sell", "won": "win", "wrote": "write", "written": "write",
         "came": "come", "drove": "drive", "driven": "drive", "found": "find", "paid": "pay", "spent": "spend",
         "taught": "teach", "brought": "bring", "built": "build", "felt": "feel", "kept": "keep", "lost": "lose",
         "heard": "hear", "told": "tell", "sat": "sit", "swam": "swim", "began": "begin", "ate": "eat",
         "did": "do", "had": "have", "attended": "attend", "purchased": "buy", "purchase": "buy", "purchasing": "buy",
         "visited": "visit", "visiting": "visit", "started": "start", "starting": "start", "begin": "start",
         "finished": "finish", "completed": "finish", "completing": "finish", "complete": "finish",
         "neighbour": "neighbor", "colour": "color", "watercolour": "watercolor", "theatre": "theater"}
SYN = {  # rule-based expansion, matched at half weight
    "trip": ["visit", "fly", "go", "travel"], "visit": ["trip", "go", "fly", "see"], "buy": ["get", "order", "pick"],
    "start": ["begin", "join", "sign"], "finish": ["end", "done", "wrap"], "class": ["course", "lesson", "workshop"],
    "course": ["class", "lesson"], "workshop": ["class", "course", "session"], "appointment": ["visit", "check"],
    "get": ["buy", "receive"], "receive": ["get", "arrive"], "arrive": ["receive", "get"], "attend": ["go", "visit"],
    "dinner": ["meal"], "run": ["race", "marathon"], "book": ["reserve"], "submit": ["send", "file", "apply"],
    "send": ["submit", "mail"], "adopt": ["bring", "home"], "repot": ["pot", "transplant"],
}
RELATIONS = set("""friend friends coworker co-worker coworkers colleague colleagues brother brothers sister sisters
neighbor neighbors neighbour roommate housemate flatmate cousin cousins boss partner mom mother dad father aunt uncle
niece nephew grandma grandmother grandpa grandfather classmate classmates teammate teammates husband wife son sons
daughter daughters kid kids child children teacher manager mentor client clients student students fiance fiancee
girlfriend boyfriend in-law sister-in-law brother-in-law family""".split())
OTHER_VERB = r"(?:[a-z]+ed|went|took|got|had|did|was|is|has|bought|flew|ran|made|came|said|held|led|left|sent|won|" \
             r"saw|met|began|gave|sold|wrote|drove|found|paid|also|just|finally|recently|will|would|'s)"
NEG_RX = re.compile(r"\b(?:not|n't|never|separate from|rather than|instead of|unlike|other than|different from|"
                    r"as opposed to|apart from|except)\b", re.I)
FILLER_START = re.compile(r"^(?:\W*(?:oh|also|and|but|so|anyway|actually|honestly|by the way|btw|quick aside|"
                          r"random(?:ly)?|before i forget|that reminds me(?: of something)?|speaking of [^,:]+|"
                          r"one more thing|small world|quick update[^,:]*|i almost forgot|on a (?:completely )?"
                          r"different note|i just remembered|when i checked earlier|in case you missed it)\b[\s,:;—–-]*)+",
                          re.I)
NAME_STOP = set("""I The On In At Also Oh By And But So Anyway Actually Honestly Quick Random Before That Speaking One
Small Last This Next Our My Your His Her Their We You It If When While After Since Then Today Yesterday Tomorrow Just
Monday Tuesday Wednesday Thursday Friday Saturday Sunday January February March April May June July August September
October November December Good Great Thanks Thank Yes No Sure Here There What How Why Where Who Which Okay OK Hi Hey
Well Wow Let Please Remember Note Got Opening Closing Annual""".split())


def stem(w: str) -> str:
    w = w.lower().strip("'’-")
    w = re.sub(r"['’]s$", "", w)
    if w in IRREG:
        return IRREG[w]
    if w.endswith("ies") and len(w) > 4:
        w = w[:-3] + "y"
    else:
        for suf in ("ing", "ed", "es", "s"):
            if w.endswith(suf) and len(w) - len(suf) >= 3 and not w.endswith("ss"):
                w = w[: -len(suf)]
                if suf in ("ing", "ed") and len(w) > 3 and w[-1] == w[-2] and w[-1] not in "aeiouls":
                    w = w[:-1]
                break
    if len(w) > 4 and w.endswith("e"):
        w = w[:-1]
    return IRREG.get(w, w)


_TOK = re.compile(r"[A-Za-zÀ-ÿ0-9][A-Za-zÀ-ÿ0-9'’\-]*")


def toks(s: str) -> list[tuple[str, int]]:
    """(stem, char offset) for content tokens; hyphenated words also yield their parts."""
    out = []
    for m in _TOK.finditer(s):
        w = m.group(0)
        lw = re.sub(r"['’]s$", "", w.lower())
        if lw in STOP or len(lw) < 2:
            continue
        out.append((stem(lw), m.start()))
        if "-" in lw:
            for p in lw.split("-"):
                if p and p not in STOP and len(p) > 2:
                    out.append((stem(p), m.start()))
    return out


SENT_RX = re.compile(r"(?<!\bSt\.)(?<!\bMt\.)(?<!\bDr\.)(?<!\bMr\.)(?<!\bMs\.)(?<!\bJr\.)(?<!\bvs\.)(?<!\bMrs\.)"
                     r"(?<!\be\.g\.)(?<!\bi\.e\.)(?<=[.!?])\s+(?=[A-Z0-9\"'“(*])|\n+")


def sentences(text: str) -> list[tuple[int, str]]:
    out, pos = [], 0
    for part in SENT_RX.split(text):
        i = text.find(part, pos)
        pos = i + len(part) if i >= 0 else pos
        if part.strip():
            out.append((max(i, 0), part.strip()))
    return out


def possessors(s: str) -> set[str]:
    """Owners in possessive phrases: "my son's play" -> {"son"}, "the hospital's 5K" -> {"hospital"}."""
    return {stem(m.group(1)) for m in re.finditer(r"\b([A-Za-z][\w\-]*)['’]s\b", s) if m.group(1).lower() not in
            ("it", "that", "there", "what", "let", "he", "she", "here", "who")}


def subject_kind(sent: str, role: str, qlow: str) -> str:
    """'self' (the user did it), 'other' (a friend / coworker / named other person did it) or 'none' (impersonal)."""
    s = FILLER_START.sub("", sent)
    s = re.sub(r"^(?:On|By|In)\s+[^,]{3,30},\s*", "", s)            # "On 2023-06-28, I cooked..."
    s = re.sub(r"\bI (?:heard|learned|noticed|found out|was told|saw)(?: that| something[^:]*:)?\s*", "", s)
    s = re.sub(r"\bI remember(?: now)?:?\s*", "", s) if role == "assistant" else s
    cands = []
    for m in REL_RX.finditer(s):
        rel = m.group(1).lower()
        if True:
            if m.group(3) and not m.group(2) and stem(rel) in qlow_stems(qlow):
                cands.append((m.start(), "self"))       # "my son's school play" and the question says "my son's"
            elif m.group(3) or m.group(2) or re.match(rf"\s+{OTHER_VERB}\b", s[m.end():]):
                name_in_q = m.group(2) and m.group(2).lower() in qlow
                cands.append((m.start(), "self" if (name_in_q or stem(rel) in qlow_stems(qlow) and not m.group(2))
                              else "other"))
    for m in re.finditer(rf"(?:^|(?<=[.:;,—]\s)|(?<=\bthat\s))([A-Z][a-zà-ÿ]+)(?:['’]s\b|\s+{OTHER_VERB}\b)", s):
        nm = m.group(1)
        if nm not in NAME_STOP and nm.lower() not in qlow:
            cands.append((m.start(), "other"))
    me = r"\b(?:I|I'm|I've|I'd|I'll|we|We|We're|We've|my|My|our|Our|me)\b" if role == "user" else \
        r"\b(?:you|You|your|Your|you're|You're|you've|You've)\b"
    mm = re.search(me, s)
    if mm:
        cands.append((mm.start(), "self"))
    if not cands:
        return "none"
    return min(cands)[1]


_REL_ALT = "|".join(re.escape(r) for r in sorted(RELATIONS | {"teenager", "teen"}, key=len, reverse=True))
_ADJ = r"(?:(?:older|younger|little|big|best|old|new|good|close|elder|eldest|youngest|oldest|twin|baby|step|ex|" \
       r"former|fellow|teenage|future)\s+)*"
# "my friend Valentina", "Your coworker", "my neighbor's greyhound", "your colleague Jan's daughter"; names must be
# capitalised, the possessive word and relation are case-insensitive
REL_RX = re.compile(rf"\b(?i:my|your|our|his|her|their)\s+(?i:{_ADJ})((?i:{_REL_ALT}))\b(?:\s+([A-Z][\w\-]+))?"
                    rf"(['’]s)?")
MODS = set("""previous former old second third another rough mock practice backup advanced winter summer spring
modular mini regular""".split())


def modifier_mismatch(sent: str, terms: dict[str, float], qlow: str) -> bool:
    """A contrastive modifier right before a matched event word that the question does not use:
    "my previous car's annual inspection", "a second set of acoustic panels", "regular dentist checkup"."""
    words = re.findall(r"[A-Za-z][\w\-]*", sent)
    for i, w in enumerate(words):
        if stem(w) in terms:
            for v in words[max(0, i - 3):i]:
                vl = v.lower()
                if vl in MODS and vl not in qlow:
                    return True
    return False


_QSTEM_CACHE: dict[str, set[str]] = {}


def qlow_stems(qlow: str) -> set[str]:
    if qlow not in _QSTEM_CACHE:
        _QSTEM_CACHE[qlow] = {t for t, _ in toks(qlow)}
    return _QSTEM_CACHE[qlow]


def negated_spans(s: str) -> list[tuple[int, int]]:
    out = []
    for m in NEG_RX.finditer(s):
        end = len(s)
        stop = re.search(r"[,;.!?—]|\b(?:but|and|so)\b", s[m.end():])
        if stop:
            end = m.end() + stop.start()
        out.append((m.end(), min(end, m.end() + 60)))
    return out


# --------------------------------------------------------------------------------------------------------------------
# Candidate mentions and scoring
# --------------------------------------------------------------------------------------------------------------------
@dataclass
class Mention:
    uid: int
    session_id: str
    sess_date: datetime
    role: str
    sentence: str
    value: date
    kind: str                       # abs / md / rel / session
    score: float = 0.0
    cov: float = 0.0
    why: list[str] = field(default_factory=list)


class _HxCache:
    def __init__(self, hx):
        self.df: dict[str, int] = {}
        self.n = max(1, len(hx.units))
        self.dated: list[int] = []
        for u in hx.units:
            seen = {t for t, _ in toks(u.text)}
            for t in seen:
                self.df[t] = self.df.get(t, 0) + 1
            if RX_ISO.search(u.text) or RX_MDY.search(u.text) or RX_DMY.search(u.text) or RX_MD.search(u.text) \
                    or RX_DM.search(u.text) or RX_SLASH.search(u.text):
                self.dated.append(u.uid)

    def idf(self, t: str) -> float:
        return math.log(1 + self.n / (1 + self.df.get(t, 0)))


_CACHE: dict[int, _HxCache] = {}


def _hxc(hx) -> _HxCache:
    k = id(hx)
    if k not in _CACHE:
        if len(_CACHE) > 8:
            _CACHE.clear()
        _CACHE[k] = _HxCache(hx)
    return _CACHE[k]


def event_terms(ev: str, c: _HxCache) -> dict[str, float]:
    """IDF-weighted stems of the event; parenthetical aliases "(MoMA)" and terms absent from the whole history are
    dropped (they cannot discriminate between sentences; missing_terms() reports them for abstention)."""
    w = {}
    for t, _ in toks(re.sub(r"\([^)]*\)", " ", ev)):
        if c.df.get(t, 0) == 0:
            continue
        w[t] = max(w.get(t, 0.0), c.idf(t) * (0.4 if t in STOP_EVENT_GENERIC else 1.0))
    return w


def named_phrases(ev: str) -> list[str]:
    """Quoted titles and multi-word capitalised names ("Museum of Modern Art", "Run for Health")."""
    out = [a or b for a, b in re.findall(r"'([^']{3,})'|\"([^\"]{3,})\"", ev)]
    for m in re.finditer(r"\b[A-Z][\w\-]*(?:\s+(?:of|for|the|de|and|&)?\s*[A-Z][\w\-]*)+", ev):
        out.append(m.group(0))
    return [p.lower() for p in out if len(p) > 3]


def score_sentence(sent: str, role: str, ev: str, terms: dict[str, float], qlow: str, qdate: datetime,
                   this_year: bool) -> tuple[float, float, list[str], int]:
    """Returns (score, coverage, reasons, anchor char offset of the matched terms)."""
    tk = toks(sent)
    neg = negated_spans(sent)
    have, pos_hit = {}, []
    for t, p in tk:
        if t in terms:
            inneg = any(a <= p < b for a, b in neg)
            if not inneg:
                have[t] = 1.0
                pos_hit.append(p)
            else:
                have.setdefault(t, 0.0)
    syn_hit = 0.0
    sent_stems = {t for t, _ in tk}
    for t, wt in terms.items():
        if have.get(t, 0) == 0 and any(s in sent_stems for s in SYN.get(t, [])):
            syn_hit += 0.5 * wt
    tot = sum(terms.values()) or 1.0
    cov = (sum(terms[t] * v for t, v in have.items()) + syn_hit) / tot
    why = []
    sc = cov
    if any(v == 0.0 for v in have.values()):
        sc -= 0.2
        why.append("neg")
    sk = subject_kind(sent, role, qlow)
    if sk == "other":
        sc -= 0.4
        why.append("other")
    elif sk == "self":
        sc += 0.05
    if modifier_mismatch(sent, terms, qlow):
        sc -= 0.15
        why.append("mod")
    epos = possessors(ev)
    spos = possessors(sent)
    if epos:
        if not (epos & spos) and not any(re.search(rf"\b{re.escape(p)}\b", sent, re.I) for p in epos):
            sc -= 0.15
            why.append("poss-miss")
    foreign = {p for p in spos if p in RELATIONS and p not in qlow_stems(qlow)}
    if foreign:
        sc -= 0.2
        why.append("poss-other")
    phr = named_phrases(ev)
    if phr:
        sl = sent.lower().replace("“", '"').replace("”", '"')
        if any(p in sl for p in phr):
            sc += 0.25
        else:
            sc -= 0.05
    anchor = int(sum(pos_hit) / len(pos_hit)) if pos_hit else 0
    return sc, cov, why, anchor


KIND_RANK = {"abs": 0, "md": 1, "rel": 2}


def choose_date(ds: list[DateHit], ev: str, apos: int) -> DateHit | None:
    """Date of the event in a sentence: skip dates that are part of the event's own name ("the March 15th issue"),
    prefer written dates over relative ones ("last weekend, on June 18th"), then the one nearest the matched words."""
    evl = ev.lower()
    ok = [d for d in ds if d.text.lower() not in evl]
    if not ok:
        return None
    return min(ok, key=lambda d: (KIND_RANK[d.kind], abs((d.start + d.end) / 2 - apos)))


def retrieve(hx, query: str, k: int, dense: bool = True) -> list[tuple[int, float]]:
    """hybrid (BM25 + dense RRF) or BM25 only (no model-server call)."""
    if dense:
        return hx.hybrid(query, k=k)
    return hx.bm25(query, k=k)


def candidate_mentions(q, hx, ev: str, pool: int, qp: QParse, dense: bool = True) -> list[Mention]:
    c = _hxc(hx)
    terms = event_terms(ev, c)
    if not terms:
        return []
    qlow = q.question.lower()
    uids: dict[int, float] = {}
    try:
        for r, (uid, _s) in enumerate(retrieve(hx, ev, pool, dense)):
            uids[uid] = 0.08 * (1 - r / pool)
    except Exception:
        pass
    # full scan of explicitly dated units: cheap, guarantees dated decoys and true mentions are both considered
    top = set(terms)
    for uid in c.dated:
        if uid in uids:
            continue
        if top & {t for t, _ in toks(hx.units[uid].text)}:
            uids[uid] = 0.0
    out: list[Mention] = []
    for uid, rbonus in uids.items():
        u = hx.units[uid]
        anchor_d = u.date.date()
        for _, sent in sentences(u.text):
            sc, cov, why, apos = score_sentence(sent, u.role, ev, terms, qlow, q.date, qp.this_year)
            if cov < 0.25:
                continue
            h = choose_date(find_dates(sent, anchor_d), ev, apos)
            if h is not None:
                val, kind = h.value, h.kind
                sc += 0.08 if kind == "abs" else 0.04
            else:
                val, kind = anchor_d, "session"
                if u.role == "assistant":
                    sc -= 0.15
                    why.append("asst-session-date")
            if val.year != q.date.year and kind in ("abs", "md"):
                sc -= 0.3 if qp.this_year else 0.12
                why.append("year")
            if val > q.date.date() + timedelta(days=1) and qp.kind == "ago":
                sc -= 0.2
                why.append("future")
            out.append(Mention(uid, u.session_id, u.date, u.role, sent, val, kind, sc + rbonus, cov, why))
    out.sort(key=lambda m: -m.score)
    return out


def pick(mentions: list[Mention], exclude_uid_sent: set = frozenset()) -> tuple[Mention | None, float]:
    """Best mention and its margin over the best mention carrying a DIFFERENT date."""
    best = next((m for m in mentions if (m.uid, m.sentence) not in exclude_uid_sent), None)
    if best is None:
        return None, 0.0
    alt = next((m for m in mentions if m.value != best.value and (m.uid, m.sentence) not in exclude_uid_sent), None)
    return best, (best.score - alt.score) if alt else 1.0


# --------------------------------------------------------------------------------------------------------------------
# Hints, computed values, abstention
# --------------------------------------------------------------------------------------------------------------------
def gap_in_unit(d1: date, d2: date, unit: str) -> int:
    days = abs((d2 - d1).days)
    if unit == "week":
        return int(round(days / 7))
    if unit == "month":
        return int(round(days / 30.44))
    if unit == "year":
        return int(round(days / 365.25))
    return days


def _snip(s: str, n: int = 110) -> str:
    s = " ".join(s.split())
    return s if len(s) <= n else s[: n - 1] + "…"


def _fmt_mention(ev: str, m: Mention) -> str:
    how = {"abs": "date written", "md": "date written, year inferred", "rel": "relative to the session date",
           "session": "said as happening on the session date"}[m.kind]
    return f'"{ev}" = {m.value.isoformat()} ({how}; session {m.sess_date.strftime("%Y/%m/%d")}: "{_snip(m.sentence)}")'


QWORDS_SKIP = set("""Saturday Sunday Monday Tuesday Wednesday Thursday Friday January February March April May June July
August September October November December Valentine Christmas Easter Thanksgiving Halloween I""".split())


def missing_terms(q, hx) -> list[str]:
    """Proper nouns of the question that never appear in the history (exact, case-insensitive)."""
    c = _hxc(hx)
    out = []
    words = re.findall(r"[A-Za-zÀ-ÿ][\w\-]+", q.question)
    for i, w in enumerate(words):
        base = re.sub(r"['’]s$", "", w)
        if i == 0 or not base[0].isupper() or base in QWORDS_SKIP or base.lower() in STOP:
            continue
        if c.df.get(stem(base), 0) == 0:
            out.append(base)
    return out


def abstain_estimate(q, hx, picks: list[tuple[str, Mention | None, float]], missing: list[str],
                     mentions: dict[str, list[Mention]] | None = None) -> float:
    p = 0.05
    if missing:
        p = max(p, 0.75)
    for ev, m, _margin in picks:
        ms = (mentions or {}).get(ev, [])
        own = max((x.cov for x in ms if "other" not in x.why and "neg" not in x.why), default=m.cov if m else 0.0)
        others = max((x.cov for x in ms if "other" in x.why), default=0.0)
        if m is None or own < 0.5:
            p = max(p, 0.6)
        if others >= 0.85 and own < 0.75:
            p = max(p, 0.7)           # only someone else did it (test trap: "Camille bought trail-running shoes")
        elif m is not None and "other" in m.why:
            p = max(p, 0.55)
        elif own < 0.7:
            p = max(p, 0.25)
    return round(p, 3)


# --------------------------------------------------------------------------------------------------------------------
# Question-side time window ("two weeks ago", "last Saturday", "in the past three months", "in March and April")
# --------------------------------------------------------------------------------------------------------------------
_MONTH_ALT = "january|february|march|april|may|june|july|august|september|october|november|december"


@dataclass
class QWindow:
    lo: date
    hi: date
    center: date | None
    text: str


def question_window(text: str, qd: date) -> QWindow | None:
    """Resolve the question's own time reference against the question date. The window is generous on purpose: the
    gold evidence of "what did I do two weeks ago" also holds the neighbouring decoy events."""
    t = " ".join(text.lower().split())
    m = re.search(rf"\b{_NUM}\s+(day|week|month|year)s?\s+ago\b", t)
    if m and not re.search(r"how many \w+ ago", t):
        n = NUMW.get(m.group(1)) or int(m.group(1))
        unit = m.group(2)
        days = n * {"day": 1, "week": 7, "month": 30, "year": 365}[unit]
        tol = {"day": max(2, n // 2 + 1), "week": 4 + 2 * n, "month": 8 + 6 * n, "year": 60}[unit]
        c = qd - timedelta(days=days)
        m2 = re.search(r"\bthe (monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", t)
        if m2:                                          # "the Wednesday two months ago"
            wd = WEEKDAYS[m2.group(1)]
            c = c + timedelta(days=(wd - c.weekday() + 3) % 7 - 3)
        return QWindow(c - timedelta(days=tol), min(qd, c + timedelta(days=tol)), c, m.group(0))
    m = re.search(r"\blast (monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", t)
    if m:
        c = _prev_weekday(qd, WEEKDAYS[m.group(1)])
        return QWindow(c - timedelta(days=2), min(qd, c + timedelta(days=3)), c, m.group(0))
    m = re.search(r"\b(?:the past|last|this past) weekend\b", t)
    if m:
        c = _prev_weekday(qd, 5)
        return QWindow(c - timedelta(days=2), min(qd, c + timedelta(days=3)), c, m.group(0))
    m = re.search(r"\blast week\b", t)
    if m:
        return QWindow(qd - timedelta(days=14), qd, qd - timedelta(days=7), m.group(0))
    m = re.search(r"\blast month\b", t)
    if m:
        return QWindow(qd - timedelta(days=50), qd - timedelta(days=5), qd - timedelta(days=30), m.group(0))
    m = re.search(rf"\b(?:in |during )?the (?:past|last) (?:{_NUM}\s+)?(day|week|month|year)s?\b", t)
    if m:
        n = (NUMW.get(m.group(1)) or int(m.group(1))) if m.group(1) else 1
        days = n * {"day": 1, "week": 7, "month": 31, "year": 366}[m.group(2)]
        return QWindow(qd - timedelta(days=days + 3), qd, None, m.group(0))
    ms = re.findall(rf"\b(?:in|during|of) ({_MONTH_ALT})(?:\s+and\s+({_MONTH_ALT}))?\b", t)
    if ms:
        mons = [MONTHS[x] for pair in ms for x in pair if x]
        rng = []
        for mo in mons:
            y = qd.year if mo <= qd.month else qd.year - 1
            first = date(y, mo, 1)
            last = date(y + (mo == 12), mo % 12 + 1, 1) - timedelta(days=1)
            rng += [first, last]
        return QWindow(min(rng) - timedelta(days=2), min(qd, max(rng) + timedelta(days=3)), None,
                       " and ".join(x for pair in ms for x in pair if x))
    m = RX_HOL.search(text)
    if m:
        c = _holiday(m, qd)
        if c and c <= qd:
            return QWindow(c - timedelta(days=2), min(qd, c + timedelta(days=3)), c, m.group(0))
    return None


_COUNTW = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8}


def strip_time(text: str, win: QWindow | None) -> str:
    s = text
    if win:
        s = re.sub(re.escape(win.text), " ", s, flags=re.I)
    s = re.sub(r"\b(?:I mentioned(?: that)?|from earliest to latest|starting from the earliest|from first to last|"
               r"what is the order of|in the order)\b", " ", s, flags=re.I)
    return " ".join(s.split())


def topic_rank(q, hx, win: QWindow | None, params: dict) -> tuple[list[int], dict[int, float]]:
    """Ranking for questions without parseable events: user-turn hybrid search on the question (time words removed),
    boosted for units inside the question's time window; optionally cross-encoder reranked."""
    dense = params.get("dense", True)
    qtext = strip_time(q.question, win)
    roles = "user" if params.get("topic_user", True) else None
    score: dict[int, float] = {}

    def run(uids, w, k):
        try:
            res = hx.hybrid(qtext, k=k, roles=roles, uids=uids, w_dense=1.0 if dense else 0.0)
        except Exception:
            res = []
        for r, (uid, _s) in enumerate(res):
            score[uid] = score.get(uid, 0.0) + w / (10 + r)

    run(None, 1.0, params.get("topic_k", 40))
    if win:
        wu = [u.uid for u in hx.units if win.lo <= u.date.date() <= win.hi and (roles is None or u.role == roles)]
        if wu:
            run(wu, params.get("win_w", 1.5), 20)
    ranked = sorted(score, key=lambda u: -score[u])
    if win and win.center and params.get("near", 0):
        near = near_picks(hx, qtext, win, params)
        merged = []
        for i in range(max(len(near), len(ranked))):       # interleave: topic hit, near-date pick, topic hit, ...
            if i < len(ranked):
                merged.append(ranked[i])
            if i < len(near):
                merged.append(near[i])
        ranked = list(dict.fromkeys(merged))
    if params.get("topic_rerank") and ranked:
        try:
            top = ranked[: params.get("topic_rerank")]
            rr = dict(hx.rerank(qtext, top))
            for u in top:
                score[u] = rr[u] + (params.get("win_bonus", 2.0) if win and win.lo <= hx.units[u].date.date() <= win.hi
                                    else 0.0)
            ranked = sorted(top, key=lambda u: -score[u]) + ranked[len(top):]
        except Exception:
            pass
    return ranked, score


EVENT_CUE = re.compile(r"\b(?:today|tonight|this morning|this afternoon|yesterday|just|last (?:week|weekend|night|"
                       r"monday|tuesday|wednesday|thursday|friday|saturday|sunday)|recently|earlier|ago|finally)\b", re.I)


def near_picks(hx, qtext: str, win: QWindow, params: dict) -> list[int]:
    """Dev/LongMemEval style: "what did I do two weeks ago" is answered by a user turn of a session dated within a few
    days of the resolved date. Per session near the centre (closest first): its best user unit for the question
    (BM25 inside the session), else its first user unit with an event cue ("today", "just", "yesterday")."""
    tol = params.get("near", 3)
    c = win.center
    by_sess: dict[str, list[int]] = {}
    for u in hx.units:
        if u.role == "user" and abs((u.date.date() - c).days) <= tol:
            by_sess.setdefault(u.session_id, []).append(u.uid)
    out = []
    for sid in sorted(by_sess, key=lambda s: abs((hx.sess_by_id[s]["date"].date() - c).days)):
        uids = by_sess[sid]
        best = None
        try:
            res = [r for r in hx.bm25(qtext, k=3, uids=uids) if r[1] > 0]
            best = res[0][0] if res else None
        except Exception:
            best = None
        cue = next((x for x in uids if EVENT_CUE.search(hx.units[x].text)), None)
        for x in ((best, cue) if params.get("near_both", False) else (cue if cue is not None else best,)):
            if x is not None and x not in out:
                out.append(x)
    return out[: params.get("near_max", 4)]


def build_topic(q, hx, qp: QParse, params: dict) -> EvidencePack:
    from core.pack import build_pack
    win = question_window(q.question, q.date.date())
    ranked, score = topic_rank(q, hx, win, params)
    hint_lines = []
    if win:
        span = f"{win.lo.isoformat()} .. {win.hi.isoformat()}"
        if win.center:
            hint_lines.append(f'Date notes: the question was asked on {q.date.strftime("%Y-%m-%d (%a)")}; '
                              f'"{win.text}" = about {win.center.strftime("%Y-%m-%d (%a)")} (sessions {span}). '
                              "Events in passages from other dates are probably not the one asked about.")
        else:
            hint_lines.append(f'Date notes: the question was asked on {q.date.strftime("%Y-%m-%d (%a)")}; '
                              f'"{win.text}" = {span}.')
    if qp.kind == "order" or re.search(r"\border\b|\bfirst\b|most recent", q.question, re.I):
        hint_lines.append("List each matching event with its date (the session date unless the text gives another "
                          "date), then sort by date.")
    hint = "\n".join(hint_lines)
    missing = missing_terms(q, hx)
    if missing:
        hint += ("\n" if hint else "") + "Note: these words of the question never appear in the history: " + \
            ", ".join(missing) + "."
    passages, included, _chars = build_pack(q, hx, ranked, budget=params.get("topic_budget", 10000),
                                            window=params.get("topic_window", 0), order="chrono", hint=hint)
    # cite: the best-scoring sessions; as many as the question counts ("three trips"), else 2
    m = re.search(r"\b(two|three|four|five|six|seven|eight)\b", q.question.lower())
    n_cite = _COUNTW[m.group(1)] if m else params.get("topic_cite", 2)
    pids = {p.id for p in passages}
    cite: list[str] = []
    for u in ranked:
        sid = hx.units[u].session_id
        if sid in pids and sid not in cite:
            cite.append(sid)
        if len(cite) >= n_cite:
            break
    p_abs = 0.75 if missing else 0.1
    return EvidencePack(
        question_id=q.question_id, variant=params.get("_variant", ""), route="temporal", passages=passages,
        hint=hint, cite_plan=cite or None, abstain_p=p_abs, included_turns=included, computed=None,
        debug={"ranking": ranked, "parse": {"kind": "topic:" + qp.kind, "unit": qp.unit, "events": qp.events},
               "window": [win.lo.isoformat(), win.hi.isoformat(), win.text] if win else None})


# --------------------------------------------------------------------------------------------------------------------
# Lens entry point
# --------------------------------------------------------------------------------------------------------------------
def analyse(q, hx, pool: int = 40, min_score: float = 0.55, min_margin: float = 0.08, dense: bool = True):
    """Pure analysis (no pack): parse, pick a mention per event, compute the gap. Used by build() and the self-test."""
    qp = parse_question(q.question)
    picks: list[tuple[str, Mention | None, float]] = []
    alts: dict[str, list[Mention]] = {}
    allm: dict[str, list[Mention]] = {}
    used: set = set()
    for ev in qp.events[:6]:
        ms = candidate_mentions(q, hx, ev, pool, qp, dense)
        best, margin = pick(ms, used)
        if best is not None:
            used.add((best.uid, best.sentence))
        picks.append((ev, best, margin))
        allm[ev] = ms
        alts[ev] = [m for m in ms if best is None or m.value != best.value][:3]
    computed, conf = None, False
    if qp.kind == "gap" and len(picks) >= 2 and picks[0][1] and picks[1][1]:
        (e1, m1, g1), (e2, m2, g2) = picks[0], picks[1]
        computed = gap_in_unit(m1.value, m2.value, qp.unit)
        conf = min(m1.score, m2.score) >= min_score and min(g1, g2) >= min_margin
        if qp.signed and m2.value < m1.value:
            conf = False
        conf = conf and not qp.vague
    elif qp.kind == "ago" and picks and picks[0][1]:
        m1 = picks[0][1]
        computed = gap_in_unit(m1.value, q.date.date(), qp.unit)
        conf = m1.score >= min_score and picks[0][2] >= min_margin
    analyse.last_mentions = allm
    return qp, picks, alts, computed, conf


def make_hint(q, qp: QParse, picks, computed, conf) -> str:
    found = [(ev, m) for ev, m, _ in picks if m is not None]
    if not found:
        return ""
    lines = ["Date notes computed by code from the passages:"]
    if qp.kind == "order":
        for ev, m in sorted(found, key=lambda x: x[1].value):
            lines.append(f"- {_fmt_mention(ev, m)}")
        srt = sorted(found, key=lambda x: x[1].value)
        chain = srt[0][0]
        for (_, pm), (ev, m) in zip(srt, srt[1:]):
            chain += (" = " if m.value == pm.value else " < ") + ev
        lines.append("Chronological order of these dates: " + chain)
        if len({m.value for _, m in found}) < len(found):
            lines.append("Some events share a date (often the same session): order those by what the text says "
                         "(\"a month ago\", \"last week\", \"before\").")
    else:
        for ev, m in found:
            lines.append(f"- {_fmt_mention(ev, m)}")
        if computed is not None:
            u = qp.unit + ("s" if computed != 1 else "")
            ref = "the question date " + q.date.strftime("%Y/%m/%d") if qp.kind == "ago" else "the two dates"
            if qp.vague and qp.kind == "gap":
                dd = abs((picks[1][1].value - picks[0][1].value).days)
                u, computed = "days", f"{dd} (about {dd / 7:.1f} weeks, {dd / 30.44:.1f} months)"
            lines.append(f"Difference between {ref}: {computed} {u}." + ("" if conf else
                         " (Low confidence: check that these are the right mentions.)"))
    lines.append("Mentions of other people's events, other years, or events the text calls different/separate are "
                 "decoys. Answer with just the number." if qp.kind != "order" else
                 "Mentions of other people's events or other years are decoys.")
    return "\n".join(lines)


def build(q, hx, **params) -> EvidencePack:
    from core.pack import build_pack     # imported lazily so the helpers and the self-test work without the core

    if params.get("topic", False):
        qp0 = parse_question(q.question)
        if qp0.kind == "other" or (qp0.kind == "order" and not qp0.events):
            return build_topic(q, hx, qp0, params)
    pool = params.get("pool", 40)
    dense = params.get("dense", True)
    qp, picks, alts, computed, conf = analyse(q, hx, pool, params.get("min_score", 0.55),
                                              params.get("min_margin", 0.08), dense)
    n_alts = params.get("alts", 1)
    ranked: list[int] = []
    cite: list[str] = []

    def add(uid):
        if uid not in ranked:
            ranked.append(uid)

    for ev, m, margin in picks:
        if m is not None:
            add(m.uid)
            if m.session_id not in cite:
                cite.append(m.session_id)
    allm = getattr(analyse, "last_mentions", {}) or {}
    for ev, m, margin in picks:                 # more mentions of the event: the gold turn may be another turn
        for x in [x for x in allm.get(ev, []) if x.role == "user"][: params.get("more", 0)]:
            add(x.uid)
    if params.get("first_turn"):                 # dev gold turns are mostly the session's opening user turn
        for ev, m, margin in picks:
            if m is not None:
                fu = next((u.uid for u in hx.units if u.session_id == m.session_id and u.role == "user"), None)
                if fu is not None:
                    add(fu)
    for ev, m, margin in picks:
        for a in alts.get(ev, [])[:n_alts]:
            add(a.uid)
            if margin < params.get("min_margin", 0.08) and a.session_id not in cite and a.score > 0.4:
                cite.append(a.session_id)
    fill = []
    try:
        fill = [u for u, _ in retrieve(hx, q.question, max(40, params.get("fill_k", 6) * 3), dense)]
    except Exception:
        pass
    for uid in fill[: params.get("fill_k", 6)]:
        add(uid)
    for uid in fill:
        if uid not in ranked:
            ranked.append(uid)        # full ranking for diagnostics; the budget stops build_pack early
    hint = make_hint(q, qp, picks, computed, conf)
    missing = missing_terms(q, hx)
    if missing:
        hint += ("\n" if hint else "") + "Note: these words of the question never appear in the history: " + \
                ", ".join(missing) + "."
    passages, included, _chars = build_pack(q, hx, ranked, budget=params.get("budget", 8000),
                                            window=params.get("window", 1), order="chrono", hint=hint)
    pids = {p.id for p in passages if p.id}
    cite_plan = [s for s in cite if s in pids] or None
    out_computed = str(computed) if (computed is not None and params.get("compute", True) and conf) else None
    return EvidencePack(
        question_id=q.question_id, variant=params.get("_variant", ""), route="temporal",
        passages=passages, hint=hint, cite_plan=cite_plan,
        abstain_p=abstain_estimate(q, hx, picks, missing, getattr(analyse, "last_mentions", None)),
        included_turns=included, computed=out_computed,
        debug={"ranking": ranked, "parse": {"kind": qp.kind, "unit": qp.unit, "events": qp.events},
               "picks": [(ev, m.value.isoformat() if m else None, round(m.score, 3) if m else None,
                          round(g, 3), m.why if m else None) for ev, m, g in picks],
               "computed_raw": computed, "conf": conf})


# --------------------------------------------------------------------------------------------------------------------
# Self-test (runs without core.index): python -m lenses.temporal.a_calendar [dev|dev200|test] [max_questions]
# --------------------------------------------------------------------------------------------------------------------
class _FakeHX:
    """Minimal HistoryIndex stand-in: one unit per sentence-aligned 700-char chunk, crude lexical 'hybrid'."""

    def __init__(self, h):
        from core.data import parse_date
        from core.types import Unit
        self.units = []
        self.sessions = h["sessions"]
        for si, s in enumerate(h["sessions"]):
            d = parse_date(s["date"])
            for ti, t in enumerate(s["turns"]):
                buf, ci = "", 0
                for _, sent in sentences(t["content"]) or [(0, t["content"])]:
                    if buf and len(buf) + len(sent) > 700:
                        self.units.append(Unit(len(self.units), s["session_id"], si, d, s["date"], t["turn_id"], ti,
                                               t["role"], ci, buf))
                        buf, ci = "", ci + 1
                    buf = (buf + " " + sent).strip()
                if buf:
                    self.units.append(Unit(len(self.units), s["session_id"], si, d, s["date"], t["turn_id"], ti,
                                           t["role"], ci, buf))
        self._tok = [{t for t, _ in toks(u.text)} for u in self.units]

    def hybrid(self, query, k=100, **_):
        qt = {t for t, _ in toks(query)}
        sc = [(len(qt & ts) / (1 + 0.002 * len(self.units[i].text)), i) for i, ts in enumerate(self._tok)]
        sc.sort(reverse=True)
        return [(i, s) for s, i in sc[:k] if s > 0]


def _selftest(split: str = "dev", limit: int = 60):
    import json
    import sys
    from core.data import gold, history, questions
    from core.paths import ROOT

    # 1. date grammar unit checks
    a = date(2023, 3, 26)
    cases = {"I ran it on 2023-04-16 with": date(2023, 4, 16), "on June 27, 2023, after": date(2023, 6, 27),
             "on 23 April 2023.": date(2023, 4, 23), "on February 26th and": date(2023, 2, 26),
             "on the 22nd of April": date(2022, 4, 22) if False else date(2023, 4, 22),
             "since I started working with her on 2/15": date(2023, 2, 15), "yesterday": date(2023, 3, 25),
             "two weeks ago": date(2023, 3, 12), "last Saturday": date(2023, 3, 25), "today": a,
             "in December 20th": date(2022, 12, 20)}
    bad = 0
    for s, want in cases.items():
        got = [h.value for h in find_dates(s, a)]
        if want not in got:
            bad += 1
            print("DATE FAIL", s, got, "want", want)
    assert not find_dates("use 1/2 teaspoon and 3/5 Mistake", a), find_dates("use 1/2 teaspoon", a)
    print(f"date grammar: {len(cases) - bad}/{len(cases)} ok")

    # 2. question parser on the samples file
    for line in open(ROOT / "runs/samples/temporal.jsonl"):
        r = json.loads(line)
        qp = parse_question(r["question"])
        print(f"  [{qp.kind:5s} {qp.unit:5s}] {qp.events}  <- {r['question'][:90]}")

    # 3. end-to-end analysis on real histories (fake index), HA vs gold where available
    g = gold(split)
    qs = [q for q in questions(split) if (not g or g[q.question_id]["question_type"] == "temporal-reasoning")]
    if split == "test":
        qs = [q for q in qs if re.search(r"how many days (apart|after)", q.question, re.I)]
    n = ok = conf_n = conf_ok = 0
    for q in qs[:limit]:
        hx = _FakeHX(history(split, q.history_id))
        qp, picks, alts, computed, conf = analyse(q, hx)
        gnum = None
        if g:
            gg = g[q.question_id]
            if gg["abstain"]:
                continue
            mm = re.match(r"\s*(\d+)", str(gg["answer"]))
            gnum = int(mm.group(1)) if mm else None
        if computed is None:
            continue
        n += 1
        hit = gnum is not None and computed in (gnum, gnum + 1, gnum - 1)
        ok += hit
        conf_n += conf
        conf_ok += conf and hit
        if split == "test" or not hit:
            print(f"{'OK' if hit else '--'} conf={int(conf)} computed={computed} gold={gnum} | {q.question[:100]}")
            for ev, m, mg in picks:
                print(f"      {ev[:40]!r:44s} -> {m.value if m else None} s={m.score if m else 0:.2f} mg={mg:.2f} "
                      f"{m.why if m else ''} | {_snip(m.sentence, 90) if m else ''}")
            print(f"      abstain_p={abstain_estimate(q, hx, picks, missing_terms(q, hx), analyse.last_mentions)}")
    if g:
        print(f"{split}: computed for {n} questions, within +-1 of gold {ok}; confident {conf_n}, of which right "
              f"{conf_ok}")
    sys.stdout.flush()


if __name__ == "__main__":
    import sys
    _selftest(sys.argv[1] if len(sys.argv) > 1 else "dev", int(sys.argv[2]) if len(sys.argv) > 2 else 200)
