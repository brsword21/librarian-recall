"""Lens B ("Semantyka") for multi-session questions: counts / sums / differences over facts spread across 2-5 sessions.

ROUND 1 BEST: `deep8_10k` (dev S 86.6 / dev200 S 85.2; see b_report.md). = session-first selection + intra-session CE
rerank of the top-15 sessions' user units + depth (8 units) for the 8 strongest sessions before more breadth + cite
plan by relative CE logit (top - 4, 2..6 sessions) + 10k budget.

Variants (see b_design.md for the reasoning):
  sweep        turn-level multi-query hybrid -> cross-encoder -> session-first diversified selection, sentence-trimmed
               snippets (many mentions per char), adaptive time-window prior, agent/negation priors, precise cite_plan,
               Chain-of-Note counting hint.
  sweep_wide   same, bigger budget / more sessions (recall first).
  sweep_tight  same, small budget (precision / chars).
  sessagg      Emergence-style: unit hits -> rerank -> session score = NDCG of its unit hits -> top sessions, rendered
               by core.pack.build_pack (window=1). Strong on dev-style long sessions.

Pure-python helpers (question analysis, date windows, agent classifier, trimming, hint) are self-tested with
    .venv/bin/python -m lenses.multi.b_sweep            # no models, no index needed
"""
from __future__ import annotations

import math
import re
from collections import defaultdict
from datetime import date, datetime, timedelta

from core.types import EvidencePack, Passage

# ------------------------------------------------------------------------------------------------------------------
VARIANTS = {
    "sweep": dict(mode="sweep", budget=12000, max_sess=22, per_sess=4, n_rerank=90, pool_k=150, asst="recap",
                  trim=320, cite_max=6, cite_delta=3.0, facets=True, window_prior=True, agent_prior=True),
    "sweep_wide": dict(mode="sweep", budget=14500, max_sess=30, per_sess=4, n_rerank=120, pool_k=200, asst="recap",
                       trim=320, cite_max=6, cite_delta=3.0, facets=True, window_prior=True, agent_prior=True),
    "sweep_tight": dict(mode="sweep", budget=8000, max_sess=12, per_sess=2, n_rerank=70, pool_k=120, asst="recap",
                        trim=260, cite_max=5, cite_delta=2.0, facets=True, window_prior=True, agent_prior=True),
    # Round 1: soft session diversity (greedy, k-th unit of a session worth decay**k) + top-k cite plan
    "decay": dict(mode="sweep", budget=12000, max_sess=22, per_sess=4, n_rerank=90, pool_k=150, asst="recap",
                  trim=320, facets=True, window_prior=True, agent_prior=False, fact_prior=False,
                  select="decay", sess_decay=0.6, cite="topk", cite_k=3, cite_d=2.0, cite_max=4),
    "decay_p": dict(mode="sweep", budget=12000, max_sess=22, per_sess=4, n_rerank=90, pool_k=150, asst="recap",
                    trim=320, facets=True, window_prior=True, agent_prior=True, fact_prior=True,
                    select="decay", sess_decay=0.6, cite="topk", cite_k=3, cite_d=2.0, cite_max=4),
    "sf_topk": dict(mode="sweep", budget=12000, max_sess=22, per_sess=4, n_rerank=90, pool_k=150, asst="recap",
                    trim=320, facets=True, window_prior=True, agent_prior=False, fact_prior=False,
                    cite="topk", cite_k=3, cite_d=2.0, cite_max=4),
    "sf_topk_in": dict(mode="sweep", budget=12000, max_sess=22, per_sess=4, n_rerank=90, pool_k=150, asst="recap",
                       trim=320, facets=True, window_prior=True, agent_prior=False, fact_prior=False,
                       cite="topk", cite_k=3, cite_d=2.0, cite_max=4, insess=15),
    "decay_in": dict(mode="sweep", budget=12000, max_sess=22, per_sess=4, n_rerank=90, pool_k=150, asst="recap",
                     trim=320, facets=True, window_prior=True, agent_prior=True, fact_prior=True,
                     select="decay", sess_decay=0.6, cite="topk", cite_k=3, cite_d=2.0, cite_max=4, insess=15),
    "sf_in6": dict(mode="sweep", budget=12000, max_sess=22, per_sess=6, n_rerank=90, pool_k=150, asst="recap",
                   trim=320, facets=True, window_prior=True, agent_prior=False, fact_prior=False,
                   cite="topk", cite_k=3, cite_d=2.0, cite_max=4, insess=15),
    "sf_in6f": dict(mode="sweep", budget=12000, max_sess=22, per_sess=6, n_rerank=90, pool_k=150, asst="recap",
                    trim=320, facets=True, window_prior=True, agent_prior=False, fact_prior=True,
                    cite="topk", cite_k=3, cite_d=2.0, cite_max=4, insess=15),
    "sf_in8": dict(mode="sweep", budget=12000, max_sess=22, per_sess=8, n_rerank=90, pool_k=150, asst="recap",
                   trim=320, facets=True, window_prior=True, agent_prior=False, fact_prior=False,
                   cite="topk", cite_k=3, cite_d=2.0, cite_max=4, insess=15),
    "sf_in8t": dict(mode="sweep", budget=12000, max_sess=22, per_sess=8, n_rerank=90, pool_k=150, asst="recap",
                    trim=240, facets=True, window_prior=True, agent_prior=False, fact_prior=False,
                    cite="topk", cite_k=3, cite_d=2.0, cite_max=4, insess=15),
    "sf_rel3": dict(mode="sweep", budget=12000, max_sess=22, per_sess=8, n_rerank=90, pool_k=150, asst="recap",
                    trim=240, facets=True, window_prior=True, agent_prior=False, fact_prior=False,
                    cite="rel", cite_d=3.0, cite_max=5, insess=15),
    "sf_rel4": dict(mode="sweep", budget=12000, max_sess=22, per_sess=8, n_rerank=90, pool_k=150, asst="recap",
                    trim=240, facets=True, window_prior=True, agent_prior=False, fact_prior=False,
                    cite="rel", cite_d=4.0, cite_max=6, insess=15),
    "deep8": dict(mode="sweep", budget=12000, max_sess=22, per_sess=8, n_rerank=90, pool_k=150, asst="recap",
                  trim=240, facets=True, window_prior=True, agent_prior=False, fact_prior=False,
                  cite="rel", cite_d=4.0, cite_max=6, insess=15, deep_top=8),
    "deep8_16": dict(mode="sweep", budget=12000, max_sess=16, per_sess=8, n_rerank=90, pool_k=150, asst="recap",
                     trim=240, facets=True, window_prior=True, agent_prior=False, fact_prior=False,
                     cite="rel", cite_d=4.0, cite_max=6, insess=15, deep_top=8),
    "deep8_10k": dict(mode="sweep", budget=10000, max_sess=22, per_sess=8, n_rerank=90, pool_k=150, asst="recap",
                      trim=240, facets=True, window_prior=True, agent_prior=False, fact_prior=False,
                      cite="rel", cite_d=4.0, cite_max=6, insess=15, deep_top=8),
    # ablations for Round 1 (same as sweep with one ingredient removed)
    "sweep_noprior": dict(mode="sweep", budget=12000, max_sess=22, per_sess=4, n_rerank=90, pool_k=150, asst="recap",
                          trim=320, cite_max=6, cite_delta=3.0, facets=True, window_prior=False, agent_prior=False,
                          fact_prior=False),
    "sweep_user": dict(mode="sweep", budget=12000, max_sess=22, per_sess=4, n_rerank=90, pool_k=150, asst="none",
                       trim=320, cite_max=6, cite_delta=3.0, facets=True, window_prior=True, agent_prior=True),
    "sessagg": dict(mode="sessagg", budget=12000, max_sess=10, per_sess=2, n_rerank=90, pool_k=150, asst="recap",
                    unit_window=1, cite_max=5, cite_frac=0.35, facets=True, window_prior=True, agent_prior=True),
}

MONTHS = {m: i + 1 for i, m in enumerate(["january", "february", "march", "april", "may", "june", "july", "august",
                                          "september", "october", "november", "december"])}
MONTHS.update({m[:3]: v for m, v in list(MONTHS.items())})
MONTHS["sept"] = 9
_MON = r"(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sept|sep|oct|nov|dec)"
_NUMW = {"a": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "few": 3, "couple of": 2, "several": 4}
SEASONS = {"spring": (3, 5), "summer": (6, 8), "fall": (9, 11), "autumn": (9, 11), "winter": (12, 2)}

# ------------------------------------------------------------------------------------------------------------------
# Question analysis (pure python)

def _month_start(y: int, m: int) -> date:
    return date(y, m, 1)


def _month_end(y: int, m: int) -> date:
    return (date(y + (m == 12), m % 12 + 1, 1) - timedelta(days=1))


def _last_occurrence(month: int, ref: date, include_current: bool = True) -> int:
    """Year of the most recent `month` not after ref."""
    if month < ref.month or (include_current and month == ref.month):
        return ref.year
    return ref.year - 1


def parse_window(text: str, qdate: datetime) -> dict | None:
    """Question time-window phrase -> {start, end (dates, inclusive), label}. Rule-based (no LLM; RESEARCH: weak models
    inventing ranges hurt). Returns None when the question has no window. 'these days/now/currently' -> latest-value
    semantics, handled by analyze() not here."""
    t = text.lower()
    ref = qdate.date()
    m = re.search(r"\bsince (?:the (?:start|beginning) of )?" + _MON + r"\b", t)
    if m:
        mo = MONTHS[m.group(1)]
        y = _last_occurrence(mo, ref)
        return dict(start=_month_start(y, mo), end=ref, label=m.group(0))
    m = re.search(r"\bsince the (?:start|beginning) of (?:the|this) year\b|\bthis year\b|\bso far this year\b", t)
    if m:
        return dict(start=date(ref.year, 1, 1), end=ref, label=m.group(0))
    m = re.search(r"\b(?:in|during|throughout|for) (?:the month of )?" + _MON + r"\b(?: (\d{4}))?", t)
    if m:
        mo = MONTHS[m.group(1)]
        y = int(m.group(2)) if m.group(2) else _last_occurrence(mo, ref)
        return dict(start=_month_start(y, mo), end=min(_month_end(y, mo), ref), label=m.group(0))
    m = re.search(r"\b(this|last|past) (spring|summer|fall|autumn|winter)\b", t)
    if m:
        a, b = SEASONS[m.group(2)]
        y = ref.year if (m.group(1) == "this" and ref.month >= a) or a == 12 else ref.year - 1
        if m.group(1) in ("last", "past") and ref.month > b and a != 12:
            y = ref.year
        start = _month_start(y, a)
        end = _month_end(y + (a == 12), b)
        return dict(start=start, end=min(end, ref), label=m.group(0))
    if re.search(r"\blast month\b", t):
        y, mo = (ref.year, ref.month - 1) if ref.month > 1 else (ref.year - 1, 12)
        return dict(start=_month_start(y, mo), end=_month_end(y, mo), label="last month")
    if re.search(r"\bthis month\b", t):
        return dict(start=_month_start(ref.year, ref.month), end=ref, label="this month")
    if re.search(r"\blast week\b", t):
        mon = ref - timedelta(days=ref.weekday() + 7)
        return dict(start=mon, end=mon + timedelta(days=6), label="last week")
    if re.search(r"\bthis week\b", t):
        return dict(start=ref - timedelta(days=ref.weekday()), end=ref, label="this week")
    m = re.search(r"\b(?:past|last) (a|one|two|three|four|five|six|few|couple of|several|\d+) (day|week|month)s?\b", t)
    if m:
        n = int(m.group(1)) if m.group(1).isdigit() else _NUMW[m.group(1)]
        days = {"day": 1, "week": 7, "month": 30}[m.group(2)] * n
        return dict(start=ref - timedelta(days=days), end=ref, label=m.group(0))
    m = re.search(r"\b(?:past|last) (week|month|year)\b", t)
    if m:
        days = {"week": 7, "month": 31, "year": 365}[m.group(1)]
        return dict(start=ref - timedelta(days=days), end=ref, label=m.group(0))
    if re.search(r"\blast year\b", t):
        return dict(start=date(ref.year - 1, 1, 1), end=date(ref.year - 1, 12, 31), label="last year")
    return None


_SCAFFOLD = re.compile(
    r"^(how (many|much)( more| less)?|what (is|was|are|were)( the)?|what's( the)?|which|when did|how long|how old|"
    r"in total,?|at which|for my)\b", re.I)
_FILLER = re.compile(
    r"\b(do|did|does|have|has|had|am|is|are|was|were|will) (i|we)\b|\b(in total|total|combined|altogether|so far|"
    r"right now|currently|these days|nowadays|now|each (morning|evening|day|week|night)|per (week|day|month)|"
    r"i lost track|that i (told|mentioned to) you( about)?|i told you about|different|number of|amount of)\b", re.I)
_LATEST = re.compile(r"\b(now|currently|these days|nowadays|right now|current|at the moment|anymore|still)\b", re.I)
_OTHERS_Q = re.compile(r"\b(friends?|family|relatives|people|members?|colleagues|coworkers|team|kids|children|"
                       r"cousins?|neighbou?rs?|guests|students|babies|animals|pets)\b", re.I)
STOP = set("""a an the i my me we our you your of in on at to for from by with and or is are was were be been do did
does have has had how many much what which when who where why that this these those it its as about since than total
times time different all any some there their they them he she his her am not no can could would should will so if but
just also each every now currently day days week weeks month months year years last past few recently right ago told
mention mentioned lost track""".split())


def content_terms(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9][a-z0-9'\-]*", text.lower()) if w not in STOP and len(w) > 2]


def _strip_window(text: str, win: dict | None) -> str:
    if not win:
        return text
    return re.sub(r"\s*\b(?:(?:in|over|during|for|within)\s+)?(?:the\s+)?" + re.escape(win["label"]) + r"\b", " ",
                  text, flags=re.I)


_VERBISH = set("""bought buy buying worked work working led lead leading assemble assembled sell sold fix fixed
learned learn cook cooked tried try trying out used use using made make visited visit attended attend done
did do am currently need pick return returned got get""".split())


def facets_of(text: str) -> list[str]:
    """'total cost of the car cover and detailing spray I purchased' -> ['car cover', 'detailing spray I purchased'].
    Coordinated noun phrases are separate facts in separate sessions; each gets its own query (coverage guarantee)."""
    t = re.sub(r"\?$", "", text.strip())
    m = re.search(r"\b(?:apart (?:were|was)|between|of|on|for|from|to|in|with|about|did i|have i)\s+"
                  r"(.+?\b(?:and|or|vs\.?|versus)\b.+)$",
                  t, re.I)
    if not m:
        return []
    span = m.group(1)
    span = re.sub(r"\b(combined|together|in total|altogether|respectively)\b", " ", span, flags=re.I)
    parts = re.split(r",\s*(?:and\s+|or\s+)?|\s+(?:and|or|vs\.?|versus|compared to)\s+", span)
    out = []
    for p in parts:
        p = re.sub(r"^(the|my|a|an|our)\s+", "", p.strip(), flags=re.I)
        p = re.sub(r"\s+(?:in|on|at|for|of|the)$", "", p.strip(), flags=re.I)
        nouns = [w for w in content_terms(p) if w not in _VERBISH]
        if nouns and len(p) >= 3:                       # coordinated verbs ("worked on or bought") are not facets
            out.append(p)
    # "flea and tick collar" style splits produce stubs; keep only if >=2 real facets
    return out if len(out) >= 2 else []


def statements_of(text: str, win: dict | None) -> list[str]:
    """LLM-free reformulation: question -> declarative statements in the user's voice (dense/rerank match the way
    facts are phrased in chat: 'I bought two orchids', 'I spent $40 on bike lights')."""
    t = _strip_window(text, win).strip().rstrip("?")
    t = re.sub(r"\b(in total|total|combined|altogether|so far|i lost track\.?)\b", " ", t, flags=re.I)
    t = re.sub(r"\s+", " ", t).strip()
    out = []
    m = re.match(r"how (many|much)\s+(?P<obj>.*?)\s*\b(?:do|did|have|has|had|am|was|were|will|can)\s+"
                 r"(?P<subj>i|we)\s+(?P<rest>.+)$", t, re.I)
    if m:
        obj, rest = m.group("obj").strip(), m.group("rest").strip()
        subj = "I" if m.group("subj").lower() == "i" else "We"
        rest = re.sub(r"^(spend|spent)\b", "spent", rest)
        if obj.lower() in ("", "money", "time"):
            out.append(f"{subj} {rest}")
        elif obj.lower().startswith("times"):
            out.append(f"{subj} {rest}")
        else:
            out.append(f"{subj} {rest} {obj}".strip())
            out.append(f"{obj} {rest}".strip())
    m = re.match(r"how (many|much)\s+(?P<obj>.+?)\s+(?:is|are|was|were)\s+(?P<rest>.+)$", t, re.I)
    if m and not out:
        out.append(f"{m.group('rest')} {m.group('obj')}")
    m = re.match(r"what (?:is|was) the (?:total )?(?P<what>[\w\s-]+?) of (?P<rest>.+)$", t, re.I)
    if m:
        out.append(f"I {m.group('rest')} {m.group('what')}")
    return [re.sub(r"\s+", " ", s).strip() for s in out if len(s) > 6]


def analyze(text: str, qdate: datetime) -> dict:
    win = parse_window(text, qdate)
    core = _FILLER.sub(" ", _SCAFFOLD.sub(" ", _strip_window(text, win)))
    core = re.sub(r"[?]", " ", core)
    core = re.sub(r"\s+", " ", core).strip()
    tl = text.lower()
    kind = ("dates" if re.search(r"\b(days?|weeks?|months?) (apart|after|before|between|passed)\b", tl) else
            "diff" if re.search(r"\b(more|less|difference|increase|decrease|left|remaining|compared)\b", tl) else
            "avg" if re.search(r"\b(average|mean)\b", tl) else
            "sum" if re.search(r"\bhow much\b|\btotal (amount|cost|number|distance|time|weight)|\bspent?\b", tl) else
            "count" if re.search(r"\bhow many\b", tl) else "other")
    return dict(window=win, core=core, facets=facets_of(_strip_window(text, win)), statements=statements_of(text, win),
                latest=(bool(_LATEST.search(text)) or bool(re.search(r"\b(?:do|am|is|are) (?:i|we|my)\b|\beach (?:week|day|morning|"
                r"evening|night)\b|\bper (?:week|day)\b", tl) and not win)) and kind != "dates",
                others_ok=bool(_OTHERS_Q.search(text) or re.search(r"\b(?:" + _KIN + r")s?\b", tl)) or kind == "dates",
                kind=kind, rate=bool(re.search(r"\beach (?:week|day|morning|evening|night|month)\b|\bper (?:week|day|"
                                               r"month)\b|\b(?:these days|nowadays|a week|a day)\b", tl)),
                terms=content_terms(core))


# ------------------------------------------------------------------------------------------------------------------
# Unit-level classifiers (pure python)

_KIN = (r"friend|brother|sister|neighbou?r|daughter|son|cousin|co-?worker|colleague|mom|mum|mother|dad|father|aunt|"
        r"uncle|wife|husband|partner|boss|roommate|niece|nephew|grandson|granddaughter|buddy|classmate|teammate|"
        r"coach|instructor|teacher|boyfriend|girlfriend|fianc[eé]e?")
_REPORT = re.compile(r"\b(told me|said|mentioned|says|tells me|explained|bragged|shared)\s+(that\s+)?(he|she|they)\b"
                     r"|\b(he|she|they)\s+(has|have|had|bought|ate|read|added|got|made|opened|took|spent|ran|swam|"
                     r"harvested|planted|adopted|visited|finished|owns?)\b", re.I)
_THIRD_SUBJ = re.compile(r"(?:^|[.!?;]\s+|,\s*|\bby the way,?\s*|\balso,?\s*|\band\s+)(?:my|our|your)\s+(?:[\w'-]+\s+){0,3}?"
                         r"(?:" + _KIN + r")s?\b(?:\s+[A-Z][a-z]+)?,?\s+(?:just\s+|also\s+|recently\s+)?"
                         r"(?:has|have|had|bought|ate|read|added|got|made|opened|took|spent|ran|swam|harvested|planted|"
                         r"adopted|visited|finished|owns?|is|was)\b", re.I)
_FIRST = re.compile(r"\b(I|I've|I'd|I'm|we|we've|my)\b")
_NEG = re.compile(r"\b(decided not to|didn't|did not|never|skipped|instead of|but (?:decided|didn't)|tried on|"
                  r"thinking (?:of|about)|plan(?:ning)? to|want(?:ed)? to|might|considering|would like to|haven't|"
                  r"hasn't|not sure (?:if|whether)|talking myself out of|need to|hoping to)\b", re.I)
_RECAP = re.compile(r"\b(you mentioned|you said|you told me|as you (?:noted|said|mentioned)|earlier you|"
                    r"you(?:'ve| have) (?:told|mentioned|said))\b", re.I)


_FACT = re.compile(
    r"\b(by the way|btw|on another note|i (?:just|recently|finally|already)|last (?:week|weekend|month|year|\w+day)|"
    r"\w+ (?:days|weeks|months) ago|on \d{1,2}/\d{1,2}|\d+(?:st|nd|rd|th)\b|"
    r"[$\u20ac\u00a3]\s?\d|\d[\d,.]*\s?(?:hours?|days?|weeks?|miles?|km|kilos?|kg|pounds?|laps?|times|pesos|dollars|"
    r"euros|minutes?|pages?|episodes?|books?|followers?)|"
    r"i(?:'ve)? (?:bought|got|visited|took|went|made|spent|finished|attended|picked up|ordered|added|planted|harvested|"
    r"read|watched|completed|tried|learned|started|sold|fixed|assembled|received|paid|earned|saved|deposited|used|"
    r"baked|cooked|swam|ran|walked|adopted|viewed|led|joined|volunteered))\b", re.I)


def fact_prior(text: str, role: str) -> float:
    """Units that state a dated / quantified first-person event are where counts live (dev: the 'By the way, I ...'
    sentence of user turn 0; test: one short sentence mid-session). Later topical turns of the same session
    otherwise crowd them out of the per-session cap."""
    return 1.3 if role == "user" and _FACT.search(text) else 1.0


def split_sents(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[\"'(A-Z0-9])", text) if s.strip()]


def _stem5(w: str) -> str:
    return w[:5] if len(w) > 5 else w


def focus_sentences(text: str, terms: list[str]) -> list[int]:
    """Indices of sentences sharing >=1 question term (prefix-5 match), best first."""
    ss = split_sents(text)
    keys = {_stem5(w) for w in terms}
    scored = []
    for i, s in enumerate(ss):
        toks = {_stem5(w) for w in re.findall(r"[a-z0-9][a-z0-9'\-]*", s.lower())}
        hit = len(keys & toks)
        if hit:
            scored.append((-hit, i))
    return [i for _, i in sorted(scored)]


def agent_label(text: str, terms: list[str]) -> str:
    """'other' (another person's fact, reported), 'neg' (planned / considered / declined), 'self' or 'none'.
    Looks only at the sentences carrying question terms, so a unit mixing my fact and a neighbour's is 'self'."""
    ss = split_sents(text)
    idx = focus_sentences(text, terms)[:2] or list(range(min(2, len(ss))))
    labs = []
    for i in idx:
        s = ss[i]
        if _REPORT.search(s) or _THIRD_SUBJ.search(s):
            # "my brother has three grandchildren" -> other, unless I also do the thing in the same sentence
            mine = re.search(r"\bI\s+(?:\w+\s+){0,2}?(bought|got|ate|read|added|made|spent|planted|harvested|picked up|"
                             r"finished|ordered|deposited|used|swam|walked|took|visited|attended)\b", s)
            labs.append("self" if mine else "other")
        elif _NEG.search(s):
            labs.append("neg")
        elif _FIRST.search(s):
            labs.append("self")
        else:
            labs.append("none")
    for lab in ("self", "none", "neg", "other"):
        if lab in labs:
            return lab
    return "none"


def trim_text(text: str, terms: list[str], max_chars: int) -> tuple[str, bool]:
    """Keep the best term-bearing sentences (+1 sentence of context before the best one) within max_chars."""
    if len(text) <= max_chars:
        return text, False
    ss = split_sents(text)
    idx = focus_sentences(text, terms)
    if not idx:
        return text[:max_chars].rsplit(" ", 1)[0] + " …", True
    keep = {idx[0]}
    if idx[0] > 0:
        keep.add(idx[0] - 1)
    for i in idx[1:]:
        if sum(len(ss[j]) + 1 for j in keep | {i}) <= max_chars:
            keep.add(i)
    while sum(len(ss[j]) + 1 for j in keep) > max_chars and len(keep) > 1:
        keep.discard(max(keep, key=lambda j: (j != idx[0], abs(j - idx[0]))))
    out, prev = [], None
    for j in sorted(keep):
        if prev is not None and j != prev + 1:
            out.append("…")
        out.append(ss[j])
        prev = j
    s = " ".join(out)
    if min(keep) > 0:
        s = "… " + s
    if max(keep) < len(ss) - 1:
        s += " …"
    return s, True


def window_prior(d: date, win: dict | None, active: bool) -> float:
    if not (win and active):
        return 1.0
    if win["start"] <= d <= win["end"]:
        return 1.0
    if d > win["end"]:
        return 0.8 if (d - win["end"]).days <= 10 else 0.5
    return 0.3 if (win["start"] - d).days > 3 else 0.9


# ------------------------------------------------------------------------------------------------------------------
# Hint (appended to the question for Reactor). Short, factual, Chain-of-Note style.

def build_hint(a: dict, qdate: datetime, abstain_p: float = 0.0) -> str:
    lines = []
    if a.get("rate") and a["kind"] in ("count", "other") and not a["window"]:
        lines.append("This asks for a current routine value (per day/week): list each statement of it with its date "
                     "and [E#]; the most recent statement wins, earlier values are outdated.")
    elif a["kind"] == "dates":
        lines.append("Find the date of each event: use a date written in the excerpt, otherwise resolve 'today', "
                     "'yesterday', 'last Friday' against that excerpt's date. Then compute the difference in days.")
    elif a["kind"] in ("count", "sum", "diff", "avg"):
        lines.append("These facts are spread over several sessions. First list every matching instance as "
                     "'date | [E#] | item | number', then compute the answer from that list.")
        lines.append("Count a thing once even if it is mentioned again in a later session (same item, same event).")
    else:
        lines.append("Combine the relevant facts from all sessions; note each with its [E#] before answering.")
    if not a["others_ok"] and a["kind"] != "dates":
        lines.append("Only count what I (the user) did or have; ignore what friends, family or others did, "
                     "and ignore things I only planned, considered, tried or decided against.")
    w = a["window"]
    if w:
        lines.append(f"Time window '{w['label']}': {w['start'].isoformat()} to {w['end'].isoformat()} "
                     f"(today is {qdate.date().isoformat()}). Resolve 'yesterday'/'last week' against each excerpt's "
                     f"date; drop instances outside the window.")
    if a["latest"] and not (a.get("rate") and a["kind"] in ("count", "other") and not a["window"]):
        lines.append("If the number changed over time, use the most recent statement.")
    if abstain_p >= 0.6:
        lines.append("If no excerpt states a matching instance for me, answer INSUFFICIENT_EVIDENCE.")
    return " ".join(lines)


# ------------------------------------------------------------------------------------------------------------------
# Abstention

_HIST_TEXT: dict[int, str] = {}


def _hist_text(hx) -> str:
    k = id(hx)
    if k not in _HIST_TEXT:
        if len(_HIST_TEXT) > 8:
            _HIST_TEXT.clear()
        _HIST_TEXT[k] = " ".join(u.text.lower() for u in hx.units)
    return _HIST_TEXT[k]


_GENERIC = set("""spend spent buy bought cost money hours minutes times many much total number amount different kind
kinds type types items things trip trips people""".split())


def missing_terms(hx, terms: list[str]) -> list[str]:
    text = _hist_text(hx)
    out = []
    for w in terms:
        if w in _GENERIC or w.isdigit():
            continue
        if not re.search(r"\b" + re.escape(_stem5(w) if len(w) > 6 else w.rstrip("s")), text):
            out.append(w)
    return out


def abstain_prob(top_rr: float, miss_frac: float, all_other: bool) -> float:
    p_rr = 1.0 / (1.0 + math.exp((top_rr + 7.0) / 1.5))
    p = 0.03 + 0.55 * miss_frac + 0.30 * p_rr + (0.35 if all_other else 0.0)
    return max(0.0, min(0.97, p))


# ------------------------------------------------------------------------------------------------------------------
# Retrieval

def _candidates(q, hx, a: dict, P: dict):
    """Multi-query RRF over hx.hybrid; returns (fused score dict, facet->top uids)."""
    queries = [(q.question, 1.0)]
    if a["core"] and a["core"].lower() != q.question.lower():
        queries.append((a["core"], 0.8))
    queries += [(s, 0.6) for s in a["statements"][:2]]
    fused: dict[int, float] = defaultdict(float)
    for qq, w in queries:
        for r, (uid, _) in enumerate(hx.hybrid(qq, k=P["pool_k"], w_bm25=1.0, w_dense=1.0, rrf_k=60, pool=300)):
            fused[uid] += w / (60 + r + 1)
    facet_top: dict[str, list[int]] = {}
    if P.get("facets"):
        for f in a["facets"][:5]:
            hits = hx.hybrid(f, k=40, roles="user", w_bm25=1.0, w_dense=1.0, rrf_k=60, pool=200)
            facet_top[f] = [u for u, _ in hits[:12]]
            for r, (uid, _) in enumerate(hits):
                fused[uid] += 0.7 / (60 + r + 1)
    return fused, facet_top


def _role_ok(u, mode: str) -> float:
    if u.role == "user":
        return 1.0
    if mode == "all":
        return 0.8
    if mode == "recap" and len(u.text) < 700 and _RECAP.search(u.text):
        return 0.7
    return 0.0


def _score_units(q, hx, a: dict, P: dict):
    fused, facet_top = _candidates(q, hx, a, P)
    order = sorted(fused, key=lambda i: -fused[i])
    order = [i for i in order if _role_ok(hx.units[i], P["asst"]) > 0]
    cand = order[:P["n_rerank"]]
    for f, us in facet_top.items():
        cand += [u for u in us[:6] if u not in cand]
    rr = dict(hx.rerank(q.question, cand)) if cand else {}
    if P.get("insess"):
        # intra-session rerank: every user unit of the top sessions gets a CE score, so a fact phrased with
        # instance words ("hearing aids" for "health devices") that missed the hybrid pool can still be picked
        top_s = []
        for u in sorted(rr, key=lambda x: -rr[x]):
            sid = hx.units[u].session_id
            if sid not in top_s:
                top_s.append(sid)
            if len(top_s) >= P["insess"]:
                break
        top_set = set(top_s)
        extra = [u.uid for u in hx.units if u.session_id in top_set and u.role == "user" and u.uid not in rr]
        if extra:
            rr.update(dict(hx.rerank(q.question, extra)))
            cand = cand + extra
    r_rank = {u: i for i, u in enumerate(sorted(rr, key=lambda u: -rr[u]))}
    f_rank = {u: i for i, u in enumerate(order)}
    span = (hx.sessions[-1]["date"] - hx.sessions[0]["date"]).days if hx.sessions else 0
    win_active = bool(P.get("window_prior")) and span >= 45          # dev histories span ~10 days: prior is off there
    final, lab = {}, {}
    for u in cand:
        unit = hx.units[u]
        s = 1.0 / (60 + r_rank.get(u, 999)) + 0.5 / (60 + f_rank.get(u, 999))
        s *= _role_ok(unit, P["asst"]) * window_prior(unit.date.date(), a["window"], win_active)
        s *= fact_prior(unit.text, unit.role) if P.get("fact_prior", True) else 1.0
        lab[u] = agent_label(unit.text, a["terms"])
        if P.get("agent_prior") and not a["others_ok"]:
            s *= {"other": 0.6, "neg": 0.85}.get(lab[u], 1.0)
        final[u] = s
    ranked = sorted(final, key=lambda u: -final[u])
    rest = [u for u in order if u not in final]                     # unranked tail for the debug ranking
    return ranked, final, rr, lab, facet_top, rest, win_active


def _select_decay(hx, ranked: list[int], final: dict, facet_top: dict, P: dict) -> list[int]:
    """Soft diversity: greedy by score, the k-th unit taken from a session is worth decay**k. Strong second/third units
    of relevant sessions (repeated gold turns) beat the weak first unit of an unrelated session; facet tops go first."""
    import heapq
    d = P.get("sess_decay", 0.6)
    out = []
    for us in facet_top.values():
        for u in us[:1]:
            if u in final and u not in out:
                out.append(u)
    per = defaultdict(int)
    for u in out:
        per[hx.units[u].session_id] += 1
    heap = [(-final[u] * d ** per[hx.units[u].session_id], u, per[hx.units[u].session_id])
            for u in ranked if u not in out]
    heapq.heapify(heap)
    sessions = {hx.units[u].session_id for u in out}
    while heap:
        neg, u, k = heapq.heappop(heap)
        sid = hx.units[u].session_id
        if per[sid] != k:                                   # stale: session got another unit since this was pushed
            heapq.heappush(heap, (-final[u] * d ** per[sid], u, per[sid]))
            continue
        if per[sid] >= P["per_sess"] or (sid not in sessions and len(sessions) >= P["max_sess"]):
            continue
        out.append(u)
        sessions.add(sid)
        per[sid] += 1
    return out


def _select(hx, ranked: list[int], facet_top: dict, P: dict) -> list[int]:
    """Session-first diversity: pass 1 = best unit of each session (facet tops first), pass 2 = depth per session."""
    seq, seen_s, out = [], set(), []
    for us in facet_top.values():
        for u in us[:2]:
            if u in ranked and u not in seq:
                seq.append(u)
    seq += [u for u in ranked if u not in seq]
    for u in seq:
        sid = hx.units[u].session_id
        if sid not in seen_s and len(seen_s) < P["max_sess"]:
            seen_s.add(sid)
            out.append(u)
    per = defaultdict(int)
    for u in out:
        per[hx.units[u].session_id] += 1
    if P.get("deep_top"):
        # depth for the strongest sessions before more breadth: a gold session's fact turn often ranks 4th-6th inside
        # its own session (on-topic chit-chat outranks the 'by the way, I ...' sentence)
        top_s = []
        for u in out:
            sid = hx.units[u].session_id
            if sid not in top_s:
                top_s.append(sid)
        top_set = set(top_s[:P["deep_top"]])
        deep = []
        for u in seq:
            sid = hx.units[u].session_id
            if u not in out and sid in top_set and per[sid] < P["per_sess"]:
                per[sid] += 1
                deep.append(u)
        k = min(len(out), P["deep_top"])
        out = out[:k] + deep + out[k:]
    for u in seq:
        sid = hx.units[u].session_id
        if u not in out and sid in seen_s and per[sid] < P["per_sess"]:
            per[sid] += 1
            out.append(u)
    return out


def _render_session(hx, uids: list[int], terms: list[str], trim: int) -> tuple[str, set[str]]:
    parts, turns, prev = [], set(), None
    for i in sorted(set(uids)):
        u = hx.units[i]
        text, _ = trim_text(u.text, terms, trim) if trim else (u.text, False)
        if prev is not None and i != prev + 1:
            parts.append("[...]")
        if u.role == "user" and u.ctx and u.chunk_idx == 0 and len(u.text) < 160 and (prev is None or i != prev + 1):
            parts.append(f"assistant: … {u.ctx}")
        parts.append(f"{u.role}: {text}")
        turns.add(u.turn_id)
        prev = i
    return "\n".join(parts), turns


def _pack_trimmed(q, hx, picks: list[int], hint: str, P: dict, terms: list[str]):
    from core.pack import message_chars
    chosen: dict[str, list[int]] = {}

    def passages(sel):
        sids = sorted(sel, key=lambda s: hx.sess_by_id[s]["sess_idx"])
        return [Passage(id=s, date=hx.sess_by_id[s]["date_str"], text=_render_session(hx, sel[s], terms, P["trim"])[0])
                for s in sids]

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
        included |= _render_session(hx, us, terms, P["trim"])[1]
    return passages(chosen), included, chars, chosen


def build(q, hx, **params) -> EvidencePack:
    P = dict(VARIANTS["sweep"])
    P.update(params)
    a = analyze(q.question, q.date)
    terms = a["terms"] + [w for f in a["facets"] for w in content_terms(f)]
    ranked, final, rr, lab, facet_top, rest, win_active = _score_units(q, hx, a, P)

    # abstention evidence
    miss = missing_terms(hx, a["terms"])
    miss_frac = len(miss) / max(1, len([w for w in a["terms"] if w not in _GENERIC and not w.isdigit()]))
    top_rr = max(rr.values()) if rr else -10.0
    head = [u for u in ranked[:10] if focus_sentences(hx.units[u].text, a["terms"])]
    all_other = bool(head) and not a["others_ok"] and all(lab[u] == "other" for u in head[:6])
    abstain_p = abstain_prob(top_rr, miss_frac, all_other)
    hint = build_hint(a, q.date, abstain_p)

    if P["mode"] == "sessagg":
        passages, included, chars, sess_rank, sess_score = _sessagg(q, hx, ranked, rr, lab, a, P, hint)
        best = sess_score[sess_rank[0]] if sess_rank else 0.0
        cite = [s for s in sess_rank if sess_score[s] >= P["cite_frac"] * best][:P["cite_max"]]
        cite += [s for s in sess_rank if s not in cite][: max(0, 2 - len(cite))]   # multi gold has >= 2 sessions
    else:
        picks = (_select_decay(hx, ranked, final, facet_top, P) if P.get("select") == "decay"
                 else _select(hx, ranked, facet_top, P))
        passages, included, chars, chosen = _pack_trimmed(q, hx, picks, hint, P, terms)
        cite = (_cite_topk(hx, chosen, final, rr, facet_top, P) if P.get("cite") == "topk"
                else _cite_plan(hx, chosen, final, rr, lab, facet_top, a, P))
    in_pack = {p.id for p in passages}
    if P.get("cite") == "rel":
        # sessions ordered by their best CE logit over ALL scored units (incl. the intra-session rerank), cite those
        # within cite_d of the best one; 2..cite_max (multi gold always spans >= 2 sessions)
        rmax: dict[str, float] = {}
        for u, sc in rr.items():
            sid = hx.units[u].session_id
            if sid in in_pack and (P["asst"] != "none" or hx.units[u].role == "user"):
                rmax[sid] = max(rmax.get(sid, -99.0), sc)
        order = sorted(rmax, key=lambda s: -rmax[s])
        top = rmax[order[0]] if order else -99.0
        cite = [s for s in order if rmax[s] >= top - P.get("cite_d", 3.0)][:P.get("cite_max", 5)]
        if len(cite) < 2:
            cite = order[:2]
    cite = [s for s in cite if s in in_pack]
    # session features for offline cite-rule analysis (scratch_b/cite_fit.py reads them from per_question.jsonl)
    sess_feat: dict[str, list] = {}
    for r, u in enumerate(ranked):
        sid = hx.units[u].session_id
        if sid not in in_pack:
            continue
        f = sess_feat.setdefault(sid, [r, rr.get(u, -99.0), lab.get(u, "none"), 0, -99.0])
        f[3] += 1 if r < 40 else 0
        f[4] = max(f[4], rr.get(u, -99.0))
    return EvidencePack(question_id=q.question_id, variant=params.get("_variant", ""), route="multi",
                        passages=passages, hint=hint, cite_plan=cite or None, abstain_p=abstain_p,
                        included_turns=included, computed=None,
                        debug={"ranking": ranked + rest, "analysis": {k: (str(v) if k == "window" else v)
                                                                       for k, v in a.items()},
                               "window_active": win_active, "top_rr": top_rr, "missing_terms": miss,
                               "all_other": all_other, "chars": chars, "sess": sess_feat})


def _cite_topk(hx, chosen, final, rr, facet_top, P) -> list[str]:
    """Top-k sessions by their best fused score, plus sessions whose best CE logit is within `cite_d` of the top one
    (up to cite_max); each facet's top session first. Dev: gold has 2-3 sessions in 75% of questions."""
    best = {sid: max(final.get(u, 0.0) for u in us) for sid, us in chosen.items()}
    rmax = {sid: max(rr.get(u, -99.0) for u in us) for sid, us in chosen.items()}
    out = []
    for us in facet_top.values():
        sid = hx.units[us[0]].session_id if us else None
        if sid in best and sid not in out:
            out.append(sid)
    k = max(P.get("cite_k", 3), len(out))
    for sid in sorted(best, key=lambda s: -best[s]):
        if len(out) >= k:
            break
        if sid not in out:
            out.append(sid)
    top = max(rmax.values(), default=-99.0)
    for sid in sorted(rmax, key=lambda s: -rmax[s]):
        if len(out) >= P.get("cite_max", 4):
            break
        if sid not in out and rmax[sid] >= top - P.get("cite_d", 2.0):
            out.append(sid)
    return out


def _cite_plan(hx, chosen, final, rr, lab, facet_top, a, P) -> list[str]:
    best: dict[str, tuple[float, float, str]] = {}
    for sid, us in chosen.items():
        u = max(us, key=lambda x: final.get(x, 0))
        best[sid] = (final.get(u, 0.0), rr.get(u, -99.0), lab.get(u, "none"))
    order = sorted(best, key=lambda s: -best[s][0])
    top = max((v[1] for v in best.values()), default=-99.0)
    tau = top - P.get("cite_delta", 3.0)                              # relative: CE logits for count questions sit
    must = []                                                         # far below 0 even on gold turns
    for us in facet_top.values():                                     # each facet's top session is cited if it is
        for u in us[:1]:                                              # a confident hit
            sid = hx.units[u].session_id
            if sid in best and sid not in must and rr.get(u, -99.0) >= tau:
                must.append(sid)
    out = list(must)
    for s in order:
        f, r, lb = best[s]
        if s in out:
            continue
        if r < tau:
            continue
        if lb == "other" and not a["others_ok"]:
            continue
        out.append(s)
        if len(out) >= P["cite_max"]:
            break
    if len(out) < 2:                                                 # multi gold always has >= 2 sessions
        out += [s for s in order if s not in out][: 2 - len(out)]
    return out


def _sessagg(q, hx, ranked, rr, lab, a, P, hint):
    from core.pack import build_pack
    by_s: dict[str, list[int]] = defaultdict(list)
    for u in sorted(rr, key=lambda x: -rr[x]):
        if lab.get(u) == "other" and P.get("agent_prior") and not a["others_ok"]:
            continue
        by_s[hx.units[u].session_id].append(u)
    sig = lambda x: 1.0 / (1.0 + math.exp(-x))
    score = {s: sum(sig(rr[u]) / math.log2(j + 2) for j, u in enumerate(us[:3])) for s, us in by_s.items()}
    sess_rank = sorted(score, key=lambda s: -score[s])[:P["max_sess"]]
    ordered = [u for s in sess_rank for u in by_s[s][:P["per_sess"]]]
    passages, included, chars = build_pack(q, hx, ordered, budget=P["budget"], window=P["unit_window"],
                                           order="chrono", hint=hint, max_sessions=P["max_sess"])
    return passages, included, chars, sess_rank, score


# ------------------------------------------------------------------------------------------------------------------
# Self-test: pure helpers on the sample questions (no index, no models)

def _selftest() -> None:
    import json
    from pathlib import Path
    F = "%Y/%m/%d (%a) %H:%M"
    d = lambda s: datetime.strptime(s, F)
    qd = d("2023/09/04 (Mon) 00:00")
    W = lambda t: parse_window(t, qd)
    assert W("How many books have I read since March?")["start"] == date(2023, 3, 1)
    w = W("How many times did my systolic reading exceed 130 in August?")
    assert (w["start"], w["end"]) == (date(2023, 8, 1), date(2023, 8, 31)), w
    w = W("How many times did I eat feijoada last month?")
    assert (w["start"], w["end"]) == (date(2023, 8, 1), date(2023, 8, 31)), w
    w = parse_window("How much honey did I harvest this summer?", d("2023/10/18 (Wed) 00:00"))
    assert (w["start"], w["end"]) == (date(2023, 6, 1), date(2023, 8, 31)), w
    w = parse_window("museums in the month of February?", d("2023/03/03 (Fri) 23:25"))
    assert (w["start"], w["end"]) == (date(2023, 2, 1), date(2023, 2, 28)), w
    w = parse_window("museums in December?", d("2023/03/03 (Fri) 23:25"))
    assert w["start"] == date(2022, 12, 1), w
    assert W("How many laps do I swim each morning these days?") is None
    w = parse_window("since the start of June", d("2023/11/07 (Tue) 00:00"))
    assert w["start"] == date(2023, 6, 1), w
    assert facets_of("What is the total cost of the car cover and detailing spray I purchased")[:2] == \
        ["car cover", "detailing spray I purchased"], facets_of("What is the total cost of the car cover and detailing spray I purchased")
    # agent classifier (synthetic sentences in the style of the near-miss decoys)
    T = ["shoes"]
    assert agent_label("By the way, I bought a pair of dance shoes for 50 euros.", T) == "self"
    assert agent_label("My friend Ana bought a pair of dance shoes for 50 euros.", T) == "other"
    assert agent_label("Anyway, Marco told me he bought new shoes last week.", T) == "other"
    assert agent_label("I tried on a pair of shoes that cost 90 but decided not to buy them.", T) == "neg"
    assert agent_label("Your sister mentioned she bought shoes twice last month.", ["shoes"]) == "other"
    assert agent_label("My brother has three grandchildren.", ["grandchildren"]) == "other"
    s, cut = trim_text("Hi there. The weather is nice. I bought two orchids at the market. They are pink. "
                       "Also I need groceries and many other unrelated things that make this long.", ["orchids"], 90)
    assert cut and "two orchids" in s, s
    # sample questions: print the analysis for eyeballing
    path = Path(__file__).resolve().parents[2] / "runs/samples/multi.jsonl"
    for line in open(path):
        r = json.loads(line)
        a = analyze(r["question"], d(r["question_date"]))
        w = a["window"]
        print(f"[{r['split']:4}] {r['question'][:90]}")
        print(f"       kind={a['kind']} latest={a['latest']} others_ok={a['others_ok']} "
              f"window={(w['label'], str(w['start']), str(w['end'])) if w else None}")
        print(f"       core={a['core']!r} facets={a['facets']} stmts={a['statements'][:2]}")
    a = analyze("How much have I spent on tango shoes in total?", d("2023/10/15 (Sun) 00:00"))
    print("\nHINT example:\n", build_hint(a, d("2023/10/15 (Sun) 00:00")))
    print("\nselftest OK")


if __name__ == "__main__":
    _selftest()
