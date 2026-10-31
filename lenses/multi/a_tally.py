"""Lens A "tally" for multi-session questions (counts, sums, differences across 2-5 sessions).

Idea: the answer is spread over several short first-person statements ("By the way, I bought a pair of tango shoes for
1,800 pesos."), each sitting in a different session among many on-topic decoys. So:
  1. recall ALL candidate statements: ranked channel (hybrid/BM25 over user units) UNION a deterministic lexical sweep
     (every user unit whose stems hit the question's head terms / phrase / category synonyms);
  2. filter by the question's time window ("since March", "in August", "last month", "this summer") on session dates,
     but only when the history's dates actually span beyond the window (test: 6-month histories; dev: ~10 days);
  3. pack SENTENCES, not sessions: per session only the matching sentences (+-1 sentence, + the assistant question
     they answer), so 20-30 sessions fit in ~8-10k chars, oldest first;
  4. tell Reactor to enumerate, then count, skipping other people's items, things not done, and repeats.

Contract: build(q, hx, **params) -> EvidencePack. Pure-python helpers (date grammar, term extraction, decoy cues,
sentence ledger) are self-tested by `python -m lenses.multi.a_tally` (no core.index needed for the self-test).
"""
from __future__ import annotations

import calendar
import re
from collections import defaultdict
from datetime import datetime, timedelta

try:  # PyStemmer is in the venv (core.index uses it); fall back to a crude suffix stripper
    import Stemmer

    _ST = Stemmer.Stemmer("english")

    def stem(w: str) -> str:
        return _ST.stemWord(w.lower())
except Exception:  # pragma: no cover
    def stem(w: str) -> str:
        w = w.lower()
        for suf in ("ies", "es", "s", "ing", "ed"):
            if len(w) > len(suf) + 2 and w.endswith(suf):
                return w[: -len(suf)]
        return w

from core.types import EvidencePack, Passage

_TIGHT = {"max_sessions": 40, "per_session": 4, "budget": 12000, "render": "sentences", "window": True, "sweep": True,
          "ctx": 0, "keep_frac": 0.8, "cite": "ssfrac", "cite_frac": 0.8, "cite_k": 5, "zl_max": 2}
VARIANTS = {
    # BEST MEASURED (round 1): BM25 + lexical sweep + window + tight sentence ledger, 12k chars, ~25-30 sessions on test
    "bm25_tight": {**_TIGHT, "rank": "bm25", "rerank": False},
    # same with hybrid (BM25 + bge-small) recall; needs the model server (not measured in round 1: server backlog)
    "ledger": {**_TIGHT, "rank": "hybrid", "rerank": False},
    # + cross-encoder on the top 60 fused units
    "ledger_rr": {**_TIGHT, "rank": "hybrid", "rerank": True},
    # ablations: wider excerpts (+-1 context sentence, 24 sessions) at 15k / 12k / 10k chars
    "bm25": {**_TIGHT, "rank": "bm25", "rerank": False, "max_sessions": 24, "budget": 15000, "ctx": 1, "keep_frac": 0.6},
    "bm25_12k": {**_TIGHT, "rank": "bm25", "rerank": False, "max_sessions": 24, "ctx": 1, "keep_frac": 0.6},
    "bm25_10k": {**_TIGHT, "rank": "bm25", "rerank": False, "max_sessions": 24, "budget": 10000, "ctx": 1,
                 "keep_frac": 0.6},
    # whole user turns instead of sentence excerpts
    "turns": {**_TIGHT, "rank": "bm25", "rerank": False, "max_sessions": 16, "per_session": 2, "render": "turns"},
}

# ----------------------------------------------------------------------------------------------------------------
# Question analysis
# ----------------------------------------------------------------------------------------------------------------
MONTHS = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}
MONTHS.update({m.lower(): i for i, m in enumerate(calendar.month_abbr) if m})
MONTHS.pop("may", None)
MONTHS["may"] = 5
_MON = r"(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sept?|oct|nov|dec)"
SEASONS = {"spring": (3, 5), "summer": (6, 8), "fall": (9, 11), "autumn": (9, 11), "winter": (12, 2)}
NUMW = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        "eleven": 11, "twelve": 12, "a couple": 2, "couple": 2, "few": 3, "several": 3}

STOP = set("""a an the of in on at to for from by with about into over after before since during and or but not no
i me my mine myself we our ours us you your yours he she it they them their his her its this that these those there here
do does did done doing have has had having be is are was were been being am will would could should can may might must
how many much what which who whom whose when where why total totally overall altogether combined combine sum number
amount count times time different kinds kind types type far so still currently current now right these days day
ever all each every any some more most less least than then just really also get got getting make made""".split())
# scaffolding verbs that carry no topic
SCAFF = set("""spend spent attend attended visit visited go went use used buy bought acquire acquired own owned
have had take took do did need needs needed""".split())

# Generic category -> instance words (written from dev failure patterns + common sense, NOT from test questions).
LEXICON = {
    "doctor": "physician dermatologist specialist ent dr appointment clinic pediatrician dentist orthodontist",
    "furniture": "table chair sofa couch bed mattress desk shelf bookshelf dresser cabinet wardrobe ottoman nightstand",
    "bake": "baked baking cake cookies bread muffins pie tart brownies cupcakes croissants scones loaf",
    "book": "novel biography memoir read reading finished chapter author paperback audiobook",
    "instrument": "guitar piano keyboard violin drum drums ukulele bass saxophone flute cello trumpet",
    "kitchen": "toaster faucet kettle blender mixer pan pot knife mat oven microwave fridge dishwasher",
    "art": "museum gallery exhibition exhibit painting sculpture tour workshop",
    "museum": "gallery exhibition exhibit tour",
    "device": "monitor tracker aids aid watch fitbit glucose thermometer scale cpap",
    "health": "blood pressure monitor glucose tracker aids medication",
    "delivery": "uber eats doordash grubhub instacart postmates deliveroo",
    "jewelry": "necklace ring bracelet earrings pendant watch brooch",
    "plant": "succulent fern cactus orchid lily ivy pothos monstera seedling",
    "clothing": "shirt jacket blazer boots shoes dress jeans coat sweater scarf",
    "festival": "film festival screening",
    "event": "concert festival show exhibition fair party conference workshop",
    "wedding": "married ceremony bride groom reception",
    "baby": "born birth newborn son daughter twins",
    "project": "lead leading led team initiative",
    "course": "class certificate certification module completed",
    "cuisine": "dish recipe cooking cooked",
    "fruit": "lemon lime orange grapefruit yuzu",
    "citrus": "lemon lime orange grapefruit yuzu",
    "pet": "dog cat puppy kitten fish hamster rabbit",
    "sibling": "brother sister brothers sisters",
    "money": "$ dollars spent paid cost",
    "expense": "$ spent paid cost bought",
    "fund": "savings account deposited saved",
    "savings": "fund account deposited saved",
    "reading": "blood pressure systolic",
    "call": "called phoned rang chat",
    "trip": "travel traveled visited flew drove",
    "game": "played playing hours",
    "subscription": "subscribe subscribed magazine",
    "vegetable": "tomato tomatoes cucumber pepper peppers carrot lettuce zucchini beans kale spinach",
    "aquarium": "tank gallon",
    "fish": "tetra tetras gourami gouramis betta pleco guppy guppies goldfish angelfish molly",
    "antique": "heirloom heirlooms vintage inherited",
    "magazine": "subscription subscribed digest journal issue",
    "fitness": "yoga pilates zumba spin bodypump crossfit gym class",
    "feed": "grains pellets layer scratch pounds",
    "graduate": "degree bachelor master diploma",
    "college": "degree bachelor university",
}

# Attribute questions whose answer is phrased without the question's words ("how old" -> "I just turned 32").
# (question pattern, statement pattern); a matching user sentence counts as a strong lexical hit.
ATTRS = [
    (r"\b(how old|years? older|years? younger|older than|younger than|\bages?\b|how many years will i be|born)\b",
     r"\b(?:i'?m|i am|i was|turned|turning)\s+(?:just\s+)?\d{2}\b|\b\d{2}[- ]years?[- ]old\b|\b\d{2} (?:is|was) "
     r"(?:considered|young|old)|\b(?:is|was|be|turned) \d{2}\b|\bin (?:my|their) (?:20|30|40|50|60|70)s\b"),
    (r"\b(each|a|per|typical|every) week\b|\bweekly\b",
     r"\b(?:on|every) (?:mondays?|tuesdays?|wednesdays?|thursdays?|fridays?|saturdays?|sundays?)\b|"
     r"\b(?:once|twice|\d+ times|three times) a week\b"),
    (r"\b(blood pressure|systolic|diastolic|bp)\b", r"\b\d{2,3}\s*(?:/|over)\s*\d{2,3}\b|\bsystolic\b"),
]
# Event verbs that separate an instance from topic chatter; they only count in a sentence that already hits the topic.
EVENTS = [
    (r"\b(add(?:ed)?|acquir\w*|bought|buy|purchas\w*|got|get|collect\w*|own)\b",
     r"\b(bought|picked up|got|purchased|ordered|adopted|received|gave me|gift(?:ed)?|added|brought home|new)\b"),
    (r"\b(spen[dt]|cost|paid|pay)\b", r"[$€£]\s?\d|\b\d[\d,.]*\s?(?:pesos|dollars|euros|pounds|reais|yen|usd|eur)\b"),
]

_PLURAL_TO = {}  # stem(alias) -> stem(category) filled lazily


def _stems(text: str) -> list[str]:
    return [stem(w) for w in re.findall(r"[a-z0-9$]+(?:'[a-z]+)?", text.lower()) if w not in STOP]


def _month_start(y: int, m: int) -> datetime:
    return datetime(y, m, 1)


def _month_end(y: int, m: int) -> datetime:
    return datetime(y, m, calendar.monthrange(y, m)[1], 23, 59)


def _last_month_on_or_before(qd: datetime, m: int) -> int:
    """Year of the most recent month m that is not after the question month."""
    return qd.year if m <= qd.month else qd.year - 1


def time_window(question: str, qd: datetime):
    """Rule-based window from the question text -> (start, end, label, phrase) or None. Generous, never invents."""
    s = question.lower()
    m = re.search(rf"\bfrom {_MON}\w* (?:to|through|until|till) {_MON}\w*", s)
    if m:
        a, b = MONTHS[m.group(1)[:3]], MONTHS[m.group(2)[:3]]
        yb = _last_month_on_or_before(qd, b)
        ya = yb if a <= b else yb - 1
        return _month_start(ya, a), _month_end(yb, b), f"{calendar.month_name[a]}-{calendar.month_name[b]} {yb}", m.group(0)
    m = re.search(rf"\bsince (?:the (?:start|beginning) of )?{_MON}\w*(?: (\d{{4}}))?", s)
    if m:
        mo = MONTHS[m.group(1)[:3]]
        y = int(m.group(2)) if m.group(2) else _last_month_on_or_before(qd, mo)
        return _month_start(y, mo), qd, f"since {calendar.month_name[mo]} 1, {y}", m.group(0)
    m = re.search(r"\bsince the (?:start|beginning) of (?:the|this) year\b|\bthis year\b|\bso far this year\b", s)
    if m:
        return datetime(qd.year, 1, 1), qd, f"{qd.year} (Jan 1 to now)", m.group(0)
    m = re.search(rf"\b(?:in|during|for) (?:the month of )?{_MON}\w* (?:and|or|&) {_MON}\w*\b", s)
    if m:  # "in January and March": span both months, say which in the label
        a, b = sorted((MONTHS[m.group(1)[:3]], MONTHS[m.group(2)[:3]]))
        yb = _last_month_on_or_before(qd, b)
        return (_month_start(yb, a), _month_end(yb, b),
                f"{calendar.month_name[a]} and {calendar.month_name[b]} {yb}", m.group(0))
    m = re.search(rf"\b(?:in|during|for) (?:the month of )?{_MON}\w*(?: (\d{{4}}))?\b", s)
    if m and not re.search(rf"{_MON}\w* \d{{1,2}}(st|nd|rd|th)?\b", s[m.start():m.end() + 6]):
        mo = MONTHS[m.group(1)[:3]]
        y = int(m.group(2)) if m.group(2) else _last_month_on_or_before(qd, mo)
        return _month_start(y, mo), _month_end(y, mo), f"{calendar.month_name[mo]} {y}", m.group(0)
    m = re.search(r"\b(?:this|last|past) (spring|summer|fall|autumn|winter)\b", s)
    if m:
        a, b = SEASONS[m.group(1)]
        y = qd.year if (qd.month >= a or a == 12) else qd.year - 1
        if m.group(0).startswith(("last", "past")) and qd.month <= b and a != 12:
            y -= 1
        start = datetime(y - 1 if a == 12 else y, a, 1)
        end = _month_end(y, b)
        return start, min(end, qd), f"{m.group(1)} {y}", m.group(0)
    m = re.search(r"\b(?:in |over |during )?the (?:last|past) (\w+|\d+)? ?(day|week|month|year)s?\b", s)
    if m:
        n = m.group(1)
        n = int(n) if n and n.isdigit() else NUMW.get(n or "", None)
        unit = m.group(2)
        if n is None:
            n = 3 if "few" in m.group(0) or "couple" in m.group(0) else 1
        days = {"day": 1, "week": 7, "month": 31, "year": 366}[unit] * n
        return qd - timedelta(days=days + 2), qd, f"last {n} {unit}(s) before {qd:%Y-%m-%d}", m.group(0)
    m = re.search(r"\blast (week|month|year)\b", s)
    if m:
        u = m.group(1)
        if u == "month":
            y, mo = (qd.year, qd.month - 1) if qd.month > 1 else (qd.year - 1, 12)
            return _month_start(y, mo), qd, f"{calendar.month_name[mo]} {y} (last month)", m.group(0)
        if u == "week":
            start = qd - timedelta(days=qd.weekday() + 7)
            return start, start + timedelta(days=6, hours=23), f"week of {start:%Y-%m-%d} (last week)", m.group(0)
        return datetime(qd.year - 1, 1, 1), datetime(qd.year - 1, 12, 31, 23, 59), f"{qd.year - 1}", m.group(0)
    m = re.search(r"\bthis (week|month)\b", s)
    if m:
        start = qd - timedelta(days=qd.weekday()) if m.group(1) == "week" else datetime(qd.year, qd.month, 1)
        return start, qd, f"this {m.group(1)}", m.group(0)
    return None


KIND_RX = [
    ("count", r"^\s*how many (times|different|of the)\b"),
    ("diff", r"\bhow (much|many) (more|less|fewer|older|younger|faster|slower|earlier|later|longer)\b|\bdifference\b|\b(left|remaining|remain)\b|\bsave\b|\bsaved\b|\bincrease\b|\bdecrease\b"),
    ("avg", r"\baverage\b|\bmean\b"),
    ("current", r"\b(now|currently|current|these days|right now|so far|at the moment|nowadays|each (morning|evening|day|week)|a week|per week|typical)\b"),
    ("sum", r"\b(total|in total|combined|altogether|sum|how much (did|have|money)|spent|spend|raise|earn|earned)\b"),
    ("count", r"^\s*(how many|what is the number|what was the number)\b"),
]


def question_kind(question: str) -> str:
    s = question.lower()
    for k, rx in KIND_RX:
        if re.search(rx, s):
            return k
    return "other"


_TITLE = re.compile(r"(?:^|(?<=[\s(]))['\"‘“]([^'\"‘’“”]{2,60}?)['\"’”](?=[\s?.,!;:)]|$)")


def analyze(question: str, qd: datetime) -> dict:
    """Head terms, phrases, synonyms, conjuncts, window, kind."""
    win = time_window(question, qd)
    s = question
    if win:
        s = re.sub(re.escape(win[3]), " ", s, flags=re.I)
    titles = [t.strip() for t in _TITLE.findall(s)]
    # proper nouns / capitalised words not at the start of the question
    caps = [w for i, w in enumerate(re.findall(r"\b[A-Z][a-zA-ZÀ-ſ'-]+", s)) if i or not s.strip().startswith(w)]
    caps = [re.sub(r"'s?$", "", c) for c in caps if c.lower() not in STOP and c.lower() not in MONTHS]
    caps = [c for c in caps if c and c.lower() not in STOP and c != "I've" and not any(c in t for t in titles)]
    toks = re.findall(r"[A-Za-z0-9À-ſ$][\wÀ-ſ'$.-]*", s)
    content = [t for t in toks if t.lower() not in STOP and t.lower() not in SCAFF and len(t) > 1]
    # noun phrase guess: maximal runs of content words (keeps "tango shoes", "bakery fund", "dulce de leche"-ish)
    phrases, run = [], []
    for t in toks + ["."]:
        if t.lower() not in STOP and t.lower() not in SCAFF and len(t) > 1 and t != ".":
            run.append(t)
        else:
            if len(run) >= 2:
                phrases.append(" ".join(run))
            run = []
    phrases += [t for t in titles if len(t.split()) >= 1]
    # numeric-unit tokens ("30-gallon") are salient
    numunits = re.findall(r"\b\d+[- ](?:gallon|pound|inch|year|kilo|kg|mile|km|liter|litre)s?\b", s, re.I)
    head = [stem(t) for t in content]
    syn = []
    for t in content:
        st = stem(t)
        for cat, words in LEXICON.items():
            if st == stem(cat) or t.lower().rstrip("s") == cat:
                syn += words.split()
    # conjuncts: "X and Y" where both sides are content -> each needs evidence
    conj = []
    for a, b in re.findall(r"\b([\w'-]+(?: [\w'-]+)?) and (?:the |my |in |at )?([\w'-]+(?: [\w'-]+)?)", s):
        if a.lower() not in STOP and b.split()[0].lower() not in STOP:
            conj.append((a, b))
    attr = [re.compile(urx, re.I) for qrx, urx in ATTRS if re.search(qrx, question, re.I)]
    events = [re.compile(urx, re.I) for qrx, urx in EVENTS if re.search(qrx, question, re.I)]
    return {"window": win, "kind": question_kind(question), "head": head, "content": content, "attr": attr, "events": events,
            "phrases": phrases, "titles": titles, "caps": caps, "numunits": numunits,
            "syn": sorted(set(syn) - set(content)), "conj": conj}


def make_query(question: str, qd: datetime | None = None) -> str:
    """Retrieval query: question minus scaffolding/time phrase, plus category synonyms."""
    qa = analyze(question, qd or datetime(2023, 6, 1))
    return " ".join(qa["content"] + qa["syn"])


# ----------------------------------------------------------------------------------------------------------------
# Sentence-level cues
# ----------------------------------------------------------------------------------------------------------------
_SENT = re.compile(r"(?<=[.!?;])\s+(?=[\"'(\[A-Z0-9*#-])|\n+")
_NUM = re.compile(r"(?:\$|€|£)?\b\d[\d,.]*(?:/\d+)?\b|\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
                  r"fifteen|twenty|thirty|forty|fifty|hundred|dozen|a couple|a pair|twice|once|first|second|third|"
                  r"another|a few|several)\b", re.I)
_OTHER = re.compile(
    r"\b(my|our|his|her)\s+(friend|sister|brother|neighbou?r|cousin|coworker|co-worker|colleague|mom|mother|dad|"
    r"father|aunt|uncle|boss|roommate|partner|husband|wife|son|daughter|niece|nephew|grandma|grandpa|classmate|"
    r"teammate)\b(?!\s*(?:'s)?\s*(?:and I|with me))|\b[A-Z][a-zÀ-ſ]+\s+(told me|said|mentioned|says)\b|"
    r"\b(he|she|they)\s+(has|have|had|bought|made|read|got|owns|drinks|averages)\b", re.I)
_NEG = re.compile(r"\b(didn'?t|did not|decided not to|not to buy|skipped|never|won'?t|wasn'?t|cancel+ed|instead of|"
                  r"but (?:I )?(?:decided|didn'?t)|talking myself out|test batch|not for|returned it)\b", re.I)
_PLAN = re.compile(r"\b(plan(?:ning)? to|thinking (?:of|about)|might|would like to|want to|hoping to|going to|"
                   r"considering|wish(?:list)?|maybe I'?ll|next (?:week|month|year))\b", re.I)


_BTW = re.compile(r"\b(by the way|btw|oh,? and|also,|speaking of|random\b|before I forget|on a (?:practical|side) note)",
                  re.I)
_FACT = re.compile(r"\bI(?:'ve| have| had)? (?:just |recently |also |finally |actually |already )?(?:\w+ed|got|bought|"
                   r"went|took|made|saw|had|did|ran|swam|rode|read|ate|met|won|lost|spent|paid|sold|gave|built|began|"
                   r"started|finished|wrote|drove|flew|brought|caught|found|kept|left|sent|set)\b", re.I)
_DATEW = re.compile(r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b|"
                    r"\b\d{1,2}/\d{1,2}\b|\b\d{4}-\d{2}-\d{2}\b|\b(last|this) (week|month|weekend|monday|tuesday|"
                    r"wednesday|thursday|friday|saturday|sunday)\b|\b(yesterday|ago|this morning|today)\b", re.I)


def fact_prior(u) -> float:
    """How much a user unit looks like a personal-fact statement (dev: gold turns hit these cues 3-11x more often)."""
    t = u.text
    return (0.35 * bool(_BTW.search(t)) + 0.35 * bool(_FACT.search(t)) + 0.25 * bool(_NUM.search(t))
            + 0.2 * bool(_DATEW.search(t)) + 0.25 * (u.turn_idx == 0 and u.chunk_idx == 0))


def sentences(text: str) -> list[str]:
    return [p.strip() for p in _SENT.split(text) if p and p.strip()]


def cue_flags(sent: str) -> list[str]:
    f = []
    if _OTHER.search(sent):
        f.append("other")
    if _NEG.search(sent):
        f.append("neg")
    if _PLAN.search(sent):
        f.append("plan")
    if _NUM.search(sent):
        f.append("num")
    return f


def match_score(sent: str, qa: dict) -> float:
    """Lexical evidence that a sentence is about the question's topic (stems of head terms, phrases, synonyms)."""
    ss = set(_stems(sent))
    low = sent.lower()
    sc = 0.0
    hs = [h for h in qa["head"] if h]
    hit = sum(1 for h in hs if h in ss)
    sc += hit
    for p in qa["phrases"]:
        pst = _stems(p)
        if pst and all(x in ss for x in pst):
            sc += 1.5
    for t in qa["titles"] + qa["caps"]:
        if t.lower() in low:
            sc += 1.0
    syn = sum(1 for w in qa["syn"] if (stem(w) in ss or (w == "$" and "$" in sent)))
    sc += min(syn, 2) * 0.6
    if any(rx.search(sent) for rx in qa.get("attr", ())):
        sc += 1.5
    if sc > 0 and any(rx.search(sent) for rx in qa.get("events", ())):
        sc += 0.8
    return sc


# ----------------------------------------------------------------------------------------------------------------
# Hints
# ----------------------------------------------------------------------------------------------------------------
def hint_text(qa: dict, window_used: bool) -> str:
    win = qa["window"]
    wtxt = ""
    if win:
        wtxt = (f" Only count what falls in {win[2]} (use each excerpt's date; 'yesterday'/'last night' refers to the "
                f"day before the excerpt's date).")
    base = ("The excerpts are the user's own messages, oldest first, from many conversations. ")
    k = qa["kind"]
    if k in ("count", "other"):
        body = ("First list every distinct instance with its date and [E#], then count. Count only the user's own "
                "(skip what friends, family, neighbours or others did or own), skip things only planned, considered, "
                "tried but not bought, or not done, and count an item mentioned in several conversations once. "
                "If the question sets a condition (a threshold, a place, a purpose), check it for each instance. "
                "If the user states the total directly (e.g. 'I'm putting five pieces in'), use the latest such statement.")
    elif k == "sum":
        body = ("First list every relevant amount with its date and [E#], then add them. Include only the user's own "
                "amounts (not friends'/family's), skip amounts only considered or not spent, and do not add the same "
                "transaction twice when it is mentioned again.")
    elif k == "diff":
        body = ("List each relevant value or transaction with its date and [E#] (deposits +, withdrawals -), then "
                "compute the difference or remaining amount. Ignore other people's values.")
    elif k == "current":
        body = ("Values may change over time: list each dated value or change (added, sold, lost, increased), and "
                "answer with the state as of the latest excerpt. Ignore other people's values.")
    else:  # avg
        body = "List each value with its date and [E#], then compute the average. Ignore other people's values."
    return base + body + wtxt


# ----------------------------------------------------------------------------------------------------------------
# Packing (own renderer: sentence ledger per session)
# ----------------------------------------------------------------------------------------------------------------
BASE_OVERHEAD = 760          # client.build_message fixed text (question date line, instructions, final-answer lines)
PER_PASSAGE = 70             # "[E12] (source: s_xxxxxxxxxxxx, date: 2023/05/20 (Sat) 02:21) "


def message_chars(q, hint: str, passages: list[Passage]) -> int:
    """Exact Reactor message length (core.pack.message_chars); estimate only if the kit is unavailable."""
    try:
        from core.pack import message_chars as mc
        return mc(q, passages, hint)
    except Exception:
        return BASE_OVERHEAD + len(q.question) + len(hint) + sum(PER_PASSAGE + len(p.text) for p in passages)


def _excerpt(hx, uid: int, qa: dict, ctx_sents: int = 1, keep_frac: float = 0.6) -> tuple[str, float, list[str]]:
    """Best sentences of a unit (+-ctx_sents) with the assistant question they answer when needed."""
    u = hx.units[uid]
    sents = sentences(u.text)
    if not sents:
        return u.text, 0.0, []
    scores = [match_score(s, qa) for s in sents]
    best = max(range(len(sents)), key=lambda i: scores[i])
    keep = set()
    for i, sc in enumerate(scores):
        if sc >= max(1.0, scores[best] * keep_frac):
            keep.update(range(max(0, i - ctx_sents), min(len(sents), i + ctx_sents + 1)))
    if not keep:  # no lexical hit: keep the personal-fact sentences ("By the way, I just got ..."), else the whole unit
        fact = [i for i, x in enumerate(sents) if _BTW.search(x) or (_FACT.search(x) and _NUM.search(x))]
        keep = set(fact[:2]) if fact else set(range(len(sents)))
    flags = sorted({f for i in keep for f in cue_flags(sents[i]) if scores[i] > 0})
    idx = sorted(keep)
    parts, prev = [], -1
    for i in idx:
        if prev >= 0 and i != prev + 1:
            parts.append("…")
        parts.append(sents[i])
        prev = i
    text = " ".join(parts)
    if idx[0] > 0:
        text = "… " + text
    if idx[-1] < len(sents) - 1:
        text += " …"
    # short replies need the assistant's question ("How was your swim?" -> "It was good, forty laps")
    if u.chunk_idx == 0 and idx[0] == 0 and u.ctx and (len(sents[0]) < 40 or re.match(
            r"(yes|no|yeah|nope|it|that|they|about|around|just|only|probably|mostly|twice|once|\d)\b", sents[0], re.I)):
        text = f"(assistant asked: …{u.ctx.strip()[-160:]}) " + text
    return text, scores[best], flags


def _sess_user_uids(hx, sid: str) -> list[int]:
    cache = hx.__dict__.setdefault("_a_multi_sess_users", {})
    if not cache:
        for u in hx.units:
            if u.role == "user":
                cache.setdefault(u.session_id, []).append(u.uid)
    return cache.get(sid, [])


def _rank(hx, query: str, params: dict, roles="user") -> list[tuple[int, float]]:
    k = params.get("pool", 300)
    if params.get("rank") == "bm25":
        return hx.bm25(query, k=k, roles=roles)
    return hx.hybrid(query, k=k, roles=roles, pool=k)


def _in_window(d: datetime, win, margin_days: int = 2) -> bool:
    return win[0] - timedelta(days=1) <= d <= win[1] + timedelta(days=margin_days)


def window_applies(hx, win) -> bool:
    """Engage the session-date filter only if the history clearly extends beyond the window and some sessions fall in it."""
    if not win:
        return False
    dates = [s["date"] for s in hx.sessions]
    inside = sum(1 for d in dates if _in_window(d, win))
    span = (max(dates) - min(dates)).days if dates else 0
    return inside >= 1 and inside <= 0.85 * len(dates) and span >= 21


def select(q, hx, qa: dict, params: dict):
    """Fused ranking of user units -> list of (uid, fused_score)."""
    query = " ".join(qa["content"] + qa["syn"]) or q.question
    fused: dict[int, float] = defaultdict(float)
    ranked = _rank(hx, query, params)
    for r, (uid, _) in enumerate(ranked):
        fused[uid] += 1.0 / (60 + r + 1)
    raw = _rank(hx, q.question, {**params, "pool": 100})  # the untouched question keeps phrasing cues
    for r, (uid, _) in enumerate(raw):
        fused[uid] += 0.5 / (60 + r + 1)
    if params.get("sweep", True):
        for u in hx.units:
            if u.role != "user":
                continue
            ms = match_score(u.text, qa)
            if ms >= 1.0:
                fused[u.uid] += 0.004 * min(ms, 4.0) + (0.002 if _NUM.search(u.text) else 0.0)
    win = qa["window"]
    use_win = params.get("window", True) and window_applies(hx, win)
    if use_win:
        for uid in list(fused):
            if not _in_window(hx.units[uid].date, win):
                fused[uid] *= 0.15  # soft: still allowed if nothing else
    order = sorted(fused.items(), key=lambda x: -x[1])
    if params.get("rerank"):
        top = [uid for uid, _ in order[:60]]
        rr = dict(hx.rerank(q.question, top))
        lo = min(rr.values()) if rr else 0.0
        order = sorted(order, key=lambda x: -(x[1] + (0.01 * (rr.get(x[0], lo) - lo) if x[0] in rr else 0.0)))
    return order, use_win


_PAIR_STOP = {stem(w) for w in """compared comparison related typical average approximate approximately percentage previous
instead between something across ago older younger total different other several consecutive simultaneously excluding
including year years week weeks month months day days""".split()}


def abstain_score(hx, qa: dict, use_win: bool, n_hits_in_win: int) -> tuple[float, list[str]]:
    """Soft abstention: salient question terms that never appear in any user turn, empty window, missing conjunct."""
    why = []
    p = 0.05
    utext = hx.__dict__.setdefault("_a_multi_usertext", " ".join(u.text for u in hx.units if u.role == "user").lower())
    for t in qa["titles"] + qa["caps"] + qa["numunits"]:
        tl = t.lower()
        if tl not in utext and tl.replace("-", " ") not in utext:
            p += 0.35
            why.append(f"absent:{t}")
    for a, b in qa["conj"]:
        for side in (a, b):
            st = _stems(side)
            if st and not any(re.search(rf"\b{re.escape(x)}", utext) for x in st):
                p += 0.25
                why.append(f"conjunct-absent:{side}")
    # head noun compound of each phrase ("egg tarts", "chili peppers"): no user unit holds both words -> likely absent
    ustems = hx.__dict__.setdefault("_a_multi_ustems", [set(_stems(u.text)) for u in hx.units if u.role == "user"])
    for ph in qa["phrases"]:
        w = [x for x in _stems(ph) if x and not x.isdigit() and x not in _PAIR_STOP][-2:]
        if len(w) == 2 and not any(w[0] in us and w[1] in us for us in ustems):
            p += 0.25
            why.append(f"pair-absent:{' '.join(w)}")
    if use_win and n_hits_in_win == 0:
        p += 0.3
        why.append("window-empty")
    return min(p, 0.95), why


def build(q, hx, **params) -> EvidencePack:
    p = {**VARIANTS["bm25_tight"], **params}
    qa = analyze(q.question, q.date)
    order, use_win = select(q, hx, qa, p)
    win = qa["window"]
    # 1) rank sessions by their best two fused unit scores (+ fact-likeness prior)
    fused = dict(order)
    top = order[0][1] if order else 1.0
    by_sess: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for uid, sc in order:
        u = hx.units[uid]
        by_sess[u.session_id].append((uid, sc / (top or 1.0) + p.get("w_prior", 0.15) * fact_prior(u)))
    sscore = {}
    for sid, lst in by_sess.items():
        v = sorted((x[1] for x in lst), reverse=True)
        sscore[sid] = v[0] + 0.3 * (v[1] if len(v) > 1 else 0.0)
    sess_rank = sorted(sscore, key=lambda s: -sscore[s])[: p["max_sessions"]]
    # 2) inside each session choose units by fused score + lexical topic match + fact cues (not fused order alone:
    #    long on-topic sessions otherwise fill the slots with chit-chat and drop the "By the way, I ..." turn)
    chosen: dict[str, list[int]] = {}
    for sid in sess_rank:
        cands = []
        pool = list(by_sess[sid])
        if p.get("all_units", True):  # every user unit of a kept session competes (a "By the way, I ..." fact turn may
            seen = {uid for uid, _ in pool}  # share no word with the question and so be absent from the fused list)
            pool += [(uid, 0.0) for uid in _sess_user_uids(hx, sid) if uid not in seen]
        zl = []  # units with no lexical hit: only fact-like ones, at most zl_max per session when lexical hits exist
        for uid, sc in pool:
            u = hx.units[uid]
            lex = match_score(u.text, qa)
            fp = fact_prior(u)
            val = sc + p.get("w_within", 1.0) * fp + 0.4 * min(lex, 3.0)
            if lex > 0:
                cands.append((val, uid))
            elif sc > 0 or fp >= p.get("zl_prior", 0.6):
                zl.append((val, uid))
        cands.sort(reverse=True)
        zl.sort(reverse=True)
        zmax = p.get("zl_max", 2) if cands else p["per_session"]
        pick = cands[: p["per_session"]]
        pick += zl[: max(0, min(zmax, p["per_session"] - len(pick)))]
        chosen[sid] = [uid for _, uid in pick] or [uid for _, uid in zl[:1]] or [pool[0][0]]
    hint = hint_text(qa, use_win)
    # render, then add sessions in fused order until the budget is hit; output in chronological order
    # fill the budget with strongly matching sessions first (phrase / several head terms), then the rest in rank order
    lex = {sid: max(match_score(hx.units[i].text, qa) for i in chosen[sid]) for sid in sess_rank}
    gbest = max(lex.values(), default=0.0)
    strong = [sid for sid in sess_rank if gbest >= 1.5 and lex[sid] >= p.get("strong_frac", 0.75) * gbest
              and (not use_win or _in_window(hx.sess_by_id[sid]["date"], win))]  # never let off-window hits jump ahead
    sess_rank = strong + [sid for sid in sess_rank if sid not in strong]
    rendered: dict[str, tuple[str, float, list[str], list[int]]] = {}
    passages: list[Passage] = []
    kept: list[str] = []
    for sid in sess_rank:
        uids = sorted(chosen[sid])
        if p["render"] == "turns":
            tids = list(dict.fromkeys(hx.units[i].turn_id for i in uids))
            texts = []
            for t in hx.sess_by_id[sid]["turns"]:
                if t["turn_id"] in tids:
                    texts.append(f"user: {t['content'].strip()}")
            text, best, flags = "\n".join(texts), 1.0, []
        else:
            parts, best, flags = [], 0.0, set()
            for i in uids:
                t, b, f = _excerpt(hx, i, qa, p.get("ctx", 1), p.get("keep_frac", 0.6))
                parts.append(f"user: {t}")
                best = max(best, b)
                flags.update(f)
            text, flags = "\n".join(parts), sorted(flags)
        s = hx.sess_by_id[sid]
        cand = Passage(id=sid, date=s["date_str"], text=text)
        trial = passages + [cand]
        if message_chars(q, hint, trial) > p["budget"]:
            continue
        passages.append(cand)
        kept.append(sid)
        rendered[sid] = (text, best, flags, uids)
    passages.sort(key=lambda x: hx.sess_by_id[x.id]["sess_idx"])
    included = {hx.units[i].turn_id for sid in kept for i in rendered[sid][3]}
    # citations: sessions with a clean (own, done) lexical hit, best first; small set keeps CF1 up
    if p.get("cite", "ssfrac") == "ssfrac":
        # sessions whose score is close to the best one (dev and dev200: CF1 0.51 -> ~0.8 vs the clean-lexical rule)
        by_s = sorted(kept, key=lambda s: -sscore[s])
        best_s = sscore[by_s[0]] if by_s else 0.0
        cite = [s for s in by_s if sscore[s] >= p.get("cite_frac", 0.8) * best_s][: p.get("cite_k", 5)]
    else:
        clean = [sid for sid in kept if rendered[sid][1] >= 1.0 and not ({"other", "neg"} & set(rendered[sid][2]))]
        cite = clean[: p.get("cite_k", 6)] or kept[:3]
    n_in_win = sum(1 for sid in kept if win and _in_window(hx.sess_by_id[sid]["date"], win) and rendered[sid][1] >= 1.0)
    ap, why = abstain_score(hx, qa, use_win, n_in_win)
    if not any(rendered[sid][1] >= 1.0 for sid in kept):
        ap = min(0.95, ap + 0.2)
        why.append("no-lexical-hit")
    if ap >= 0.5 and why:
        hint += " Note: " + "; ".join(w.replace("absent:", "'") + ("' never appears in the history" if w.startswith("absent:") else "")
                                      for w in why if w.startswith("absent:")) + "." if any(w.startswith("absent:") for w in why) else ""
    return EvidencePack(question_id=q.question_id, variant=params.get("_variant", ""), route="multi",
                        passages=passages, hint=hint, cite_plan=cite, abstain_p=ap, included_turns=included,
                        computed=None,
                        debug={"ranking": [uid for uid, _ in order], "window": win[2] if win else None,
                               "window_used": use_win, "kind": qa["kind"], "abstain_why": why,
                               "query": " ".join(qa["content"] + qa["syn"]),
                               "sess": [[sid, round(sscore[sid], 4), round(rendered[sid][1], 2), rendered[sid][2]]
                                        for sid in kept], "inc": sorted(included)})


# ----------------------------------------------------------------------------------------------------------------
# Self-test (pure helpers only): python -m lenses.multi.a_tally
# ----------------------------------------------------------------------------------------------------------------
def _selftest() -> None:
    qd = datetime(2023, 9, 4)
    cases = {
        "How many books have I read since March?": "since March 1, 2023",
        "How many times did my systolic reading exceed 130 in August?": "August 2023",
        "How many times did I eat feijoada last month?": "August 2023 (last month)",
        "How much honey did I harvest in total from my two hives this summer?": "summer 2023",
        "How many rides from July to October?": "July-October 2022",
        "How many plants did I acquire in the last month?": "last 1 month(s) before 2023-09-04",
        "How many pieces of furniture did I buy in the past few months?": "last 3 month(s) before 2023-09-04",
        "How many weddings have I attended in this year?": "2023 (Jan 1 to now)",
        "How many different museums or galleries did I visit in the month of February?": "February 2023",
        "How many museums did I visit in December?": "December 2022",
        "What was the page count of the two novels I finished in January and March?": "January and March 2023",
        "How many laps do I swim each morning these days?": None,
    }
    bad = 0
    for qq, want in cases.items():
        w = time_window(qq, qd)
        got = w[2] if w else None
        ok = got == want
        bad += not ok
        print(("ok  " if ok else "BAD ") + f"{qq[:70]:70s} -> {got}")
    kinds = {"How much is left in my bakery fund after those transactions I told you about?": "diff",
             "How much have I spent on tango shoes in total?": "sum",
             "How many laps do I swim each morning these days?": "current",
             "How many books have I read since March?": "count",
             "How many times did my systolic reading exceed 130 in August?": "count",
             "What is the average age of me, my parents, and my grandparents?": "avg"}
    for qq, want in kinds.items():
        got = question_kind(qq)
        bad += got != want
        print(("ok  " if got == want else "BAD ") + f"kind {qq[:60]:60s} -> {got}")
    flags = {"My friend Valentina bought a pair of tango shoes for 1,800 pesos.": "other",
             "Carlos told me he has read three books since March.": "other",
             "I tried on a pair of tango shoes that cost 3,000 pesos but decided not to buy them.": "neg",
             "I bought a third pair of tango shoes on sale for 2,000 pesos.": None,
             "My sister has 9 Icelandic horses on her farm.": "other"}
    for s, want in flags.items():
        f = cue_flags(s)
        ok = (want in f) if want else not ({"other", "neg"} & set(f))
        bad += not ok
        print(("ok  " if ok else "BAD ") + f"cue {s[:60]:60s} -> {f}")
    qa = analyze("How much have I spent on tango shoes in total?", qd)
    print("analyze:", {k: qa[k] for k in ("kind", "head", "phrases", "syn", "conj")})
    s1 = match_score("By the way, I bought a pair of tango shoes from a shop in San Telmo for 1,800 pesos.", qa)
    s2 = match_score("We drilled molinetes for an hour.", qa)
    bad += not (s1 > 1.5 and s2 == 0)
    print(f"match tango={s1} other={s2}")
    qa = analyze("How many days did I spend in total traveling in Hawaii and in Seattle?", qd)
    print("conj/caps:", qa["conj"], qa["caps"])
    print("SELFTEST", "PASS" if not bad else f"FAIL ({bad})")


if __name__ == "__main__":
    _selftest()
