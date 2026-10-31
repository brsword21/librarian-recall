"""Basic date grammar: explicit dates (ISO, 'June 27, 2023', 'June 27th', '23 April 2023', 'Jun 27') and common
relative expressions resolved against a reference datetime (the session date). Lens agents extend this in their files.

extract_dates(text, ref) -> list[DateHit]; days_between(a, b) -> int
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta

MONTHS = {m: i for i, m in enumerate(
    "january february march april may june july august september october november december".split(), 1)}
MONTHS.update({m[:3]: i for m, i in list(MONTHS.items())})
MONTHS["sept"] = 9
WEEKDAYS = {d: i for i, d in enumerate("monday tuesday wednesday thursday friday saturday sunday".split())}
NUMWORDS = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve".split())}
NUMWORDS.update({"a": 1, "an": 1, "a couple of": 2, "couple of": 2, "a few": 3, "few": 3, "several": 3})

_MON = r"(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sept|sep|oct|nov|dec)\.?"
_ORD = r"(\d{1,2})(?:st|nd|rd|th)?"
ISO = re.compile(r"\b(20\d\d|19\d\d)-(\d\d)-(\d\d)\b")
MDY = re.compile(rf"\b{_MON}\s+{_ORD}(?:,?\s+(\d{{4}}))?\b", re.I)
DMY = re.compile(rf"\b{_ORD}\s+(?:of\s+)?{_MON}(?:,?\s+(\d{{4}}))?\b", re.I)
SLASH = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b")
REL_AGO = re.compile(r"\b(\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|a couple of|a few|few|several)"
                     r"\s+(day|week|month|year)s?\s+ago\b", re.I)
REL_WD = re.compile(r"\b(last|this|next|this past|this coming|past|coming)\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", re.I)
REL_ON_WD = re.compile(r"\b(?:on|since)\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", re.I)
REL_SIMPLE = re.compile(r"\b(today|yesterday|tomorrow|the day before yesterday|last night|tonight|this morning)\b", re.I)
REL_PERIOD = re.compile(r"\b(last|this|next)\s+(week|month|year|weekend)\b", re.I)


@dataclass
class DateHit:
    text: str
    start: int
    end: int
    date: date
    kind: str          # iso | mdy | dmy | slash | ago | weekday | simple | period
    explicit_year: bool


def _mk(y: int, m: int, d: int) -> date | None:
    try:
        return date(y, m, d)
    except ValueError:
        return None


def _infer_year(m: int, d: int, ref: datetime) -> date | None:
    """No year given: pick the occurrence closest to the reference date."""
    cands = [x for x in (_mk(ref.year + k, m, d) for k in (-1, 0, 1)) if x]
    return min(cands, key=lambda x: abs((x - ref.date()).days)) if cands else None


def _num(s: str) -> int:
    s = s.lower()
    return int(s) if s.isdigit() else NUMWORDS.get(s, 1)


def extract_dates(text: str, ref: datetime | None = None, relative: bool = True) -> list[DateHit]:
    hits: list[DateHit] = []
    taken: list[tuple[int, int]] = []

    def free(a: int, b: int) -> bool:
        return all(b <= x or a >= y for x, y in taken)

    def add(m, d: date | None, kind: str, explicit: bool):
        if d and free(m.start(), m.end()):
            hits.append(DateHit(m.group(0), m.start(), m.end(), d, kind, explicit))
            taken.append((m.start(), m.end()))

    for m in ISO.finditer(text):
        add(m, _mk(int(m.group(1)), int(m.group(2)), int(m.group(3))), "iso", True)
    for m in SLASH.finditer(text):
        add(m, _mk(int(m.group(3)), int(m.group(1)), int(m.group(2))), "slash", True)
    for m in MDY.finditer(text):
        mon, day, yr = MONTHS[m.group(1).lower().rstrip(".")], int(m.group(2)), m.group(3)
        d = _mk(int(yr), mon, day) if yr else (_infer_year(mon, day, ref) if ref else None)
        add(m, d, "mdy", bool(yr))
    for m in DMY.finditer(text):
        day, mon, yr = int(m.group(1)), MONTHS[m.group(2).lower().rstrip(".")], m.group(3)
        d = _mk(int(yr), mon, day) if yr else (_infer_year(mon, day, ref) if ref else None)
        add(m, d, "dmy", bool(yr))
    if relative and ref is not None:
        r = ref.date()
        for m in REL_SIMPLE.finditer(text):
            w = m.group(1).lower()
            off = {"today": 0, "tonight": 0, "this morning": 0, "yesterday": -1, "last night": -1, "tomorrow": 1,
                   "the day before yesterday": -2}[w]
            add(m, r + timedelta(days=off), "simple", False)
        for m in REL_AGO.finditer(text):
            n, unit = _num(m.group(1)), m.group(2).lower()
            days = {"day": 1, "week": 7, "month": 30, "year": 365}[unit] * n
            add(m, r - timedelta(days=days), "ago", False)
        for m in REL_WD.finditer(text):
            which, wd = m.group(1).lower(), WEEKDAYS[m.group(2).lower()]
            delta = (r.weekday() - wd) % 7
            if which in ("last", "past", "this past"):
                d = r - timedelta(days=delta or 7)
            elif which in ("next", "coming", "this coming"):
                d = r + timedelta(days=((wd - r.weekday()) % 7) or 7)
            else:  # "this <weekday>": nearest, prefer past within the week
                d = r - timedelta(days=delta) if delta <= 3 else r + timedelta(days=(wd - r.weekday()) % 7)
            add(m, d, "weekday", False)
        for m in REL_ON_WD.finditer(text):
            wd = WEEKDAYS[m.group(1).lower()]
            delta = (r.weekday() - wd) % 7
            add(m, r - timedelta(days=delta), "weekday", False)
        for m in REL_PERIOD.finditer(text):
            which, unit = m.group(1).lower(), m.group(2).lower()
            sign = {"last": -1, "this": 0, "next": 1}[which]
            days = {"week": 7, "weekend": 7, "month": 30, "year": 365}[unit] * sign
            add(m, r + timedelta(days=days), "period", False)
    hits.sort(key=lambda h: h.start)
    return hits


def days_between(a, b) -> int:
    a = a.date() if isinstance(a, datetime) else a
    b = b.date() if isinstance(b, datetime) else b
    return abs((b - a).days)


if __name__ == "__main__":
    ref = datetime(2023, 6, 1, 18, 54)
    for s in ["My son's school play is on 2023-06-10, so", "Our anniversary was June 21, 2022.",
              "on 23 April 2023 we", "the appointment is June 24th", "I went there two weeks ago",
              "last Saturday I baked", "May I ask", "since May 5 I have", "on Friday we met", "yesterday"]:
        print(s, "->", [(h.text, h.date.isoformat(), h.kind) for h in extract_dates(s, ref)])
