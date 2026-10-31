"""Turn an uploaded file into a Librarian history (the challenge format), so the demo can index any archive.

Accepted inputs:
- the challenge format: one history as JSON, or a histories.jsonl (its first history);
- a ChatGPT export (conversations.json), a list of sessions, or one list of {role, content} messages;
- a WhatsApp-style chat export ("12/05/2023, 14:03 - Anna: ..."): one session per day, every turn a "user" turn
  prefixed with the speaker's name;
- plain text or Markdown: a heading that is a date ("## 3 May 2024") starts a new session; without dates, paragraphs
  are grouped into ~3,000-character sessions dated at upload time.
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timedelta

from dateutil import parser as dparser

from core.data import DATE_FMT

MAX_CHARS = 6_000_000
SESSION_CHARS = 3000
ASSISTANT_ROLES = {"assistant", "model", "bot", "ai", "gpt", "chatgpt", "claude"}
SKIP_ROLES = {"system", "tool", "function"}
ASSISTANT_PREFIX = re.compile(r"^(assistant|ai|bot|chatgpt|gpt|claude|model)\s*:\s*", re.I)
USER_PREFIX = re.compile(r"^(user|me|human|you)\s*:\s*", re.I)
CHAT_LINE = re.compile(r"^\[?(\d{1,4})[./-](\d{1,2})[./-](\d{1,4}),?\s+(\d{1,2}:\d{2})(?::\d{2})?\s*([AaPp]\.?\s?[Mm]\.?)?\]?"
                       r"\s*(?:-\s*)?([^:\n]{1,40}?):\s(.*)$")
DATE_WORDS = re.compile(r"\b(19|20)\d{2}\b|\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b", re.I)


def _naive(d: datetime) -> datetime:
    return d.replace(tzinfo=None) if d.tzinfo else d


def _date(value, fallback: datetime) -> datetime:
    if isinstance(value, (int, float)) and value > 0:
        return datetime.fromtimestamp(value / 1000 if value > 1e11 else value)
    if isinstance(value, str) and value.strip():
        try:
            return datetime.strptime(value.strip(), DATE_FMT)
        except ValueError:
            pass
        try:
            return _naive(dparser.parse(value, fuzzy=True, default=fallback.replace(hour=12, minute=0)))
        except (ValueError, OverflowError):
            pass
    return fallback


def _role(raw) -> str | None:
    r = str(raw or "user").lower()
    if r in SKIP_ROLES:
        return None
    return "assistant" if r in ASSISTANT_ROLES else "user"


def _text(content) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):   # OpenAI-style parts: strings or {"type": "text", "text": ...}
        return "\n".join(p if isinstance(p, str) else str(p.get("text") or "") for p in content
                         if isinstance(p, (str, dict))).strip()
    if isinstance(content, dict):
        return _text(content.get("parts") or content.get("text") or "")
    return ""


# ---- structured inputs ----------------------------------------------------------------------------------------------
def _from_messages(msgs: list, date: datetime, title: str = "") -> tuple[datetime, str, list]:
    turns = []
    for m in msgs:
        if not isinstance(m, dict):
            continue
        role = _role(m.get("role") or (m.get("author") or {}).get("role") or m.get("sender"))
        text = _text(m.get("content") if "content" in m else m.get("text"))
        if role and text:
            turns.append((role, text))
    return date, title, turns


def _chatgpt(convs: list, now: datetime) -> list:
    sessions = []
    for c in convs:
        mapping = c.get("mapping") or {}
        chain, node = [], c.get("current_node")
        while node in mapping and len(chain) < 100_000:
            chain.append(mapping[node])
            node = mapping[node].get("parent")
        nodes = list(reversed(chain)) if chain else sorted(
            mapping.values(), key=lambda n: ((n.get("message") or {}).get("create_time") or 0))
        msgs = [n["message"] for n in nodes if n.get("message")]
        sessions.append(_from_messages(msgs, _date(c.get("create_time"), now), c.get("title") or ""))
    return sessions


def _session_list(items: list, now: datetime) -> list:
    out = []
    for i, s in enumerate(items):
        d = _date(s.get("date") or s.get("timestamp") or s.get("created_at") or s.get("create_time"),
                  now + timedelta(minutes=i))
        out.append(_from_messages(s.get("turns") or s.get("messages") or [], d, s.get("title") or ""))
    return out


def _structured(obj, now: datetime) -> tuple[str, list]:
    if isinstance(obj, dict) and isinstance(obj.get("sessions"), list):
        return "challenge history", _session_list(obj["sessions"], now)
    if isinstance(obj, dict) and isinstance(obj.get("messages"), list):
        return "message list", [_from_messages(obj["messages"], _date(obj.get("date"), now))]
    if isinstance(obj, list) and obj and all(isinstance(x, dict) for x in obj):
        if "mapping" in obj[0]:
            return "ChatGPT export", _chatgpt(obj, now)
        if "turns" in obj[0] or "messages" in obj[0]:
            return "session list", _session_list(obj, now)
        if "role" in obj[0] or "author" in obj[0]:
            return "message list", [_from_messages(obj, now)]
    raise ValueError("unrecognised JSON: expected a history with 'sessions', a ChatGPT export, or a message list")


# ---- plain text ---------------------------------------------------------------------------------------------------
def _chat_export(lines: list[str], now: datetime) -> list:
    rows, cur = [], None
    for line in lines:
        m = CHAT_LINE.match(line.strip())
        if m:
            cur = list(m.groups())
            rows.append(cur)
        elif cur is not None and line.strip():
            cur[6] += "\n" + line.strip()
    dayfirst = not any(int(r[1]) > 12 for r in rows)   # "05/13/2023" means month first; otherwise assume day first
    by_day: dict[str, list] = {}
    for d1, d2, d3, hm, ampm, name, msg in rows:
        name = name.strip()
        try:
            d = _naive(dparser.parse(f"{d1}/{d2}/{d3} {hm} {ampm or ''}".strip(), dayfirst=dayfirst and len(d1) <= 2))
        except (ValueError, OverflowError):
            d = now
        # every human turn is "user" with the speaker's name: on LoCoMo, labelling the second person "assistant" cost
        # 6.6 S (runs/ood/LOCOMO.md)
        by_day.setdefault(d.strftime("%Y-%m-%d"), []).append((d, "user", f"{name}: {msg.strip()}"))
    return [(rows_[0][0], "", [(r, t) for _, r, t in rows_]) for rows_ in by_day.values()]


def _date_heading(par: str) -> datetime | None:
    s = par.strip().strip("#*_=- ").strip()
    if "\n" in par.strip() or not (4 <= len(s) <= 40) or not DATE_WORDS.search(s):
        return None
    try:
        return _naive(dparser.parse(s, fuzzy=False, default=datetime(2000, 1, 1, 12, 0)))
    except (ValueError, OverflowError):
        return None


def _plain(text: str, now: datetime) -> tuple[str, list]:
    lines = text.splitlines()
    nonblank = [x for x in lines if x.strip()]
    if sum(1 for x in nonblank if CHAT_LINE.match(x.strip())) >= max(3, 0.3 * len(nonblank)):
        return "chat export", _chat_export(lines, now)
    sessions: list = []
    cur: list = []
    cur_date, cur_chars, dated = now, 0, False

    def flush():
        if cur:
            sessions.append((cur_date, "", list(cur)))
            cur.clear()

    for par in re.split(r"\n\s*\n", text):
        par = par.strip()
        if not par:
            continue
        d = _date_heading(par)
        if d is not None:
            flush()
            cur_date, cur_chars, dated = d, 0, True
            continue
        if not dated and cur_chars > SESSION_CHARS:
            flush()
            cur_date, cur_chars = cur_date + timedelta(minutes=1), 0
        role, body = "user", par
        if ASSISTANT_PREFIX.match(par):
            role, body = "assistant", ASSISTANT_PREFIX.sub("", par, count=1)
        elif USER_PREFIX.match(par):
            body = USER_PREFIX.sub("", par, count=1)
        cur.append((role, body))
        cur_chars += len(body)
    flush()
    return ("dated notes" if dated else "plain text"), sessions


# ---- entry point --------------------------------------------------------------------------------------------------
def to_history(raw: str, name: str = "upload") -> tuple[dict, dict]:
    """Returns (history in the challenge format, summary for the UI). Raises ValueError on unusable input."""
    raw = raw.lstrip("﻿")
    if not raw.strip():
        raise ValueError("the file is empty")
    now = datetime.now().replace(second=0, microsecond=0)
    stripped = raw.lstrip()
    fmt, sessions = None, None
    if stripped[:1] in "[{":
        try:
            fmt, sessions = _structured(json.loads(raw), now)
        except json.JSONDecodeError:
            first = next((x for x in raw.splitlines() if x.strip()), "")
            try:
                fmt, sessions = _structured(json.loads(first), now)
                fmt += " (first line of a JSONL file)"
            except (json.JSONDecodeError, ValueError):
                fmt = None
    if fmt is None:
        fmt, sessions = _plain(raw, now)
    sessions = sorted((s for s in sessions if s[2]), key=lambda s: s[0])
    chars = sum(len(t) for s in sessions for _, t in s[2])
    if not sessions:
        raise ValueError("no messages found in the file")
    if chars > MAX_CHARS:
        raise ValueError(f"the archive has {chars:,} characters; the demo limit is {MAX_CHARS:,}")
    hid = "u_" + hashlib.sha1(raw.encode("utf-8", "replace")).hexdigest()[:12]
    hist = {"history_id": hid, "sessions": [
        {"session_id": f"s_{hid[2:8]}{i:04d}", "date": d.strftime(DATE_FMT),
         "turns": [{"turn_id": f"t_{hid[2:8]}{i:04d}{j:04d}", "role": r, "content": t} for j, (r, t) in enumerate(turns)]}
        for i, (d, _, turns) in enumerate(sessions)]}
    info = {"history_id": hid, "name": name, "format": fmt, "sessions": len(sessions),
            "turns": sum(len(s[2]) for s in sessions), "chars": chars,
            "first_date": hist["sessions"][0]["date"], "last_date": hist["sessions"][-1]["date"],
            "assistant_turns": sum(1 for s in sessions for r, _ in s[2] if r == "assistant")}
    return hist, info
