"""Turn -> sentence-aligned chunks (Units). One chunking for everyone, so embedding caches are shared."""
from __future__ import annotations

import re

from core.data import parse_date
from core.types import Unit

CHUNK_CHARS = 700
CTX_CHARS = 220
_SENT = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[A-Z0-9*#-])|\n\s*\n|\n(?=\s*(?:[-*•]|\d+[.)])\s)")


def sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _SENT.split(text) if p and p.strip()]
    out = []
    for p in parts:  # hard-split monsters (tables, code)
        while len(p) > CHUNK_CHARS * 1.5:
            cut = p.rfind(" ", 0, CHUNK_CHARS)
            cut = cut if cut > CHUNK_CHARS // 2 else CHUNK_CHARS
            out.append(p[:cut].strip())
            p = p[cut:].strip()
        if p:
            out.append(p)
    return out


def chunk(text: str, limit: int = CHUNK_CHARS) -> list[str]:
    chunks, cur = [], ""
    for s in sentences(text):
        if cur and len(cur) + 1 + len(s) > limit:
            chunks.append(cur)
            cur = s
        else:
            cur = f"{cur} {s}" if cur else s
    if cur:
        chunks.append(cur)
    return chunks or [text.strip()]


def last_sentence(text: str) -> str:
    ss = sentences(text)
    s = ss[-1] if ss else text
    return s[-CTX_CHARS:]


def build_units(history: dict) -> tuple[list[Unit], list[dict]]:
    units: list[Unit] = []
    sessions = []
    for si, s in enumerate(history["sessions"]):
        d = parse_date(s["date"])
        sessions.append({"session_id": s["session_id"], "date": d, "date_str": s["date"], "sess_idx": si,
                         "turns": s["turns"]})
        prev = ""
        for ti, t in enumerate(s["turns"]):
            for ci, c in enumerate(chunk(t["content"])):
                units.append(Unit(uid=len(units), session_id=s["session_id"], sess_idx=si, date=d, date_str=s["date"],
                                  turn_id=t["turn_id"], turn_idx=ti, role=t["role"], chunk_idx=ci, text=c,
                                  ctx=last_sentence(prev) if (ci == 0 and prev) else ""))
            prev = t["content"]
    return units, sessions
