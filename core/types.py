"""Shared data contract between the core, the lenses and the harness. Changing it is the integrator's call."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

TYPES = ("single-session-user", "single-session-assistant", "multi-session", "knowledge-update", "temporal-reasoning")
SHORT = {"single-session-user": "ssu", "single-session-assistant": "ssa", "multi-session": "multi",
         "knowledge-update": "ku", "temporal-reasoning": "temporal"}
LONG = {v: k for k, v in SHORT.items()}


@dataclass
class Question:
    question_id: str
    history_id: str
    question: str
    date: datetime          # question date (wall clock, no tz)
    date_str: str           # as written in questions.jsonl


@dataclass
class Unit:
    """Smallest retrievable piece: a chunk of one turn (default <= ~700 chars, sentence-aligned)."""
    uid: int                # index in HistoryIndex.units (chronological order)
    session_id: str
    sess_idx: int           # position of the session in this history (0 = oldest)
    date: datetime          # session date in THIS history
    date_str: str
    turn_id: str
    turn_idx: int           # position of the turn in its session
    role: str               # "user" | "assistant"
    chunk_idx: int          # position of the chunk in its turn
    text: str
    ctx: str = ""           # for chunk 0 of a turn: last sentence of the previous turn (often the assistant's question)

    @property
    def index_text(self) -> str:
        """What BM25 and the embedder see."""
        return f"{self.ctx} {self.text}".strip() if self.ctx else self.text


@dataclass
class Passage:
    id: str | None          # session_id (a citation); None = our own computed note (never a citation)
    date: str | None        # the session's date string in this history
    text: str


@dataclass
class EvidencePack:
    question_id: str
    variant: str
    route: str                                   # question type this pack was built for
    passages: list[Passage]
    hint: str = ""                               # appended to the question text sent to Reactor
    cite_plan: list[str] | None = None           # session ids we expect Reactor to cite (default: all passage ids)
    abstain_p: float = 0.0                       # our estimate that the history does NOT answer the question
    included_turns: set[str] = field(default_factory=set)   # turn_ids (fully or partly) present in passages
    computed: str | None = None                  # an answer our code computed (days, count, value), if any
    debug: dict = field(default_factory=dict)    # free-form; "ranking": [uid, ...] enables rank diagnostics

    def session_ids(self) -> list[str]:
        out = []
        for p in self.passages:
            if p.id and p.id not in out:
                out.append(p.id)
        return out

    def planned_citations(self) -> list[str]:
        return list(self.cite_plan) if self.cite_plan is not None else self.session_ids()
