"""Librarian: from months of chat history, the few passages that answer a question.

    from librarian import Librarian
    lib = Librarian()
    hx = lib.index(history)                      # a history dict, see README "Input format"
    pack = lib.pack(hx, "What is my dog's name?")
    print(pack["prompt"])                        # ready for any LLM

Needs the model server (python -m core.model_server) for embeddings and reranking.
"""
from __future__ import annotations

import time
from datetime import datetime

from core.data import DATE_FMT
from core.index import HistoryIndex
from core.loader import load_lens
from core.pack import render_prompt
from core.types import Question

DEFAULT_CONFIG = "configs/best.json"


class Librarian:
    def __init__(self, config: str = DEFAULT_CONFIG):
        self.config = config
        self._pipe = load_lens("lenses/pipeline.py")

    @staticmethod
    def index(history: dict, namespace: str = "api") -> HistoryIndex:
        """Splits the history into retrievable units. BM25 and embeddings are built lazily on the first question,
        or up front with warm()."""
        return HistoryIndex(namespace, history["history_id"], history)

    @staticmethod
    def warm(hx: HistoryIndex, step: int = 256, progress=None) -> None:
        """Builds BM25 and embeds every unit now, so the first question is as fast as the next ones."""
        hx.bm25_scores("warm up")
        uids = list(range(len(hx.units)))
        for i in range(0, len(uids), step):
            hx.vectors(uids[i:i + step])
            if progress:
                progress(min(1.0, (i + step) / max(1, len(uids))))

    def pack(self, hx: HistoryIndex, question: str, question_date: str | datetime | None = None,
             question_id: str = "q") -> dict:
        """The evidence pack for one question. question_date ("YYYY/MM/DD (Ddd) HH:MM" or a datetime) is "now" for
        the question; it defaults to the current time."""
        if question_date is None:
            d = datetime.now().replace(second=0, microsecond=0)
        elif isinstance(question_date, datetime):
            d = question_date
        else:
            d = datetime.strptime(question_date, DATE_FMT)
        q = Question(question_id, hx.history_id, question, d, d.strftime(DATE_FMT))
        t0 = time.perf_counter()
        p = self._pipe.build(q, hx, config=self.config)
        ms = int((time.perf_counter() - t0) * 1000)
        return {
            "question": question, "question_date": q.date_str, "route": p.route,
            "passages": [{"label": f"E{i}", "id": x.id, "date": x.date, "text": x.text}
                         for i, x in enumerate(p.passages, 1)],
            "notes": p.hint, "computed": p.computed, "cite_plan": p.planned_citations(),
            "abstain_p": round(float(p.abstain_p or 0.0), 4),
            "chars": sum(len(x.text) for x in p.passages), "build_ms": ms,
            "prompt": render_prompt(question, p.passages, p.hint, q.date_str),
        }

