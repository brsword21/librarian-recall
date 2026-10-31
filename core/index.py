"""Per-history retrieval: BM25 (bm25s), dense (bge-small via the model server cache), RRF hybrid, cross-encoder rerank."""
from __future__ import annotations

from collections import OrderedDict, defaultdict

import bm25s
import numpy as np
import Stemmer

from core import models
from core.data import history
from core.types import Unit
from core.units import build_units

_STEMMER = Stemmer.Stemmer("english")


def tokenize(texts: list[str]):
    return bm25s.tokenize(texts, stopwords="en", stemmer=_STEMMER, show_progress=False)


class HistoryIndex:
    _cache: "OrderedDict[tuple[str, str], HistoryIndex]" = OrderedDict()
    MAX_CACHED = 4

    @classmethod
    def load(cls, split: str, history_id: str) -> "HistoryIndex":
        key = (split, history_id)
        if key in cls._cache:
            cls._cache.move_to_end(key)
            return cls._cache[key]
        hx = cls(split, history_id, history(split, history_id))
        cls._cache[key] = hx
        while len(cls._cache) > cls.MAX_CACHED:
            cls._cache.popitem(last=False)
        return hx

    def __init__(self, split: str, history_id: str, hist: dict):
        self.split, self.history_id = split, history_id
        self.units: list[Unit]
        self.units, self.sessions = build_units(hist)
        self.sess_by_id = {s["session_id"]: s for s in self.sessions}
        self._turn_units: dict[str, list[int]] = defaultdict(list)
        for u in self.units:
            self._turn_units[u.turn_id].append(u.uid)
        self._bm25: bm25s.BM25 | None = None
        self._vecs: dict[int, np.ndarray] = {}

    # ---- navigation -------------------------------------------------------------------------------------------
    def turn_units(self, turn_id: str) -> list[int]:
        return list(self._turn_units.get(turn_id, []))

    def neighbors(self, uid: int, before: int = 1, after: int = 1) -> list[int]:
        u = self.units[uid]
        out = []
        for j in range(max(0, uid - before), min(len(self.units), uid + after + 1)):
            if self.units[j].session_id == u.session_id:
                out.append(j)
        return out

    def _pool(self, roles, uids=None) -> list[int]:
        pool = range(len(self.units)) if uids is None else uids
        if roles:
            roles = {roles} if isinstance(roles, str) else set(roles)
            return [i for i in pool if self.units[i].role in roles]
        return list(pool)

    # ---- lexical ----------------------------------------------------------------------------------------------
    def bm25_scores(self, query: str) -> np.ndarray:
        if self._bm25 is None:
            self._bm25 = bm25s.BM25()
            self._bm25.index(tokenize([u.index_text for u in self.units]), show_progress=False)
        qt = tokenize([query])
        ids = [qt.vocab_inv[i] for i in qt.ids[0]] if hasattr(qt, "vocab_inv") else None
        if ids is None:
            inv = {v: k for k, v in qt.vocab.items()}
            ids = [inv[i] for i in qt.ids[0]]
        if not ids:
            return np.zeros(len(self.units), dtype=np.float32)
        return np.asarray(self._bm25.get_scores(ids), dtype=np.float32)

    def bm25(self, query: str, k: int = 200, roles=None, uids=None) -> list[tuple[int, float]]:
        s = self.bm25_scores(query)
        pool = self._pool(roles, uids)
        pool = [i for i in pool if s[i] > 0]
        pool.sort(key=lambda i: -s[i])
        return [(i, float(s[i])) for i in pool[:k]]

    # ---- dense ------------------------------------------------------------------------------------------------
    def vectors(self, uids: list[int]) -> np.ndarray:
        missing = [i for i in uids if i not in self._vecs]
        if missing:
            vecs = models.embed([self.units[i].index_text for i in missing])
            for i, v in zip(missing, vecs):
                self._vecs[i] = v
        return np.stack([self._vecs[i] for i in uids]) if uids else np.zeros((0, models.DIM), np.float32)

    def dense(self, query: str, k: int = 200, roles=None, uids=None) -> list[tuple[int, float]]:
        pool = self._pool(roles, uids)
        if not pool:
            return []
        qv = models.embed([query], query=True)[0]
        s = self.vectors(pool) @ qv
        order = np.argsort(-s)[:k]
        return [(pool[j], float(s[j])) for j in order]

    # ---- hybrid + rerank --------------------------------------------------------------------------------------
    def hybrid(self, query: str, k: int = 100, roles=None, w_bm25: float = 1.0, w_dense: float = 1.0,
               rrf_k: int = 60, pool: int = 300, uids=None) -> list[tuple[int, float]]:
        score: dict[int, float] = defaultdict(float)
        if w_bm25:
            for r, (i, _) in enumerate(self.bm25(query, k=pool, roles=roles, uids=uids)):
                score[i] += w_bm25 / (rrf_k + r + 1)
        if w_dense:
            for r, (i, _) in enumerate(self.dense(query, k=pool, roles=roles, uids=uids)):
                score[i] += w_dense / (rrf_k + r + 1)
        return sorted(score.items(), key=lambda x: -x[1])[:k]

    def rerank(self, query: str, uids: list[int]) -> list[tuple[int, float]]:
        uids = list(dict.fromkeys(uids))
        s = models.rerank(query, [self.units[i].index_text for i in uids])
        return sorted(zip(uids, map(float, s)), key=lambda x: -x[1])

    @staticmethod
    def rerank_texts(query: str, texts: list[str]) -> list[float]:
        """Cross-encoder logits for arbitrary strings (sentences, trimmed snippets), in input order. Same sqlite cache as
        rerank() (keyed by model, query and text), so repeated sentence-level calls are free."""
        return [float(x) for x in models.rerank(query, list(texts))] if texts else []

    @staticmethod
    def dense_texts(query: str, texts: list[str]) -> list[float]:
        """bge-small cosine between the query and arbitrary strings, in input order (cached embeddings)."""
        if not texts:
            return []
        qv = models.embed([query], query=True)[0]
        return [float(x) for x in models.embed(list(texts)) @ qv]
