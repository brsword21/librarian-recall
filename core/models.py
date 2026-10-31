"""Client side of the shared model server: embeddings and cross-encoder scores, read from the sqlite cache directly
and computed by the server (core.model_server) on a miss. No torch in this module."""
from __future__ import annotations

import base64
import hashlib
import json
import os
import sqlite3
import time
import urllib.error
import urllib.request

import numpy as np

from core.paths import CACHE

DB = CACHE / "models.sqlite"
URL = os.environ.get("LIBRARIAN_MODELS", "http://127.0.0.1:8765")
DIM = 384
EMB_MODEL = "BAAI/bge-small-en-v1.5"
RR_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def sha(*parts: str) -> str:
    return hashlib.sha1("\x00".join(parts).encode("utf-8")).hexdigest()


def connect(readonly: bool = True) -> sqlite3.Connection:
    DB.parent.mkdir(parents=True, exist_ok=True)
    if readonly and DB.exists():
        con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True, timeout=30, check_same_thread=False)
    else:
        con = sqlite3.connect(DB, timeout=30, check_same_thread=False)
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("CREATE TABLE IF NOT EXISTS emb (h TEXT PRIMARY KEY, v BLOB)")
        con.execute("CREATE TABLE IF NOT EXISTS rr (h TEXT PRIMARY KEY, s REAL)")
        con.commit()
    return con


_con: sqlite3.Connection | None = None


def _db() -> sqlite3.Connection:
    global _con
    if _con is None:
        if not DB.exists():
            connect(readonly=False).close()
        _con = connect(readonly=True)
    return _con


def _post(path: str, payload: dict, retries: int = 30) -> dict:
    data = json.dumps(payload).encode("utf-8")
    for i in range(retries):
        try:
            req = urllib.request.Request(URL + path, data=data, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=1800) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"model server error {exc.code}: {exc.read()[:500]!r}") from exc
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            if i == retries - 1:
                raise RuntimeError(f"model server not reachable at {URL}: start `python -m core.model_server`")
            time.sleep(min(2 * (i + 1), 10))
    raise AssertionError


def _decode(b64: str, n: int) -> np.ndarray:
    return np.frombuffer(base64.b64decode(b64), dtype=np.float16).reshape(n, DIM).astype(np.float32)


def _lookup(table: str, keys: list[str]) -> dict[str, bytes | float]:
    out: dict = {}
    con = _db()
    col = "v" if table == "emb" else "s"
    for i in range(0, len(keys), 500):
        part = keys[i:i + 500]
        q = f"SELECT h, {col} FROM {table} WHERE h IN ({','.join('?' * len(part))})"
        for h, v in con.execute(q, part):
            out[h] = v
    return out


def embed(texts: list[str], query: bool = False) -> np.ndarray:
    """L2-normalised float32 embeddings, one row per text (bge-small; query prefix added for queries)."""
    if not texts:
        return np.zeros((0, DIM), dtype=np.float32)
    full = [(QUERY_PREFIX + t) if query else t for t in texts]
    keys = [sha(EMB_MODEL, t) for t in full]
    found = _lookup("emb", list(dict.fromkeys(keys)))
    missing = [i for i, k in enumerate(keys) if k not in found]
    if missing:
        uniq = list(dict.fromkeys(full[i] for i in missing))
        for j in range(0, len(uniq), 512):
            batch = uniq[j:j + 512]
            res = _post("/embed", {"texts": batch})
            vecs = np.frombuffer(base64.b64decode(res["vecs"]), dtype=np.float16).reshape(len(batch), DIM)
            for t, v in zip(batch, vecs):
                found[sha(EMB_MODEL, t)] = v.tobytes()
    return np.stack([np.frombuffer(found[k], dtype=np.float16) for k in keys]).astype(np.float32)


def rerank(query: str, texts: list[str]) -> np.ndarray:
    """Cross-encoder relevance logits for (query, text) pairs."""
    if not texts:
        return np.zeros(0, dtype=np.float32)
    keys = [sha(RR_MODEL, query, t) for t in texts]
    found = _lookup("rr", list(dict.fromkeys(keys)))
    missing = list(dict.fromkeys(t for t, k in zip(texts, keys) if k not in found))
    for j in range(0, len(missing), 256):
        batch = missing[j:j + 256]
        res = _post("/rerank", {"query": query, "texts": batch})
        for t, s in zip(batch, res["scores"]):
            found[sha(RR_MODEL, query, t)] = s
    return np.array([found[k] for k in keys], dtype=np.float32)
