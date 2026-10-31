"""The only process that loads torch: serves embeddings and cross-encoder scores over localhost and fills the shared
sqlite cache (cache/models.sqlite). Start once:  .venv/bin/python -m core.model_server
"""
from __future__ import annotations

import base64
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer

from core.models import EMB_MODEL, RR_MODEL, connect, sha

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
LOCK = threading.Lock()
STATS = {"embedded": 0, "reranked": 0, "since": time.time()}

emb_model = SentenceTransformer(EMB_MODEL, device=DEVICE)
emb_model.max_seq_length = 320
rr_model = CrossEncoder(RR_MODEL, device=DEVICE, max_length=512)
db = connect(readonly=False)


def do_embed(texts: list[str]) -> np.ndarray:
    with LOCK:
        vecs = emb_model.encode(texts, batch_size=64, normalize_embeddings=True, convert_to_numpy=True,
                                show_progress_bar=False).astype(np.float16)
        db.executemany("INSERT OR REPLACE INTO emb (h, v) VALUES (?, ?)",
                       [(sha(EMB_MODEL, t), v.tobytes()) for t, v in zip(texts, vecs)])
        db.commit()
        STATS["embedded"] += len(texts)
    return vecs


def do_rerank(query: str, texts: list[str]) -> list[float]:
    with LOCK:
        scores = rr_model.predict([(query, t) for t in texts], batch_size=64, show_progress_bar=False)
        scores = [float(s) for s in np.asarray(scores).reshape(-1)]
        db.executemany("INSERT OR REPLACE INTO rr (h, s) VALUES (?, ?)",
                       [(sha(RR_MODEL, query, t), s) for t, s in zip(texts, scores)])
        db.commit()
        STATS["reranked"] += len(texts)
    return scores


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def _send(self, obj: dict, code: int = 200) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._send({"ok": True, "device": DEVICE, **STATS})

    def do_POST(self):
        try:
            payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            if self.path == "/embed":
                vecs = do_embed(payload["texts"])
                self._send({"vecs": base64.b64encode(vecs.tobytes()).decode()})
            elif self.path == "/rerank":
                self._send({"scores": do_rerank(payload["query"], payload["texts"])})
            else:
                self._send({"error": "unknown path"}, 404)
        except Exception as exc:  # report, keep serving
            self._send({"error": repr(exc)}, 500)


if __name__ == "__main__":
    print(f"model server on 127.0.0.1:8765 ({DEVICE})", flush=True)
    ThreadingHTTPServer(("127.0.0.1", 8765), Handler).serve_forever()
