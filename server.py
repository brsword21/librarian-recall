"""HTTP API for Librarian (standard library only).

    python -m server --port 8080                 # needs: python -m core.model_server

Endpoints (JSON in, JSON out):
  GET  /health                                 {"ok": true}
  POST /v1/histories   {"history": {...}}      a history in the challenge format
                       {"content": "...", "name": "chat.txt"}   any supported file as text (see ingest.py)
                       -> {"history_id", "sessions", "turns", "chars", "state": "indexing"}
  GET  /v1/histories                           every stored history with its indexing state
  POST /v1/pack        {"history_id", "question", "question_date"?}
                       -> {"route", "passages": [{"label", "id", "date", "text"}], "notes", "computed",
                           "cite_plan", "abstain_p", "chars", "build_ms", "prompt"}

Set LIBRARIAN_API_KEY to require "Authorization: Bearer <key>" on every /v1 call. Histories are kept in
cache/hist/api/ and survive restarts.
"""
from __future__ import annotations

import argparse
import hmac
import json
import os
import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from core.data import DATE_FMT, shard_dir
from core.index import HistoryIndex
from ingest import to_history
from librarian import Librarian

STORE = shard_dir("api")
STATE: dict[str, dict] = {}
LIB: Librarian | None = None
API_KEY = os.environ.get("LIBRARIAN_API_KEY", "")
MAX_BODY = 40_000_000
_lock = threading.Lock()


def _load(hid: str) -> HistoryIndex:
    path = STORE / f"{hid}.json"
    if not path.exists():
        raise KeyError(f"unknown history_id {hid}")
    return HistoryIndex.load("api", hid)


def _warm(hid: str) -> None:
    st = STATE[hid]
    try:
        Librarian.warm(_load(hid), progress=lambda f: st.update(progress=round(f, 3)))
        st.update(state="ready", progress=1.0)
    except Exception as exc:
        st.update(state="error", error=repr(exc))


def add_history(body: dict) -> dict:
    if isinstance(body.get("history"), dict):
        hist, info = to_history(json.dumps(body["history"]), body.get("name") or "history")
    elif isinstance(body.get("content"), str):
        hist, info = to_history(body["content"], body.get("name") or "upload")
    else:
        raise ValueError("send {'history': {...}} or {'content': '<file text>', 'name': '...'}")
    hid = hist["history_id"]
    with _lock:
        if hid in STATE and STATE[hid]["state"] != "error":
            return STATE[hid]
        STORE.mkdir(parents=True, exist_ok=True)
        info["added"] = datetime.now().strftime(DATE_FMT)
        (STORE / f"{hid}.json").write_text(json.dumps({**hist, "upload_info": info}, ensure_ascii=False),
                                           encoding="utf-8")
        STATE[hid] = {**info, "state": "indexing", "progress": 0.0}
    threading.Thread(target=_warm, args=(hid,), daemon=True).start()
    return STATE[hid]


def pack(body: dict) -> dict:
    if not body.get("question"):
        raise ValueError("question is required")
    return LIB.pack(_load(body["history_id"]), body["question"], body.get("question_date"))


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, obj, code=200):
        data = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _authorised(self) -> bool:
        if not API_KEY:
            return True
        return hmac.compare_digest(self.headers.get("Authorization", ""), f"Bearer {API_KEY}")

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        if self.path == "/health":
            return self._send({"ok": True})
        if not self._authorised():
            return self._send({"error": "unauthorised"}, 401)
        if self.path.startswith("/v1/histories"):
            return self._send({"histories": sorted(STATE.values(), key=lambda s: s.get("added", ""), reverse=True)})
        self._send({"error": "not found"}, 404)

    def do_POST(self):
        if not self._authorised():
            return self._send({"error": "unauthorised"}, 401)
        try:
            n = int(self.headers.get("Content-Length", 0))
            if n > MAX_BODY:
                return self._send({"error": f"body too large (limit {MAX_BODY // 1_000_000} MB)"}, 413)
            body = json.loads(self.rfile.read(n) or b"{}")
            if self.path.startswith("/v1/histories"):
                self._send(add_history(body))
            elif self.path.startswith("/v1/pack"):
                self._send(pack(body))
            else:
                self._send({"error": "not found"}, 404)
        except (ValueError, KeyError) as exc:
            self._send({"error": str(exc)}, 400)
        except Exception as exc:
            self._send({"error": repr(exc)}, 500)


def main() -> None:
    global LIB
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--config", default="configs/best.json")
    a = ap.parse_args()
    LIB = Librarian(a.config)
    if STORE.exists():   # histories stored before a restart; their embeddings are cached
        for f in STORE.glob("*.json"):
            info = json.loads(f.read_text(encoding="utf-8")).get("upload_info") or {"history_id": f.stem}
            STATE[f.stem] = {**info, "state": "ready", "progress": 1.0}
    print(f"librarian API on http://{a.host}:{a.port}" + (" (API key required)" if API_KEY else ""), flush=True)
    ThreadingHTTPServer((a.host, a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
