"""Where things live: the repo root and the cache (history shards, model cache). LIBRARIAN_CACHE moves the cache."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = Path(os.environ.get("LIBRARIAN_CACHE", ROOT / "cache"))
