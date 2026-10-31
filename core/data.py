"""Dates and stored histories. A history is kept as one JSON file: cache/hist/<namespace>/<history_id>.json."""
from __future__ import annotations

import json
from datetime import datetime
from functools import lru_cache
from pathlib import Path

from core.paths import CACHE

DATE_FMT = "%Y/%m/%d (%a) %H:%M"


def parse_date(s: str) -> datetime:
    return datetime.strptime(s, DATE_FMT)


def shard_dir(namespace: str) -> Path:
    return CACHE / "hist" / namespace


@lru_cache(maxsize=8)
def history(namespace: str, history_id: str) -> dict:
    return json.loads((shard_dir(namespace) / f"{history_id}.json").read_text(encoding="utf-8"))
