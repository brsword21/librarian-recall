"""Loads a lens module from its path (relative to the repo root), as the configs name them."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from core.paths import ROOT


def load_lens(path: str):
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    spec = importlib.util.spec_from_file_location(f"lens_{p.parent.name}_{p.stem}", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod
