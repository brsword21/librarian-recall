"""Round-4 merge helper: combine gated post-fix wrappers that all sit on the same frozen base lens.

first_changed(q, hx, base, candidates): build the base pack, then each candidate in priority order; the first
candidate whose Reactor-visible content (passage ids + texts, hint) differs from the base wins. If none differs,
the base pack is returned. A candidate that raises is skipped (logged in debug["merge_errors"]).
Each spec is (lens_path, variant, extra_params). Used by lenses/<type>/merged_r4.py.
"""
from __future__ import annotations

_loaded: dict = {}


def _lens(path: str):
    if path not in _loaded:
        from core.loader import load_lens
        _loaded[path] = load_lens(path)
    return _loaded[path]


def run_spec(q, hx, spec):
    path, variant, extra = (list(spec) + [{}])[:3]
    lens = _lens(path)
    return lens.build(q, hx, _variant=variant, **{**lens.VARIANTS[variant], **(extra or {})})


def visible_key(pack) -> tuple:
    return tuple((p.id, p.text) for p in pack.passages), (pack.hint or "").strip()


def first_changed(q, hx, base, candidates, variant: str = ""):
    bp = run_spec(q, hx, base)
    bkey = visible_key(bp)
    errors, pick = [], None
    for spec in candidates:
        try:
            cp = run_spec(q, hx, spec)
        except Exception as exc:                      # a failing fix never breaks the base pack
            errors.append(f"{spec[0]}:{spec[1]}: {exc!r}"[:200])
            continue
        if visible_key(cp) != bkey:
            pick = (spec, cp)
            break
    out = pick[1] if pick else bp
    dbg = dict(out.debug or {}) if isinstance(out.debug, dict) else {}
    dbg["merge_pick"] = f"{pick[0][0]}:{pick[0][1]}" if pick else "base"
    if errors:
        dbg["merge_errors"] = errors
    out.debug = dbg
    out.variant = variant or out.variant
    return out
