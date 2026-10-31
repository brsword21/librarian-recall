"""Integrator merge after round 2: the multi ROUTE with the SSA-B role hand-off (do not edit the source lenses here).

reactorq.build_packs routes by text only, so the hand-off lives in the route's lens. role_guess (lenses/ssa/b_role.py)
>= 0.9 -> ssa lens ("How many mummies does the party face in the Lost Temple ...?": content the assistant produced);
otherwise the multi lens. dev: TP 5 / FP 0, dev200: TP 2 / FP 0 at 0.9. A wrong move off multi is expensive, hence the
stricter threshold than on the ssu route (0.8).
"""
from __future__ import annotations

from core.loader import load_lens

_loaded: dict = {}


def _lens(path: str):
    if path not in _loaded:
        _loaded[path] = load_lens(path)
    return _loaded[path]


_BASE = {"multi_lens": "lenses/multi/b_tight.py", "multi_variant": "p08c35_aor",
         "ssa_lens": "lenses/ssa/a_recsent.py", "ssa_variant": "r2_conf2",
         "role_module": "lenses/ssa/b_role.py", "thr": 0.9}
VARIANTS = {
    "tight_role9": dict(_BASE),
    "deep8_role9": dict(_BASE, multi_lens="lenses/multi/b_sweep.py", multi_variant="deep8_10k"),
}


def build(q, hx, _variant: str = "", multi_lens: str = "", multi_variant: str = "", ssa_lens: str = "",
          ssa_variant: str = "", role_module: str = "lenses/ssa/b_role.py", thr: float = 0.9, **_):
    info: dict = {}
    r = "multi"
    try:
        p = _lens(role_module).role_guess(q, hx)
        info["p_assistant"] = round(p, 4)
        if p >= thr:
            r = "ssa"
    except Exception as exc:
        info["role_error"] = repr(exc)[:200]
    path, v = (ssa_lens, ssa_variant) if r == "ssa" else (multi_lens, multi_variant)
    lens = _lens(path)
    pack = lens.build(q, hx, _variant=v, **lens.VARIANTS[v])
    pack.route = r
    pack.debug = {**(pack.debug or {}), "route_final": r, "inner_lens": path, "inner_variant": v, **info}
    return pack
