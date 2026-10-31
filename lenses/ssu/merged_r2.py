"""Integrator merge after round 2: the ssu ROUTE with evidence-based hand-offs (do not edit the source lenses here).

reactorq.build_packs routes by text only (core.router.route) and then calls the lens configured for that route, so
a hand-off has to live inside the route's lens to reach Reactor. This wrapper is that lens for the ssu route:
  1. role_guess (lenses/ssa/b_role.py, SSA-B r2): p(question asks what the ASSISTANT said) >= thr -> ssa lens.
     dev: 50 of 144 ssu-routed questions are gold ssa; FP 0 on dev and dev200 at 0.8 (max non-ssa p 0.65).
  2. optional ku_pair_p (lenses/ku/b_r2.py, KU-B r2): two own sessions state two different values of the
     attribute -> ku lens. Only on ssu-routed questions (never multi: a multi -> ku misroute costs ~70 S).
  3. otherwise the ssu lens.
debug: route_final, p_assistant, ku_pair. A failing detector never changes the route.
"""
from __future__ import annotations

from core.loader import load_lens

_loaded: dict = {}


def _lens(path: str):
    if path not in _loaded:
        _loaded[path] = load_lens(path)
    return _loaded[path]


_BASE = {"ssu_lens": "lenses/ssu/a_struct.py", "ssu_variant": "nb3t0",
         "ssa_lens": "lenses/ssa/a_recsent.py", "ssa_variant": "r2_conf2",
         "ku_lens": "lenses/ku/b_r2.py", "ku_variant": "r2b",
         "role_module": "lenses/ssa/b_role.py", "thr": 0.8, "ku_pair": False}
VARIANTS = {
    "role8": dict(_BASE),
    "role8_r3rn": dict(_BASE, ssu_lens="lenses/ssu/a_struct2.py", ssu_variant="r3rn"),
    "role8_ku": dict(_BASE, ku_pair=True),
    "role8_r3rn_ku": dict(_BASE, ssu_lens="lenses/ssu/a_struct2.py", ssu_variant="r3rn", ku_pair=True),
}


def choose(q, hx, thr: float = 0.8, ku_pair: bool = False, role_module: str = "lenses/ssa/b_role.py",
           ku_module: str = "lenses/ku/b_r2.py", min_score: float = 6.0, min_cov: float = 0.5) -> tuple[str, dict]:
    info: dict = {}
    try:
        p = _lens(role_module).role_guess(q, hx)
        info["p_assistant"] = round(p, 4)
        if p >= thr:
            return "ssa", info
    except Exception as exc:
        info["role_error"] = repr(exc)[:200]
    if ku_pair:
        try:
            hit, _f = _lens(ku_module).ku_pair_p(q, hx, min_score=min_score, min_cov=min_cov)
            info["ku_pair"] = bool(hit)
            if hit:
                return "ku", info
        except Exception as exc:
            info["ku_pair_error"] = repr(exc)[:200]
    return "ssu", info


def build(q, hx, _variant: str = "", ssu_lens: str = "", ssu_variant: str = "", ssa_lens: str = "",
          ssa_variant: str = "", ku_lens: str = "", ku_variant: str = "", role_module: str = "lenses/ssa/b_role.py",
          thr: float = 0.8, ku_pair: bool = False, **_):
    r, info = choose(q, hx, thr=thr, ku_pair=ku_pair, role_module=role_module, ku_module=ku_lens or "lenses/ku/b_r2.py")
    path, v = {"ssa": (ssa_lens, ssa_variant), "ku": (ku_lens, ku_variant)}.get(r, (ssu_lens, ssu_variant))
    lens = _lens(path)
    pack = lens.build(q, hx, _variant=v, **lens.VARIANTS[v])
    pack.route = r
    pack.debug = {**(pack.debug or {}), "route_final": r, "inner_lens": path, "inner_variant": v, **info}
    return pack
