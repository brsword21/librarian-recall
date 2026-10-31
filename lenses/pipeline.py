"""The full system as one lens: router -> lens chosen per route in a config (default configs/best.json).

Round 2: evidence-based hand-offs after the text router, all driven by the config's optional "handoff" block
(a config without it, e.g. configs/best_r1.json, behaves exactly as in round 1):
  "role":    {"module": "lenses/ssa/b_role.py", "thr": {"ssu": 0.8, "multi": 0.9}}
             role_guess(q, hx) = p(the question asks what the ASSISTANT said); above the per-route threshold the
             question goes to the ssa route. Never applied to ku / temporal (12 FP on temporal at 0.7).
  "ku_pair": {"module": "lenses/ku/b_r2.py", "routes": ["ssu"], "min_score": 6.0, "min_cov": 0.5}
             ku_pair_p(q, hx): the two best own-attribute sessions state two different values -> ku route.
             Only for ssu-routed questions (a multi -> ku misroute costs ~70 S per question).
debug carries route_text (text router), route (final), p_assistant and ku_pair so audits can see every hand-off.
"""
from __future__ import annotations

import json

from core.paths import ROOT
from core.router import route

VARIANTS = {"routed": {"config": "configs/best.json"}}
_loaded: dict = {}
_cfg_cache: dict = {}


def _lens(path: str):
    if path not in _loaded:
        from core.loader import load_lens
        _loaded[path] = load_lens(path)
    return _loaded[path]


def _cfg(config: str) -> dict:
    p = ROOT / config
    key = (config, p.stat().st_mtime)
    if key not in _cfg_cache:
        _cfg_cache[key] = json.loads(p.read_text())
    return _cfg_cache[key]


def final_route(q, hx, cfg: dict) -> tuple[str, dict]:
    """Text router + evidence hand-offs from cfg["handoff"]. Returns (route, info for debug)."""
    r = route(q.question)[0]
    info = {"route_text": r}
    ho = cfg.get("handoff") or {}
    role = ho.get("role")
    if role and r in role.get("thr", {}):
        try:
            p = _lens(role["module"]).role_guess(q, hx)
            info["p_assistant"] = round(p, 4)
            if p >= role["thr"][r]:
                r = "ssa"
        except Exception as exc:                      # a failing detector never changes the route
            info["role_error"] = repr(exc)[:200]
    kp = ho.get("ku_pair")
    if kp and r in kp.get("routes", []):
        try:
            hit, f = _lens(kp["module"]).ku_pair_p(q, hx, min_score=kp.get("min_score", 6.0),
                                                   min_cov=kp.get("min_cov", 0.5))
            info["ku_pair"] = bool(hit)
            if hit:
                r = "ku"
        except Exception as exc:
            info["ku_pair_error"] = repr(exc)[:200]
    info["route"] = r
    return r, info


def build(q, hx, _variant: str = "", config: str = "configs/best.json", force_route: str | None = None, **_):
    cfg = _cfg(config)
    if force_route:
        r, info = force_route, {"route_text": force_route, "route": force_route, "forced": True}
    else:
        r, info = final_route(q, hx, cfg)
    spec = cfg["routes"].get(r) or cfg["fallback"]
    lens = _lens(spec["lens"])
    params = {**lens.VARIANTS[spec["variant"]], **spec.get("params", {})}
    try:
        pack = lens.build(q, hx, _variant=spec["variant"], **params)
    except Exception as exc:
        info["lens_error"] = repr(exc)[:200]
        spec = cfg["fallback"]
        lens = _lens(spec["lens"])
        pack = lens.build(q, hx, _variant=spec["variant"], **{**lens.VARIANTS[spec["variant"]], **spec.get("params", {})})
    # a route lens may hand the question off itself (lenses/ssu/merged_r2.py, lenses/multi/merged_r2.py): keep its route
    pack.route = (pack.debug or {}).get("route_final", r) if isinstance(pack.debug, dict) else r
    pack.debug = {**(pack.debug or {}), "lens": spec["lens"], "lens_variant": spec["variant"], **info}
    if pack.route != r:
        pack.debug["route"] = pack.route
    return pack
