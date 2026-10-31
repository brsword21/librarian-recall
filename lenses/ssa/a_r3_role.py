"""SSA-A round 3: refit role detector + route wrappers for the ssu and multi routes (evidence-based ssa hand-off).

Why: on dev, 13 gold-ssa questions still go to the ssu lens and 1 to the multi lens. The ssu lens appends
"Answer from what the user said about their own X ..." to them. In the real-Reactor calibration (runs/reactor/
e1b_results.jsonl, q_73a1f68b2421 "Which company does a great job with environmentally responsible supply chains?",
gold Patagonia) exactly that hint made Reactor answer INSUFFICIENT_EVIDENCE although the pack held the answer.
v1 still routes that question to ssu (p_b 0.73 < 0.8). Proxy PA says 1.0 for these questions; Reactor does not.

role_guess(q, hx) = max(p_b, p_fit)  (drop-in for lenses/ssa/b_role.py: merged_r2 wrappers take role_module=...)
  p_b   = SSA-B r2 detector (sigmoid(0.35*clip(d,+-8) + 2.2*nf0 - 1.2)), unchanged.
  p_fit = sigmoid(0.469*clip(d,+-8) + 5.666*nf0 + b): logistic fit on dev ssu+multi-routed questions (features d =
          CE(best assistant unit) - CE(best user unit), nf0 = no first-person word). Fitted weights; the intercept
          b is set so that p_fit = 0.8 at d = -0.73 for nf0 questions, the midpoint between the nearest dev negative
          ("How long did it take to move to the new apartment?", d -1.04) and the lowest positive above it (d -0.42).
          Questions WITH a first-person word never pass through p_fit (p_fit <= 0.46 at d = 8 < 0.8).
  advice rule: "should I/my/we" and d >= 2 -> p >= 0.85 (passes the ssu threshold 0.8, not the multi one 0.9).
  Adding bm_diff / sess_cov / top5_a / advice / route features does not raise the zero-FP TP count (fit_role.py).
Soft band (ssu route only, variants *_s55 / *_nh55): nf0 and soft_d <= d < -0.73 -> either the ssa lens recqs_cf7
  (s55: dev +0.4 / dev200 -0.9 on the ssu-routed slice, REJECTED: the apartment question cites the wrong session) or
  the ssu pack with the user-only hint removed (nh55: tie on both splits; optional, Reactor-motivated only).

Measured (tag r3-ssa-a / r3-ssa-a-routed; lens a_r3_routed.py = pipeline with this role module on both wrappers):
  routed dev 86.0 -> 86.3 (Sall 84.9 -> 85.2), gold-ssa rows 91.8 -> 93.0 (PA 0.973 -> 0.991, 97 -> 104 on ssa);
  routed dev200 84.9 -> 84.9 (ssa 21 -> 22 on ssa). FP 0 on dev and dev200 (no non-ssa question changes route).
  Test: harness.audit --compare runs/final/v1_test_packs.jsonl -> 0 of 146 packs change (route, cite, passages,
  hint); every ssu/multi-routed test question carries I/my/our and none says "should I".
RECOMMENDED integration: configs/best.json routes.ssu.params = routes.multi.params =
  {"role_module": "lenses/ssa/a_r3_role.py"} (see lenses/ssa/scratch_a/r3/routed_r3fit.json).
"""
from __future__ import annotations

import math
import re

from core.loader import load_lens

_loaded: dict = {}


def _lens(path: str):
    if path not in _loaded:
        _loaded[path] = load_lens(path)
    return _loaded[path]


_W_D, _W_NF, _D0 = 0.469, 5.666, -0.73
_B = math.log(0.8 / 0.2) - _W_NF - _W_D * _D0          # p_fit(nf0, d = _D0) = 0.8


def _features(q, hx) -> dict:
    f = _lens("lenses/ssa/b_role.py").role_features(q, hx)
    return {"d": f["d"], "nf0": f["n_first"] == 0}


def p_fit(d: float, nf0: bool) -> float:
    x = _W_D * max(-8.0, min(8.0, d)) + _W_NF * nf0 + _B
    return 1.0 / (1.0 + math.exp(-x))


_ADVICE = re.compile(r"\bshould (?:i|my|we)\b", re.I)


def role_guess(q, hx) -> float:
    """P(the question asks what the ASSISTANT said): max of SSA-B's p_b, the refit p_fit and the advice rule."""
    br = _lens("lenses/ssa/b_role.py")
    f = br.role_features(q, hx)
    pb = br.p_from_features(f["d"], f["n_first"], c=-1.2)
    p = max(pb, p_fit(f["d"], f["n_first"] == 0))
    # "Which X should I use / learn ...?" asks for the assistant's advice; with an assistant unit ahead on the CE
    # (d >= 2) it passes the ssu threshold (0.8) but never the multi one (0.9). dev: 3/3 such ssu-routed questions
    # are ssa (d 4.1 / 8.2 / 14.2); the "do I need to" questions on the multi route (d <= 1.3) are not ssa.
    if _ADVICE.search(q.question) and f["d"] >= 2.0:
        p = max(p, 0.85)
    return p


_SSU = {"base": "lenses/ssu/merged_r2.py", "base_variant": "role8_r3rn", "route": "ssu"}
_MUL = {"base": "lenses/multi/merged_r2.py", "base_variant": "tight_role9", "route": "multi"}
VARIANTS = {
    # controls: the frozen v1 route lenses, untouched
    "ssu_v1": dict(_SSU, role_module="lenses/ssa/b_role.py"),
    "multi_v1": dict(_MUL, role_module="lenses/ssa/b_role.py"),
    # refit hand-off only (role_module swap; same thresholds 0.8 / 0.9)
    "ssu_fit": dict(_SSU, role_module="lenses/ssa/a_r3_role.py"),
    "multi_fit": dict(_MUL, role_module="lenses/ssa/a_r3_role.py"),
    # + soft band on the ssu route: nf0 and d >= soft_d -> ssa lens recqs_cf7 (session fill, no user-only hint)
    "ssu_fit_s55": dict(_SSU, role_module="lenses/ssa/a_r3_role.py", soft_d=-5.5),
    # soft band variant: keep the ssu pack, only drop the user-only hint
    "ssu_fit_nh55": dict(_SSU, role_module="lenses/ssa/a_r3_role.py", soft_d=-5.5, soft_mode="nohint"),
}


def build(q, hx, _variant: str = "", base: str = "", base_variant: str = "", route: str = "ssu",
          role_module: str = "lenses/ssa/b_role.py", soft_d: float | None = None, soft_mode: str = "ssa",
          soft_lens: str = "lenses/ssa/a_recsent.py", soft_variant: str = "recqs_cf7", **_):
    wrapper = _lens(base)
    params = {**wrapper.VARIANTS[base_variant], "role_module": role_module}
    pack = wrapper.build(q, hx, _variant=base_variant, **params)
    info = {}
    if soft_d is not None and route == "ssu" and pack.route != "ssa":
        f = _features(q, hx)
        info = {"soft_d": round(f["d"], 3), "soft_nf0": f["nf0"]}
        if f["nf0"] and f["d"] >= soft_d:
            if soft_mode == "nohint":
                pack.hint = ""
                info["soft"] = "nohint"
            else:
                lens = _lens(soft_lens)
                inner = lens.build(q, hx, _variant=soft_variant, **lens.VARIANTS[soft_variant])
                inner.route = "ssa"
                inner.debug = {**(inner.debug or {}), "route_final": "ssa", "inner_lens": soft_lens,
                               "inner_variant": soft_variant}
                pack = inner
                info["soft"] = "ssa"
    pack.variant = _variant
    pack.debug = {**(pack.debug or {}), "r3_role": info}
    return pack
