"""Integrator merge after round 2 (do not edit the source lenses from here).

Temporal A (a_gap ct_evgba) improved the gap route and the questions that NAME their events ("how many days ago did I
X", "which first, X or Y"); temporal B (b_merge ev1_mix_g) kept merged_r1's gap route and improved the questions
without parsed events (session-centric whole-question route + cite_wide) and the event route (ev_fill).
The two gains sit on different subsets, so:
  * evA_mixB : questions A claims (_GAP or a_kinds with events) -> a_gap ct_evgba, everything else -> b_merge ev1_mix_g
  * gapA_mixB: only pure gap questions (_GAP) -> a_gap ct_evgba, everything else -> b_merge ev1_mix_g
"""
from __future__ import annotations

from core.loader import load_lens

_A = load_lens("lenses/temporal/a_gap.py")
_B = load_lens("lenses/temporal/b_merge.py")

VARIANTS = {
    "evA_mixB": {"a_variant": "ct_evgba", "b_variant": "ev1_mix_g", "a_scope": "kinds"},
    "gapA_mixB": {"a_variant": "ct_evgba", "b_variant": "ev1_mix_g", "a_scope": "gap"},
}


def build(q, hx, _variant: str = "", a_variant: str = "ct_evgba", b_variant: str = "ev1_mix_g", a_scope: str = "kinds",
          **_):
    ap = _A.VARIANTS[a_variant]
    use_a = bool(_A._GAP.search(q.question)) if a_scope == "gap" else _A._use_a(q, ap)
    if use_a:
        pack = _A.build(q, hx, _variant=a_variant, **ap)
        src = f"a_gap:{a_variant}"
    else:
        pack = _B.build(q, hx, _variant=b_variant, **_B.VARIANTS[b_variant])
        src = f"b_merge:{b_variant}"
    pack.variant = _variant
    pack.debug = {**(pack.debug or {}), "merged_r2_from": src}
    return pack
