"""Integrator round 3: the inner ssu lens for lenses/ssu/a_r3_route.py.

= SSU-B lenses/ssu/b_r3_ssu.py (a_struct2 r3rn + KU-cue recall backstop, ku11fn4s_p5), which ignores extra a_struct2
params, plus pass-through of SSU-A's `ru_cite` (compositional runner-up cite / runner gate), so A's wrapper
(role hand-off, typed value-disagreement ku gate, comp, rgate) and B's recall backstop compose. Imports both
frozen files unchanged; the swap of b_r3_ssu._R3RN is per call (sweep workers are processes, not threads).
"""
from __future__ import annotations

from lenses.ssu import b_r3_ssu

VARIANTS = {"ku11fn4s_p5": dict(b_r3_ssu.VARIANTS["ku11fn4s_p5"])}


def build(q, hx, ru_cite: str | None = None, **params):
    if not ru_cite or ru_cite == "no":
        return b_r3_ssu.build(q, hx, **params)
    saved = b_r3_ssu._R3RN
    b_r3_ssu._R3RN = dict(saved, ru_cite=ru_cite)
    try:
        return b_r3_ssu.build(q, hx, **params)
    finally:
        b_r3_ssu._R3RN = saved
