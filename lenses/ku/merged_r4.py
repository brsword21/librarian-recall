"""KU route, round-4 merge (integrator). Base = frozen v2 ku route (lenses/ku/a_r4.py r3w == a_r3 r3w).
Gated post-fix wrappers, first one that changes the Reactor-visible pack wins (core/merge.py):
  1. lenses/ku/a_r4.py r4          KU-A: clause-level value attribution (long run), conflict-gated decoy-session drop
                                   (rent), absence line when no statement names the question noun (cats).
  2. lenses/ku/b_r4_ku.py nolate_c KU-B: drop uncited sessions dated after the newest cited session (job title, rent,
                                   tractors); skipped on count-of-instances questions. cnt2 (15-session instance pack)
                                   NOT taken: it imports noise and a neighbour's 'four rooms' decoy.
`v2` returns the base pack unchanged."""
from __future__ import annotations

from core.merge import first_changed

BASE = ("lenses/ku/a_r4.py", "r3w", {})
CANDS = [("lenses/ku/a_r4.py", "r4", {}),
         ("lenses/ku/b_r4_ku.py", "nolate_c", {})]
VARIANTS = {"v2": {"cands": 0}, "r4": {"cands": 2}, "r4_a": {"cands": 1}}


def build(q, hx, _variant: str = "", cands: int = 2, **_):
    return first_changed(q, hx, BASE, CANDS[:cands], variant=_variant)
