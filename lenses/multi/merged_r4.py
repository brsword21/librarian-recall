"""Multi route, round-4 merge (integrator). Base = frozen v2 multi route (lenses/multi/a_r4_route.py v2 == merged_r2
tight_role9 + ssa/a_r3_role.py). Instance-rescue wrappers, first one that changes the Reactor-visible pack wins:
  1. lenses/multi/a_r4_route.py vlbhc6  MULTI-A: verb-class rescue + hyponyms + balance class (books: 3 own reads,
                                        orchids, bakery fund opening deposit, vegetables).
  2. lenses/multi/b_r4_inst.py vc2      MULTI-B: quantity-bearing own instance rescue (book club March '7 people').
`v2` returns the base pack unchanged."""
from __future__ import annotations

from core.merge import first_changed

BASE = ("lenses/multi/a_r4_route.py", "v2", {})
CANDS = [("lenses/multi/a_r4_route.py", "vlbhc6", {}),
         ("lenses/multi/b_r4_inst.py", "vc2", {})]
VARIANTS = {"v2": {"cands": 0}, "r4": {"cands": 2}, "r4_a": {"cands": 1}}


def build(q, hx, _variant: str = "", cands: int = 2, **_):
    return first_changed(q, hx, BASE, CANDS[:cands], variant=_variant)
