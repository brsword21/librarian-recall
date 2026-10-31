"""SSU route, round-4 merge (integrator). Base = frozen v2 ssu route (lenses/ssu/a_r4_route.py v2 == configs/best_v2.json
ssu spec). Gated post-fix wrappers, first one that changes the Reactor-visible pack wins (core/merge.py):
  1. lenses/ssu/a_r4_route.py hpd       SSU-A: hint head-noun fix, residence hop (Lisbon 'I live in Graça'),
                                        before/after delta rebuilt from the two volume-pot sessions (R-77).
  2. lenses/temporal/b_r4_last.py last  TEMP-B: recency guard for 'the last X I mentioned VERB' (beer, book).
  3. lenses/ssu/b_r4_sent.py g_res_lv_d SSU-B: gated sentence hop (Van Nelle Factory mixed unit); its residence /
                                        delta / last-verb hops only act if 1 and 2 did not change the pack.
`v2` returns the base pack unchanged."""
from __future__ import annotations

from core.merge import first_changed

BASE = ("lenses/ssu/a_r4_route.py", "v2", {})
CANDS = [("lenses/ssu/a_r4_route.py", "hpd", {}),
         ("lenses/temporal/b_r4_last.py", "last", {}),
         ("lenses/ssu/b_r4_sent.py", "g_res_lv_d", {})]
VARIANTS = {"v2": {"cands": 0}, "r4": {"cands": 3}, "r4_a": {"cands": 1}, "r4_al": {"cands": 2}}


def build(q, hx, _variant: str = "", cands: int = 3, **_):
    return first_changed(q, hx, BASE, CANDS[:cands], variant=_variant)
