"""Multi ROUTE wrapper for round 4: the frozen merged_r2 hand-off (role >= 0.9 -> ssa, role detector ssa/a_r3_role.py)
with the inner multi lens swapped for lenses/multi/a_r4_verb.py. Variant 'v2' == configs/best_v2.json multi route."""
from __future__ import annotations

from lenses.multi import merged_r2 as M

_B = dict(M._BASE, role_module="lenses/ssa/a_r3_role.py")
VARIANTS = {
    "v2": dict(_B),
    "vc": dict(_B, multi_lens="lenses/multi/a_r4_verb.py", multi_variant="vc"),
    "vcc": dict(_B, multi_lens="lenses/multi/a_r4_verb.py", multi_variant="vcc"),
    "vc_dp": dict(_B, multi_lens="lenses/multi/a_r4_verb.py", multi_variant="vc_dp"),
    "vl": dict(_B, multi_lens="lenses/multi/a_r4_verb.py", multi_variant="vl"),
    "vlc": dict(_B, multi_lens="lenses/multi/a_r4_verb.py", multi_variant="vlc"),
    "vlb": dict(_B, multi_lens="lenses/multi/a_r4_verb.py", multi_variant="vlb"),
    "vlbc": dict(_B, multi_lens="lenses/multi/a_r4_verb.py", multi_variant="vlbc"),
    "vlbhc": dict(_B, multi_lens="lenses/multi/a_r4_verb.py", multi_variant="vlbhc"),
    "vlbhc6": dict(_B, multi_lens="lenses/multi/a_r4_verb.py", multi_variant="vlbhc6"),
}


def build(q, hx, **params):
    return M.build(q, hx, **params)
