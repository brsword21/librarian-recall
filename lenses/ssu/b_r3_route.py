"""SSU ROUTE lens B round 3: the frozen v1 route wrapper (lenses/ssu/merged_r2.py, integrator) with the inner ssu lens
swapped for lenses/ssu/b_r3_ssu.py. role_guess >= 0.8 -> ssa lens exactly as in v1. Run with --route --type ssu.
"""
from __future__ import annotations

from lenses.ssu import merged_r2

_B = dict(merged_r2._BASE, ssu_lens="lenses/ssu/b_r3_ssu.py")
VARIANTS = {
    "v1": dict(merged_r2._BASE, ssu_lens="lenses/ssu/a_struct2.py", ssu_variant="r3rn"),
    "base": dict(_B, ssu_variant="base"),
    "ku11": dict(_B, ssu_variant="ku11"),
    "ku11b9": dict(_B, ssu_variant="ku11b9"),
    "ku11s": dict(_B, ssu_variant="ku11s"),
    "ku8": dict(_B, ssu_variant="ku8"),
    "ku11_6k": dict(_B, ssu_variant="ku11_6k"),
    "ku11f": dict(_B, ssu_variant="ku11f"),
    "ku11fn": dict(_B, ssu_variant="ku11fn"),
    "ku11fn4": dict(_B, ssu_variant="ku11fn4"),
    "ku11f10n4": dict(_B, ssu_variant="ku11f10n4"),
    "ku11fn4s": dict(_B, ssu_variant="ku11fn4s"),
    "ku11f10n4s": dict(_B, ssu_variant="ku11f10n4s"),
    "ku11fn4s_p5": dict(_B, ssu_variant="ku11fn4s_p5"),
    "ku11fn4s_p11": dict(_B, ssu_variant="ku11fn4s_p11"),
    "abz": dict(_B, ssu_variant="abz"),
    "ku11_abz": dict(_B, ssu_variant="ku11_abz"),
}


def build(q, hx, **params):
    return merged_r2.build(q, hx, **params)
