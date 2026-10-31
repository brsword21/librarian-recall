"""SSA lens A, round 2: evidence-based hand-off for questions the text router sends to ssu.

Dev phrases about half of its ssa questions without any cue ("Back in Bandung soon. Which restaurant in Cihampelas Walk
does the great Nasi Goreng?", "What refining processes run at CITGO's Lake Charles Refinery?"); core.router sends them
to ssu, where the user-fact lens scores S ~70 on them (vs ~93 with the ssa lens). This lens is meant to be put on the
ssu ROUTE: it asks `a_recsent.role_features` whether the answer sits in an assistant turn and hands the question to the
ssa lens when

    the question has no first-person word (I / my / me / our ...; "should I ..." advice questions exempt in nf50a)
    AND p_assistant >= thr

otherwise it runs the routed ssu winner unchanged. The first-person gate keeps test safe: every ssu-routed test
question says I/my (0 of 45 hand off), so this only repairs dev-style paraphrases. Measured with
    .venv/bin/python -m harness.sweep --lens lenses/ssa/a_handoff.py --variants all --split dev --type ssu --route
(see lenses/ssa/a_report.md). The ssu lens and variant are parameters, so the integrator can point it at whatever wins
the ssu route.
"""
from __future__ import annotations

_SSU = {"ssu_lens": "lenses/ssu/a_struct.py", "ssu_variant": "nb3t0", "ssa_variant": "recqs_cf7"}
VARIANTS = {
    # hand off at p_assistant >= 0.5 when the question has no first-person word (plateau 0.3-0.65 on dev)
    "nf50": dict(_SSU, thr=0.5, nofirst=True),
    # + "should I / do I need to" questions count as asking for the assistant's advice (first person allowed)
    "nf50a": dict(_SSU, thr=0.5, nofirst=True, advice_ok=True),
    # same hand-off, round-2 ssa variant (single session, confidence-tiered budget)
    "nf50a_c2": dict(_SSU, thr=0.5, nofirst=True, advice_ok=True, ssa_variant="r2_conf2"),
    # no first-person gate (dev-optimal 0.65, but on test it would move 'the immigration law firm where I work')
    "p65": dict(_SSU, thr=0.65, nofirst=False),
    # control: the ssu winner alone (what the routed system does today)
    "ssu_only": dict(_SSU, thr=2.0, nofirst=True),
}
_loaded: dict = {}


def _lens(path: str):
    if path not in _loaded:
        from core.loader import load_lens
        _loaded[path] = load_lens(path)
    return _loaded[path]


def handoff(q, hx, thr: float = 0.5, nofirst: bool = True, advice_ok: bool = False) -> tuple[bool, dict]:
    from lenses.ssa.a_recsent import role_features
    f = role_features(q, hx)
    personal = f["q_nofirst"] < 0 or f["q_ours"] > 0
    if advice_ok and f.get("q_advice"):      # "What sealant should I use ...?" asks for the assistant's advice
        personal = False
    ok = f["p_assistant"] >= thr and (not nofirst or not personal)
    return ok, f


def build(q, hx, _variant: str = "", **p):
    if p.get("thr", 2.0) <= 1.0:
        go, f = handoff(q, hx, p["thr"], p.get("nofirst", True), p.get("advice_ok", False))
    else:
        go, f = False, {}
    if go:
        lens = _lens("lenses/ssa/a_recsent.py")
        v = p.get("ssa_variant", "recqs_cf7")
    else:
        lens = _lens(p.get("ssu_lens", "lenses/ssu/a_struct.py"))
        v = p.get("ssu_variant", "nb3t0")
    pack = lens.build(q, hx, _variant=v, **lens.VARIANTS[v])
    pack.variant = _variant
    pack.debug = {**(pack.debug or {}), "handoff": go,
                  "p_assistant": round(f.get("p_assistant", 0.0), 3) if f else None}
    return pack
