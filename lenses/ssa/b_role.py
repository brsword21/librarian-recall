"""SSA-B round 2: evidence-based role detector for cue-less questions + a routed wrapper to measure it.

role_guess(q, hx) -> p_assistant in [0, 1]: probability that the question asks about something the ASSISTANT said
(single-session-assistant) rather than something the user said. Meant for questions the text router sends to ssu
(no "you" / "remind me" cue). 50 of 144 such dev questions are gold ssa ("Which dessert shop in Orlando had the
giant milkshakes?", "In the chess game, what move came after 27. Kg2 Bd5+?").

Two signals:
  d   = best cross-encoder logit over assistant units - best over user units, in the CE-reranked top 40 of
        hx.hybrid(question) (same query and pool as lenses/generic.py hybrid_rr, so the rerank pairs are cached).
        Content questions about what the assistant produced match an assistant unit far better (d ~ +2..+16);
        personal-fact questions match the user's own statement (d ~ -15..+3).
  nf0 = the question has no first-person word (I, my, me, mine, our, us ...). Cue-less ssa questions are about
        content ("the dinosaur children's book", "the Portland indie venues list"), ssu questions about the user.
p_assistant = sigmoid(0.35 * clip(d, -8, 8) + 2.2 * nf0 - 1.2)   (c = -1.2: margin version, see b_report.md)
Switch rule (variant switch_m8m9, recommended): router says ssu and p >= 0.8 -> ssa lens;
                                                router says multi and p >= 0.9 -> ssa lens. Never on ku / temporal.
On ssu-routed questions p >= 0.8 means (nf0 and d >= 1.10) or d >= 7.39. Thresholds 0.7 and 0.8 give the same S on
dev and dev200; 0.8 keeps a wider margin to the nearest ssu question (max p of a non-ssa question: 0.65).

Measured (lenses/ssa/scratch_b/role_eval.py offline; harness.sweep end to end):
  ssu-routed dev    (144 q, 50 gold ssa): AUC 0.974 (ssa vs ssu only 0.978, acc@0.7 0.919); TP 42 / 37, FP 0 at 0.7 / 0.8
  ssu-routed dev200 ( 40 q,  7 gold ssa): AUC 1.000; TP 7 / 6, FP 0 (the TPs dropped at 0.8 gain ~0 S: ssu lens
  already answers them)
  multi-routed @0.9: dev TP 5 FP 0, dev200 TP 2 FP 0 (temporal-routed would get 12 FP at 0.7 -> not used there)
  routed system (lenses/pipeline.py = noswitch): dev S 80.4 -> 83.3 (Sall 74.9 -> 78.0), dev200 79.9 -> 82.8
  (Sall 74.2 -> 77.2), abstain AUC 0.719 -> 0.750 / 0.808 -> 0.825.
  Test: 0 of 45 ssu-routed and 0 of 20 multi-routed questions switch (test ssa questions all carry the "you
  recommended" cue), so the switch is insurance plus dev/dev200 gain, no test change today.

VARIANTS (this file is also a lens: router -> configs/best.json, but a switched question goes to the ssa lens).
Sweep it with --route --type ssu / multi, or on all questions for the routed system.
"""
from __future__ import annotations

import json
import math
import re

from core.paths import ROOT
from core.router import route

_FIRST = re.compile(r"\b(I|my|me|mine|I'm|I've|I'd|I'll|myself|our|ours|us)\b", re.I)

VARIANTS = {
    "switch_a": {"thr": 0.7, "ssa_lens": "lenses/ssa/a_recsent.py", "ssa_variant": "recqs_cf7"},
    "switch_b": {"thr": 0.7, "ssa_lens": "lenses/ssa/b_semantic.py", "ssa_variant": "session10k1_rr"},
    "switch_a_m": {"thr": 0.7, "c": -1.2, "ssa_lens": "lenses/ssa/a_recsent.py", "ssa_variant": "recqs_cf7"},
    # also switch multi-routed questions (6 dev / 2 dev200 gold ssa are routed to multi), at a stricter threshold
    "switch_am9": {"thr": 0.7, "c": -1.2, "thr_multi": 0.9, "ssa_lens": "lenses/ssa/a_recsent.py",
                   "ssa_variant": "recqs_cf7"},
    # FINAL recommendation: ssu route at 0.8 (same S as 0.7 on both splits, wider margin to the nearest ssu
    # question), multi route at 0.9
    "switch_m8m9": {"thr": 0.8, "c": -1.2, "thr_multi": 0.9, "ssa_lens": "lenses/ssa/a_recsent.py",
                    "ssa_variant": "recqs_cf7"},
    "switch_am95": {"thr": 0.7, "c": -1.2, "thr_multi": 0.95, "ssa_lens": "lenses/ssa/a_recsent.py",
                    "ssa_variant": "recqs_cf7"},
    "switch_a5": {"thr": 0.5, "ssa_lens": "lenses/ssa/a_recsent.py", "ssa_variant": "recqs_cf7"},
    # threshold sensitivity (ssu route, margin version)
    "switch_m6": {"thr": 0.6, "c": -1.2, "ssa_lens": "lenses/ssa/a_recsent.py", "ssa_variant": "recqs_cf7"},
    "switch_m8": {"thr": 0.8, "c": -1.2, "ssa_lens": "lenses/ssa/a_recsent.py", "ssa_variant": "recqs_cf7"},
    # evidence ablation: text feature only (no first-person word), no cross-encoder margin
    "switch_nf0": {"thr": 0.5, "a": 0.0, "c": -1.2, "ssa_lens": "lenses/ssa/a_recsent.py", "ssa_variant": "recqs_cf7"},
    "noswitch": {"thr": 9.0, "ssa_lens": "lenses/ssa/a_recsent.py", "ssa_variant": "recqs_cf7"},
}


def role_features(q, hx, n: int = 40) -> dict:
    text = q.question if hasattr(q, "question") else str(q)
    cand = hx.hybrid(text, k=100, pool=300)
    rr = hx.rerank(text, [u for u, _ in cand[:n]])
    best = {"assistant": -20.0, "user": -20.0}
    for u, s in rr:
        r = hx.units[u].role
        if s > best.get(r, -20.0):
            best[r] = s
    d = best["assistant"] - best["user"]
    nf = len(_FIRST.findall(text))
    return {"d": d, "n_first": nf, "ce_a": best["assistant"], "ce_u": best["user"],
            "top_uid": rr[0][0] if rr else None}


def p_from_features(d: float, n_first: int, a: float = 0.35, b: float = 2.2, c: float = -1.0) -> float:
    x = a * max(-8.0, min(8.0, d)) + b * (n_first == 0) + c
    return 1.0 / (1.0 + math.exp(-x))


def role_guess(q, hx, c: float = -1.2, a: float = 0.35) -> float:
    """p_assistant: probability the question is about what the assistant said (see module docstring)."""
    f = role_features(q, hx)
    return p_from_features(f["d"], f["n_first"], a=a, c=c)


_loaded: dict = {}


def _lens(path: str):
    if path not in _loaded:
        from core.loader import load_lens
        _loaded[path] = load_lens(path)
    return _loaded[path]


def build(q, hx, _variant: str = "", thr: float = 0.7, c: float = -1.0, ssa_lens: str = "", ssa_variant: str = "",
          config: str = "configs/best.json", thr_multi: float = 9.0, a: float = 0.35, **_):
    cfg = json.loads((ROOT / config).read_text())
    r = route(q.question)[0]
    p = None
    if r == "ssu":
        p = role_guess(q, hx, c=c, a=a)
        if p >= thr:
            r = "ssa"
    elif r == "multi" and thr_multi <= 1.0:
        p = role_guess(q, hx, c=c, a=a)
        if p >= thr_multi:
            r = "ssa"
    if r == "ssa" and ssa_lens:
        spec = {"lens": ssa_lens, "variant": ssa_variant}
    else:
        spec = cfg["routes"].get(r) or cfg["fallback"]
    lens = _lens(spec["lens"])
    params = {**lens.VARIANTS[spec["variant"]], **spec.get("params", {})}
    pack = lens.build(q, hx, _variant=spec["variant"], **params)
    pack.route = r
    pack.variant = _variant
    pack.debug = {**(pack.debug or {}), "lens": spec["lens"], "lens_variant": spec["variant"], "p_assistant": p}
    return pack
