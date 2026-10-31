"""SSU lens B, round 3: a_struct2 r3rn (frozen v1 ssu lens, lens A, imported unchanged) + semantic recall backstop.

BEST: `ku11fn4s` (route lens: lenses/ssu/b_r3_route.py ku11fn4s). Test packs identical to v1 (0 of 45 changed).
One hypothesis per variant (see lenses/ssu/b_report.md):
  * base     r3rn exactly (control; reproduces v1).
  * ku11     KU-cue recall backstop: when the question carries a knowledge-update cue (b_merge.KU_CUE: previous, before,
             still, now, same, do I / have I been, ...), widen the competitor window from 6 to 11 logits and the budget
             to 8000 chars. The old value of an update sits 4-11 logits under the new one and was cut by the window
             or by the 6000-char budget (dev200: mom grocery list -10.2, 5K personal best and Saturday wake-up time
             cut by budget). The cite set is unchanged (still the r3rn primary / runner rule).
  * ku11s    ku11 + re-pack with every kept session's lead unit first (budget starvation). No effect: ku11 already fits.
  * ku8 / ku11b9 / ku11_6k   window 8 / budget 9000 / budget 6000.
  * ku11f    widened window floored at an absolute score of -6 (no junk sessions at -10 when the top is weak).
  * ku11fn   + no backstop on recency / drift phrasing (REC_RX: "end up", "the last X I mentioned").
  * ku11fn4  + up to 4 sessions under the backstop.  ku11f10n4: floor -10.
  * ku11fn4s / ku11f10n4s  strict: the backstop acts only when the floor actually widens the window (top session
             confident), otherwise the pack is exactly r3rn's.
  * abz      abstain_p from a pool-relative z-score of the best SENTENCE-level CE logit among the top-12 user units
             (hx.rerank_texts) (brief item 2). Packs identical to base. Rejected: routed AUC 0.931 -> 0.384 (dev).
debug (feats=True) carries the features (u_best, sent_best, z_user, z_sent, gap2) for offline calibration.
"""
from __future__ import annotations

import math
import re

from core.types import EvidencePack
from lenses.ssu import a_struct2
from lenses.ssu.b_merge import KU_CUE

_R3RN = dict(a_struct2.VARIANTS["r3rn"])

_BASE = dict(ku=False, ku_gap=11.0, ku_budget=8000, ku_per=None, ku_lead=False, abz=False, abz_w=1.0,
             abz_a=1.2, abz_b=-2.5, feats=False, sent_k=12, ku_floor=None, ku_norec=False, ku_max=None,
             ku_strict=False, ku_pair=None)

# recency / drift phrasing ("what did I end up using", "the last X I mentioned"): r3rn's attribute-checked recency
# pick handles these; test plants sibling-slot drift there (R-77), so the KU backstop stays off on them (ku_norec)
REC_RX = re.compile(r"\bend(?:ed)? up\b|\b(?:last|latest|most recent)\b[\w\s'-]{0,40}\bI\b[\w\s'-]{0,20}"
                    r"\b(?:mention|told|said|talk)", re.I)

VARIANTS = {
    "base": dict(_BASE),
    "ku11": dict(_BASE, ku=True),
    "ku11b9": dict(_BASE, ku=True, ku_budget=9000),
    "ku11s": dict(_BASE, ku=True, ku_lead=True),
    "ku8": dict(_BASE, ku=True, ku_gap=8.0),
    "ku11_6k": dict(_BASE, ku=True, ku_budget=6000),
    # floor: a session enters the widened window only if its score is above ku_floor (dev200: the Air Fryer
    # question added two -10.5 junk sessions)
    "ku11f": dict(_BASE, ku=True, ku_floor=-6.0),
    "ku11fn": dict(_BASE, ku=True, ku_floor=-6.0, ku_norec=True),
    "ku11fn4": dict(_BASE, ku=True, ku_floor=-6.0, ku_norec=True, ku_max=4),
    "ku11f10n4": dict(_BASE, ku=True, ku_floor=-10.0, ku_norec=True, ku_max=4),
    # strict backstop: act only when the floor actually widens the window (top session confident); otherwise the pack
    # is exactly r3rn's (test: the low-confidence "physiotherapist after the Toronto Marathon" pack got a 4th decoy
    # session under ku11fn4)
    "ku11fn4s": dict(_BASE, ku=True, ku_floor=-6.0, ku_norec=True, ku_max=4, ku_strict=True),
    "ku11f10n4s": dict(_BASE, ku=True, ku_floor=-10.0, ku_norec=True, ku_max=4, ku_strict=True),
    # + cite the best other packed session within ku_pair logits of the top when the strict backstop fires (old/new
    # pair of a KU misroute; ku->ssu CF1 is 0.64 on dev with a one-session cite)
    "ku11fn4s_p5": dict(_BASE, ku=True, ku_floor=-6.0, ku_norec=True, ku_max=4, ku_strict=True, ku_pair=5.0),
    "ku11fn4s_p11": dict(_BASE, ku=True, ku_floor=-6.0, ku_norec=True, ku_max=4, ku_strict=True, ku_pair=11.0),
    "abz": dict(_BASE, abz=True, feats=True),
    "ku11_abz": dict(_BASE, ku=True, abz=True, feats=True),
}


def _sig(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, x))))


def build(q, hx, **params) -> EvidencePack:
    p = {**_BASE, **{k: v for k, v in params.items() if not k.startswith("_")}}
    inner = dict(_R3RN)
    ku_cue = bool(KU_CUE.search(q.question))
    if p["ku_norec"] and REC_RX.search(q.question):
        ku_cue = False
    if p["ku"] and ku_cue:
        gap = p["ku_gap"]
        if p["ku_floor"] is not None:
            b0 = a_struct2.build(q, hx, **_R3RN)
            top = b0.debug["sess"][0][1] if b0.debug.get("sess") else -12.0
            gap = max(_R3RN.get("sess_gap", a_struct2.DEFAULTS["sess_gap"]), min(gap, top - p["ku_floor"]))
        base_gap = _R3RN.get("sess_gap", a_struct2.DEFAULTS["sess_gap"])
        if p["ku_strict"] and gap <= base_gap:
            ku_cue = False
    if p["ku"] and ku_cue:
        inner["sess_gap"] = gap
        inner["budget"] = p["ku_budget"]
        if p["ku_max"]:
            inner["max_sessions"] = p["ku_max"]
        inner["budget"] = p["ku_budget"]
        if p["ku_per"]:
            inner["per_sess"] = p["ku_per"]
    if p["ku"] and ku_cue and p["ku_lead"]:
        pack = _build_lead(q, hx, inner)
    else:
        pack = a_struct2.build(q, hx, _variant=params.get("_variant", ""), **inner)
    pack.variant = params.get("_variant", "")
    dbg = pack.debug or {}
    dbg["ku_cue"] = ku_cue
    if p["ku"] and ku_cue and p["ku_pair"] is not None and pack.cite_plan and dbg.get("sess"):
        top = dbg["sess"][0][1]
        in_pack = {ps.id for ps in pack.passages}
        other = [s for s, v in dbg["sess"] if s in in_pack and s not in pack.cite_plan and v >= top - p["ku_pair"]]
        if other and len(pack.cite_plan) == 1:
            pack.cite_plan = pack.cite_plan + other[:1]
            dbg["ku_pair_cite"] = other[0]

    if p["feats"] or p["abz"]:
        feats = _abst_feats(q, hx, dbg.get("ranking", []), p)
        dbg.update(feats)
        dbg["ab_r3rn"] = pack.abstain_p
        if p["abz"] and feats.get("z_sent") is not None:
            pz = _sig(-(p["abz_a"] * feats["z_sent"] + p["abz_b"]))
            pack.abstain_p = (1 - p["abz_w"]) * pack.abstain_p + p["abz_w"] * pz
    pack.debug = dbg
    return pack


def _abst_feats(q, hx, ranking: list[int], p) -> dict:
    """Pool-relative strength of the best user evidence: z-score of the best unit logit and of the best sentence
    logit (cross-encoder on the sentences of the top-k user units) against the user-unit logits of the pool."""
    from lenses.ssu.a_owner import sentences
    users = [u for u in ranking if hx.units[u].role == "user"]
    if not users:
        return {"z_user": None, "z_sent": None}
    rr = dict(hx.rerank(q.question, users))
    vals = sorted(rr.values(), reverse=True)
    mu = sum(vals) / len(vals)
    sd = (sum((v - mu) ** 2 for v in vals) / max(1, len(vals) - 1)) ** 0.5 or 1.0
    top = sorted(users, key=lambda u: -rr[u])[: p["sent_k"]]
    texts = []
    for u in top:
        for st in (sentences(hx.units[u].text) or [hx.units[u].text]):
            if len(st) >= 12:
                texts.append(st)
    sb = max(hx.rerank_texts(q.question, texts)) if texts else vals[0]
    gap2 = vals[0] - vals[1] if len(vals) > 1 else 0.0
    return {"u_best": round(vals[0], 3), "sent_best": round(float(sb), 3), "pool_mu": round(mu, 3),
            "pool_sd": round(sd, 3), "z_user": round((vals[0] - mu) / sd, 3),
            "z_sent": round((float(sb) - mu) / sd, 3), "gap2": round(gap2, 3), "n_pool": len(vals)}


def _build_lead(q, hx, inner) -> EvidencePack:
    """r3rn with a two-pass unit order: every kept session's lead unit (+/-1) first, then the rest in r3rn order.
    Implemented by calling r3rn twice: once to learn the kept sessions, then re-packing with lead units pinned."""
    from core.pack import build_pack
    pk = a_struct2.build(q, hx, **inner)
    sess = [s for s, _ in pk.debug.get("sess", [])]
    ranking = pk.debug.get("ranking", [])
    U = hx.units
    in_pack = {ps.id for ps in pk.passages}
    # sessions inside the widened window that the budget dropped
    top = pk.debug["sess"][0][1] if pk.debug.get("sess") else 0.0
    full = {**a_struct2.DEFAULTS, **inner}
    want = [s for s, v in pk.debug.get("sess", []) if v >= top - full["sess_gap"]][: full["max_sessions"]]
    missing = [s for s in want if s not in in_pack]
    if not missing:
        return pk
    lead = []
    for s in want:
        u = next((x for x in ranking if U[x].session_id == s), None)
        if u is not None:
            lead += [u] + [x for x in hx.neighbors(u, 1, 1) if x != u]
    order = list(dict.fromkeys(lead + [u for u in ranking if U[u].session_id in want]))
    passages, included, _ = build_pack(q, hx, order, budget=full["budget"], window=0, order="chrono",
                                       hint=pk.hint, max_sessions=max(full["max_sessions"], len(pk.cite_plan or [])))
    ids = {ps.id for ps in passages}
    cite = [c for c in (pk.cite_plan or []) if c in ids] or None
    return EvidencePack(question_id=pk.question_id, variant=pk.variant, route=pk.route, passages=passages,
                        hint=pk.hint, cite_plan=cite, abstain_p=pk.abstain_p, included_turns=included,
                        computed=pk.computed, debug={**pk.debug, "lead_repack": missing})
