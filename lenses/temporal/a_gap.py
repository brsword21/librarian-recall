"""Temporal lens A, round 2 ("Struktura"): gap questions through an improved event-pair calendar, everything else
through lens B (b_sem sem_turns9), exactly like the integrator's merged_r1 split (credit: merged_r1 routing, B's
whole-question route). Only the gap route changes here.

Gap route = a_calendar (round 1) plus:
  * conf2   - confidence of the computed gap:
              - an alternative mention whose only date is its session date ("Then he asked me about varroa treatment
                timing", "I think our anniversary is Wednesday") is discounted when the pick carries an explicit or
                relative date: it does not say WHEN the event happened;
              - no computed value when a pick is someone else's event or negated, when the gap is 0 (two different
                events on the same day is almost always a wrong pick), or when a week / month count sits near a
                rounding boundary (the hint then gives the day count and the fractional value).
  * tight   - when the computed gap is confident, the pack holds only the two event sessions (pick sentence, the
              session's other user mentions of the event, its first user turn) plus at most one runner-up session
              when an event's margin is small. Passages ~= cite set, so the metric does not depend on whether
              Reactor's citations are post-filtered (Sall).
  * ce_tb   - cross-encoder tie-break (hx.rerank_texts on the event phrase vs the top mention sentences) for events
              whose rule margin is small.
  * abst    - abstain_p 0.8 when an event word never occurs in the history ("iPad").
  * a_kinds - non-gap questions that NAME their events ("how many days ago did I X", "which first, X or Y", "how
              long had I been X when I Y") also take this route instead of B; questions without events stay with B.
  * abst_b  - for those extra questions abstain_p comes from B's pipeline (B's continuous scale ranks better).
  * ago_gate - no computed "N ago" from an assistant sentence dated only by its session, or a 0 from a session date.
Best variant (round 2): ct_evgba. Report: lenses/temporal/a_report.md.
"""
from __future__ import annotations

import math
import re

from core.types import EvidencePack
from core.loader import load_lens
from lenses.temporal import a_calendar as A

_GAP = re.compile(r"how many (days|weeks|months)\b.*\b(apart|after|before|between|passed)|days? (apart|after|before)",
                  re.I)
_B = None

_R1 = dict(A.VARIANTS["ftb_bm25"])
VARIANTS = {
    "r1": {**_R1},                                                      # = merged_r1 gapA_semB
    "conf2": {**_R1, "conf2": True},
    "tight": {**_R1, "tight": True},
    "ct": {**_R1, "conf2": True, "tight": True},
    "ctc": {**_R1, "conf2": True, "tight": True, "ce_tb": True},
    "ct45": {**_R1, "conf2": True, "tight": True, "tight_budget": 4500},
    "ctn": {**_R1, "conf2": True, "tight": True, "fill_k": 2, "budget": 6500},
    # + event-route for non-gap questions that name their events ("how many days ago did I X", "which did I do
    #   first, X or Y?"); questions without parsed events stay with B
    "ct_ev": {**_R1, "conf2": True, "tight": True, "a_kinds": ["ago", "order"]},
    "ct_ago": {**_R1, "conf2": True, "tight": True, "a_kinds": ["ago"]},
    "ct_ord": {**_R1, "conf2": True, "tight": True, "a_kinds": ["order"]},
    "ct_evg": {**_R1, "conf2": True, "tight": True, "a_kinds": ["ago", "order", "gap"]},
    # same, abstain_p of the extra (non-gap) questions taken from B's whole pipeline (B's scale separates better)
    "ct_evgb": {**_R1, "conf2": True, "tight": True, "a_kinds": ["ago", "order", "gap"], "abst_b": True},
    "ct_evgbc": {**_R1, "conf2": True, "tight": True, "a_kinds": ["ago", "order", "gap"], "abst_b": True,
                 "ce_tb": True},
    "ct_evgba": {**_R1, "conf2": True, "tight": True, "a_kinds": ["ago", "order", "gap"], "abst_b": True,
                 "ago_gate": True},
    # + order questions: pack only the event sessions when every event has a clear own pick (rejected: dev PA drop)
    "ct_evo": {**_R1, "conf2": True, "tight": True, "a_kinds": ["ago", "order"], "tight_order": 0.08},
}


def _b():
    global _B
    if _B is None:
        _B = load_lens("lenses/temporal/b_sem.py")
    return _B


# --------------------------------------------------------------------------------------------------------------------
# picks with the round-2 confidence rules
# --------------------------------------------------------------------------------------------------------------------
def _margin(best, ms, used, sess_disc: float) -> float:
    alt = None
    for m in ms:
        if (m.uid, m.sentence) in used or m.value == best.value:
            continue
        s = m.score
        if sess_disc and m.kind == "session" and best.kind != "session":
            s -= sess_disc
        if alt is None or s > alt:
            alt = s
    return (best.score - alt) if alt is not None else 1.0


def _ce_tiebreak(hx, ev: str, ms: list, top: int, w: float) -> list:
    """Re-order the top mentions of one event by rule score + w * sigmoid(CE(event, sentence))."""
    head = ms[:top]
    try:
        ce = hx.rerank_texts(ev, [m.sentence for m in head])
    except Exception:
        return ms
    for m, c in zip(head, ce):
        m.score = m.score + w * (1 / (1 + math.exp(-c)) - 0.5)
        m.why = list(m.why) + [f"ce{c:+.1f}"]
    head.sort(key=lambda m: -m.score)
    return head + ms[top:]


def _absent_terms(hx, events: list[str]) -> list[str]:
    """Content words of the events that never occur in the history ("iPad"): the event cannot be in it."""
    c = A._hxc(hx)
    out = []
    for ev in events:
        out += [t for t, _ in A.toks(re.sub(r"\([^)]*\)", " ", ev)) if c.df.get(t, 0) == 0]
    return out


def analyse2(q, hx, params: dict):
    qp = A.parse_question(q.question)
    pool = params.get("pool", 40)
    dense = params.get("dense", False)
    min_score, min_margin = params.get("min_score", 0.55), params.get("min_margin", 0.08)
    sess_disc = params.get("sess_disc", 0.15) if params.get("conf2") else 0.0
    picks, alts, allm, used = [], {}, {}, set()
    for ev in qp.events[:6]:
        ms = A.candidate_mentions(q, hx, ev, pool, qp, dense)
        best, margin = A.pick(ms, used)
        if params.get("ce_tb") and best is not None and margin < params.get("ce_margin", 0.15) and len(ms) > 1:
            ms = _ce_tiebreak(hx, ev, ms, params.get("ce_top", 6), params.get("ce_w", 0.6))
            best, margin = A.pick(ms, used)
        if best is not None:
            if params.get("conf2"):
                margin = _margin(best, ms, used, sess_disc)
            used.add((best.uid, best.sentence))
        picks.append((ev, best, margin))
        allm[ev] = ms
        alts[ev] = [m for m in ms if best is None or m.value != best.value][:3]
    computed, conf, gate = None, False, None
    if qp.kind == "gap" and len(picks) >= 2 and picks[0][1] and picks[1][1]:
        (e1, m1, g1), (e2, m2, g2) = picks[0], picks[1]
        computed = A.gap_in_unit(m1.value, m2.value, qp.unit)
        conf = min(m1.score, m2.score) >= min_score and min(g1, g2) >= min_margin
        if qp.signed and m2.value < m1.value:
            conf, gate = False, "sign"
        if qp.vague:
            conf, gate = False, "vague"
        absent = _absent_terms(hx, qp.events[:2]) if params.get("conf2") else []
        if absent:
            conf, gate = False, "absent:" + absent[0]
        if params.get("conf2") and conf:
            days = abs((m2.value - m1.value).days)
            bad = [w for _, m, _ in picks[:2] for w in m.why if w in ("other", "neg", "poss-other")]
            if bad:
                conf, gate = False, "who:" + bad[0]
            elif days == 0:
                conf, gate = False, "zero"
            elif qp.unit in ("week", "month"):
                per = 7 if qp.unit == "week" else 30.44
                frac = days / per - math.floor(days / per)
                if abs(frac - 0.5) < params.get("round_tol", 0.15):
                    conf, gate = False, "rounding"
    elif qp.kind == "ago" and picks and picks[0][1]:
        m1 = picks[0][1]
        computed = A.gap_in_unit(m1.value, q.date.date(), qp.unit)
        conf = m1.score >= min_score and picks[0][2] >= min_margin
        if params.get("ago_gate") and conf:
            bad = [w for w in m1.why if w in ("other", "neg", "poss-other", "asst-session-date")]
            if bad:
                conf, gate = False, "who:" + bad[0]
            elif computed == 0 and m1.kind == "session":
                conf, gate = False, "zero"          # "how many months ago" answered by the question's own day
    return qp, picks, alts, allm, computed, conf, gate


# --------------------------------------------------------------------------------------------------------------------
# gap route
# --------------------------------------------------------------------------------------------------------------------
def _sess_units(hx, sid: str) -> list[int]:
    s = hx.sess_by_id[sid]
    out = []
    for t in s["turns"]:
        out += hx.turn_units(t["turn_id"])
    return out


def build_gap(q, hx, params: dict) -> EvidencePack:
    from core.pack import build_pack
    if params.get("topic", False):
        qp0 = A.parse_question(q.question)
        if qp0.kind == "other" or (qp0.kind == "order" and not qp0.events):
            return A.build_topic(q, hx, qp0, params)
    qp, picks, alts, allm, computed, conf, gate = analyse2(q, hx, params)
    ranked: list[int] = []
    cite: list[str] = []

    def add(uid):
        if uid is not None and uid not in ranked:
            ranked.append(uid)

    for ev, m, margin in picks:
        if m is not None:
            add(m.uid)
            if m.session_id not in cite:
                cite.append(m.session_id)

    tight = params.get("tight") and conf and len(cite) >= 1
    if params.get("tight_order") is not None and qp.kind == "order" and picks and all(m is not None for _, m, _ in picks):
        tight = tight or (min(g for _, _, g in picks) >= params["tight_order"] and
                          not any(w in ("other", "neg", "poss-other") for _, m, _ in picks for w in m.why))
    if tight:
        allowed = set(cite)
        # runner-up: the best alternative session of an event whose margin is small
        for ev, m, margin in picks:
            if m is not None and margin < params.get("runner_margin", 0.25):
                a = next((x for x in allm.get(ev, []) if x.session_id not in allowed and "other" not in x.why), None)
                if a is not None and len(allowed) < len(cite) + params.get("max_runner", 1):
                    allowed.add(a.session_id)
                    add(a.uid)
        for ev, m, margin in picks:                 # the session's other user mentions of the event
            for x in [x for x in allm.get(ev, []) if x.role == "user" and x.session_id in allowed][:params.get("more", 2)]:
                add(x.uid)
        for sid in list(dict.fromkeys(cite)):       # dev gold turns are mostly the opening user turn
            fu = next((u for u in _sess_units(hx, sid) if hx.units[u].role == "user"), None)
            add(fu)
        budget = params.get("tight_budget", 6000)
    else:
        for ev, m, margin in picks:
            for x in [x for x in allm.get(ev, []) if x.role == "user"][: params.get("more", 0)]:
                add(x.uid)
        if params.get("first_turn"):
            for ev, m, margin in picks:
                if m is not None:
                    fu = next((u for u in _sess_units(hx, m.session_id) if hx.units[u].role == "user"), None)
                    add(fu)
        for ev, m, margin in picks:
            for a in alts.get(ev, [])[: params.get("alts", 1)]:
                add(a.uid)
                if margin < params.get("min_margin", 0.08) and a.session_id not in cite and a.score > 0.4:
                    cite.append(a.session_id)
        fill = []
        try:
            fill = [u for u, _ in A.retrieve(hx, q.question, max(40, params.get("fill_k", 6) * 3),
                                             params.get("dense", False))]
        except Exception:
            pass
        for uid in fill[: params.get("fill_k", 6)]:
            add(uid)
        budget = params.get("budget", 8000)
    hint = make_hint2(q, qp, picks, computed, conf, gate) if params.get("conf2") else \
        A.make_hint(q, qp, picks, computed, conf)
    missing = A.missing_terms(q, hx)
    if missing:
        hint += ("\n" if hint else "") + "Note: these words of the question never appear in the history: " + \
            ", ".join(missing) + "."
    passages, included, _chars = build_pack(q, hx, ranked, budget=budget, window=params.get("window", 1),
                                            order="chrono", hint=hint)
    pids = {p.id for p in passages if p.id}
    cite_plan = [s for s in cite if s in pids] or None
    out_computed = str(computed) if (computed is not None and params.get("compute", True) and conf) else None
    return EvidencePack(
        question_id=q.question_id, variant=params.get("_variant", ""), route="temporal",
        passages=passages, hint=hint, cite_plan=cite_plan,
        abstain_p=max(A.abstain_estimate(q, hx, picks, missing, allm),
                      0.8 if (params.get("conf2") and (gate or "").startswith("absent")) else 0.0),
        included_turns=included, computed=out_computed,
        debug={"ranking": ranked, "parse": {"kind": qp.kind, "unit": qp.unit, "events": qp.events},
               "picks": [(ev, m.value.isoformat() if m else None, round(m.score, 3) if m else None,
                          round(g, 3), m.why if m else None) for ev, m, g in picks],
               "computed_raw": computed, "conf": conf, "gate": gate, "tight": bool(tight)})


def make_hint2(q, qp, picks, computed, conf, gate) -> str:
    h = A.make_hint(q, qp, picks, computed, conf)
    if gate and (gate in ("sign", "zero") or gate.startswith(("who:", "absent:"))):
        # the pair is inconsistent (wrong order, same day, someone else's event, unknown word): do not show a number
        # that Reactor would copy; keep the list of mentions found
        h = "\n".join(x for x in h.splitlines() if not x.startswith("Difference between"))
    if gate == "rounding" and len(picks) >= 2 and picks[0][1] and picks[1][1]:
        dd = abs((picks[1][1].value - picks[0][1].value).days)
        per = 7 if qp.unit == "week" else 30.44
        h = h.replace(" (Low confidence: check that these are the right mentions.)",
                      f" ({dd} days = {dd / per:.1f} {qp.unit}s.)")
    return h


def _use_a(q, params: dict) -> bool:
    if _GAP.search(q.question):
        return True
    kinds = params.get("a_kinds")
    if not kinds:
        return False
    qp = A.parse_question(q.question)
    return qp.kind in kinds and len(qp.events) >= (1 if qp.kind == "ago" else 2)


def build(q, hx, **params) -> EvidencePack:
    if not _use_a(q, params):
        B = _b()
        v = params.get("b_variant", "sem_turns9")
        pack = B.build(q, hx, _variant=v, **B.VARIANTS[v])
        pack.variant = params.get("_variant", "")
        pack.debug = {**(pack.debug or {}), "merged_from": f"b_sem:{v}"}
        return pack
    pack = build_gap(q, hx, params)
    pack.debug["merged_from"] = "a_gap"
    if params.get("abst_b") and not _GAP.search(q.question):
        B = _b()
        v = params.get("b_variant", "sem_turns9")
        try:
            pack.abstain_p = B.build(q, hx, _variant=v, **B.VARIANTS[v]).abstain_p
            pack.debug["abstain_from"] = "b_sem"
        except Exception:
            pass
    return pack
