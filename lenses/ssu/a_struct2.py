"""Lens A "Struktura" round 2 for single-session-user (ssu). Builds on round-1 `a_struct.py` nb3t0 (routed winner).

BEST: `r3rn` (see lenses/ssu/a_report.md). Round-2 changes, one hypothesis per variant:
  * r2      nb3t0 with no abstention line in the hint (integrator policy: never tell Reactor to abstain).
  * pair*   cite the runner-up session too when it scores within `pair_gap` logits of the best (KU update pair).
            Wins on routed dev, but it fires on 10 of 12 audited TEST ssu packs (planted drift / decoys), so it is
            OFF in the final variant. *t = pack only cited sessions; *h = dated "later value is current" hint.
  * ru      outlier runner-up: s2 > -6 and s2 - s3 > 3 -> the evidence spans two sessions (ku / multi misroute);
            pack its best units (not cited). Fires on 28/34 ku, 9/13 multi, 3/57 ssa, 8/62 ssu routed dev+dev200
            questions; the runner-up is gold in 27/9/0/1 of them (scratch_a/r2_s2.py).
  * ab_all  abstention prior sees max(best user logit, best ALL-roles logit - 1) (generic hybrid_rr pool, cached);
            ab_platt = Platt scaling fitted on pooled dev+dev200 routed-ssu rows (scratch_a/r2_platt.py).
  * asst_m  the best ALL-roles unit is an assistant unit in ANOTHER session and beats the best user logit by 4:
            cite that session (ssa misroute). asst_add: pack the best assistant unit (+/-1) of a kept session when
            it beats the best user logit by 2 (ssa misroutes cite the right session but missed the answer turn).
  * rec_need  recency questions ("end up", "now", "last ... I mentioned") jump to a later near-top session only if
            that session itself states the attribute (affirmed own unit with a head / modifier hit). Test: the
            "speaker impedance / AM IF did I end up using" packs jumped to a later resistor-only session.
"""
import math
import re

from core.types import EvidencePack
from lenses.ssu.a_owner import _snip, _why, build_hint, head_presence, parse_question, unit_class
from lenses.ssu.a_struct import (DEFAULTS as _D1, Q_PAST_RX, abstain_prob, adjust, candidate_pool, tense_adjust)

DEFAULTS = dict(_D1, nb="smart", per_sess=(8, 3, 2), pen_fut=2.0, second_w=0.0,
                ab_hint=False, pair_gap=None, pair_units=8, tight=False, pair_hint=False, latest_hint=False,
                ru=False, ru_abs=-6.0, ru_out=3.0, ru_units=3, ru_cite="no", pair_mode="both", rec_need=None,
                ab_all=None, ab_platt=None, asst_m=None, asst_add=None)

VARIANTS = {
    "r2": dict(),
    "pair2": dict(pair_gap=2.0),
    "pair3": dict(pair_gap=3.0),
    "pair3t": dict(pair_gap=3.0, tight=True),
    "pair3h": dict(pair_gap=3.0, pair_hint=True),
    "pair3th": dict(pair_gap=3.0, tight=True, pair_hint=True),
    # outlier runner-up: s2 > ru_abs and s2 - s3 > ru_out -> the evidence spans two sessions (ku / multi misroute:
    # fires on 28/34 ku, 9/13 multi, 3/57 ssa, 8/62 ssu routed dev+dev200 questions; the runner-up is gold in 27/9/0/1)
    "ru": dict(pair_gap=3.0, ru=True),
    "ruc": dict(pair_gap=3.0, ru=True, ru_cite="all"),
    "ruch": dict(pair_gap=3.0, ru=True, ru_cite="head"),
    "ruct": dict(pair_gap=3.0, ru=True, ru_cite="all", tight=True),
    "rucht": dict(pair_gap=3.0, ru=True, ru_cite="head", tight=True),
    # abstention prior also sees the best ALL-roles logit (generic hybrid_rr pool, cached): an answer the assistant
    # restated (or an ssa misroute) is evidence that the history covers the question
    "ru_ab0": dict(pair_gap=3.0, ru=True, ab_all=0.0),
    "ru_ab1": dict(pair_gap=3.0, ru=True, ab_all=1.0),
    "ru_ab2": dict(pair_gap=3.0, ru=True, ab_all=2.0),
    # best: ru + all-roles abstention prior + Platt calibration fitted on pooled dev+dev200 routed-ssu rows
    # (scratch_a/r2_platt.py: log-loss 0.299 -> 0.164; monotone, AUC unchanged)
    "r2best": dict(pair_gap=3.0, ru=True, ab_all=1.0, ab_platt=(0.5, -2.0)),
    # assistant-content detector: the best ALL-roles unit is an assistant unit in another session and beats the best
    # user logit by asst_m -> the question is about what the assistant said (ssa misroute): cite that session
    "r2best_a2": dict(pair_gap=3.0, ru=True, ab_all=1.0, ab_platt=(0.5, -2.0), asst_m=2.0),
    "r2best_a4": dict(pair_gap=3.0, ru=True, ab_all=1.0, ab_platt=(0.5, -2.0), asst_m=4.0),
    # + the best assistant unit (+/-1) whenever it beats the best user logit by asst_add, in any kept session: the
    # cited session is right on ssa misroutes (CF1 1.0) but the answer sits in an assistant turn we did not pack
    "r2best_a4x4": dict(pair_gap=3.0, ru=True, ab_all=1.0, ab_platt=(0.5, -2.0), asst_m=4.0, asst_add=4.0),
    "r2best_a4x2": dict(pair_gap=3.0, ru=True, ab_all=1.0, ab_platt=(0.5, -2.0), asst_m=4.0, asst_add=2.0),
    # round-2 final candidates: no 2-session pair cite (it fires on most test ssu packs, see report)
    "r3": dict(ru=True, ab_all=1.0, ab_platt=(0.5, -2.0), asst_m=4.0, asst_add=2.0),
    # FINAL (round 2): r3 + attribute-checked recency jump
    "r3rn": dict(ru=True, ab_all=1.0, ab_platt=(0.5, -2.0), asst_m=4.0, asst_add=2.0, rec_need="head"),
    "r3rn_x0": dict(ru=True, ab_all=1.0, ab_platt=(0.5, -2.0), asst_m=4.0, asst_add=0.0, rec_need="head"),
    "r3rn_x1": dict(ru=True, ab_all=1.0, ab_platt=(0.5, -2.0), asst_m=4.0, asst_add=1.0, rec_need="head"),
    "r3g4": dict(ru=True, ab_all=1.0, ab_platt=(0.5, -2.0), asst_m=4.0, asst_add=2.0, sess_gap=4.0),
    "r3g3": dict(ru=True, ab_all=1.0, ab_platt=(0.5, -2.0), asst_m=4.0, asst_add=2.0, sess_gap=3.0),
    "r3new": dict(pair_gap=3.0, pair_mode="newest", ru=True, ab_all=1.0, ab_platt=(0.5, -2.0), asst_m=4.0,
                  asst_add=2.0),
    "r3new2": dict(pair_gap=2.0, pair_mode="newest", ru=True, ab_all=1.0, ab_platt=(0.5, -2.0), asst_m=4.0,
                   asst_add=2.0),
    # r2best + dated pair hint (later value current; earlier for 'previous' questions) - Reactor-side only
    "r2best_h": dict(pair_gap=3.0, ru=True, ab_all=1.0, ab_platt=(0.5, -2.0), pair_hint=True),
}


def build(q, hx, **params) -> EvidencePack:
    p = {**DEFAULTS, **{k: v for k, v in params.items() if not k.startswith("_")}}
    from core.pack import build_pack

    qp = parse_question(q.question)
    q_past = bool(Q_PAST_RX.search(q.question))
    pool = candidate_pool(q, hx, qp, p)
    rr = dict(hx.rerank(q.question, pool)) if pool else {}

    cls, score = {}, {}
    for uid in pool:
        base = rr.get(uid, -12.0)
        sc = unit_class(hx.units[uid], qp)
        cls[uid] = sc
        score[uid] = base + adjust(sc, qp, p) + tense_adjust(hx.units[uid], sc, q_past, p)
    ranked = sorted(pool, key=lambda x: -score[x])

    def is_decoy(uid):
        sc = cls.get(uid)
        return sc is not None and (sc.head_hit or sc.verb_hit) and not sc.valid

    by_sess: dict[str, list[int]] = {}
    for uid in ranked:
        by_sess.setdefault(hx.units[uid].session_id, []).append(uid)
    sess_score = {}
    for sid, uids in by_sess.items():
        clean = [u for u in uids if not (p["drop_decoys"] and is_decoy(u))]
        if clean:
            sess_score[sid] = score[clean[0]]
    sess_rank = sorted(sess_score, key=lambda s: -sess_score[s])
    sidx = lambda s: hx.sess_by_id[s]["sess_idx"]

    top_user = max(rr.values()) if rr else -12.0
    ra = {}
    if p["ab_all"] is not None or p["asst_m"] is not None:
        ra = dict(hx.rerank(q.question, [u for u, _ in hx.hybrid(q.question, k=100, pool=300)]))
    asst_sess, asst_uid = None, None
    a_uid = max((u for u in ra if hx.units[u].role == "assistant"), key=lambda u: ra[u], default=None) if ra else None
    if p["asst_m"] is not None and a_uid is not None and sess_rank:
        if ra[a_uid] > top_user + p["asst_m"] and hx.units[a_uid].session_id != sess_rank[0]:
            asst_uid, asst_sess = a_uid, hx.units[a_uid].session_id

    primary = sess_rank[0] if sess_rank else None
    if primary and p["recency"] and qp.recency:
        best = sess_score[primary]
        near = [s for s in sess_rank if sess_score[s] >= best - p["rec_gap"]]
        if p["rec_need"]:
            # the later session must itself state the attribute: an affirmed own unit with a head / modifier hit
            # (test: "speaker impedance did I end up using" jumped to a later session about a resistor that only
            # shares "National R-77"); the best session always stays eligible
            def states(sid):
                for u in by_sess[sid][:3]:
                    c = cls.get(u)
                    if c is not None and c.valid and c.owner != "mismatch" and c.polarity not in ("stale",) \
                            and (c.head_hit or c.mods_hit > 0):
                        return True
                return False
            near = [s for s in near if s == primary or states(s)]
        if near:
            primary = max(near, key=lambda s: (sidx(s), sess_score[s]))

    # ---- KU-style pair: the runner-up is nearly as good as the best -> the attribute was stated twice
    pair = None
    if primary and p["pair_gap"] is not None and len(sess_rank) > 1:
        a, b = sess_rank[0], sess_rank[1]
        if sess_score[a] - sess_score[b] < p["pair_gap"]:
            pair = sorted({a, b} if primary in (a, b) else {primary, a}, key=sidx)
    if pair and p["pair_mode"] == "newest":
        # test histories plant drift/decoys: a close runner-up is usually NOT a second gold session there
        # (pair fired on 10/12 audited test ssu packs) -> pack both, cite one: the newest (oldest for 'previous')
        primary = pair[0] if qp.past_wanted else pair[-1]
    cite = list(pair) if (pair and p["pair_mode"] == "both") else ([primary] if primary else [])
    if asst_sess:
        pair, cite = None, [asst_sess]

    # ---- outlier runner-up (evidence spread over two sessions: update pair or a second instance)
    runner = None
    if primary and p["ru"] and len(sess_rank) > 1 and not asst_sess:
        sc3 = [sess_score[s] for s in sess_rank[:3]] + [-12.0] * 3
        if sc3[1] > p["ru_abs"] and sc3[1] - sc3[2] > p["ru_out"]:
            runner = sess_rank[1] if primary == sess_rank[0] else sess_rank[0]
            if runner not in cite:
                ru_u = next((u for u in by_sess[runner] if not is_decoy(u)), None)
                ru_sc = cls.get(ru_u)
                if p["ru_cite"] == "all" or (p["ru_cite"] == "head" and ru_sc is not None and ru_sc.head_hit
                                             and ru_sc.valid and ru_sc.owner != "mismatch"):
                    cite = sorted(cite + [runner], key=sidx)

    keep = []
    if primary:
        keep = ([asst_sess] if asst_sess else []) + [primary] + [s for s in (pair or []) if s != primary]
        if runner and runner not in keep:
            keep.append(runner)
        keep += [s for s in sess_rank if s not in keep and sess_score[s] >= sess_score[sess_rank[0]] - p["sess_gap"]]
    keep = keep[: max(p["max_sessions"], len(cite))]
    if p["tight"]:
        keep = [s for s in keep if s in cite or s == runner]

    order = []
    if asst_sess:
        a_units = sorted((u for u in ra if hx.units[u].session_id == asst_sess), key=lambda u: -ra[u])[:3]
        for j, uid in enumerate(a_units):
            order.extend([uid] + [x for x in hx.neighbors(uid, 1, 1 if j == 0 else 0) if x != uid])
        keep = [k for k in keep if k != asst_sess]
    for i, sid in enumerate(keep):
        n = p["pair_units"] if (pair and sid in pair) else p["ru_units"] if sid == runner else \
            p["per_sess"][min(i, len(p["per_sess"]) - 1)]
        units = by_sess[sid][:n]
        for j, uid in enumerate(units):
            if j == 0:
                order.extend([uid] + [x for x in hx.neighbors(uid, 1, 1) if x != uid])
            elif cls.get(uid) is not None and cls[uid].reply:
                order.extend([uid] + [x for x in hx.neighbors(uid, 1, 0) if x != uid])
            else:
                order.append(uid)
        if p["asst_add"] is not None and a_uid is not None and hx.units[a_uid].session_id == sid \
                and ra[a_uid] > top_user + p["asst_add"]:
            order.extend([a_uid] + [x for x in hx.neighbors(a_uid, 1, 1) if x != a_uid])

    decoys = []
    if p["drop_decoys"] and primary:
        top = sess_score[sess_rank[0]]
        for uid in ranked:
            if is_decoy(uid) and len(decoys) < 2 and rr.get(uid, -12) >= top - 3.0:
                u = hx.units[uid]
                decoys.append({"uid": uid, "date": u.date_str[:10], "snip": _snip(cls[uid].sent, 90),
                               "why": _why(cls[uid])})

    top_uid = by_sess[primary][0] if primary else None
    top_sc = cls.get(top_uid) if top_uid is not None else None
    top_logit = top_user
    if p["ab_all"] is not None and ra:
        top_logit = max(top_logit, max(ra.values()) - p["ab_all"])
    near_top = [u for u in ranked[:6] if rr.get(u, -12) >= top_logit - 2.0 and cls.get(u) and cls[u].head_hit]
    all_decoy = bool(near_top) and all(is_decoy(u) for u in near_top)
    pres = head_presence(hx, qp) if qp.head else {"user": 1, "any": 1, "full": 1, "chain": 1}
    abstain_p = abstain_prob(qp, pres, top_logit, top_sc, all_decoy, p)
    if p["ab_platt"]:
        a, c = p["ab_platt"]
        z = math.log(min(max(abstain_p, 1e-4), 1 - 1e-4) / (1 - min(max(abstain_p, 1e-4), 1 - 1e-4)))
        abstain_p = 1.0 / (1.0 + math.exp(-(a * z + c)))

    # abstain_p is NOT passed to the hint (no abstention line to Reactor)
    hint = build_hint(qp, {"hint": p["hint"], "recency": p["recency"]}, decoys,
                      abstain_p if p["ab_hint"] else 0.0, pres, [])
    two = pair or (sorted([primary, runner], key=sidx) if runner else None)
    if two and p["pair_hint"]:
        d1, d2 = (hx.sess_by_id[s]["date_str"][:10] for s in two)
        if qp.past_wanted:
            hint += f" The user mentioned this on {d1} and again on {d2}; if the values differ, {d1} is the earlier one."
        elif not qp.recency:
            hint += f" The user mentioned this on {d1} and again on {d2}; if the values differ, the later one ({d2}) is current."

    passages, included, chars = build_pack(q, hx, order, budget=p["budget"], window=0, order="chrono", hint=hint,
                                           max_sessions=max(p["max_sessions"], len(cite)))
    cite = [c for c in cite if any(ps.id == c for ps in passages)] or None
    computed = None
    if top_sc is not None and qp.attr == "name" and top_sc.values and top_sc.valid and not pair:
        computed = top_sc.values[0]
    dbg = {"ranking": ranked, "qparse": {k: getattr(qp, k) for k in ("anchor", "head", "mods", "chain", "verbs",
                                                                       "attr", "recency", "past_wanted", "source")},
           "presence": pres, "top_logit": round(top_logit, 2), "top_user": round(top_user, 2), "decoys": [d["uid"] for d in decoys],
           "pair": pair, "runner": runner, "asst": asst_sess, "sess": [(s, round(sess_score[s], 2)) for s in sess_rank[:5]],
           "top": [(uid, round(score[uid], 2), round(rr.get(uid, -12), 2),
                    cls[uid].owner if uid in cls else "", cls[uid].polarity if uid in cls else "")
                   for uid in ranked[:6]]}
    return EvidencePack(question_id=q.question_id, variant=params.get("_variant", ""), route="ssu",
                        passages=passages, hint=hint, cite_plan=cite, abstain_p=abstain_p,
                        included_turns=included, computed=computed, debug=dbg)
