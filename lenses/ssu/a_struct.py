"""Lens A "Struktura" round 1 for single-session-user (ssu): logit-space structural rerank + compact pack.

Builds on the round-0 draft `a_owner.py` (question grammar, owner/polarity sentence classifier, value extraction are
imported from there). Changes, one hypothesis each, measured with harness.sweep:
  * scores live in cross-encoder LOGIT space: decoy penalties are large (owner mismatch, negated, plan, stale), the
    positive structure bonuses are small, so the reranker decides between clean candidates (round-0 sigmoid scores
    let +0.15/+0.10 bonuses overrule the reranker when every logit was low: "internet plan" picked a reply to an
    assistant question that mentioned "plan").
    The logit-space feature weighting and max-pooled session score follow lens B's design (b_mentions.py), credited.
  * session score = best unit + 0.1 * second unit (max-pool: the ssu fact is a side remark, topical sums hurt).
  * compact pack: top `per_sess` units of each kept session (+/-1 neighbour gives the assistant question before a
    reply and the restatement after), primary session first, decoy-only sessions dropped.
  * abstain_p = noisy-OR of a reranker-logit prior (the generic baseline's signal, AUC 0.92 on dev) and the
    structural gates (head noun / relation absent from the user's messages, every near-top candidate a decoy).
"""
import math
import re

from core.types import EvidencePack
from lenses.ssu.a_owner import (_form_rx, _snip, _why, build_hint, head_presence, norm, parse_question, unit_class)

DEFAULTS = dict(pool=60, bm25_k=40, scan=True, scan_cap=80, rerank_cap=140,
                structure=True, hint="owner", recency=True, drop_decoys=True,
                pen_other=3.0, pen_neg=3.0, pen_plan=2.0, pen_stale=1.5, pen_stale_soft=0.3,
                b_owner=0.3, b_head=0.3, b_mod=0.15, b_verb=0.15, b_aside=0.2, b_reply=0.5, b_new=0.5,
                rec_gap=2.5, second_w=0.1, max_sessions=3, per_sess=(4, 2, 1), window=1, budget=6000,
                sess_gap=6.0, cite_second_gap=None, nb="window",
                pen_fut=0.0, b_past=0.5, pen_ask=0.0, ab_bias=2.0, ab_temp=1.0, ab_gate=True, ab_conf=0.0)


Q_PAST_RX = re.compile(r"^\s*(?:\w+\s+){0,6}?(?:did|was|were|had)\s+(?:i|we|my)\b", re.I)
FUT_RX = re.compile(r"\b(?:i'll|i will|we'll|i'm going to|i am going to|i'm planning|i plan to|i'm considering|"
                    r"i am considering|i'm thinking (?:of|about) \w+ing|i want to|i'd like to|i need to|i'm hoping to|"
                    r"i'm looking to)\b")
PAST_MARK_RX = re.compile(r"\b(?:was|were|went|spent|visited|stayed|traveled|travelled|returned|came back|got back|"
                          r"finished|completed|graduated|bought|did|ago|last (?:week|month|year|summer|winter|spring|"
                          r"fall|weekend|time))\b")
WH_ASK_RX = re.compile(r"^\s*(?:what|which|where|who|how|should|can|could|do|does|is|are|any)\b[^.!]*\?\s*$", re.I)


def tense_adjust(u, sc, q_past: bool, p) -> float:
    """Tense agreement: a past-tense question ("How long was I in Japan?") is answered by a past statement, not by a
    future plan ("I'll need a Japanese visa", "I'm considering the train"). User questions to the assistant
    ("What else should I get for my dog?") state nothing."""
    d = 0.0
    tn = norm(u.text)
    if p["pen_fut"] and q_past:
        fut, past = bool(FUT_RX.search(tn)), bool(PAST_MARK_RX.search(tn))
        if fut and not past:
            d -= p["pen_fut"]
        elif past:
            d += p["b_past"]
    if p["pen_ask"] and sc is not None and WH_ASK_RX.match(sc.sent) and not re.search(r"\d|\b[A-Z][a-z]+\b",
                                                                                     sc.sent[1:]):
        d -= p["pen_ask"]
    return d


def _sig(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, x))))


def adjust(sc, qp, p) -> float:
    d = 0.0
    if sc.owner == "mismatch":
        d -= p["pen_other"]
    elif sc.owner == "match" and sc.head_hit:
        d += p["b_owner"]
    if sc.polarity == "neg":
        d -= p["pen_neg"]
    elif sc.polarity == "plan":
        d -= p["pen_plan"]
    elif sc.polarity == "stale" and not qp.past_wanted:
        d -= p["pen_stale"] if (qp.recency or qp.attr == "name") else p["pen_stale_soft"]
    elif sc.polarity == "new" and qp.recency:
        d += p["b_new"]
    if sc.head_hit:
        d += p["b_head"]
    d += min(sc.mods_hit, 2) * p["b_mod"]
    if sc.verb_hit:
        d += p["b_verb"]
    if sc.aside:
        d += p["b_aside"]
    if sc.reply:
        d += p["b_reply"]
    return d


def candidate_pool(q, hx, qp, p) -> list[int]:
    pool, seen = [], set()

    def add(uids):
        for uid in uids:
            if uid not in seen:
                seen.add(uid)
                pool.append(uid)

    add(uid for uid, _ in hx.hybrid(q.question, k=p["pool"], roles=["user"]))
    forms = qp.head_forms()
    if len(forms) > 2 or qp.mods:
        add(uid for uid, _ in hx.bm25(" ".join([q.question] + forms + qp.mods), k=p["bm25_k"], roles=["user"]))
    if p["scan"] and qp.head:
        rxs = [re.compile(_form_rx(f)) for f in forms if f]
        hits = [u.uid for u in hx.units if u.role == "user" and any(r.search(norm(u.text)) for r in rxs)]
        if qp.mods:
            hits.sort(key=lambda uid: -sum(bool(re.search(_form_rx(m), norm(hx.units[uid].text))) for m in qp.mods))
        add(hits[: p["scan_cap"]])
    return pool[: p["rerank_cap"]]


def abstain_prob(qp, pres, top_logit: float, top_sc, all_decoy: bool, p) -> float:
    prior = _sig(-(top_logit + p["ab_bias"]) / p["ab_temp"])     # low best logit -> likely unanswerable
    if not p["ab_gate"]:
        return prior
    gate = 0.0
    if qp.strong and pres["any"] == 0:
        gate = 0.9
    elif qp.strong and pres["user"] == 0:
        gate = 0.7
    elif qp.chain and pres.get("chain", 1) == 0:
        gate = 0.8
    elif qp.strong and qp.mods and pres["full"] == 0:
        gate = 0.25
    if top_logit > p["ab_conf"]:
        # a confident reranker overrides the lexical gates: the question's wording may simply not be the user's
        # ("AM intermediate frequency" vs "AM IF ... 465 kHz"); keeps the INSUFFICIENT_EVIDENCE line out of the hint
        gate = min(gate, 0.3)
    if all_decoy:
        gate = max(gate, 0.4)
    elif top_sc is not None and not top_sc.valid:
        gate = max(gate, 0.15)
    return 1.0 - (1.0 - prior) * (1.0 - gate)


VARIANTS = {
    # logit-space structure, 3 sessions, compact (4/2/1 units per session), +/-1 window on every unit
    "s3": dict(),
    # wider pack (recall insurance at 200 sessions)
    "s4w": dict(max_sessions=4, per_sess=(6, 3, 2, 1), budget=9000),
    # explicit neighbours only around the best unit / reply units; more user units per session
    "nb3": dict(nb="smart", per_sess=(8, 3, 2)),
    "nb4": dict(nb="smart", max_sessions=4, per_sess=(8, 4, 2, 2), budget=7000),
    # + tense agreement (past question -> past statement beats future plan)
    "nb3t": dict(nb="smart", per_sess=(8, 3, 2), pen_fut=2.0),
    # session score = best unit only (no second-unit bonus: topical sessions beat side remarks)
    "nb3_0": dict(nb="smart", per_sess=(8, 3, 2), second_w=0.0),
    "nb3t0": dict(nb="smart", per_sess=(8, 3, 2), pen_fut=2.0, second_w=0.0),
    # + user wh-questions to the assistant carry no fact
    "nb3q": dict(nb="smart", per_sess=(8, 3, 2), pen_ask=1.0),
    # control: same pool + nb3 pack, no structure, no hint
    "plain_nb3": dict(nb="smart", per_sess=(8, 3, 2), structure=False, hint="none", drop_decoys=False,
                      recency=False),
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
        if p["structure"]:
            sc = unit_class(hx.units[uid], qp)
            cls[uid] = sc
            score[uid] = base + adjust(sc, qp, p) + tense_adjust(hx.units[uid], sc, q_past, p)
        else:
            score[uid] = base
    ranked = sorted(pool, key=lambda x: -score[x])

    def is_decoy(uid):
        sc = cls.get(uid)
        return sc is not None and (sc.head_hit or sc.verb_hit) and not sc.valid

    # ---- session scores (max-pool over units, decoy units excluded when structure is on)
    by_sess: dict[str, list[int]] = {}
    for uid in ranked:
        by_sess.setdefault(hx.units[uid].session_id, []).append(uid)
    sess_score = {}
    for sid, uids in by_sess.items():
        clean = [u for u in uids if not (p["drop_decoys"] and is_decoy(u))]
        if not clean:
            continue
        s = score[clean[0]] + (p["second_w"] * max(0.0, score[clean[1]] + 5.0) if len(clean) > 1 else 0.0)
        sess_score[sid] = s
    sess_rank = sorted(sess_score, key=lambda s: -sess_score[s])

    # ---- primary session (recency cue -> latest clean session within rec_gap of the best)
    primary = sess_rank[0] if sess_rank else None
    if primary and p["structure"] and p["recency"] and qp.recency:
        best = sess_score[primary]
        near = [s for s in sess_rank if sess_score[s] >= best - p["rec_gap"]]
        if near:
            primary = max(near, key=lambda s: (hx.sess_by_id[s]["sess_idx"], sess_score[s]))
    keep = []
    if primary:
        keep = [primary] + [s for s in sess_rank if s != primary
                            and sess_score[s] >= sess_score[sess_rank[0]] - p["sess_gap"]]
    keep = keep[: p["max_sessions"]]

    order = []
    for i, sid in enumerate(keep):
        n = p["per_sess"][min(i, len(p["per_sess"]) - 1)]
        units = by_sess[sid][:n]          # decoy units of a kept session stay (penalised, so ranked lower)
        if p["nb"] == "smart":
            # explicit neighbours, packed with window=0: +/-1 around the session's best unit (assistant question
            # before a reply, restatement after), the preceding assistant chunk for reply units, nothing else
            for j, uid in enumerate(units):
                if j == 0:
                    order.extend([uid] + [x for x in hx.neighbors(uid, 1, 1) if x != uid])
                elif cls.get(uid) is not None and cls[uid].reply:
                    order.extend([uid] + [x for x in hx.neighbors(uid, 1, 0) if x != uid])
                else:
                    order.append(uid)
        else:
            order.extend(units)

    decoys = []
    if p["structure"] and p["drop_decoys"] and primary:
        top = sess_score[sess_rank[0]]
        for uid in ranked:
            if is_decoy(uid) and len(decoys) < 2 and rr.get(uid, -12) >= top - 3.0:
                u = hx.units[uid]
                decoys.append({"uid": uid, "date": u.date_str[:10], "snip": _snip(cls[uid].sent, 90),
                               "why": _why(cls[uid])})

    top_uid = by_sess[primary][0] if primary else None
    top_sc = cls.get(top_uid) if top_uid is not None else None
    top_logit = max(rr.values()) if rr else -12.0
    near_top = [u for u in ranked[:6] if rr.get(u, -12) >= top_logit - 2.0 and cls.get(u) and cls[u].head_hit]
    all_decoy = bool(near_top) and all(is_decoy(u) for u in near_top)
    pres = head_presence(hx, qp) if qp.head else {"user": 1, "any": 1, "full": 1, "chain": 1}
    abstain_p = abstain_prob(qp, pres, top_logit, top_sc, all_decoy, {**p, "ab_gate": p["ab_gate"] and p["structure"]})

    hint = build_hint(qp, {"hint": p["hint"], "recency": p["recency"]}, decoys, abstain_p, pres, []) \
        if p["structure"] else ""

    passages, included, chars = build_pack(q, hx, order, budget=p["budget"],
                                           window=0 if p["nb"] == "smart" else p["window"], order="chrono",
                                           hint=hint, max_sessions=p["max_sessions"])
    cite = [primary] if primary else []
    if p["cite_second_gap"] is not None and len(keep) > 1 and \
            sess_score[keep[0]] - sess_score[keep[1]] < p["cite_second_gap"]:
        cite.append(keep[1])
    cite = [c for c in cite if any(ps.id == c for ps in passages)] or None
    computed = None
    if top_sc is not None and qp.attr == "name" and top_sc.values and top_sc.valid:
        computed = top_sc.values[0]
    dbg = {"ranking": ranked, "qparse": {k: getattr(qp, k) for k in ("anchor", "head", "mods", "chain", "verbs",
                                                                       "attr", "recency", "source")},
           "presence": pres, "top_logit": round(top_logit, 2), "decoys": [d["uid"] for d in decoys],
           "sess": [(s, round(sess_score[s], 2)) for s in sess_rank[:5]],
           "top": [(uid, round(score[uid], 2), round(rr.get(uid, -12), 2),
                    cls[uid].owner if uid in cls else "", cls[uid].polarity if uid in cls else "")
                   for uid in ranked[:6]]}
    return EvidencePack(question_id=q.question_id, variant=params.get("_variant", ""), route="ssu",
                        passages=passages, hint=hint, cite_plan=cite, abstain_p=abstain_p,
                        included_turns=included, computed=computed, debug=dbg)
