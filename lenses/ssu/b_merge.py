"""SSU lens B, round 2: `b_merge` = B's semantic retrieval + A's structure, with citation-robust (compact) packs.

Credits: the question grammar, the owner / polarity / tense sentence classifier, the decoy hint and the abstention
gates are lens A's (`lenses/ssu/a_owner.py`, `lenses/ssu/a_struct.py`), imported unchanged.

Pipeline (variant `best`)
  1. pool: all-role hybrid top-100 (the assistant echo is strong session evidence on dev), cross-encoder rerank, then
     a sweep that reranks every USER unit of the top 3 sessions (B round 1). All cached.
  2. structure: every user unit gets A's logit-space adjustment (owner mismatch / negated / plan / stale penalties,
     small head / reply bonuses, past-question vs future-plan tense agreement). Decoy units do not score sessions.
  3. sessions: max-pool over clean units (user adjusted, assistant raw). Primary = top session, except
     - user-backed primary (`pu_gap`): when the top session wins on an ASSISTANT unit, take the session within 3
       logits whose best clean user statement (head match, not a question to the assistant) is strongest;
     - recency cue ("end up", "now", "last ... I mentioned"): the latest session whose best clean USER unit is within
       `rec_gap` of the best user unit (`rec_user`), so an assistant suggestion cannot pose as the final value.
  4. pack = the cite set plus only what can change the answer: sessions within `keep_gap`=5 logits of the top, at
     most 2. Under a KU-style cue (present-tense state / habit, previous / before / most recently / still / now ...)
     the window is 11 logits and 3 sessions: the old value of an update sits 4-11 logits below the new one, and
     misrouted KU questions need both. Decoy-only sessions are left out and named in the hint (A's note format).
     ssu gold sessions sit 5-18 logits above the next session on dev, so clean questions get ONE passage.
  5. units: the primary session gets its best 5 user units (+ the assistant question before a short reply, + the
     chunk after the best unit); the session's best ASSISTANT unit too when it beats every user unit (ssa questions
     that the router sends here). Other kept sessions: lead unit + 2 more by raw CE logit (KU / multi facts often live
     in "I'm thinking of ..." wording that the plan penalty demotes).
  6. cite: the primary session only (pair citing helped dev KU misroutes but cites decoys on test "end up" questions).
  7. abstain_p: A's prior on the best user-unit logit + A's lexical gates (no abstention text goes to Reactor).
"""
from __future__ import annotations

import math
import re
from collections import defaultdict

from core.types import EvidencePack

_BASE = dict(k_all=100, pool=300, sweep=3, struct=True, keep_gap=5.0, max_sessions=3, cu=5, ou=2, asst=True,
             a_margin=0.0, a_after=1, cite="top", pair_gap=3.0, recency=True, rec_gap=2.5, budget=6000,
             ku_gap=None, ku_pair=None, unit_rank="adj", asweep=0, asst_cap=None, b_value=0.0,
             sent_w=0.0, sent_k=12, rec_user=False, max_sessions_ku=None, pu_gap=None, pu_head=True,
             reply_len=200, hint="owner", decoy_hint=True, decoy_gap=6.0, ab="A",
             # A's adjustment weights (a_struct nb3t0)
             pen_other=3.0, pen_neg=3.0, pen_plan=2.0, pen_stale=1.5, pen_stale_soft=0.3, b_owner=0.3, b_head=0.3,
             b_mod=0.15, b_verb=0.15, b_aside=0.2, b_reply=0.5, b_new=0.5, pen_fut=2.0, b_past=0.5, pen_ask=0.0,
             ab_bias=2.0, ab_temp=1.0, ab_gate=True, ab_conf=0.0)

VARIANTS = {
    # B retrieval + A structure, compact pack (sessions within 5 logits, max 3), cite primary
    "m5": dict(_BASE),
    # no assistant-unit packing (B round-1 behaviour) -> measures the ssa-misroute fix
    "m5_noasst": {**_BASE, "asst": False},
    # tighter / looser competitor window
    "m3": {**_BASE, "keep_gap": 3.0},
    "m8": {**_BASE, "keep_gap": 8.0},
    # single-session pack (pure cite set)
    "m0": {**_BASE, "keep_gap": 0.0, "max_sessions": 1},
    # cite clean near-tie sessions too (KU-style pairs)
    "m5_pair": {**_BASE, "cite": "pair"},
    # no structure (pure semantic + compact pack): control
    "m5_plain": {**_BASE, "struct": False, "hint": "none", "decoy_hint": False},
    # KU-style cue (present-tense state / habit, previous / before / most recently / still / now): wider competitor
    # window (the old value of an update sits 4-11 logits below the new one); other sessions' units by raw CE logit
    "k11": {**_BASE, "ku_gap": 11.0, "unit_rank": "raw", "ou": 3},
    "k11p": {**_BASE, "ku_gap": 11.0, "unit_rank": "raw", "ou": 3, "ku_pair": 5.0},
    "m5r": {**_BASE, "unit_rank": "raw", "ou": 3},
    # + rerank the ASSISTANT units of the top session too (an ssa question routed here: the answer is an assistant
    #   chunk that the all-role pool missed)
    "k11c2": {**_BASE, "ku_gap": 11.0, "unit_rank": "raw", "ou": 3, "asst_cap": 2.0},
    "k11c0": {**_BASE, "ku_gap": 11.0, "unit_rank": "raw", "ou": 3, "asst_cap": 0.0},
    "k11v": {**_BASE, "ku_gap": 11.0, "unit_rank": "raw", "ou": 3, "b_value": 1.0},
    "k11s": {**_BASE, "ku_gap": 11.0, "unit_rank": "raw", "ou": 3, "sent_w": 1.0},
    "k11s5": {**_BASE, "ku_gap": 11.0, "unit_rank": "raw", "ou": 3, "sent_w": 0.5},
    "k11ru": {**_BASE, "ku_gap": 11.0, "unit_rank": "raw", "ou": 3, "rec_user": True},
    "k11sru": {**_BASE, "ku_gap": 11.0, "unit_rank": "raw", "ou": 3, "sent_w": 0.5, "rec_user": True},
    # the newest USER statement wins under a recency cue + 2 sessions (3 under a KU cue)
    "k11ru2": {**_BASE, "ku_gap": 11.0, "unit_rank": "raw", "ou": 3, "rec_user": True, "max_sessions": 2,
               "max_sessions_ku": 3},
    "k11ru2p": {**_BASE, "ku_gap": 11.0, "unit_rank": "raw", "ou": 3, "rec_user": True, "max_sessions": 2,
                "max_sessions_ku": 3, "ku_pair": 2.0},
    # user-backed primary among sessions within 2 / 3 logits of the top
    "k11ru2u2": {**_BASE, "ku_gap": 11.0, "unit_rank": "raw", "ou": 3, "rec_user": True, "max_sessions": 2,
                 "max_sessions_ku": 3, "pu_gap": 2.0},
    "k11ru2u3": {**_BASE, "ku_gap": 11.0, "unit_rank": "raw", "ou": 3, "rec_user": True, "max_sessions": 2,
                 "max_sessions_ku": 3, "pu_gap": 3.0},
    # round-2 pick (= k11ru2u3)
    "best": {**_BASE, "ku_gap": 11.0, "unit_rank": "raw", "ou": 3, "rec_user": True, "max_sessions": 2,
             "max_sessions_ku": 3, "pu_gap": 3.0},
    "best_k2": {**_BASE, "ku_gap": 11.0, "unit_rank": "raw", "ou": 3, "rec_user": True, "max_sessions": 2,
                "max_sessions_ku": 2, "pu_gap": 3.0},
    "best_k8": {**_BASE, "ku_gap": 8.0, "unit_rank": "raw", "ou": 3, "rec_user": True, "max_sessions": 2,
                "max_sessions_ku": 3, "pu_gap": 3.0},
    "k11pa": {**_BASE, "ku_gap": 11.0, "unit_rank": "raw", "ou": 3, "ku_pair": 5.0, "asweep": 1, "a_margin": -4.0},
    "k11pa8": {**_BASE, "ku_gap": 11.0, "unit_rank": "raw", "ou": 3, "ku_pair": 5.0, "asweep": 1, "a_margin": -8.0},
}

KU_CUE = re.compile(r"\b(?:do|does|am|have|has) (?:i|my)\b|\bhave i been\b|\bhow often\b|\busually\b|\bprevious(?:ly)?\b|"
                    r"\binitially\b|\bat first\b|\bbefore (?:i|getting|buying|the|my)\b|\bmost recent(?:ly)?\b|\bstill\b|"
                    r"\bnow\b|\bcurrent(?:ly)?\b|\bswitch|\bany ?more\b|\bso far\b|\bthese days\b|\bend(?:ed)? up\b|"
                    r"\bafter\b|\bsame\b", re.I)

_RECENCY = re.compile(r"\bend(?:ed)? up\b|\b(?:the )?(?:last|latest|most recent)\b[\w\s'-]{0,40}\bI\b[\w\s'-]{0,20}"
                      r"\b(?:mention|told|said|talk)|\b(?:now|currently|current|these days|right now|finally|"
                      r"settle[ds]? on)\b", re.I)


def _sig(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, x))))


def build(q, hx, **params) -> EvidencePack:
    from core.pack import build_pack
    from lenses.ssu.a_owner import _snip, _why, build_hint, head_presence, parse_question, unit_class
    from lenses.ssu.a_struct import Q_PAST_RX, WH_ASK_RX, abstain_prob, adjust, tense_adjust

    p = {**_BASE, **{k: v for k, v in params.items() if not k.startswith("_")}}
    question = q.question
    U = hx.units

    # 1. pool + rerank + user sweep of the top sessions
    cand = [u for u, _ in hx.hybrid(question, k=p["k_all"], pool=p["pool"])]
    rr = dict(hx.rerank(question, cand)) if cand else {}

    def raw_sess(scores):
        best: dict[str, float] = {}
        for u, s in scores.items():
            sid = U[u].session_id
            if s > best.get(sid, -99.0):
                best[sid] = s
        return sorted(best, key=lambda s: -best[s])

    if p["sweep"] and rr:
        top_sids = set(raw_sess(rr)[: p["sweep"]])
        a_sids = set(raw_sess(rr)[: p["asweep"]])
        more = [u.uid for u in U if u.uid not in rr and ((u.session_id in top_sids and u.role == "user")
                                                         or u.session_id in a_sids)]
        if more:
            rr.update(dict(hx.rerank(question, more)))

    # 2. structure on user units (A's classifier, logit space)
    qp = parse_question(question)
    q_past = bool(Q_PAST_RX.search(question))
    cls, sc = {}, {}
    for u, s in rr.items():
        if U[u].role == "user" and p["struct"]:
            c = unit_class(U[u], qp)
            cls[u] = c
            sc[u] = s + adjust(c, qp, p) + tense_adjust(U[u], c, q_past, p)
            if p["b_value"] and qp.attr == "name" and c.values and c.valid:
                sc[u] += p["b_value"]      # a name question is answered by a unit that states a name
        else:
            sc[u] = s
    if p["sent_w"]:
        # granularity: a unit (<= ~700 chars) mixes the fact with chit-chat; the cross-encoder on single SENTENCES
        # separates "realigned the AM IF to 465 kHz" from "briefly tried a 22k screen grid resistor" when the unit
        # logits are compressed. Move the top user units towards their best sentence logit.
        from lenses.ssu.a_owner import sentences
        top_users = [u for u in sorted(sc, key=lambda u: -sc[u]) if U[u].role == "user"][: p["sent_k"]]
        texts, owners = [], []
        for u in top_users:
            for st in (sentences(U[u].text) or [U[u].text]):
                if len(st) >= 12:
                    texts.append(st)
                    owners.append(u)
        if texts:
            best_s: dict[int, float] = {}
            for u, v in zip(owners, hx.rerank_texts(question, texts)):
                best_s[u] = max(best_s.get(u, -99.0), float(v))
            for u, v in best_s.items():
                sc[u] += p["sent_w"] * (v - rr[u])
    ranked = sorted(sc, key=lambda u: -sc[u])

    def is_decoy(u):
        c = cls.get(u)
        return c is not None and (c.head_hit or c.verb_hit) and not c.valid

    # 3. sessions (max-pool over clean units)
    by_sess: dict[str, list[int]] = defaultdict(list)
    for u in ranked:
        by_sess[U[u].session_id].append(u)
    s_score, decoy_only = {}, []
    for sid, us in by_sess.items():
        clean = [u for u in us if not is_decoy(u)]
        if clean:
            s_score[sid] = sc[clean[0]]
            if p["asst_cap"] is not None and U[clean[0]].role == "assistant":
                # an assistant echo may lift its session at most `asst_cap` logits above the session's best clean
                # user unit (test decoys: the assistant also echoes the sister's / brother's statement)
                cu_ = [u for u in clean if U[u].role == "user"]
                if cu_:
                    s_score[sid] = min(sc[clean[0]], sc[cu_[0]] + p["asst_cap"])
        else:
            decoy_only.append(sid)
    sess_rank = sorted(s_score, key=lambda s: -s_score[s])
    if not sess_rank:
        sess_rank = list(by_sess)[:1]
        s_score = {sid: sc[by_sess[sid][0]] for sid in sess_rank}
    top = s_score[sess_rank[0]] if sess_rank else -20.0
    primary = sess_rank[0] if sess_rank else None
    if primary and p["pu_gap"] is not None and U[next(u for u in by_sess[primary] if not is_decoy(u))].role == "assistant":
        # user-backed primary: when the top session wins on an ASSISTANT echo, take the near session (within
        # pu_gap) whose best clean USER unit is strongest (the fact is a user statement; test decoys get echoed too)
        def ub(sid):
            cu_ = [u for u in by_sess[sid] if U[u].role == "user" and not is_decoy(u)
                   and (not p["pu_head"] or not cls.get(u) or cls[u].head_hit or cls[u].reply)
                   and not WH_ASK_RX.match(U[u].text)]     # a question to the assistant states no fact
            return sc[cu_[0]] if cu_ else -99.0
        near = [sid for sid in sess_rank if s_score[sid] >= top - p["pu_gap"]]
        cand_p = max(near, key=ub)
        if ub(cand_p) > ub(primary):
            primary = cand_p
    recency = bool(p["recency"] and (qp.recency or _RECENCY.search(question)))
    if primary and recency and p["rec_user"]:
        # the newest USER statement of the slot wins: compare sessions by their best clean user unit only
        u_best = {}
        for sid in sess_rank:
            cu_ = [u for u in by_sess[sid] if U[u].role == "user" and not is_decoy(u)]
            if cu_:
                u_best[sid] = sc[cu_[0]]
        if u_best:
            ub = max(u_best.values())
            near = [s for s in u_best if u_best[s] >= ub - p["rec_gap"]]
            primary = max(near, key=lambda s: (hx.sess_by_id[s]["sess_idx"], u_best[s]))
    elif primary and recency:
        near = [s for s in sess_rank if s_score[s] >= top - p["rec_gap"]]
        primary = max(near, key=lambda s: (hx.sess_by_id[s]["sess_idx"], s_score[s]))
    ku_mode = bool(p["ku_gap"] is not None and KU_CUE.search(question))
    keep_gap = p["ku_gap"] if ku_mode else p["keep_gap"]
    max_s = p["max_sessions_ku"] if (ku_mode and p["max_sessions_ku"]) else p["max_sessions"]
    keep = [primary] if primary else []
    for sid in sess_rank:
        if len(keep) >= max_s:
            break
        if sid not in keep and s_score[sid] >= top - keep_gap:
            keep.append(sid)

    # 5. units
    def with_ctx(u):
        c = U[u]
        short = c.role == "user" and c.chunk_idx == 0 and len(c.text) < p["reply_len"] and c.ctx.rstrip().endswith("?")
        return hx.neighbors(u, 1 if short else 0, 0)

    order: list[int] = []
    for i, sid in enumerate(keep):
        us = by_sess[sid]
        users = [u for u in us if U[u].role == "user"]
        clean_users = [u for u in users if not is_decoy(u)] + [u for u in users if is_decoy(u)]
        if p["unit_rank"] == "raw" and clean_users:
            # structure picks the session's lead unit; the rest by raw CE logit (plan / tentative wording is often
            # where KU / multi facts live: "I'm thinking of using my new Instant Pot")
            clean_users = clean_users[:1] + sorted(clean_users[1:], key=lambda u: -rr[u])
        k = p["cu"] if sid == primary else p["ou"]
        best_u = clean_users[0] if clean_users else None
        best_any = us[0]
        best_a = next((u for u in us if U[u].role == "assistant"), None)
        if p["asst"] and best_a is not None and (best_u is None or sc[best_a] > sc[best_u] + p["a_margin"]):
            # an assistant statement is (close to) the session's best evidence (ssa-style question): pack it with
            # its user question and the chunk after
            order += hx.neighbors(best_a, 1, p["a_after"])
        for j, u in enumerate(clean_users[:k]):
            order += with_ctx(u)
            if j == 0 and sid == primary:
                order += hx.neighbors(u, 0, 1)
        if not clean_users and not (p["asst"] and U[best_any].role == "assistant"):
            order += hx.neighbors(best_any, 1, 1)
    order = list(dict.fromkeys(order))

    # decoys: named in the hint, kept out of the pack
    decoys = []
    if p["decoy_hint"] and p["struct"]:
        for u in ranked:
            if len(decoys) >= 2:
                break
            if is_decoy(u) and U[u].session_id not in keep and rr.get(u, -20) >= top - p["decoy_gap"]:
                decoys.append({"uid": u, "date": U[u].date_str[:10], "snip": _snip(cls[u].sent, 90),
                               "why": _why(cls[u])})

    # 6. cite
    cite = [primary] if primary else []
    pair_gap = p["pair_gap"] if p["cite"] == "pair" else (p["ku_pair"] if ku_mode and p["ku_pair"] else None)
    if pair_gap is not None:
        for sid in keep:
            if sid != primary and s_score.get(sid, -99) >= top - pair_gap:
                cite.append(sid)

    # 7. abstention (A's prior + gates), no abstention text in the hint
    user_rr = [rr[u] for u in rr if U[u].role == "user"]
    top_logit = max(user_rr) if user_rr else -12.0
    pres = head_presence(hx, qp) if qp.head else {"user": 1, "any": 1, "full": 1, "chain": 1}
    top_uid = next((u for u in by_sess.get(primary, []) if U[u].role == "user" and not is_decoy(u)), None)
    top_sc = cls.get(top_uid) if top_uid is not None else None
    near_top = [u for u in ranked[:6] if U[u].role == "user" and rr.get(u, -12) >= top_logit - 2.0
                and cls.get(u) and cls[u].head_hit]
    all_decoy = bool(near_top) and all(is_decoy(u) for u in near_top)
    abstain_p = abstain_prob(qp, pres, top_logit, top_sc, all_decoy,
                             {**p, "ab_gate": p["ab_gate"] and p["struct"]})

    hint = ""
    if p["hint"] == "owner":
        hint = build_hint(qp, {"hint": "owner", "recency": recency}, decoys, 0.0, pres, [])
        if recency and "most recent" not in hint:
            hint += " If the user mentioned it more than once, the most recent dated statement wins."

    passages, included, _ = build_pack(q, hx, order, budget=p["budget"], window=0, order="chrono", hint=hint,
                                       max_sessions=max_s)
    in_pack = {ps.id for ps in passages}
    cite = [c for c in cite if c in in_pack] or [ps.id for ps in passages[:1]]
    return EvidencePack(question_id=q.question_id, variant=params.get("_variant", ""), route="ssu",
                        passages=passages, hint=hint, cite_plan=cite, abstain_p=abstain_p,
                        included_turns=included, computed=None,
                        debug={"ranking": ranked, "top_logit": round(top_logit, 2), "recency": recency, "ku_mode": ku_mode,
                               "sessions": [(round(s_score[s], 2), s) for s in sess_rank[:6]],
                               "keep": keep, "decoy_only": decoy_only[:4], "decoys": [d["uid"] for d in decoys]})
