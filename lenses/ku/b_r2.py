"""KU lens B, round 2 ("Semantyka" r2): pair_r1b retrieval/pair logic (lenses/ku/b_semantic.py, unchanged) with a
citation-robust pack.

Round-1 winner pair_r1b packs the cited pair + 2 context sessions: S 93.2 but Sall 83.9 (if Reactor cites every
passage we send, the 2 context sessions cost ~0.33 CF1). Round 2:
  * the pack holds ONLY the pair when the pair is confident (gate below), else the pair + `extra_lo` sessions;
  * statements from the dropped context sessions stay readable in the timeline hint (dated quotes), so Reactor still
    sees a newer own value if the pair missed it, but cannot cite those sessions (hint is not a passage);
  * a phrase-level (bigram) abstention signal for entity swaps ('Italian restaurants' vs Korean, 'Software Engineer
    Manager'); idea and helper credited to lens A (a_timeline.missing_phrases, pkl_b1).

Confidence gate (all must hold to drop the context sessions):
  - both pair sentences are the user's own (own >= 0) and match the attribute (cov >= gate_cov),
  - the pair carries two different values, or the pair members are well ahead of the next session (margin),
  - the next session is not a stronger own same-attribute statement NEWER than the pair's new member.
"""
from __future__ import annotations

import math
import re

from lenses.ku import b_semantic as B

try:
    from core.types import EvidencePack
except Exception:  # pragma: no cover
    EvidencePack = None

_R1B = dict(B.VARIANTS["pair_r1b"])
VARIANTS = {
    # control: round-1 winner, re-run through this file (pair + 2 context sessions)
    "r1b": dict(_R1B),
    # pair only, always (upper bound for Sall, shows the PA cost of no context)
    "only": dict(_R1B, extra=0, budget=4500, units_per_sess=3, tl_extra=2),
    # gated: pair only when confident, else pair + 1 context session
    "gate": dict(_R1B, extra=0, extra_lo=1, gate=True, budget=4500, units_per_sess=3, tl_extra=2),
    # gated + bigram abstention (lens A's missing_phrases)
    "gate_bg": dict(_R1B, extra=0, extra_lo=1, gate=True, budget=4500, units_per_sess=3, tl_extra=2, bigram_w=0.6),
    # r2 gate (margin only): pair alone when its weaker member leads the next session by >= 3.5 (score units)
    "m35": dict(_R1B, extra=0, extra_lo=1, gate=True, gate_v=2, gate_margin=3.5, budget=4500, units_per_sess=3,
                tl_extra=2),
    "m35x2": dict(_R1B, extra=0, extra_lo=2, gate=True, gate_v=2, gate_margin=3.5, budget=4500, units_per_sess=3,
                  tl_extra=2),
    "m35_bg": dict(_R1B, extra=0, extra_lo=1, gate=True, gate_v=2, gate_margin=3.5, budget=4500, units_per_sess=3,
                   tl_extra=2, bigram_w=0.6),
    "m35_bg1": dict(_R1B, extra=0, extra_lo=1, gate=True, gate_v=2, gate_margin=3.5, budget=4500, units_per_sess=3,
                    tl_extra=2, bigram_w=1.0),
    "m35_bg15": dict(_R1B, extra=0, extra_lo=1, gate=True, gate_v=2, gate_margin=3.5, budget=4500, units_per_sess=3,
                     tl_extra=2, bigram_w=1.5),
    # r2b = m35_bg1 without the INSUFFICIENT_EVIDENCE note in the hint (round-2 rule: no abstention hints to Reactor)
    "r2b": dict(_R1B, extra=0, extra_lo=1, gate=True, gate_v=2, gate_margin=3.5, budget=4500, units_per_sess=3,
                tl_extra=2, bigram_w=1.0, ab_hint=9.0),
    # r2c = r2b + bare-name subject ownership in quantity questions ('Carlos drinks three cups of coffee every morning')
    "r2c": dict(_R1B, extra=0, extra_lo=1, gate=True, gate_v=2, gate_margin=3.5, budget=4500, units_per_sess=3,
                tl_extra=2, bigram_w=1.0, ab_hint=9.0, own_name=True),
    "m5": dict(_R1B, extra=0, extra_lo=1, gate=True, gate_v=2, gate_margin=5.0, budget=4500, units_per_sess=3,
               tl_extra=2),
}

# copied from lens A (a_timeline.py) so a mid-round edit there cannot change this lens
_STOPW = set("""a an the of in on at to for from with by about as and or but if is are was were be been being do does
did have has had i me my mine myself we our ours us you your yours what which who whom whose when where why how many
much often time times currently current now these days day right recently most recent so far still previous
previously initially originally before after since that this there their they them it its am can could would should
will just any some other first last get got go going total number new old ago ever usually each every per tell than i'm i've i'd i'll you're you've it's that's don't didn't doesn't can't won't isn't aren't wasn't
we're we've they're
remind""".split())
_CUE_RE = re.compile(r"\b(now|currently|current|these days|right now|at the moment|most recently|most recent|"
                     r"recently|so far|still|anymore|latest|today|lately)\b", re.I)


def missing_phrases(q: str, utext: str) -> list[str]:
    """Copied from lens A (a_timeline.missing_phrases): adjacent content-word pairs of the question that never occur
    as a phrase in the user's turns ('Italian restaurants', 'table tennis', 'Engineer Manager')."""
    ws = re.findall(r"[A-Za-z][\w&'’\-]*|\||\d+", _CUE_RE.sub(" | ", q))
    out = []
    for a, b in zip(ws, ws[1:]):
        a, b = a.lower(), b.lower()
        if "|" in (a, b) or a in _STOPW or b in _STOPW or len(a) < 3 or len(b) < 3:
            continue
        pa = re.escape(a[:-1] if a.endswith("s") and len(a) > 4 else a)
        pb = re.escape(b[:-1] if b.endswith("s") and len(b) > 4 else b)
        if not re.search(rf"\b{pa}\w*\s+{pb}", utext):
            out.append(f"{a} {b}")
    return out


def _user_text(hx) -> str:
    c = hx.__dict__.setdefault("_ku_b2_cache", {})
    if "ut" not in c:
        c["ut"] = " ".join(u.text for u in hx.units if u.role == "user").lower()
    return c["ut"]


def confidence(r: dict, hx, question: str, p: dict) -> tuple[bool, dict]:
    """Is the cited pair safe enough to send alone? Returns (confident, features)."""
    pair, rank, bbs = r["pair"], r["sess_rank"], r["best_by_sess"]
    f: dict = {}
    if len(pair) < 2:
        return False, {"why": "single"}
    so, sn = bbs[pair[0]], bbs[pair[1]]
    vo, vn = B.values(so["raw"], question), B.values(sn["raw"], question)
    f["distinct"] = bool(vo and vn and vo != vn)
    f["own_ok"] = so["own"] >= 0 and sn["own"] >= 0
    f["cov_min"] = round(min(so["cov"], sn["cov"]), 3)
    rest = [sid for sid in rank[: p.get("top_m", 6)] if sid not in pair]
    pmin = min(so["score"], sn["score"])
    f["margin"] = round(pmin - (bbs[rest[0]]["score"] if rest else -20.0), 3)
    # a strong own same-attribute statement newer than the pair's new member: the pair may have missed the update
    new_idx = hx.sess_by_id[pair[1]]["sess_idx"]
    threat = [sid for sid in rest[:3] if hx.sess_by_id[sid]["sess_idx"] > new_idx and bbs[sid]["own"] > 0
              and bbs[sid]["val"] and bbs[sid]["cov"] >= p.get("gate_cov", 0.5)
              and bbs[sid]["score"] >= pmin - p.get("threat_gap", 2.0)]
    f["threat"] = threat
    # features for an evidence-based KU detector (router): absolute scores of the top sessions, pair cov
    f["top3"] = [round(bbs[sid]["score"], 2) for sid in rank[:3]]
    f["pair_sc"] = [round(so["score"], 2), round(sn["score"], 2)]
    f["cov_max"] = round(max(so["cov"], sn["cov"]), 3)
    if p.get("gate_v") == 2:   # r2 gate: the score margin alone (dev: all 5 pair misses / PA risks have margin < 3.5)
        ok = f["margin"] >= p.get("gate_margin", 3.5) and not (p.get("gate_threat") and threat)
    else:
        ok = (f["own_ok"] and f["cov_min"] >= p.get("gate_cov", 0.5) and not threat
              and (f["distinct"] or f["margin"] >= p.get("gate_margin", 2.0)))
    return ok, f


def build(q, hx, **p):
    from core.pack import build_pack

    r = B.analyze(q, hx, **p)
    question = q.question
    pair, sess_rank, sents, best_by_sess = r["pair"], r["sess_rank"], r["sents"], r["best_by_sess"]

    extra = p.get("extra", 2)
    feats: dict = {}
    if p.get("gate"):
        ok, feats = confidence(r, hx, question, p)
        if not ok:
            extra = p.get("extra_lo", 1)
        feats["confident"] = ok
    rest = [sid for sid in sess_rank if sid not in pair]
    keep = list(pair) + rest[:extra]

    unit_best: dict[int, float] = {}
    for s in sents:
        unit_best[s["uid"]] = max(unit_best.get(s["uid"], -1e9), s["score"])
    pinned, pack_uids = [], []
    for sid in keep:
        su = sorted({s["uid"] for s in sents if s["sid"] == sid and not s.get("asst")}, key=lambda uid: -unit_best[uid])
        n = p.get("units_per_sess", 2) if sid in pair else p.get("units_extra", 2)
        if sid in pair and su:
            pinned.append(su[0])           # the pair's best unit always enters first
        pack_uids.extend(su[:n])
        if best_by_sess[sid].get("asst"):
            (pinned if sid in pair else pack_uids).append(best_by_sess[sid]["uid"])

    # timeline hint: pair + shown sessions + (tl_extra) dropped context sessions, dated, oldest first
    tl_sids = list(keep) + [sid for sid in rest[extra: extra + p.get("tl_extra", 0)]]
    tl = []
    for sid in sorted(set(tl_sids), key=lambda x: hx.sess_by_id[x]["sess_idx"]):
        s = best_by_sess[sid]
        if p.get("tl_min_cov") is not None and not s.get("linked") and (
                s["cov"] < p["tl_min_cov"] or (s["raw"].rstrip().endswith("?") and not s["val"])):
            continue
        if sid in pair or (s["own"] > 0 and s["val"] and s["cov"] >= p.get("tl_cov", 0.6)):
            tl.append((hx.sess_by_id[sid]["date_str"], s["raw"]))

    ap = r["abstain_p"]
    miss_ph = []
    if p.get("bigram_w"):
        miss_ph = missing_phrases(question, _user_text(hx))
        z = math.log(ap / (1 - ap)) + p["bigram_w"] * min(2, len(miss_ph))
        ap = 1 / (1 + math.exp(-z))

    raw_absent = [w for w in re.findall(r"[A-Za-z][\w'-]+", question)[1:] if w[0].isupper() and B._stem(w) in r["absent"]]
    hint = B.build_hint(tl[:4], r["mode"], raw_absent if r["abstain_p"] > p.get("ab_hint", 0.6) else None) \
        if p.get("hint", True) else ""

    passages, included_turns, _chars = build_pack(q, hx, pack_uids, budget=p.get("budget", 6000),
                                                  window=p.get("window", 0), order="chrono", hint=hint,
                                                  max_sessions=len(keep), whole_turns=False, pinned=pinned)
    pack_ids = {ps.id for ps in passages if ps.id}
    cite = [sid for sid in pair if sid in pack_ids]
    return EvidencePack(question_id=q.question_id, variant=p.get("_variant", ""), route="ku", passages=passages,
                        hint=hint, cite_plan=cite or None, abstain_p=round(ap, 4),
                        included_turns=included_turns, computed=None,
                        debug={"ranking": r["ranked_units"], "pair": pair, "sess_rank": sess_rank[:8],
                               "mode": r["mode"], "qd": r["qd"], "miss": round(r["miss"], 3), "absent": r["absent"],
                               "gate": feats, "miss_ph": miss_ph, "n_pass": len(passages)})


def ku_pair_p(q, hx, min_score: float = 6.0, min_cov: float = 0.5) -> tuple[bool, dict]:
    """Evidence-based KU detector for the router (brief item 4). True when the two best own-attribute sessions state
    two DIFFERENT values, both with a strong score (>= min_score) and attribute coverage (>= min_cov).
    Measured with r2b features (dev, 40-q samples per type): ku 40/78, multi 2/40, ssu 0/40. Because a multi -> ku
    misroute costs ~70 S per question, apply it ONLY to questions the text router sends to ssu."""
    p = dict(VARIANTS["r2b"])
    r = B.analyze(q, hx, **p)
    ok, f = confidence(r, hx, q.question, dict(p, gate_v=2))
    if "pair_sc" not in f:
        return False, f
    hit = bool(f["distinct"] and min(f["pair_sc"]) >= min_score and f["cov_min"] >= min_cov)
    return hit, f
