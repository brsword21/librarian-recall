"""KU lens A, round 4 ("Struktura" r4): frozen a_r3 r3w (v2 ku route) plus three test-visible fixes, each a flag.
a_r3.py / b_r2.py / b_semantic.py are imported, never edited. `r3w` here == frozen a_r3 r3w (control).

  attr   value attribution of the pair's NEW member: a sentence whose value sits only in clauses that name neither the
         user (I/my/me/we/our) nor an attribute word of the question gives somebody else's value
         ("The toddler's Sunday playgroup got moved to 9:00 a.m., so I need to keep my long-run start early").
         It is dropped as the new member: the old member becomes the new one and the best-ranked own, valued session
         older than it becomes the old one (test long run: [06/24, 07/08] -> [03/12, 06/24]).
  dsx    decoy-session extras: an extra (non-pair) session that carries another person's value for the same attribute
         (own < 0, value, coverage >= 0.6: "My brother Brian pays 22,000 shillings a month") is a decoy session; it
         is not added to the pack. Own remarks inside it are generated chatter around the decoy (test rent: real
         Reactor answered "9,000" from "My current rent is only 9,000, shared." in that session while the pair and
         hint said 18,000 -> 22,000).
  head   'how many <noun> do I have' where no pair statement names the noun (test cats: the pair is "9 Icelandic
         horses" + "4 beehives"): the timeline is replaced by a factual absence line and abstain_p is raised.
"""
from __future__ import annotations

import math
import re

from core.types import EvidencePack
from lenses.ku import a_r3 as A3
from lenses.ku import b_r2 as R2
from lenses.ku import b_semantic as B

_R3W = dict(A3.VARIANTS["r3w"])
VARIANTS = {
    "r3w": dict(_R3W),                                       # frozen v2 ku route (control)
    "attr": dict(_R3W, attr=True),
    "dsx": dict(_R3W, dsx=True),
    "head": dict(_R3W, head=True),
    "r4a": dict(_R3W, attr=True, dsx=True),
    "r4": dict(_R3W, attr=True, dsx=True, head=True),
}

_FP = re.compile(r"\b(?:I|I'm|I've|I'd|I'll|my|me|mine|we|our|us)\b", re.I)
_CLAUSE = re.compile(r"[,;:—–]|\.(?=\s)|\b(?:so|because|but|while|although|since|whereas|which|and then)\b", re.I)


def _clauses(sent: str) -> list[str]:
    return [c for c in _CLAUSE.split(sent) if c and c.strip()]


def misattributed(sent: str, question: str, attr_terms: list[str]) -> bool:
    """True when every clause carrying a value names neither the user nor an attribute word of the question."""
    vcl = [c for c in _clauses(sent) if B.values(c, question)]
    if not vcl:
        return False
    for c in vcl:
        if _FP.search(c):
            return False
        terms = set(B.content_terms(c))
        if any(t == a or (len(a) > 3 and (t.startswith(a) or a.startswith(t))) for t in terms for a in attr_terms):
            return False
    return True


def fix_attr(q, hx, r, dbg) -> list | None:
    pair, bbs = r["pair"], r["best_by_sess"]
    if len(pair) != 2 or r["mode"] != "new":
        return None
    attr_terms = [t for t in B.content_terms(r["qd"]) if len(t) > 2]
    sn = bbs[pair[1]]
    if not misattributed(sn["raw"], q.question, attr_terms):
        return None
    new = pair[0]
    new_idx = hx.sess_by_id[new]["sess_idx"]
    for sid in r["sess_rank"]:
        s = bbs[sid]
        if sid in pair or hx.sess_by_id[sid]["sess_idx"] >= new_idx or s["own"] < 1.0 or not s["val"]:
            continue
        if s["cov"] < 0.3 or misattributed(s["raw"], q.question, attr_terms):
            continue
        dbg["attr_drop"] = pair[1]
        return [sid, new]
    return None


def decoy_sessions(r, question: str, cov_min: float = 0.6) -> set:
    """Sessions with another person's value for the attribute: coverage >= cov_min, or the same value as a pair
    statement ("My brother Brian pays 22,000 shillings a month" next to the user's "now pay 22,000 shillings")."""
    pv = set()
    for sid in r["pair"]:
        pv |= {v for v in B.values(r["best_by_sess"][sid]["raw"], question) if re.search(r"\d", v)}
    out = set()
    for s in r["sents"]:
        if s["own"] >= 0 or not s["val"] or s.get("asst"):
            continue
        if s["cov"] >= cov_min or (pv & B.values(s["raw"], question)):
            out.add(s["sid"])
    return out


# possession counts only ('how many cats do I have / own'); 'how many times have I met ...' is not one
_HEAD_Q = re.compile(r"^\s*how many\s+((?:[a-z][\w'-]*\s+){0,2}?[a-z][\w'-]*?s)\s+(?:do|did)\s+(?:I|we)\s+"
                     r"(?:currently\s+|still\s+)?(?:have|own|keep|raise)\b", re.I)


def head_absent(q, r) -> str | None:
    """'How many cats do I have ...' -> 'cats' when no pair statement names it (stem match)."""
    m = _HEAD_Q.search(q.question)
    if not m:
        return None
    head = m.group(1).split()[-1]
    hs = B._stem(head)
    if len(hs) < 3:
        return None
    for sid in r["pair"]:
        raw = r["best_by_sess"][sid]["raw"]
        if any(B._stem(w) == hs or w.lower().startswith(hs) for w in re.findall(r"[A-Za-z][\w'-]*", raw)):
            return None
    return head


def build(q, hx, **p):
    from core.pack import build_pack

    r = B.analyze(q, hx, **p)
    question = q.question
    dbg: dict = {}
    if p.get("hab"):
        np_ = A3.habitual(q, hx, r, dbg)
        if np_:
            r["pair"] = np_
    if p.get("adm"):
        np_ = A3.admit(q, hx, r, p, dbg)
        if np_:
            r["pair"] = np_
            r["sess_rank"] = list(dict.fromkeys(np_ + r["sess_rank"]))
    if p.get("attr"):
        np_ = fix_attr(q, hx, r, dbg)
        if np_:
            r["pair"] = np_
            r["sess_rank"] = list(dict.fromkeys(np_ + r["sess_rank"]))
    ot = A3.owner_total(q, hx, r) if p.get("owner") else None
    if ot:
        return A3._owner_pack(q, hx, r, ot, p, dbg)
    pair, sess_rank, sents, best_by_sess = r["pair"], r["sess_rank"], r["sents"], r["best_by_sess"]

    extra = p.get("extra", 2)
    feats: dict = {}
    if p.get("gate"):
        ok, feats = R2.confidence(r, hx, question, p)
        if not ok:
            extra = p.get("extra_lo", 1)
        feats["confident"] = ok
    rest = [sid for sid in sess_rank if sid not in pair]
    ext = rest[:extra]
    if p.get("dsx"):
        dec = decoy_sessions(r, question)
        if p.get("dsx_conflict", True):
            # only decoy sessions that ALSO carry an own value differing from the new member's (the remark that can
            # pull Reactor off the pair); a decoy session without one is harmless context and stays
            nv = B.values(best_by_sess[pair[-1]]["raw"], question) if pair else set()
            dec = {sid for sid in dec if any(
                s["sid"] == sid and s["own"] >= 1.0 and s["val"] and s["cov"] >= 0.4 and not s.get("asst")
                and not (B.values(s["raw"], question) & nv) for s in sents)}
        dropped = [s for s in ext if s in dec]
        if dropped:
            dbg["dsx_drop"] = dropped
        ext = [s for s in ext if s not in dec]
    keep = list(pair) + ext

    unit_best: dict[int, float] = {}
    for s in sents:
        unit_best[s["uid"]] = max(unit_best.get(s["uid"], -1e9), s["score"])
    pinned, pack_uids = [], []
    for sid in keep:
        su = sorted({s["uid"] for s in sents if s["sid"] == sid and not s.get("asst")}, key=lambda uid: -unit_best[uid])
        n = p.get("units_per_sess", 2) if sid in pair else p.get("units_extra", 2)
        if sid in pair and su:
            pinned.append(su[0])
        pack_uids.extend(su[:n])
        if best_by_sess[sid].get("asst"):
            (pinned if sid in pair else pack_uids).append(best_by_sess[sid]["uid"])

    tl_sids = list(keep) + [sid for sid in rest[extra: extra + p.get("tl_extra", 0)]]
    tl = []
    for sid in sorted(set(tl_sids), key=lambda x: hx.sess_by_id[x]["sess_idx"]):
        s = best_by_sess[sid]
        if p.get("tl_min_cov") is not None and not s.get("linked") and (
                s["cov"] < p["tl_min_cov"] or (s["raw"].rstrip().endswith("?") and not s["val"])):
            continue
        if p.get("attr") and dbg.get("attr_drop") == sid:
            continue
        if sid in pair or (s["own"] > 0 and s["val"] and s["cov"] >= p.get("tl_cov", 0.6)):
            tl.append((sid, hx.sess_by_id[sid]["date_str"], s["raw"]))
    if p.get("tl_pair") and len(tl) > 4:
        others = [x for x in tl if x[0] not in pair]
        slots = 4 - sum(1 for x in tl if x[0] in pair)
        keep_o = {x[0] for x in others[-slots:]} if slots > 0 else set()
        tl = [x for x in tl if x[0] in pair or x[0] in keep_o]
    tl = tl[:4]

    ap = r["abstain_p"]
    miss_ph = []
    if p.get("bigram_w"):
        miss_ph = R2.missing_phrases(question, R2._user_text(hx))
        z = math.log(ap / (1 - ap)) + p["bigram_w"] * min(2, len(miss_ph))
        ap = 1 / (1 + math.exp(-z))

    raw_absent = [w for w in re.findall(r"[A-Za-z][\w'-]+", question)[1:] if w[0].isupper() and B._stem(w) in r["absent"]]
    hint = B.build_hint([(d, s) for _, d, s in tl], r["mode"],
                        raw_absent if r["abstain_p"] > p.get("ab_hint", 0.6) else None) if p.get("hint", True) else ""

    head = head_absent(q, r) if p.get("head") else None
    if head:
        dbg["head_absent"] = head
        hint = (f"None of the statements found gives a number of {head} that you yourself have; the passages below "
                f"are about other things. If no passage states it, answer INSUFFICIENT_EVIDENCE.")
        ap = max(ap, p.get("head_ap", 0.75))

    computed = None
    if p.get("rel") and not head and len(pair) == 2 and r["mode"] not in ("old", "both", "compare"):
        olds = [best_by_sess[pair[0]]["raw"]] + [s["raw"] for s in sorted(sents, key=lambda s: -s["score"])
                                                 if s["sid"] == pair[0] and s["own"] >= 0 and s["cov"] >= 0.3]
        ru = None
        for o_raw in olds[:4]:
            ru = A3.rel_update(o_raw, best_by_sess[pair[1]]["raw"], question)
            if ru:
                break
        if ru:
            computed = ru[2]
            hint = hint.replace("\nCite both", f"\nThe later statement adds {ru[1]} to the earlier {ru[0]}: "
                                               f"{ru[0]} + {ru[1]} = {ru[2]}.\nCite both", 1)
            dbg["rel"] = ru

    passages, included_turns, _chars = build_pack(q, hx, pack_uids, budget=p.get("budget", 6000),
                                                  window=p.get("window", 0), order="chrono", hint=hint,
                                                  max_sessions=len(keep), whole_turns=False, pinned=pinned)
    pack_ids = [ps.id for ps in passages if ps.id]
    cite = [sid for sid in pair if sid in pack_ids]
    return EvidencePack(question_id=q.question_id, variant=p.get("_variant", ""), route="ku", passages=passages,
                        hint=hint, cite_plan=cite or None, abstain_p=round(ap, 4),
                        included_turns=included_turns, computed=computed,
                        debug={"ranking": r["ranked_units"], "pair": pair, "sess_rank": sess_rank[:8],
                               "mode": r["mode"], "qd": r["qd"], "miss": round(r["miss"], 3), "absent": r["absent"],
                               "gate": feats, "miss_ph": miss_ph, "n_pass": len(passages), **dbg})
