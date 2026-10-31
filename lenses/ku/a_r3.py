"""KU lens A, round 3 ("Struktura" r3): the frozen v1 ku route (lenses/ku/b_r2.py r2b, retrieval and pair logic by
lens B) plus structural fixes to the pair and the hint. b_r2.py / b_semantic.py are imported, never edited.

Changes (each one a variant flag, default off, so `r2b` here == frozen v1):
  tl_pair  the hint timeline always quotes both cited pair statements. r2b keeps the OLDEST 4 statements (tl[:4]), so
           with several older restatements the newest value is cut from the hint (test laps: 4 x '40 laps' quoted,
           the '30 laps' update dropped although its session is in the pack).
  lab      the citation line names the pair's passage labels ("... then [E1][E2]"). The client's format suffix says to
           cite "the passage(s) that state it", and real Reactor on ku cites only the newest (evidence F1 0.5 on the
           calibration; every ku gold has exactly 2 sessions: old + new).
  rel      relative update: the newest statement adds a count to the older absolute value ("I added three new apple
           trees" after "four apple trees"; "added a new coin" after "a total of 37 coins") -> computed = old + added,
           stated in the hint.
  adm      CE admission (brief item 1): an own, first-person change statement NEWER than the pair's new member whose
           sentence cross-encoder score beats the new member's by `adm_gap` becomes the new member, and the old new
           member becomes the old one ("I got another promotion, so now I am Director of Clinical Services" for
           'current job title', where neither 'job' nor 'title' occurs).
  hab      habitual questions ('each morning', 'these days', 'usually'): a one-off episodic sentence ('40 laps today',
           'just finished') cannot displace a change statement ('increased my morning swim to 30 laps') as the new
           member (brief item 2).
"""
from __future__ import annotations

import math
import re

from core.types import EvidencePack
from lenses.ku import b_r2 as R2
from lenses.ku import b_semantic as B

_R2B = dict(R2.VARIANTS["r2b"])
VARIANTS = {
    "r2b": dict(_R2B),                                              # frozen v1 (control)
    "tl": dict(_R2B, tl_pair=True),
    "lab": dict(_R2B, tl_pair=True, lab=True),
    "rel": dict(_R2B, tl_pair=True, lab=True, rel=True),
    "adm": dict(_R2B, tl_pair=True, lab=True, rel=True, adm=True, adm_gap=2.0, adm_abs=0.0),
    "adm4": dict(_R2B, tl_pair=True, lab=True, rel=True, adm=True, adm_gap=4.0, adm_abs=2.0),
    "hab": dict(_R2B, tl_pair=True, lab=True, rel=True, adm=True, adm_gap=2.0, adm_abs=0.0, hab=True),
    # wider assistant-recap pool (test KU updates planted only in assistant turns: "You mentioned your long run has
    # shifted to Sundays at 6:30 a.m."; r2b looks at the top 20 of 30 assistant units only)
    "wa": dict(_R2B, tl_pair=True, lab=True, rel=True, hab=True, asst_k=80, asst_depth=40),
    "wa2": dict(_R2B, tl_pair=True, lab=True, rel=True, hab=True, asst_k=120, asst_depth=60),
    # labels only when the pair is confident (b_r2 margin gate); otherwise a generic 'cite two passages' line
    "wag": dict(_R2B, tl_pair=True, lab=True, lab_gate=True, rel=True, hab=True, asst_k=80, asst_depth=40),
    # + per-owner totals ('my neighbor Bjørn and I in total'): one count timeline per owner, summed (brief item 3)
    "wago": dict(_R2B, tl_pair=True, lab=True, lab_gate=True, rel=True, hab=True, asst_k=80, asst_depth=40,
                 owner=True),
    # + labels only when abstain_p < 0.5, worded 'If you answer, ...' (5 of 6 dev ku unanswerables got labels in wago)
    "wagoa": dict(_R2B, tl_pair=True, lab=True, lab_gate=True, lab_ap=0.5, lab_if=True, rel=True, hab=True,
                  asst_k=80, asst_depth=40, owner=True),
    # RECOMMENDED: content fixes only, B's proven citation line kept (real Reactor cited both pair passages on the
    # first 2 of 2 ku-routed v1 answers, so `lab` is not needed there)
    "r3a": dict(_R2B, tl_pair=True, rel=True, asst_k=80, asst_depth=40, owner=True),
    # + weak-pair rescue: when both pair members score < 0 (off-attribute pair), an own first-person change statement
    #   newer than the old member that the sentence CE prefers over the new member (gap 0) becomes the new member
    "r3w": dict(_R2B, tl_pair=True, rel=True, asst_k=80, asst_depth=40, owner=True, adm=True, adm_weak_only=True,
                adm_gap=0.0, adm_abs=-99.0),
}

_CHANGE = re.compile(r"\b(now|switched|switch|changed|moved|increased|decreased|upped|bumped|cut (?:back|down)|reduced|"
                     r"promot\w*|renamed|shifted|no longer|anymore|instead|upgraded|downgraded|replaced|went from|"
                     r"these days|started|new)\b", re.I)
_EPISODIC = re.compile(r"\b(today|tonight|this morning|yesterday|just finished|last (?:sunday|monday|tuesday|"
                       r"wednesday|thursday|friday|saturday|week|weekend|night)|this week)\b", re.I)
_HABIT_Q = re.compile(r"\b(each|every|per|usually|typically|normally|these days|a (?:day|week|month)|daily|weekly|"
                      r"nightly)\b", re.I)
_ADD = re.compile(r"\b(?:added|bought|adopted|planted|acquired|purchased)\s+"
                  r"(?:(?:a|an|one|another)\b|(\d+|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\b)"
                  r"(?:\s+(?:new|more|additional|extra))?", re.I)
# cumulative / absolute statements: the count in the newest sentence is already the total
_TOTAL = re.compile(r"\b(?:total|in all|altogether|brings|bringing|so far|since|already|currently|now (?:have|own|has)|"
                    r"so now|that makes|which makes|I'll have|I will have|I have|I've got)\b", re.I)
_W = {"a": 1, "an": 1, "one": 1, "another": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
      "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12}


def _num(v: str):
    try:
        return int(v) if re.fullmatch(r"\d+", v) else None
    except ValueError:
        return None


def rel_update(old_raw: str, new_raw: str, question: str):
    """'I added three new apple trees' after 'I currently have four apple trees' -> (4, 3, 7). None when the newest
    sentence already states a total, or the older one has no single plain count."""
    if not re.match(r"\s*how\s+many\b", question, re.I):
        return None
    m = _ADD.search(new_raw)
    if not m or _TOTAL.search(new_raw):
        return None
    tok = (m.group(1) or re.search(r"\b(a|an|one|another)\b", m.group(0), re.I).group(1)).lower()
    add = _W.get(tok) if not tok.isdigit() else int(tok)
    qnums = set(re.findall(r"\d+", question))
    def counts(raw):
        return [n for n in (_num(v) for v in B.values(raw, question)) if n is not None and n < 1000
                and str(n) not in qnums]
    olds = counts(old_raw)
    news = [n for n in counts(new_raw) if n != add]
    if add is None or len(set(olds)) != 1 or news:
        return None
    return olds[0], add, olds[0] + add


def admit(q, hx, r, p, dbg):
    """CE admission of a newer own change statement as the pair's new member (returns the new pair or None)."""
    pair, bbs = r["pair"], r["best_by_sess"]
    if len(pair) < 2:
        return None
    new_idx = hx.sess_by_id[pair[1]]["sess_idx"]
    weak = r["mode"] == "new" and max(bbs[pair[0]]["score"], bbs[pair[1]]["score"]) < p.get("weak_thr", 0.0)
    if p.get("adm_weak_only") and not weak:
        return None
    if weak:   # weak pair: the whole pair may be off-attribute, so a candidate newer than the OLD member may replace
        new_idx = hx.sess_by_id[pair[0]]["sess_idx"]   # the new member ("current job title" -> promotion sentence)
        dbg["weak"] = True
    k = p.get("adm_k", 60)
    uids = [u for u, _ in hx.dense(r["qs"], k=k, roles="user")]
    uids += [u for u, _ in hx.bm25(r["qd"], k=k, roles="user") if u not in set(uids)]
    cands = []
    for uid in uids:
        u = hx.units[uid]
        if u.sess_idx <= new_idx or u.session_id in pair:
            continue
        for s in B.sentences(u.text) or [u.text]:
            if len(s) > 400 or s.rstrip().endswith("?") or not _CHANGE.search(s):
                continue
            if B.ownership(s, q.question) < 1.0 or not B.has_value(s, q.question):
                continue
            cands.append((uid, u.session_id, s))
    if not cands:
        return None
    try:
        ce = hx.rerank_texts(r["qs"], [bbs[pair[1]]["raw"]] + [c[2] for c in cands])
    except Exception as e:  # model server down -> no admission
        dbg["adm_err"] = repr(e)[:120]
        return None
    base, scores = ce[0], ce[1:]
    i = max(range(len(cands)), key=lambda j: scores[j])
    dbg["adm_best"] = (round(scores[i], 2), round(base, 2), cands[i][2][:120])
    if scores[i] < p.get("adm_abs", 0.0) or scores[i] <= base + p.get("adm_gap", 2.0):
        return None
    uid, sid, s = cands[i]
    bbs[sid] = {"uid": uid, "sid": sid, "text": s, "raw": s, "own": 1.0, "val": True, "cov": 0.0, "score": scores[i],
                "unit_rr": None, "sent_rr": scores[i], "admitted": True,
                "linked": True}   # linked: quoted in the hint despite cov 0 (no question word in a paraphrase)
    r["sents"].append(bbs[sid])
    dbg["admitted"] = sid
    if weak:
        return sorted([pair[0], sid], key=lambda x: hx.sess_by_id[x]["sess_idx"])
    return [pair[1], sid]


def habitual(q, hx, r, dbg):
    """'each morning / these days': when the pair's new member is a one-off episodic sentence and an own change
    statement in another top session is newer than the old member, pair = [old, change statement's session]."""
    if not _HABIT_Q.search(q.question):
        return None
    pair, bbs = r["pair"], r["best_by_sess"]
    if len(pair) < 2:
        return None
    sn = bbs[pair[1]]
    if not _EPISODIC.search(sn["raw"]) or _CHANGE.search(sn["raw"]):
        return None
    old_idx = hx.sess_by_id[pair[0]]["sess_idx"]
    for sid in r["sess_rank"][:6]:
        s = bbs[sid]
        if sid in pair or s["own"] < 1.0 or not s["val"] or hx.sess_by_id[sid]["sess_idx"] <= old_idx:
            continue
        if _CHANGE.search(s["raw"]) and not _EPISODIC.search(s["raw"]):
            dbg["habitual"] = sid
            return sorted([pair[0], sid], key=lambda x: hx.sess_by_id[x]["sess_idx"])
    return None


_OWNER_Q = re.compile(rf"\b(?:my\s+)?(?:(?:{B._REL})\s+)?([A-Z][\wÀ-ſ]+)\s+and\s+I\b")
_AGG_Q = re.compile(r"\b(?:in total|total|combined|together|altogether|between us|both)\b", re.I)
_MORE = re.compile(r"\b(?:\d+|a|an|one|another|two|three|four|five|six|seven|eight|nine|ten)\s+(?:more|new|additional|extra)\b"
                   r"|\b(?:added|bought|adopted|got|acquired|purchased)\s+(?:another|an? (?:new|extra))\b", re.I)


def _count(sent: str, question: str):
    qnums = set(re.findall(r"\d+", question))
    ns = {n for n in (_num(v) for v in B.values(sent, question)) if n is not None and n < 1000 and str(n) not in qnums}
    return ns.pop() if len(ns) == 1 else None


def owner_total(q, hx, r):
    """'How many Icelandic horses do my neighbor Bjørn and I have in total?': one count timeline per owner (absolute
    statements set the value, 'bought 2 more' adds to it), latest value per owner, summed. Returns None unless both
    owners have a count."""
    question = q.question
    m = _OWNER_Q.search(question)
    if not m or not _AGG_Q.search(question):
        return None
    name = m.group(1)
    nl = name.lower()
    attr = [t for t in B.content_terms(r["qd"]) if t not in (nl, B._stem(nl)) and not re.fullmatch(B._REL, t)
            and t not in ("total", "combined", "together", "altogether")]
    if not attr:
        return None
    own_rx = re.compile(rf"\b{re.escape(name)}\b", re.I)
    other_poss = re.compile(rf"\b{re.escape(name)}['’]s\s+(?:{B._REL})\b", re.I)
    tl = {"self": [], "other": []}
    for u in hx.units:
        for sent in B.sentences(u.text) or [u.text]:
            terms = set(B.content_terms(sent))
            if not all(any(t == a or t.startswith(a) or a.startswith(t) for t in terms) for a in attr):
                continue
            n = _count(sent, question)
            if n is None or "?" in sent:
                continue
            rel = bool(_MORE.search(sent))
            if own_rx.search(sent):
                if other_poss.search(sent):
                    continue
                tl["other"].append((u.sess_idx, u.uid, u.session_id, u.date_str, sent, n, rel))
            elif u.role == "user" and B.ownership(sent, question) >= 1.0:
                tl["self"].append((u.sess_idx, u.uid, u.session_id, u.date_str, sent, n, rel))
    out = {}
    for who, items in tl.items():
        val, used, expr = None, [], []
        for it in sorted(items):
            if it[6]:
                if val is None:
                    continue
                val += it[5]
                expr.append(str(it[5]))
            else:
                val = it[5]
                used, expr = [], [str(it[5])]
            used.append(it)
        if val is None:
            return None
        out[who] = (val, used, " + ".join(expr))
    out["name"] = name
    return out


def build(q, hx, **p):
    from core.pack import build_pack, message_chars

    r = B.analyze(q, hx, **p)
    question = q.question
    dbg: dict = {}
    if p.get("hab"):
        np_ = habitual(q, hx, r, dbg)
        if np_:
            r["pair"] = np_
    if p.get("adm"):
        np_ = admit(q, hx, r, p, dbg)
        if np_:
            r["pair"] = np_
            r["sess_rank"] = list(dict.fromkeys(np_ + r["sess_rank"]))
    ot = owner_total(q, hx, r) if p.get("owner") else None
    if ot:
        return _owner_pack(q, hx, r, ot, p, dbg)
    pair, sess_rank, sents, best_by_sess = r["pair"], r["sess_rank"], r["sents"], r["best_by_sess"]

    extra = p.get("extra", 2)
    feats: dict = {}
    if p.get("gate"):
        ok, feats = R2.confidence(r, hx, question, p)
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
        if sid in pair or (s["own"] > 0 and s["val"] and s["cov"] >= p.get("tl_cov", 0.6)):
            tl.append((sid, hx.sess_by_id[sid]["date_str"], s["raw"]))
    if p.get("tl_pair") and len(tl) > 4:
        # keep both pair statements; fill the other slots with the most recent remaining statements
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

    computed = None
    if p.get("rel") and len(pair) == 2 and r["mode"] not in ("old", "both", "compare"):
        # the older count can sit in another sentence of the old session ("I have a total of 37 coins in that
        # collection" next to the best-scoring "organizing my pre-1920 American coins")
        olds = [best_by_sess[pair[0]]["raw"]] + [s["raw"] for s in sorted(sents, key=lambda s: -s["score"])
                                                 if s["sid"] == pair[0] and s["own"] >= 0 and s["cov"] >= 0.3]
        ru = None
        for o_raw in olds[:4]:
            ru = rel_update(o_raw, best_by_sess[pair[1]]["raw"], question)
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
    if p.get("lab") and len(cite) == 2 and hint:
        old_line = "Cite both the session with the earlier value and the session with the latest value."
        if (feats.get("confident", True) or not p.get("lab_gate")) and ap < p.get("lab_ap", 1.01):
            lo, ln = (pack_ids.index(s) + 1 for s in cite)
            if p.get("lab_if"):   # keep the abstention option open ('If you answer, ...')
                new_line = (f"If you answer, cite both [E{lo}] (the earlier value) and [E{ln}] (the latest value) "
                            f"on the final answer line: the answer, then [E{lo}][E{ln}].")
            else:
                new_line = (f"On the final answer line, cite both [E{lo}] (the earlier value) and [E{ln}] (the latest "
                            f"value): the answer, then [E{lo}][E{ln}].")
        else:   # pair not confident: do not designate passages, but still ask for both labels
            new_line = ("When you answer, cite two passages on the final answer line: the one with the earlier value "
                        "and the one with the latest value, not only the latest.")
        hint = hint.replace(old_line, new_line)
    return EvidencePack(question_id=q.question_id, variant=p.get("_variant", ""), route="ku", passages=passages,
                        hint=hint, cite_plan=cite or None, abstain_p=round(ap, 4),
                        included_turns=included_turns, computed=computed,
                        debug={"ranking": r["ranked_units"], "pair": pair, "sess_rank": sess_rank[:8],
                               "mode": r["mode"], "qd": r["qd"], "miss": round(r["miss"], 3), "absent": r["absent"],
                               "gate": feats, "miss_ph": miss_ph, "n_pass": len(passages), **dbg})


def _owner_pack(q, hx, r, ot, p, dbg):
    from core.pack import build_pack
    name = ot["name"]
    items = ot["self"][1] + ot["other"][1]
    sids = list(dict.fromkeys(it[2] for it in sorted(items)))
    lines = ["Counts per owner in the history, oldest first:"]
    for who, lab in (("self", "me"), ("other", name)):
        for it in ot[who][1]:
            lines.append(f'- {it[3][:10]} ({lab}): "{B._short(it[4], 180)}"')
    sv, ov = ot["self"][0], ot["other"][0]
    se, oe = ot["self"][2], ot["other"][2]
    se = se if se == str(sv) else f"{se} = {sv}"
    oe = oe if oe == str(ov) else f"{oe} = {ov}"
    lines.append(f"Latest counts: me {se}; {name} {oe}; total {sv} + {ov} = {sv + ov}.")
    hint = "\n".join(lines)
    pinned = [it[1] for it in items]
    passages, included_turns, _c = build_pack(q, hx, pinned, budget=p.get("budget", 4500), window=0, order="chrono",
                                              hint=hint + "\nCite every passage listed above.", max_sessions=len(sids),
                                              whole_turns=False, pinned=pinned)
    pack_ids = [ps.id for ps in passages if ps.id]
    labs = "".join(f"[E{pack_ids.index(s) + 1}]" for s in sids if s in pack_ids)
    hint += f"\nOn the final answer line, give the total, then {labs}."
    dbg["owner_total"] = (sv, ov, name)
    return EvidencePack(question_id=q.question_id, variant=p.get("_variant", ""), route="ku", passages=passages,
                        hint=hint, cite_plan=[s for s in sids if s in pack_ids] or None,
                        abstain_p=round(r["abstain_p"], 4), included_turns=included_turns, computed=sv + ov,
                        debug={"ranking": r["ranked_units"], "pair": sids, "n_pass": len(passages), **dbg})
