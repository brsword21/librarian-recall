"""KU lens B, round 4: post-processing wrapper over the frozen ku lens `lenses/ku/a_r3.py` r3w (KU-A, credited).

Two general fixes for defects visible on the real-Reactor test pass (no per-question rules):

1. `nolate`: drop uncited context sessions dated AFTER the newest cite_plan session.
   The cite_plan's newest session is the lens's claim for "the latest value". A later session that the lens did not
   cite is either a decoy session (sibling value, planted 'my brother pays 22,000') or noise; when it carries a
   contradicting first-person line Reactor follows the 'MOST RECENT value' instruction and answers from it
   (test rent: v1 answered "9,000 shillings" from an uncited brother-decoy session dated after the 22,000 update).
   Earlier context sessions are kept (they cannot override the latest value).

2. Count-of-instances questions. "How many <things> have I/we <done> (so far / in total)" questions whose statement
   list carries no number (digits, number words, once/twice, second..tenth) are accumulations of events ("we finished
   the main bedroom", "the bathroom being done"), not a stated value that was updated.
   - `nolate_c` (RECOMMENDED): nolate, but skipped on these questions (there is no latest-value claim to protect).
   - `nolate_c_cnt2` (optional, +1 test PA fix): on these questions build an instance pack instead: own user units with
     the question's verb stem or a completion verb, ranked by bge similarity to the question's statement form
     ("we renovated in the bungalow the rooms"), pinned by rank under a 6000-char budget, rendered chronologically,
     with a list-then-count hint. abstain_p stays on the ku lens's scale.
   - `cnt` / `nolate_cnt` (REJECTED): hand-off to the frozen multi route lens; its pack was 9.7k chars of renovation
     chatter with none of the completion lines.
"""
from __future__ import annotations

import importlib.util
import re
from pathlib import Path

from core.pack import message_chars
from core.types import EvidencePack

ROOT = Path(__file__).resolve().parents[2]
_MODS: dict = {}


def _lens(path: str):
    if path not in _MODS:
        p = ROOT / path
        spec = importlib.util.spec_from_file_location(f"b_r4_{p.parent.name}_{p.stem}", p)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _MODS[path] = mod
    return _MODS[path]


KU_LENS, KU_VARIANT = "lenses/ku/a_r3.py", "r3w"
MULTI_LENS, MULTI_VARIANT = "lenses/multi/merged_r2.py", "tight_role9"
MULTI_PARAMS = {"role_module": "lenses/ssa/a_r3_role.py"}

VARIANTS = {
    "r3w": {},                                   # control = frozen v2 ku route
    "nolate": {"nolate": True},
    "cnt": {"cnt": True},
    "nolate_cnt": {"nolate": True, "cnt": True},
    "nolate_v": {"nolate": True, "need_value": True},   # nolate only when the newest cited statement states a value
    "nolate_c": {"nolate": True, "skip_cnt": True},     # nolate except on count-of-instances questions (RECOMMENDED)
    # + count-of-instances pack: own event/completion sentences ranked by dense similarity to the question's statement
    #   form, pinned chronologically, with a list-then-count hint (replaces the empty latest-value statement list)
    "nolate_c_cnt2": {"nolate": True, "skip_cnt": True, "cnt2": True, "cnt2_k": 24, "cnt2_budget": 6000},
}

_DONE = r"finished|done|wrapped up|completed?|finally"
_CNT_HINT = ("These facts are spread over several sessions. First list every matching instance as 'date | [E#] | item', "
             "then count them. Count each item once. Only count what I (the user) actually did or have; ignore what "
             "friends, family, neighbours or others did, and ignore things only planned, started, not finished or "
             "that I said do not count.")


def _instance_pack(q, hx, pk: EvidencePack, p: dict, dbg: dict) -> EvidencePack:
    from core.pack import build_pack
    m = re.match(r"\s*how many\s+(.+?)\s+(?:have|has|did)\s+(i|we)\s+(.+?)\??\s*$", q.question, re.I)
    if not m:
        return pk
    rest = re.sub(r"\b(so far|in total|completely|already)\b", " ", m.group(3), flags=re.I)
    stmt = re.sub(r"\s+", " ", f"{m.group(2)} {rest} the {m.group(1)}").strip()
    verb = re.sub(r"(ed|d)$", "", m.group(3).split()[0].lower()) if m.group(3).split() else ""
    vb = re.escape(verb[:6]) + r"\w*" if len(verb) >= 3 else ""
    ev = re.compile(r"\b(" + "|".join(x for x in (vb, _DONE) if x) + r")\b", re.I)
    cand = [u for u in hx.units if u.role == "user" and ev.search(u.text)]
    if len(cand) < 3:
        return pk
    ds = hx.dense_texts(stmt, [u.text for u in cand])
    # pinned in similarity order (build_pack renders chronologically), so the char budget, not k, trims the tail
    top = [cand[i].uid for i in sorted(range(len(cand)), key=lambda i: -ds[i])[: p.get("cnt2_k", 24)]]
    passages, inc, _c = build_pack(q, hx, top, budget=p.get("cnt2_budget", 6000), window=0, order="chrono",
                                   hint=_CNT_HINT, pinned=top)
    d = dict(pk.debug or {})
    d["b_r4"] = dict(dbg, cnt2={"stmt": stmt, "n_cand": len(cand), "top": top})
    return EvidencePack(question_id=q.question_id, variant=p.get("_variant", ""), route="ku", passages=passages,
                        hint=_CNT_HINT, cite_plan=[ps.id for ps in passages if ps.id] or None,
                        abstain_p=pk.abstain_p, included_turns=set(inc), computed=None, debug=d)

_NUMW = (r"\b(?:\d[\d,.]*|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|"
         r"sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|a dozen|dozen|single|both|couple|"
         r"once|twice|thrice|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\b")
# "how many rooms have we completely renovated", "how many books have I read so far", "how many ... did I finish in total"
_CNT_Q = re.compile(r"^\s*how many\s+(?:[a-z'-]+\s+){1,4}?(?:have|has|did)\s+(?:i|we)\s+(?:\w+ly\s+)?(?:already\s+)?"
                    r"([a-z]+ed|done|made|read|run|seen|built|grown|sold|bought|written|taken|won|eaten|met|visited)\b",
                    re.I)
_STMT = re.compile(r'^- \d{4}/\d\d/\d\d: "(.*)"\s*$', re.M)


def _statements(hint: str) -> list[str]:
    return _STMT.findall(hint or "")


def _sess_date_idx(hx, sid):
    s = hx.sess_by_id.get(sid)
    return s["sess_idx"] if s else -1


_STMT_D = re.compile(r'^- (\d{4}/\d\d/\d\d): "(.*)"\s*$', re.M)
# a stated value: a number, or a multi-word proper name not at sentence start ("Director of Clinical Services")
_NAME = re.compile(r"(?<=[a-z,;:] )[A-Z][a-z]+(?:\s+(?:of|de|the|and)?\s*[A-Z][a-z]+)+|['‘\"][A-Z][^'’\"]{1,30}['’\"]")


def _newest_has_value(hx, pk: EvidencePack, newest_sid: str) -> bool:
    """True when the hint's statement for the newest cited session carries a value (the lens's latest-value claim).
    Count-of-instances questions ('how many rooms have we renovated') have no such claim: nothing to protect."""
    d = hx.sess_by_id[newest_sid]["date_str"][:10]
    st = [s for dd, s in _STMT_D.findall(pk.hint or "") if dd == d]
    return any(re.search(_NUMW, s, re.I) or _NAME.search(s) for s in st)


def _drop_late(q, hx, pk: EvidencePack, dbg: dict, need_value: bool = False) -> EvidencePack:
    cite = [s for s in (pk.cite_plan or []) if s in hx.sess_by_id]
    if not cite:
        return pk
    newest = max(_sess_date_idx(hx, s) for s in cite)
    if need_value:
        nsid = max(cite, key=lambda s: _sess_date_idx(hx, s))
        if not _newest_has_value(hx, pk, nsid):
            dbg["nolate_skip"] = "newest cited statement carries no value"
            return pk
    keep, drop = [], []
    for ps in pk.passages:
        if ps.id is None or ps.id in cite or _sess_date_idx(hx, ps.id) <= newest:
            keep.append(ps)
        else:
            drop.append(ps.id)
    if not drop:
        return pk
    dropped = set(drop)
    turns = {u.turn_id for u in hx.units if u.session_id in dropped}
    pk.passages = keep
    pk.included_turns = set(t for t in (pk.included_turns or []) if t not in turns)
    dbg["nolate_dropped"] = drop
    return pk


def build(q, hx, **p) -> EvidencePack:
    dbg: dict = {}
    if p.get("cnt") and _CNT_Q.search(q.question):
        ku_pk = _lens(KU_LENS).build(q, hx, **dict(_lens(KU_LENS).VARIANTS[KU_VARIANT], _variant=KU_VARIANT))
        stm = _statements(ku_pk.hint)
        if not any(re.search(_NUMW, s, re.I) for s in stm) and ku_pk.computed is None:
            mp = _lens(MULTI_LENS).build(q, hx, **dict(_lens(MULTI_LENS).VARIANTS[MULTI_VARIANT], **MULTI_PARAMS,
                                                         _variant=MULTI_VARIANT))
            mp.variant = p.get("_variant", "")
            mp.route = "ku"
            d = dict(mp.debug or {})
            d["b_r4"] = {"cnt_handoff": True, "ku_statements": len(stm)}
            mp.debug = d
            mp.abstain_p = ku_pk.abstain_p          # ku-calibrated scale (min() cost AUC on an unanswerable count)
            return mp
        pk = ku_pk
    else:
        pk = _lens(KU_LENS).build(q, hx, **dict(_lens(KU_LENS).VARIANTS[KU_VARIANT], _variant=KU_VARIANT))
    if p.get("nolate"):
        if p.get("skip_cnt") and _CNT_Q.search(q.question) and pk.computed is None \
                and not any(re.search(_NUMW, s, re.I) for s in _statements(pk.hint)):
            dbg["nolate_skip"] = "count-of-instances question: no latest-value claim to protect"
            if p.get("cnt2"):
                return _instance_pack(q, hx, pk, p, dbg)
        else:
            pk = _drop_late(q, hx, pk, dbg, need_value=bool(p.get("need_value")))
    pk.variant = p.get("_variant", "")
    if dbg:
        d = dict(pk.debug or {})
        d["b_r4"] = dbg
        pk.debug = d
    return pk
