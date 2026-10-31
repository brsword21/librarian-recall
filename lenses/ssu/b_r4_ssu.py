"""SSU lens B, round 4: the frozen v2 ssu route (lenses/ssu/a_r3_route.py kg3crga + inner merged_r3 ku11fn4s_p5, params
from configs/best_v2.json) + a SENTENCE-LEVEL second hop (round-3 brief B item 1: score evidence by its best clean own
SENTENCE, not by the unit).

Why: the v2 ssu lens ranks UNITS with the cross-encoder. A long mixed unit ("Oh, and one piece of good news on the
family front: We finally booked the Van Nelle Factory for my daughter's wedding reception. ...") scores low as a unit,
so the answer sentence never enters the pack (test: venue pack holds only a 'trying to figure out venues' session).
Fix: CE over the SENTENCES of the user units in the hybrid pool (hx.rerank_texts, cached); if the best affirmed own
sentence is outside the pack and scores >= sh_abs and >= (best in-pack sentence - sh_delta), add its unit to the pack
(its session is inserted in chrono order; frozen passages are otherwise byte-identical). If it beats the best in-pack
sentence by >= sh_lead logits, it becomes the cite (the frozen cite pointed at a weaker sentence).

`res`: answer-type hop for "which neighbourhood / area" questions (vocabulary gap: "I live in Graça" never mentions
"neighbourhood"; CE -11, bge rank 454 -> no model finds it). Lexical first-person residence statements
("I live in X", "my flat/apartment is in X", "new apartment in X"), own-checked, best by CE, added when absent.
Idea credited to the round-3 brief (SSU-A item 1, category -> instance second hop).

Hand-offs (route_final ssa / ku) are returned unchanged.
"""
from __future__ import annotations

import json
import re

from core.pack import message_chars, render
from core.types import Passage
from core.loader import load_lens
from lenses.ssu.a_owner import REJECT_RX, classify, parse_question, sentences

from pathlib import Path

_V2 = json.load(open(Path(__file__).resolve().parents[2] / "configs" / "best_v2.json"))["routes"]["ssu"]
_loaded: dict = {}


def _lens(path):
    if path not in _loaded:
        _loaded[path] = load_lens(path)
    return _loaded[path]


_B = dict(sh=False, sh_k=150, sh_abs=2.0, sh_delta=1.5, sh_lead=2.0, sh_cite=True, res=False, budget=12000)
VARIANTS = {
    "v2": dict(_B),                                       # control: the frozen v2 ssu route
    "sh": dict(_B, sh=True),                              # sentence second hop
    "sh_nc": dict(_B, sh=True, sh_cite=False),            # ... pack only, never re-cite
    "sh_a1": dict(_B, sh=True, sh_abs=1.0),
    "sh_a3": dict(_B, sh=True, sh_abs=3.0),
    "sh_d0": dict(_B, sh=True, sh_delta=0.0),
    "res": dict(_B, res=True),                            # residence answer-type hop only
    "sh_res": dict(_B, sh=True, res=True),
}

RES_Q = re.compile(r"\b(?:neighbou?rhoods?|district|borough|quarter|part of (?:town|the city)|area)\b", re.I)
_NAME = r"([A-ZÀ-ž][\wÀ-ž'-]+(?:\s+[A-ZÀ-ž][\wÀ-ž'-]+){0,2})"
RES_S = re.compile(r"\b(?:I(?:'m| am)? (?:now )?(?:live|living|lives|settled)|we (?:now )?live|"
                   r"(?:my|our) (?:new )?(?:flat|apartment|place|house|home|neighbou?rhood) (?:is |'s )?)\s+in\s+"
                   r"(?:the\s+)?" + _NAME + r"|\bnew (?:flat|apartment|place|house|home) in " + _NAME)


def _own_ok(sent: str, qp) -> bool:
    sc = classify(sent, qp)
    if sc.owner == "mismatch" or sc.polarity in ("neg", "plan"):
        return False
    return not REJECT_RX.search(sent)


def _pack_uids(hx, pack) -> dict[str, set[int]]:
    """Reconstruct the uid set of every frozen passage (unit texts are substrings of the rendered passage)."""
    out = {}
    for ps in pack.passages:
        txt = ps.text
        s = hx.sess_by_id[ps.id]
        uids = set()
        for t in s["turns"]:
            for u in hx.turn_units(t["turn_id"]):
                if hx.units[u].text and hx.units[u].text in txt:
                    uids.add(u)
        out[ps.id] = uids
    return out


def _repack(q, hx, pack, sel: dict[str, set[int]], add_uid: int):
    sid = hx.units[add_uid].session_id
    new = {k: set(v) for k, v in sel.items()}
    new.setdefault(sid, set()).add(add_uid)
    keep = {ps.id: ps for ps in pack.passages}
    sids = sorted(new, key=lambda s: hx.sess_by_id[s]["sess_idx"])
    out = []
    for s in sids:
        if s in keep and s != sid:
            out.append(keep[s])          # untouched frozen passage
        else:
            text, _ = render(hx, sorted(new[s]))
            out.append(Passage(id=s, date=hx.sess_by_id[s]["date_str"], text=text))
    included = set(pack.included_turns or set()) | {hx.units[add_uid].turn_id}
    return out, included


def build(q, hx, **params):
    p = {**_B, **{k: v for k, v in params.items() if not k.startswith("_")}}
    L = _lens(_V2["lens"])
    pack = L.build(q, hx, _variant=_V2["variant"], **{**L.VARIANTS[_V2["variant"]], **_V2["params"]})
    pack.variant = params.get("_variant", "")
    dbg = pack.debug or {}
    if dbg.get("route_final", "ssu") != "ssu" or not (p["sh"] or p["res"]):
        return pack
    qp = parse_question(q.question)
    sel = _pack_uids(hx, pack)
    in_pack = {u for v in sel.values() for u in v}
    # in-pack best user sentence
    pin_t = [s for u in in_pack if hx.units[u].role == "user"
             for s in (sentences(hx.units[u].text) or [hx.units[u].text]) if len(s) >= 12]
    best_in = max(hx.rerank_texts(q.question, pin_t)) if pin_t else -12.0
    dbg["sh_best_in"] = round(float(best_in), 2)
    add_uid, lead = None, False

    if p["sh"]:
        pool = [u for u, _ in hx.hybrid(q.question, k=p["sh_k"], roles=["user"], pool=400)]
        pool += [u for u in dbg.get("ranking", [])[:100] if u not in set(pool)]
        texts, own = [], []
        for u in pool:
            for s in sentences(hx.units[u].text) or [hx.units[u].text]:
                if len(s) >= 12:
                    texts.append(s)
                    own.append(u)
        sc = hx.rerank_texts(q.question, texts) if texts else []
        order = sorted(range(len(texts)), key=lambda i: -sc[i])
        for i in order[:8]:
            if sc[i] < p["sh_abs"] or sc[i] < best_in - p["sh_delta"]:
                break
            if own[i] in in_pack:
                continue
            if not _own_ok(texts[i], qp):
                continue
            add_uid = own[i]
            lead = sc[i] >= best_in + p["sh_lead"]
            dbg["sh_add"] = (own[i], round(float(sc[i]), 2), texts[i][:120])
            break

    if add_uid is None and p["res"] and RES_Q.search(q.question) and re.search(r"\b(?:my|I)\b", q.question):
        qwords = set(re.findall(r"[A-Z][\wÀ-ž]+", q.question))
        cands = []
        for u in hx.units:
            if u.role != "user":
                continue
            for s in sentences(u.text) or [u.text]:
                m = RES_S.search(s)
                if not m:
                    continue
                name = (m.group(1) or m.group(2) or "").strip()
                if not name or name in qwords or name.split()[0] in ("I", "The", "A"):
                    continue
                if _own_ok(s, qp):
                    cands.append((u.uid, s))
        if cands:
            rs = hx.rerank_texts(q.question, [s for _, s in cands])
            j = max(range(len(cands)), key=lambda i: rs[i])
            if cands[j][0] not in in_pack:
                add_uid = cands[j][0]
                lead = best_in < 0.0
                dbg["res_add"] = (cands[j][0], round(float(rs[j]), 2), cands[j][1][:120])

    if add_uid is None:
        return pack
    passages, included = _repack(q, hx, pack, sel, add_uid)
    if message_chars(q, passages, pack.hint) > p["budget"]:
        return pack
    pack.passages, pack.included_turns = passages, included
    if p["sh_cite"] and lead:
        pack.cite_plan = [hx.units[add_uid].session_id]
    dbg["sh_lead"] = lead
    pack.debug = dbg
    return pack
