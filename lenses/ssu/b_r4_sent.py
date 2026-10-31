"""SSU lens B, round 4 (final): the frozen v2 ssu route + a GATED sentence-level second hop (+ residence hop,
+ recency-verb hop). BEST: `g_res_lv` (lenses/ssu/b_report.md). dev / dev200 (gold and routed) identical to v2 (no hop
fires there); 3 of 45 test ssu packs change: q_607d3cf5b4f1 (neighbourhood, Graça), q_6fa4f8620294 (venue, Van Nelle
Factory), q_4b0e37d9cd0c (last kegged beer, peach saison).

Base: lenses/ssu/b_r4_ssu.py (same repack machinery; its `sh` variant was too loose on test: it added a planted
running-partner decoy, wrong-attribute R-77 sentences and a no-name 'younger son' line). This file tightens the gate:
  1. CE(sentence) >= sh_abs and >= best in-pack user sentence + sh_margin (it must clearly beat what the pack holds);
  2. a_owner.classify says valid (owner not mismatch, polarity not neg / plan, head or verb hit), no REJECT_RX
     ('briefly tried ... removed it');
  3. no third-party subject: '(Oh,) my <relation> is/was/...' where the relation noun is not in the question
     (running partner's half marathon, sister's physiotherapist);
  4. name questions (qp.attr == 'name'): the sentence carries a capitalised name that is not in the question.
If the added sentence beats the best in-pack sentence by >= sh_lead logits it becomes the cite.
`res` (residence hop) as in b_r4_ssu. Hand-offs (route_final ssa / ku) unchanged.
"""
from __future__ import annotations

import re

from core.pack import message_chars
from lenses.ssu import b_r4_ssu as B
from lenses.ssu.a_owner import CAP_STOP, REJECT_RX, classify, parse_question, sentences

_D = dict(sh=True, sh_k=150, sh_abs=2.0, sh_margin=1.0, sh_lead=2.0, res=False, budget=12000, name_need=True,
          third=True, lastv=False, lastv_rare=8, delta=False)
VARIANTS = {
    "g": dict(_D),
    "g_res": dict(_D, res=True),
    "g_m0": dict(_D, sh_margin=0.0),
    "g_m0_res": dict(_D, sh_margin=0.0, res=True),
    "res": dict(_D, sh=False, res=True),
    # + recency-verb hop for "the last X I mentioned <verb>ing / I'd <verb>ed": the latest own affirmed sentence with
    # an inflected form of that verb (CE misses 'I finally kegged my peach saison' at -4.4 for the beer question)
    "g_res_lv": dict(_D, res=True, lastv=True),
    "lv": dict(_D, sh=False, lastv=True),
    # + delta hop for "By how many <unit> did I <change> <attribute> (in my <object>)": every own sentence that names
    # the attribute and carries a value in that unit (and the object, if any) enters (test: volume pot 500k -> 250k
    # missing; the v2 pack held only cathode-resistor 150 -> 180 sentences, an invitation to answer 30)
    "g_res_lv_d": dict(_D, res=True, lastv=True, delta=True),
}
DELTA_Q = re.compile(r"\bby how (?:many|much) ([\w%-]+) did (?:I|we) (\w+) (?:the |my |our )?(.+?)(?: (?:in|on|of|for) "
                     r"(?:my|our|the) (.+?))?\?", re.I)
_GENERIC = {"resistance", "value", "amount", "number", "level", "size", "time", "total", "price", "cost", "weight",
            "distance", "speed", "count", "rate"}
LAST_R1 = re.compile(r"\bI\s+(?:last|most recently)\s+mentioned\s+(?:(?:I'd|I had|I've|I have|I was|I'm|I am|I)\s+)?"
                     r"(\w+)", re.I)
LAST_R2 = re.compile(r"\b(?:last|latest|most recent)\b[^?]{0,40}?\bI\s+mentioned\s+(?:(?:I'd|I had|I've|I have|I was|"
                     r"I'm|I am|I)\s+)?(\w+)", re.I)
PROG_RX = re.compile(r"\b(?:was|am|'m|’m|been|were|are|'re|’re)\s+(?:\w+\s+)?$", re.I)

REL = (r"partner|friend|sister|brother|mom|mother|dad|father|wife|husband|son|daughter|neighbou?r|coworker|"
       r"co-worker|colleague|boss|cousin|aunt|uncle|niece|nephew|roommate|flatmate|grandma|grandpa|grandmother|"
       r"grandfather|kid|kids|child|children|teen|boyfriend|girlfriend|fianc[eé]e?|in-laws?|nurse|manager|teacher|"
       r"classmate|teammate|landlord|mentor")
THIRD_RX = re.compile(r"^\W*(?:(?:oh|and|also|anyway|by the way|so|well|btw|plus|actually)\W+)*(?:my|our)\s+"
                      r"(?:\w+\s+){0,2}?(" + REL + r")(?:'s|’s)?\s+(?:\w+\s+){0,1}?(?:is|was|has|had|just|said|"
                      r"saw|went|got|booked|bought|tried|named|called|recommended|told|invited|joined|loves?|"
                      r"swears|uses?|runs?|trains?|wants?)\b", re.I)
_CAL = {"Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday", "January", "February", "March",
        "April", "May", "June", "July", "August", "September", "October", "November", "December", "I", "I'm", "I've",
        "I'd", "I'll", "Oh", "OK", "Ok"}


def _names(sent: str, question: str) -> list[str]:
    toks = re.findall(r"(?<![\w'])([A-Z][\wÀ-ž'-]+)", sent)
    first = re.match(r"\W*([\w'-]+)", sent)
    first = first.group(1) if first else ""
    return [t for t in toks if t != first and t not in _CAL and t not in CAP_STOP and t.lower() not in question.lower()]


def _gate(sent: str, qp, question: str, p) -> str | None:
    """None if the sentence may enter, else the reason it may not."""
    sc = classify(sent, qp)
    if not sc.valid:
        return "invalid"
    if REJECT_RX.search(sent):
        return "reject"
    if p["third"]:
        m = THIRD_RX.search(sent)
        if m and not re.search(r"\b" + re.escape(m.group(1)) + r"\b", question, re.I):
            return "third:" + m.group(1)
    if p["name_need"] and qp.attr == "name" and not _names(sent, question):
        return "noname"
    return None


def build(q, hx, **params):
    p = {**_D, **{k: v for k, v in params.items() if not k.startswith("_")}}
    pack = B.build(q, hx, **B.VARIANTS["v2"])
    pack.variant = params.get("_variant", "")
    dbg = pack.debug or {}
    if dbg.get("route_final", "ssu") != "ssu" or not (p["sh"] or p["res"]):
        return pack
    qp = parse_question(q.question)
    sel = B._pack_uids(hx, pack)
    in_pack = {u for v in sel.values() for u in v}
    pin_t = [s for u in in_pack if hx.units[u].role == "user"
             for s in (sentences(hx.units[u].text) or [hx.units[u].text]) if len(s) >= 12]
    best_in = float(max(hx.rerank_texts(q.question, pin_t))) if pin_t else -12.0
    dbg["sh_best_in"] = round(best_in, 2)
    add_uid, lead, why = None, False, []
    if p["sh"]:
        pool = [u for u, _ in hx.hybrid(q.question, k=p["sh_k"], roles=["user"], pool=400)]
        seen = set(pool)
        pool += [u for u in dbg.get("ranking", [])[:100] if u not in seen]
        texts, own = [], []
        for u in pool:
            for s in sentences(hx.units[u].text) or [hx.units[u].text]:
                if len(s) >= 12:
                    texts.append(s)
                    own.append(u)
        sc = hx.rerank_texts(q.question, texts) if texts else []
        order = sorted(range(len(texts)), key=lambda i: -sc[i])
        for i in order[:8]:
            if sc[i] < p["sh_abs"] or sc[i] < best_in + p["sh_margin"]:
                break
            if own[i] in in_pack:
                break                      # the pack already holds the best remaining sentence
            r = _gate(texts[i], qp, q.question, p)
            if r:
                why.append((own[i], round(float(sc[i]), 2), r))
                continue
            add_uid = own[i]
            lead = sc[i] >= best_in + p["sh_lead"]
            dbg["sh_add"] = (own[i], round(float(sc[i]), 2), texts[i][:120])
            break
    if why:
        dbg["sh_skip"] = why[:4]
    extra = []
    if add_uid is None and p.get("delta"):
        extra = _delta(q, hx, qp, p, in_pack)
        if extra:
            dbg["delta_add"] = [(u, hx.units[u].date_str[:10], hx.units[u].text[:100]) for u in extra]
    if add_uid is None and not extra and p["lastv"]:
        hit = _lastv(q, hx, qp, p)
        if hit is not None:
            dbg["lv"] = (hit, hx.units[hit].date_str[:10], hx.units[hit].text[:120])
            if hit not in in_pack:
                add_uid, lead = hit, True
    if add_uid is None and p["res"] and B.RES_Q.search(q.question) and re.search(r"\b(?:my|I)\b", q.question):
        qwords = set(re.findall(r"[A-Z][\wÀ-ž]+", q.question))
        cands = []
        for u in hx.units:
            if u.role != "user":
                continue
            for s in sentences(u.text) or [u.text]:
                m = B.RES_S.search(s)
                if not m:
                    continue
                name = (m.group(1) or m.group(2) or "").strip()
                if not name or name in qwords or name.split()[0] in ("I", "The", "A"):
                    continue
                if B._own_ok(s, qp):
                    cands.append((u.uid, s))
        if cands:
            rs = hx.rerank_texts(q.question, [s for _, s in cands])
            j = max(range(len(cands)), key=lambda i: rs[i])
            if cands[j][0] not in in_pack:
                add_uid = cands[j][0]
                lead = best_in < 0.0
                dbg["res_add"] = (cands[j][0], round(float(rs[j]), 2), cands[j][1][:120])
    if extra:
        passages, included = pack.passages, pack.included_turns
        for u in extra:
            passages, included = B._repack(q, hx, pack, sel, u)
            pack.passages, pack.included_turns = passages, included
            sel = B._pack_uids(hx, pack)
        if message_chars(q, passages, pack.hint) <= p["budget"]:
            pack.cite_plan = sorted({hx.units[u].session_id for u in extra},
                                    key=lambda x: hx.sess_by_id[x]["sess_idx"])
        pack.debug = dbg
        return pack
    if add_uid is None:
        pack.debug = dbg
        return pack
    passages, included = B._repack(q, hx, pack, sel, add_uid)
    if message_chars(q, passages, pack.hint) > p["budget"]:
        return pack
    pack.passages, pack.included_turns = passages, included
    if lead:
        pack.cite_plan = [hx.units[add_uid].session_id]
    dbg["sh_lead"] = lead
    pack.debug = dbg
    return pack


def _lastv(q, hx, qp, p):
    """Latest own affirmed user sentence carrying an inflected form (past / progressive) of the question's verb in
    'the last X I mentioned <verb>'. Generic verbs (buy, try, read) need the question's head noun in the sentence;
    a rare verb (<= lastv_rare accepted sentences: kegged, camped) does not. Generic verbs also need a named instance."""
    from lenses.ssu.a_owner import stem
    m = LAST_R1.search(q.question) or LAST_R2.search(q.question)
    if not m:
        return None
    verb = m.group(1).lower()
    vs = stem(verb)
    if vs in ("be", "have", "get", "do", "go", "say", "mention", "the", "a", "my"):
        return None
    heads = [f for f in qp.head_forms() if f]
    acc, n_stem = [], 0
    for u in hx.units:
        if u.role != "user":
            continue
        for s in sentences(u.text) or [u.text]:
            if any(stem(w.lower()) == vs for w in re.findall(r"[A-Za-z']+", s)):
                n_stem += 1
            ok = False
            for t in re.finditer(r"[A-Za-z']+", s):
                w = t.group(0).lower()
                if stem(w) != vs or w in (vs, vs + "s", vs + "es"):
                    continue
                if w.endswith("ing") and not PROG_RX.search(s[: t.start()]):
                    continue
                ok = True
                break
            if not ok or _gate_lv(s, qp, q.question, p):
                continue
            acc.append((u.uid, s))
    if not acc:
        return None
    rare = n_stem <= p["lastv_rare"]      # every user sentence with any form of the verb (keg: 7, read: dozens)
    if not rare:
        # a generic verb (buy, try, read) needs the head noun AND a named instance ('self-help books' is no title)
        acc = [(u, s) for u, s in acc if any(re.search(r"\b" + re.escape(h) + r"s?\b", s, re.I) for h in heads)
               and (_names(s, q.question) or re.search(r"['\"“‘][^'\"”’]{3,}['\"”’]", s))]
    if not acc:
        return None
    return max(acc, key=lambda x: (hx.units[x[0]].sess_idx, x[0]))[0]


def _gate_lv(sent, qp, question, p):
    sc = classify(sent, qp)
    if sc.owner == "mismatch" or sc.polarity in ("neg", "plan"):
        return "own"
    if REJECT_RX.search(sent):
        return "reject"
    m = THIRD_RX.search(sent)
    if m and not re.search(r"\b" + re.escape(m.group(1)) + r"\b", question, re.I):
        return "third"
    if not re.search(r"\b(?:I|I'm|I've|I'd|we|we've|we're|my|our)\b", sent, re.I):
        return "noperson"
    return None


def _delta(q, hx, qp, p, in_pack) -> list[int]:
    """Own user sentences naming the attribute with a value in the asked unit (and the object, if named): the before /
    after statements of a 'by how many <unit> did I <change>' question. Returns the missing units (<= 3, chrono)."""
    m = DELTA_Q.search(q.question)
    if not m:
        return []
    attr = [w for w in re.findall(r"[a-z]+", m.group(3).lower()) if len(w) > 3 and w not in _GENERIC]
    uw = m.group(1).lower().rstrip("s")
    val_rx = re.compile(r"\b\d[\d,.]*\s*(?:[kKmM]\b|[kKmM](?=-))?\s*-?\s*" + re.escape(uw) + r"s?\b", re.I)
    obj = (m.group(4) or "").strip()
    if not attr:
        return []
    hits = []
    for u in hx.units:
        if u.role != "user":
            continue
        for s in sentences(u.text) or [u.text]:
            sl = s.lower()
            if not any(re.search(r"\b" + re.escape(a), sl) for a in attr):
                continue
            if obj and obj.lower() not in sl:
                continue
            if not val_rx.search(s):
                continue
            if _gate_lv(s, qp, q.question, p):
                continue
            hits.append(u.uid)
            break
    hits = sorted(set(hits))
    if not hits or len(hits) > 3:
        return []
    if all(h in in_pack for h in hits):
        return []
    return [h for h in hits if h not in in_pack]
