"""Temporal lens B, round 4: recency guard for "the last X I mentioned VERB" / "which X did I VERB most recently".

Why: these are order / recency questions, but the text router sends them to ssu (route conf 0.4). On test, 5 questions
of this form sit in one history. The frozen ssu route answers 3 of them from the right session, but 2 packs miss
the user's latest own statement:
  * "style of the last beer I mentioned I'd kegged": "I finally kegged my peach saison last weekend" (2023/05/21) is
    absent; the pack cites the 2023/03/18 "Maybe I'll just keg the wheat" (a maybe-plan).
  * "last book I mentioned I was reading": "I switched to reading 'The Civil War: A Narrative' by Shelby Foote"
    (2023/05/10) is absent; the pack holds January / February books only.
Dev analog: "Which streaming service did I start using most recently?" (temporal gold -> ssu route, PA 0 in v2).

What it does (wrapper, the inner pack is the frozen v2 pack byte for byte unless the guard fires):
  1. parse the question into a category ("beer") and an action ("kegged");
  2. candidates = user sentences that contain a form of the action verb, are the user's own done statement
     (a_calendar.subject_kind == self, a_r3_gap._undone false: no plan / negation / "maybe"), and are relevant to the
     statement "I <verb> <category>" by the cross-encoder (hx.rerank_texts, >= ce_thr);
  3. the latest candidate by session date (then turn order) is the recency answer;
  4. if its session is already in the inner pack's cite_plan, the inner pack is returned unchanged (no re-ask);
     otherwise the latest statement's unit is pinned first, the inner ranking follows (same budget), the hint names
     the latest own statement with its date, and the cite_plan becomes that session.
Imports the frozen lenses; edits nothing.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from core.types import EvidencePack
from core.loader import load_lens
from lenses.temporal import a_calendar as A
from lenses.temporal import a_r3_gap as R3

_ROOT = Path(__file__).resolve().parents[2]


def _spec(config: str, route: str) -> dict:
    return json.loads((_ROOT / config).read_text())["routes"][route]


VARIANTS = {
    # wire as routes.ssu: inner = the v2 ssu route spec (configs/best_v2.json)
    "last": {"inner_route": "ssu", "config": "configs/best_v2.json"},
    # evaluation: inner = the whole v2 pipeline; the guard only acts when the final route is ssu
    "last_pipe": {"inner_pipe": True, "config": "configs/best_v2.json"},
    # stricter relevance floor
    "last_ce4": {"inner_route": "ssu", "config": "configs/best_v2.json", "ce_thr": -4.0},
    "last_pipe_lo": {"inner_pipe": True, "config": "configs/best_v2.json", "ce_thr": -10.0},
    "last_pipe_ce4": {"inner_pipe": True, "config": "configs/best_v2.json", "ce_thr": -4.0},
}

# "What was the style of the last beer I mentioned I'd kegged?" / "What was the brand of the grill I last mentioned
# buying?" / "Which streaming service did I start using most recently?"
_P1 = re.compile(r"\blast\s+(?P<cat>[a-z][a-z \-]{1,40}?)\s+I\s+(?:had\s+)?(?:mentioned|told you(?: about)?|said)\s+"
                 r"(?P<verb>.+?)\s*\?*$", re.I)
_P2 = re.compile(r"\b(?:the\s+)?(?P<cat>[a-z][a-z \-]{1,40}?)\s+I\s+last\s+(?:mentioned|told you about)\s+"
                 r"(?P<verb>.+?)\s*\?*$", re.I)
_P3 = re.compile(r"\b(?:which|what)\s+(?:type of\s+|kind of\s+)?(?P<cat>[a-z][a-z \-]{1,40}?)\s+(?:did|have)\s+I\s+"
                 r"(?P<verb>[a-z][a-z \-]{1,40}?)\s+most recently\b", re.I)
_AUX = {"i", "i'd", "i've", "i'm", "was", "were", "had", "have", "been", "did", "do", "at", "to", "in", "on", "of",
        "for", "with", "a", "an", "the", "my", "about", "from", "out", "into"}
_IRR = {"buy": {"bought", "buy", "buying", "buys"}, "read": {"read", "reading", "reads"},
        "get": {"got", "get", "getting", "gotten"}, "go": {"went", "go", "going", "gone"},
        "eat": {"ate", "eat", "eating", "eaten"}, "see": {"saw", "see", "seen", "seeing"},
        "take": {"took", "take", "taking", "taken"}, "make": {"made", "make", "making"},
        "drink": {"drank", "drink", "drinking", "drunk"}, "begin": {"began", "begin", "beginning", "begun"},
        "fly": {"flew", "fly", "flying", "flown"}, "try": {"tried", "try", "trying", "tries"},
        "purchase": {"purchased", "purchase", "purchasing", "bought"}}


def parse_last(text: str) -> dict | None:
    t = text.strip().split("\n")[0]
    for rx in (_P1, _P2, _P3):
        m = rx.search(t)
        if m:
            cat = re.sub(r"^(?:what|which)\s+(?:was|is)\s+", "", m.group("cat").strip(), flags=re.I)
            cat = re.sub(r"^.*?\bof\s+(?:the|my)\s+", "", cat, flags=re.I)
            cat = re.sub(r"^(?:the|a|an)\s+", "", cat, flags=re.I)
            verb = m.group("verb").strip().rstrip("?").strip()
            vtoks = [w for w in re.findall(r"[a-z']+", verb.lower()) if w not in _AUX]
            if not vtoks or not cat:
                return None
            return {"cat": cat, "verb": verb, "vtoks": vtoks}
    return None


def _base(w: str) -> str:
    w = w.lower()
    for b, forms in _IRR.items():
        if w in forms:
            return b
    for suf, rep in (("ied", "y"), ("ing", ""), ("ed", ""), ("es", ""), ("s", "")):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            w = w[: -len(suf)] + rep
            if suf in ("ing", "ed") and len(w) > 3 and w[-1] == w[-2] and w[-1] not in "ls":
                w = w[:-1]
            break
    return w


def _verb_forms(vtoks: list[str]) -> set[str]:
    out = set()
    for v in vtoks:
        b = _base(v)
        out |= _IRR.get(b, set())
        for x in (b, b.rstrip("e")):
            out |= {x, x + "s", x + "es", x + "ed", x + "d", x + "ing", x + x[-1:] + "ed", x + x[-1:] + "ing"}
        if b.endswith("y"):
            out |= {b[:-1] + "ied", b[:-1] + "ies"}
    return out


def _statement(pq: dict) -> str:
    v = re.sub(r"^(?:I'd|I had|I was|I)\s+", "", pq["verb"], flags=re.I)
    return f"I {v} {pq['cat']}".strip() if not v.lower().endswith((" at", " in", " on")) else f"I {v} a {pq['cat']}"


_PAST_IRR = {"buy": ["bought"], "read": ["read"], "get": ["got"], "go": ["went"], "eat": ["ate"], "see": ["saw"],
             "take": ["took"], "make": ["made"], "drink": ["drank"], "begin": ["began"], "fly": ["flew"],
             "purchase": ["purchased", "bought"], "try": ["tried"]}
_OBJ_BAD = re.compile(r"^(?:a lot|a bit|a little|more|less|to\b|it\b|them\b|that\b|this\b|some of|any)", re.I)
_OBJ_OK = re.compile(r"^(?:(?:at|to|in|on|from|into)\s+)?(?:['\"‘“]|[A-Z0-9]|(?:my|our|a|an|the|this|his|her)\s+\S)")


def _forms(vtoks: list[str]) -> tuple[list[str], list[str]]:
    past, ing = set(), set()
    for v in vtoks:
        b = _base(v)
        past |= set(_PAST_IRR.get(b, []))
        stems = {b, b[:-1]} if b.endswith("e") else {b, b + b[-1:]}
        for x in stems:
            past |= {x + "ed", x + "d"} if x.endswith("e") or x == b else {x + "ed"}
            ing.add(x + "ing")
        if b.endswith("y"):
            past.add(b[:-1] + "ied")
        if b.endswith("e"):
            past.add(b + "d")
    return sorted(past, key=len, reverse=True), sorted(ing, key=len, reverse=True)


def candidates(q, hx, pq: dict, params: dict) -> list[dict]:
    """Own, done statements of the action with a specific object: "I finally kegged my peach saison", "I switched to
    reading 'The Civil War: A Narrative'", "We camped at Cloudland Canyon State Park", "I bought a Big Green Egg"."""
    past, ing = _forms(pq["vtoks"])
    vx = re.compile(rf"\b(?:{'|'.join(map(re.escape, past))})\b|\b(?:been|started|start|switched to|began|am|was|were|"
                    rf"'m|’m)\s+(?:{'|'.join(map(re.escape, ing))})\b", re.I)
    qlow = q.question.lower()
    rows = []
    for u in hx.units:
        if u.role != "user":
            continue
        for _, s in A.sentences(u.text):
            m = vx.search(s)
            if not m:
                continue
            obj = s[m.end():].lstrip(" ,")
            if _OBJ_BAD.match(obj) or not _OBJ_OK.match(obj):
                continue
            rows.append((u, s))
    if not rows:
        return []
    qq = re.sub(r"\b(?:last|most recently)\b ?", "", q.question.split("\n")[0])
    try:
        ce = hx.rerank_texts(qq, [s for _, s in rows])
    except Exception:
        return []
    vterms = {A.stem(v): 1.0 for v in pq["vtoks"]} | {A.stem(f): 1.0 for f in past + ing}
    out = []
    for (u, s), c in zip(rows, ce):
        own = A.subject_kind(s, "user", qlow) == "self" and not R3._undone(s, vterms) and \
            not re.search(r"\b(?:maybe|might|perhaps|probably|thinking about|planning|going to|will)\b|'ll\b|’ll\b",
                          s, re.I)
        out.append({"uid": u.uid, "sid": u.session_id, "date": u.date_str[:10], "sent": s, "ce": float(c),
                    "own": own, "key": (u.date, u.turn_idx, u.chunk_idx)})
    return out


def latest(cands: list[dict], params: dict) -> dict | None:
    thr = params.get("ce_thr", -7.0)
    ok = [c for c in cands if c["own"] and c["ce"] >= thr]
    return max(ok, key=lambda c: c["key"]) if ok else None


def _inner(q, hx, params: dict) -> EvidencePack:
    cfg = params.get("config", "configs/best_v2.json")
    if params.get("inner_pipe"):
        lens = load_lens("lenses/pipeline.py")
        return lens.build(q, hx, _variant="routed_v2", config=cfg)
    spec = _spec(cfg, params.get("inner_route", "ssu"))
    lens = load_lens(spec["lens"])
    return lens.build(q, hx, _variant=spec["variant"], **{**lens.VARIANTS[spec["variant"]], **spec.get("params", {})})


def build(q, hx, **params) -> EvidencePack:
    from core.pack import build_pack, message_chars
    pack = _inner(q, hx, params)
    pq = parse_last(q.question)
    if pq is None or (params.get("inner_pipe") and pack.route != "ssu"):
        return pack
    d = dict(pack.debug or {})
    cands = candidates(q, hx, pq, params)
    best = latest(cands, params)
    d["last"] = {"parse": pq, "n": len(cands), "best": best and (best["date"], best["sid"], best["sent"][:120])}
    pack.debug = d
    if best is None:
        return pack
    if best["sid"] in (pack.cite_plan or []):
        return pack
    note = (f"Most recent own statement found by code: the {best['date'].replace('-', '/')} line "
            f"\"{A._snip(best['sent'], 160)}\". Earlier mentions are older values.")
    hint = (pack.hint + "\n" + note).strip() if pack.hint else note
    # the inner pack's sessions stay (they show the older values); the latest statement's session is added first
    keep = set(pack.session_ids())
    inner_rank = [u for u in d.get("ranking", []) if 0 <= u < len(hx.units) and hx.units[u].session_id in keep]
    ranking = [best["uid"]] + [u for u in inner_rank if u != best["uid"]]
    budget = message_chars(q, pack.passages, pack.hint) + params.get("extra", 2500)
    passages, included, _ = build_pack(q, hx, ranking, budget=budget, window=1, order="chrono", hint=hint,
                                       max_sessions=len(keep) + 1)
    pids = {p.id for p in passages}
    cite = [best["sid"]] if best["sid"] in pids else pack.cite_plan
    return EvidencePack(question_id=pack.question_id, variant=params.get("_variant", pack.variant), route=pack.route,
                        passages=passages, hint=hint, cite_plan=cite, abstain_p=min(pack.abstain_p or 0.0, 0.2),
                        included_turns=included, computed=pack.computed,
                        debug={**d, "ranking": ranking, "last_fired": True})
