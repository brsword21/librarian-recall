"""SSA lens B ("Semantyka"): what the ASSISTANT said / recommended.

Pipeline (see lenses/ssa/b_design.md):
  question -> (clean query, topic cue, category, purpose, declarative form)
  -> hybrid pool over assistant units (+ a weaker user-unit channel: ~9% of dev gold turns are user turns)
  -> cross-encoder rerank of the top pool
  -> unit score = rerank z-score + round bonus (preceding user turn matches) + topic bonus (session's user turns
     match the "our conversation about X" cue) + recommendation-act features (first-person "I recommend/I'd use X",
     penalties for recommendations to other people, third-party or hedged/negative recommendations, purpose overlap)
  -> session score = NDCG-style sum of its unit hits (Emergence) -> pack top sessions, best unit +/- neighbours
  -> hint = short list of the assistant's own recommendation sentences found in the pack (Chain-of-Note, precomputed)
  -> abstain_p from (no matching first-person recommendation) x (flat score distribution).

Pure-python helpers live at the top and have a self-test: `.venv/bin/python -m lenses.ssa.b_semantic`.
Core imports are lazy so the helpers run before core.index / core.pack exist.
"""
from __future__ import annotations

import math
import re
from collections import defaultdict

_B = {"pool": 100, "rerank_n": 0, "w_rec": 1.0, "w_round": 0.35, "w_topic": 0.25, "user_w": 0.6, "w_clean": 0.5,
      "w_decl": 0.0, "n_sess": 2, "mode": "turns", "top_budget": 4000, "extra": 1, "budget": 12000, "hint": True}
VARIANTS = {
    # no cross-encoder: hybrid(question) + BM25(cleaned, assistant) RRF, semantic features, turn-granular top session
    "turns": dict(_B),
    # same ranking, whole top session (A-style granularity) -> PA ceiling vs chars
    "session": dict(_B, mode="session", n_sess=1),
    # turn-granular, top session only (no runner-up contrast unit)
    "turns1": dict(_B, n_sess=1),
    # cross-encoder on top-40 of the pool (query = raw question, cache-compatible with lenses/generic.py)
    "turns_rr": dict(_B, rerank_n=40, w_clean=0.0),
    # tighter turn budget
    "turns_tight": dict(_B, top_budget=2500),
    # ablation: no speech-act features / hint
    "plain": dict(_B, w_rec=0.0, hint=False),
    # whole top session capped at 8k (fused-order first, so the cut drops the weakest units) + runner-up's best unit
    "session8k": dict(_B, mode="session", n_sess=2, extra=1, budget=8000),
    "session8k1": dict(_B, mode="session", n_sess=1, budget=8000),
    "session6k": dict(_B, mode="session", n_sess=2, extra=1, budget=6000),
    # cross-encoder (top-40, raw-question query = cache-compatible with the generic lens) for the session/unit order
    "session8k1_rr": dict(_B, mode="session", n_sess=1, budget=8000, rerank_n=40, w_clean=0.0),
    "session10k1_rr": dict(_B, mode="session", n_sess=1, budget=10000, rerank_n=40, w_clean=0.0),
    # round 2: + recommendation-sentence pool (assistant units with a first-person recommendation sharing a
    # category/purpose token) added to the cross-encoder pool
    "session10k1_rrp": dict(_B, mode="session", n_sess=1, budget=10000, rerank_n=40, w_clean=0.0, rec_pool=30),
    # round 2: CE-trimmed top session (best turn + its request whole, the rest in CE order) cut at 4k / 5k / 6k
    "trim4k": dict(_B, mode="trim", n_sess=1, budget=4000, rerank_n=40, w_clean=0.0),
    "trim5k": dict(_B, mode="trim", n_sess=1, budget=5000, rerank_n=40, w_clean=0.0),
    "trim6k": dict(_B, mode="trim", n_sess=1, budget=6000, rerank_n=40, w_clean=0.0),
    "trim5k_cq": dict(_B, mode="trim", n_sess=1, budget=5000, rerank_n=40, w_clean=0.0, trim_query="clean"),
    "trim6k_cq": dict(_B, mode="trim", n_sess=1, budget=6000, rerank_n=40, w_clean=0.0, trim_query="clean"),
    # sentence-level CE (max over a unit's sentences, cleaned question) orders the non-best units
    "trim5k_s": dict(_B, mode="trim", n_sess=1, budget=5000, rerank_n=40, w_clean=0.0, trim_query="clean",
                     unit_score="sent"),
    "trim6k_s": dict(_B, mode="trim", n_sess=1, budget=6000, rerank_n=40, w_clean=0.0, trim_query="clean",
                     unit_score="sent"),
    "trim5k_sb": dict(_B, mode="trim", n_sess=1, budget=5000, rerank_n=40, w_clean=0.0, trim_query="clean",
                      unit_score="sent", best="sent"),
    "trim4k_nohead": dict(_B, mode="trim", n_sess=1, budget=4000, rerank_n=40, w_clean=0.0, head=False),
}
DEFAULT = "session10k1_rr"  # best after rounds 1 and 2 (see b_report.md)

# ----------------------------------------------------------------------------------------------------------------
# Question parsing (pure python)
# ----------------------------------------------------------------------------------------------------------------
_CUE_PATTERNS = [
    # "I'm going back to / I wanted to follow up on / I was looking back at ..." (the cue, not its object)
    r"\b(?:i'?m|i am|i was|i wanted to|i want to|i'?d like to|i would like to|just wanted to|i)\s+"
    r"(?:just\s+)?(?:checking(?: back on)?|check(?:ing)? back (?:on|in)|going back (?:to|over)|going through|"
    r"looking back (?:at|on)|looking at|thinking about|thinking back (?:to|on)|following up on|follow up on|"
    r"revisiting|reviewing|trying to (?:recall|remember)|remember(?:ed)?|recall)\b",
    r"\b(?:in |from |during |about )?our (?:previous|last|earlier|past|recent|prior) "
    r"(?:chat|conversation|discussion|session|talk|exchange)s?(?: we had)?\b",
    r"\b(?:can|could|would) you (?:please )?(?:remind me|tell me|help me remember|confirm|refresh my memory)(?: of| on| about)?\b",
    r"\bi was wondering(?: if| whether)?(?: you could)?(?: remind me)?(?: of)?\b",
    r"\b(?:do|did) you (?:remember|recall)\b",
    r"\bremind me(?: of| again)?\b",
    r"\b(?:i )?wanted to (?:confirm|clarify|ask|check)\b",
    r"\b(?:last time|earlier|previously|the other day|a while ago|before)\b",
    r"\b(?:again)\b",
]
_CUE_RX = [re.compile(p, re.I) for p in _CUE_PATTERNS]

_TOPIC_RX = [
    re.compile(r"(?:chat|conversation|discussion|talk|exchange)s?(?: we had)? (?:about|on|regarding|where (?:you|we|I)) "
               r"(?P<t>.+?)(?=[.?!;]|,| and I\b| - | – |\. |$)", re.I),
    re.compile(r"\bwe (?:discussed|talked about|chatted about) (?P<t>.+?)(?= earlier| before| last time|[.?!,;]|$)", re.I),
    re.compile(r"\byou (?:told me|taught me) (?:about )?(?P<t>.+?)(?= earlier| before| last time|[.?!,;]|$)", re.I),
]

# "What was the name of that <cat> you recommended ...", "Which <cat> did you recommend ...", "What <cat> did you ..."
_CAT_RX = re.compile(
    r"(?:(?:what|which)(?: was| is| were| are)? (?:the )?name of (?:that|the|this|those|these)|"
    r"(?:which|what)(?: was| is)?(?: that| the)?)\s+"
    r"(?P<cat>[a-z0-9][\w'’\- ]{1,48}?)\s+(?:did |that |which )?you(?: had)?\s+"
    r"(?P<verb>recommend|suggest|mention|propose|advise|point|told|said|gave|provided|list)\w*\b(?P<rest>.*)$",
    re.I)
_PURPOSE_RX = re.compile(r"\b(?P<prep>for|in|on|with|during|to fix|to repair|to use for)\s+(?P<p>.+?)\s*[?.!]*$", re.I)

_PRON = [(r"\bmy\b", "your"), (r"\bI'm\b", "you're"), (r"\bI\b", "you"), (r"\bme\b", "you"), (r"\bmine\b", "yours"),
         (r"\bours?\b", "your")]


def to_second_person(s: str) -> str:
    for a, b in _PRON:
        s = re.sub(a, b, s)
    return s


def clean_question(q: str) -> str:
    """Drop conversational meta-cues ("I'm going back to our previous chat about", "can you remind me") that match
    every session equally; keep content words. Leaves 'recommend' in place (it matches the target speech act)."""
    s = q
    for rx in _CUE_RX:
        s = rx.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip(" ,.-–")
    return s or q


def split_topic(q: str) -> tuple[str, str]:
    """(topic, detail): topic = object of "our previous conversation about X" style cues (dev questions carry one);
    detail = the remaining ask. Topic is '' when there is no cue."""
    for rx in _TOPIC_RX:
        m = rx.search(q)
        if m:
            t = m.group("t").strip(" ,.")
            t = re.sub(r"^(?:the|a|an)\s+", "", t, flags=re.I)
            if 2 <= len(t) <= 120:
                head = re.sub(r"\b(?:in |from |about )?our (?:previous|last|earlier|past|recent|prior)\s*$", " ",
                              q[:m.start()], flags=re.I)
                detail = clean_question(head + " " + q[m.end():])
                return t, detail
    return "", clean_question(q)


def parse_rec(q: str) -> dict:
    """Category / purpose of a 'what X did you recommend for my Y' question (test style, ~17% of test).
    Returns {"cat": str, "purpose": str, "is_rec": bool}."""
    out = {"cat": "", "purpose": "", "is_rec": False}
    m = _CAT_RX.search(q)
    if m:
        cat = re.sub(r"^(?:name of |the |that |a |an )+", "", m.group("cat").strip(), flags=re.I)
        out["cat"] = cat
        out["is_rec"] = m.group("verb").lower().startswith(("recommend", "suggest", "propose", "advise", "point"))
        rest = m.group("rest") or ""
        rest = re.sub(r"^\s*(?:to )?(?:me|I|us)\b\s*(?:use|try|get|buy|go with|pick|read|watch)?\s*", " ", rest)
        pm = _PURPOSE_RX.search(rest)
        if pm:
            out["purpose"] = (pm.group("prep") + " " + pm.group("p")).strip()
        elif rest.strip(" ?.!"):
            out["purpose"] = rest.strip(" ?.!")
    elif re.search(r"\byou\s+(?:had\s+)?(?:recommend|suggest)\w*", q, re.I):
        out["is_rec"] = True
    return out


def declarative(q: str) -> str:
    """LLM-free question -> statement as the assistant would have said it ("For your Y, I recommend X").
    Used as an extra dense query (statement-to-statement similarity beats question-to-statement for bi-encoders)."""
    r = parse_rec(q)
    if r["cat"]:
        purpose = to_second_person(r["purpose"]).strip()
        cat = r["cat"]
        if purpose:
            p = purpose[0].upper() + purpose[1:]
            return f"{p}, I recommend the {cat}".replace("  ", " ")
        return f"I recommend this {cat}"
    s = clean_question(q)
    s = re.sub(r"^(?:what|which|who|where|when|how many|how much|how)\b\s*(?:was|were|is|are|did|do|does)?\s*", "",
               s, flags=re.I)
    s = re.sub(r"\byou (?:recommend|suggest|mention|say|said|told)\w*\b", "", s, flags=re.I)
    return to_second_person(s).strip(" ?.")


# ----------------------------------------------------------------------------------------------------------------
# Recommendation speech-act tagging on unit text (pure python)
# ----------------------------------------------------------------------------------------------------------------
_SELF_REC = re.compile(
    r"\b(?:I(?:'d|’d| would)?|I can)\s+(?:strongly |highly |definitely |personally |really |also |still )?"
    r"(?:recommend|suggest|use|go with|pick|choose|opt for|point you to|steer you toward)\b"
    r"|\bmy (?:top |personal |first )?(?:pick|recommendation|suggestion|choice) (?:is|would be)\b"
    r"|\b(?:you should|you could|you might want to|you'll want to|try)\s+(?:try|use|check out|go with|visit|look at)\b"
    r"|\byou (?:might|may|would|'d|will|'ll) (?:really )?(?:like|love|enjoy)\b",
    re.I)
_PAST_REC = re.compile(r"\bI (?:recommended|suggested|mentioned)\b", re.I)
_OTHER_PERSON = (r"(?:friend|neighbou?r|sister|brother|son|daughter|mom|mother|dad|father|wife|husband|partner|"
                 r"coworker|co-worker|colleague|client|cousin|aunt|uncle|boss|roommate|niece|nephew|grandson|"
                 r"granddaughter|someone else|another (?:user|client|person))")
_FOR_OTHER = re.compile(
    rf"\b(?:to (?:a|my|your|another|one of your)\s+(?:\w+\s+)?{_OTHER_PERSON}|for (?:a|my|your)\s+(?:\w+\s+)?"
    rf"{_OTHER_PERSON}(?:'s|’s)?)\b", re.I)
_THIRD_PARTY = re.compile(
    r"\b(?:[A-Z][a-z]+(?: [A-Z][a-z]+)?|[Tt]he [a-z]+(?: [a-z]+)?|(?:my|your|his|her) [a-z]+) "
    r"(?:sometimes |often |usually |also )?(?:recommended|recommends|suggested|suggests|swears by|sends \w+ to)\b")
_HEDGE = re.compile(
    r"\bI haven't (?:been|tried|used)\b|\bI'd avoid\b|\bI would avoid\b|\bwouldn't recommend\b|\bnot recommend\b|"
    r"\bbut that(?:'s| is) (?:more|less|not)\b|\bavoid \w+ for now\b", re.I)
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _stem(w: str) -> str:
    for suf in ("ing", "ed", "es", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[:-len(suf)]
    return w


_STOP = set("a an the and or of to in on at for from by with my your our me you i it its this that is was be "
            "what which who did do does name recommend recommended suggest suggested".split())


def content_tokens(s: str) -> set[str]:
    return {_stem(w) for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in _STOP and len(w) > 1}


def rec_sentences(text: str) -> list[str]:
    """Sentences of an assistant turn that carry a recommendation speech act (any kind)."""
    out = []
    for s in _SENT_SPLIT.split(text):
        if _SELF_REC.search(s) or _PAST_REC.search(s) or _THIRD_PARTY.search(s):
            out.append(s.strip())
    return out


GENERIC_CATS = {"value", "name", "one", "thing", "option", "place", "spot", "item", "product", "number", "amount",
                "setting", "size", "brand", "type", "kind", "model", "tool", "resource", "website", "site"}


def cat_is_generic(cat: str) -> bool:
    toks = [t for t in re.findall(r"[a-z]+", cat.lower()) if t not in _STOP]
    return not toks or toks[-1] in GENERIC_CATS


_IDF_CACHE: dict[int, dict[str, float]] = {}


def history_idf(hx) -> dict[str, float]:
    """Token IDF over the history's units (cached per index object); used to weight purpose overlap so generic words
    ("data", "project", "my") count less than the distinctive ones ("cleaning", "speedwork")."""
    key = id(hx)
    if key not in _IDF_CACHE:
        df: dict[str, int] = defaultdict(int)
        for un in hx.units:
            for t in content_tokens(un.text):
                df[t] += 1
        n = max(1, len(hx.units))
        _IDF_CACHE.clear()  # one history at a time is enough
        _IDF_CACHE[key] = {t: math.log(1 + n / (1 + c)) for t, c in df.items()}
        _IDF_CACHE[key]["__default__"] = math.log(1 + n)
    return _IDF_CACHE[key]


def _overlap(pt: set[str], st: set[str], idf: dict[str, float] | None) -> float:
    if not pt:
        return 0.0
    if not idf:
        return len(pt & st) / len(pt)
    d = idf.get("__default__", 5.0)
    tot = sum(idf.get(t, d) for t in pt)
    return sum(idf.get(t, d) for t in pt & st) / tot if tot else 0.0


def rec_features(text: str, purpose: str = "", cat: str = "", idf: dict[str, float] | None = None) -> dict:
    """Score how much a unit looks like 'the assistant itself recommended <cat> for <purpose>'.
    Returns {"rec": float in [-1, 1], "purpose": overlap in [0, 1], "cat": 0/1, "sent": best sentence}."""
    best = {"rec": 0.0, "purpose": 0.0, "cat": 0.0, "sent": ""}
    pt, ct = content_tokens(purpose), content_tokens(cat)
    best_val = -9.0
    for s in _SENT_SPLIT.split(text):
        self_rec = bool(_SELF_REC.search(s))
        past = bool(_PAST_REC.search(s))
        third = bool(_THIRD_PARTY.search(s)) and not self_rec
        if not (self_rec or past or third):
            continue
        r = 1.0 if self_rec else 0.4 if past else 0.0
        if _FOR_OTHER.search(s):
            r -= 1.2
        if past and not self_rec:
            r -= 0.2
        if third:
            r -= 0.6
        if _HEDGE.search(s):
            r -= 0.8
        st = content_tokens(s)
        po = _overlap(pt, st, idf)
        low = s.lower()
        co = 1.0 if ct and any(len(c) >= 3 and c in low for c in ct) else 0.0
        val = r + po + 0.3 * co
        if val > best_val:
            best_val = val
            best = {"rec": max(-1.0, min(1.0, r)), "purpose": po, "cat": co, "sent": s.strip()[:300]}
    return best


# ----------------------------------------------------------------------------------------------------------------
# Lens
# ----------------------------------------------------------------------------------------------------------------
def _z(scores: dict[int, float]) -> dict[int, float]:
    if not scores:
        return {}
    v = list(scores.values())
    mu = sum(v) / len(v)
    sd = (sum((x - mu) ** 2 for x in v) / len(v)) ** 0.5 or 1.0
    return {k: (x - mu) / sd for k, x in scores.items()}


def _role_hits(hx, fn, query, k, role, **kw):
    """Call hx.bm25/hx.dense/hx.hybrid restricted to one role; falls back to post-filtering if `roles` is unsupported."""
    try:
        return fn(query, k=k, roles=[role], **kw)
    except TypeError:
        return [(u, s) for u, s in fn(query, k=k * 3, **kw) if hx.units[u].role == role][:k]


def _user_asks_own(q: str) -> bool:
    """Question asks about something the USER said ("my example", "I mentioned", "I told you") -> user turns are fine."""
    return bool(re.search(r"\b(?:I|we) (?:mentioned|said|told you|asked|shared|described|gave|used|wrote)\b|"
                          r"\bmy (?:example|question|list|description)\b", q, re.I))


def rec_pool_units(hx, rp: dict, limit: int = 30) -> list[int]:
    """Round 2 (idea shared with lens A's recommendation-sentence scan): every assistant unit with a first-person
    recommendation sentence that shares a distinctive category/purpose token with the question. These enter the
    cross-encoder pool even when hybrid retrieval ranks them low ("You should check out Peachtree Brew Supply"
    for "the homebrew supply shop you recommended" sat at hybrid rank 99)."""
    want = {t for t in content_tokens(rp.get("cat", "") + " " + rp.get("purpose", "")) if t not in GENERIC_CATS}
    if not want:
        return []
    idf = history_idf(hx)
    d = idf.get("__default__", 5.0)
    hits = []
    for un in hx.units:
        if un.role != "assistant" or not _SELF_REC.search(un.text):
            continue
        best = 0.0
        for snt in _SENT_SPLIT.split(un.text):
            if _SELF_REC.search(snt):
                ov = sum(idf.get(t, d) for t in want & content_tokens(snt))
                best = max(best, ov)
        if best > 0:
            hits.append((best, un.uid))
    hits.sort(key=lambda x: -x[0])
    return [u for _, u in hits[:limit]]


def rank_units(q, hx, *, pool=100, rerank_n=40, w_rec=1.0, w_round=0.35, w_topic=0.25, user_w=0.6,
               w_clean=0.5, w_decl=0.0, rr_query="raw", rec_pool=0, **_):
    """Candidate pool = hx.hybrid(question) (same call as the generic lens: its dense query + rerank pairs are cached)
    plus a BM25 channel on the cleaned question restricted to assistant units; optional declarative dense channel.
    Scores = cross-encoder z-score (rerank_n > 0) or fused-RRF z-score (rerank_n == 0), plus semantic features."""
    text = q.question
    cq = clean_question(text)
    topic, detail = split_topic(text)
    rp = parse_rec(text)
    units = hx.units
    own = _user_asks_own(text)

    cand: dict[int, float] = defaultdict(float)
    for r, (u, _s) in enumerate(hx.hybrid(text, k=pool, pool=300)):
        cand[u] += 1.0 / (60 + r)
    if w_clean:
        for r, (u, _s) in enumerate(hx.bm25(cq, k=pool // 2, roles=["assistant"])):
            cand[u] += w_clean / (60 + r)
    decl = declarative(text) if w_decl else ""
    if decl and decl.lower() != cq.lower():
        for r, (u, _s) in enumerate(hx.dense(decl, k=pool // 2, roles=["assistant"])):
            cand[u] += w_decl / (60 + r)

    # round bonus: an assistant unit whose preceding user turn matches the question (the user asked for it)
    user_turn_score: dict[str, float] = {}
    if w_round:
        for r, (u, _s) in enumerate(hx.bm25(topic or cq, k=60, roles=["user"])):
            tid = units[u].turn_id
            user_turn_score[tid] = max(user_turn_score.get(tid, 0.0), 1.0 / (1 + r / 5))
    sess_topic: dict[str, float] = defaultdict(float)
    if topic and w_topic:
        for r, (u, _s) in enumerate(hx.bm25(topic, k=40, roles=["user"])):
            sid = units[u].session_id
            sess_topic[sid] = max(sess_topic[sid], 1.0 / (1 + r / 3))

    order = sorted(cand, key=lambda u: -cand[u])
    if rerank_n:
        top = order[:rerank_n]
        if rec_pool and rp["is_rec"]:
            top += [u for u in rec_pool_units(hx, rp, rec_pool) if u not in top]
        rr = dict(hx.rerank(text if rr_query == "raw" else cq, top))
        z = _z(rr)
    else:
        top = order[:max(60, pool)]
        z = _z({u: cand[u] for u in top})

    prev_turn: dict[str, str] = {}
    for s in hx.sessions:
        for a, b in zip(s["turns"], s["turns"][1:]):
            prev_turn[b["turn_id"]] = a["turn_id"]

    scored, feats = {}, {}
    idf = history_idf(hx) if (w_rec and rp["is_rec"]) else None
    for u in top:
        un = units[u]
        sc = z.get(u, 0.0)
        if un.role == "assistant":
            sc += w_round * user_turn_score.get(prev_turn.get(un.turn_id, ""), 0.0)
            if w_rec and rp["is_rec"]:
                f = rec_features(un.text, rp["purpose"], rp["cat"], idf=idf)
                feats[u] = f
                sc += w_rec * (0.6 * f["rec"] + 0.6 * f["purpose"] * (f["rec"] > 0) + 0.2 * f["cat"])
        elif not own:
            sc -= (1.0 - user_w) * 0.75  # SSA: the answer is usually the assistant's; user turns stay as a weaker channel
        sc += w_topic * sess_topic.get(un.session_id, 0.0)
        scored[u] = sc
    ranking = sorted(scored, key=lambda u: -scored[u])
    ranking += [u for u in order if u not in scored]
    return ranking, scored, feats, {"clean": cq, "topic": topic, "decl": decl, "rec": rp, "own": own}


def session_scores(ranking, scored, hx, depth=20, per_session=2):
    """Emergence-style NDCG of unit hits, but each session contributes at most its `per_session` best units: on dev a
    plain sum let long, broadly related sessions (8 moderate hits) outvote the gold session's one strong hit."""
    ss: dict[str, float] = defaultdict(float)
    cnt: dict[str, int] = defaultdict(int)
    for r, u in enumerate(ranking[:depth]):
        sid = hx.units[u].session_id
        if cnt[sid] >= per_session:
            continue
        cnt[sid] += 1
        gain = math.exp(min(5.0, scored.get(u, 0.0)))  # positive gain from the fused score
        ss[sid] += gain / math.log2(r + 2) * (1.0 if cnt[sid] == 1 else 0.5)
    return sorted(ss.items(), key=lambda kv: -kv[1])


def abstain_probability(ranking, scored, feats, rp) -> float:
    """High when (a) a recommendation question finds no first-person recommendation whose purpose matches, and/or
    (b) the fused score distribution is flat (top-1 barely above the pool)."""
    if not ranking:
        return 0.9
    top = [scored.get(u, 0.0) for u in ranking[:10]]
    gap = top[0] - (sum(top[1:]) / max(1, len(top) - 1))
    p_flat = 1.0 / (1.0 + math.exp(3.0 * (gap - 1.0)))   # gap 1.0 -> 0.5, gap 2.0 -> ~0.05
    if rp.get("is_rec"):
        m = rec_match_strength(feats, rp)
        return round(min(0.95, 0.05 + 0.65 * (1.0 - m) + 0.25 * p_flat), 3)
    return round(0.6 * p_flat, 3)


def rec_match_strength(feats, rp) -> float:
    """max over candidate assistant units of: first-person rec x purpose overlap x category agreement, in [0, 1]."""
    generic = cat_is_generic(rp.get("cat", ""))
    best = 0.0
    for f in feats.values():
        if f["rec"] <= 0:
            continue
        pur = f["purpose"] if rp.get("purpose") else 0.7
        cat = 1.0 if (f["cat"] or generic or not rp.get("cat")) else 0.6
        best = max(best, min(1.0, f["rec"]) * min(1.0, pur / 0.6) * cat)
    return best


def make_hint(feats, chosen_uids, hx, rp, none_found: bool = False) -> str:
    """Precomputed Chain-of-Note: the assistant's own recommendation sentences in the pack, labelled by session date."""
    if not rp.get("is_rec"):
        return ""
    lines = []
    for u in chosen_uids:
        f = feats.get(u)
        if f and f["sent"] and f["rec"] > 0 and f["purpose"] + f["cat"] > 0:
            lines.append(f'({hx.units[u].date_str[:10]}) "{f["sent"][:180]}"')
        if len(lines) >= 4:
            break
    what = rp["cat"] or "item"
    purpose = f" {rp['purpose']}" if rp.get("purpose") else ""
    rule = (f"Answer with the {what} the assistant itself recommended{purpose}. Recommendations for other purposes "
            f"or other people, options only listed, and things the user mentioned do not count.")
    if lines:
        return rule + " Recommendation sentences found: " + " | ".join(lines)
    if none_found:
        return rule + (f" Note: no assistant turn found recommending a {what}{purpose}; if the passages do not show "
                       f"one, answer INSUFFICIENT_EVIDENCE.")
    return rule


def _approx_chars(hx, uids) -> int:
    return sum(len(hx.units[u].text) + 12 for u in uids)


def select_units(hx, ranking, scored, keep, mode="turns", top_budget=4000, extra=1) -> list[int]:
    """Granularity of the pack (build_pack is called with window=0 and walks this list in order):
    mode="session": the whole top session (units in fused order first so a budget cut drops the weakest).
    mode="turns":   whole TURNS of the top session in turn-score order (turn score = its best unit), starting with the
                    best turn and the user turn before it (the request the answer responds to), until ~top_budget chars;
                    a session that fits top_budget goes in whole (test-style injected sessions are ~2k chars).
    Then `extra` best units (+/-1) from each runner-up session in `keep` (near-miss contrast / recall insurance)."""
    if not keep:
        return []
    top_sid = keep[0]
    sess = hx.sess_by_id[top_sid]
    sess_units = [u.uid for u in hx.units if u.session_id == top_sid] if False else \
        [j for t in sess["turns"] for j in hx.turn_units(t["turn_id"])]
    ranked_top = [u for u in ranking if hx.units[u].session_id == top_sid]
    chosen: list[int] = []
    runner: list[int] = []
    for sid in keep[1:]:
        for b in [u for u in ranking if hx.units[u].session_id == sid][:extra]:
            runner += [x for x in hx.neighbors(b, 1, 1) if x not in runner]
    if mode == "session" or _approx_chars(hx, sess_units) <= top_budget:
        # best turn of the top session, then the runner-up contrast units (so a budget cut never drops them), then
        # the rest of the top session in fused order, then chronologically
        best_turn = hx.turn_units(hx.units[ranked_top[0]].turn_id) if ranked_top else []
        return list(dict.fromkeys(best_turn + runner + ranked_top + sess_units))
    else:
        turn_order: list[str] = []
        for u in ranked_top:
            t = hx.units[u].turn_id
            if t not in turn_order:
                turn_order.append(t)
        prev = {}
        tids = [t["turn_id"] for t in sess["turns"]]
        for a, b in zip(tids, tids[1:]):
            prev[b] = a
        used = 0
        for i, t in enumerate(turn_order):
            seq = [t]
            if i == 0 and t in prev:
                seq = [t, prev[t]]
            for tt in seq:
                us = [x for x in hx.turn_units(tt) if x not in chosen]
                c = _approx_chars(hx, us)
                if used and used + c > top_budget:
                    continue
                chosen += us
                used += c
            if used >= top_budget:
                break
    for sid in keep[1:]:
        best = [u for u in ranking if hx.units[u].session_id == sid][:extra]
        for b in best:
            chosen += [x for x in hx.neighbors(b, 1, 1) if x not in chosen]
    return list(dict.fromkeys(chosen))


_SENT_NL = re.compile(r"(?<=[.!?])\s+|\n+")


def unit_sentence_scores(hx, query: str, uids: list[int]) -> dict[int, float]:
    """Unit score = max cross-encoder logit over its sentences (hx.rerank_texts, cached). Sharper than scoring the
    whole <=700-char unit: one answer sentence inside a list or a long recipe is not diluted by its neighbours."""
    sents, owner = [], []
    for u in uids:
        for snt in _SENT_NL.split(hx.units[u].text):
            snt = snt.strip()
            if len(snt) >= 12:
                sents.append(snt[:400])
                owner.append(u)
    out = {u: -99.0 for u in uids}
    if sents:
        for u, sc in zip(owner, hx.rerank_texts(query, sents)):
            out[u] = max(out[u], sc)
    return out


def trim_units(q, hx, ranking, keep, head=True, prev_user=True, trim_query="raw", unit_score="unit",
               best="rank") -> list[int]:
    """Round 2: CE-trimmed top session. Priority order for build_pack(window=0), which cuts at `budget`:
    1. the best turn of the top session (turn of the best-ranked unit there), whole;
    2. the user turn right before it (the request it answers), when prev_user;
    3. every other unit of the top session in cross-encoder order (hx.rerank on all session units, raw question;
       cached), each preceded by its turn's first unit when head (chunk 0 carries the turn's topic sentence).
    The pack is rendered chronologically, '[...]' marks dropped material."""
    if not keep:
        return []
    top_sid = keep[0]
    sess = hx.sess_by_id[top_sid]
    sess_units = [j for t in sess["turns"] for j in hx.turn_units(t["turn_id"])]
    ranked_top = [u for u in ranking if hx.units[u].session_id == top_sid]
    tq = q.question if trim_query == "raw" else (split_topic(q.question)[1] or clean_question(q.question))
    sscore = unit_sentence_scores(hx, tq, sess_units) if unit_score == "sent" else {}
    if best == "sent" and sscore:
        best_tid = hx.units[max(sess_units, key=lambda x: sscore[x])].turn_id
    else:
        best_tid = hx.units[ranked_top[0]].turn_id if ranked_top else sess["turns"][0]["turn_id"]
    tids = [t["turn_id"] for t in sess["turns"]]
    out = list(hx.turn_units(best_tid))
    i = tids.index(best_tid)
    if prev_user and i > 0 and hx.units[hx.turn_units(tids[i - 1])[0]].role == "user":
        out += hx.turn_units(tids[i - 1])
    rest = [u for u in sess_units if u not in out]
    if sscore:
        ce = {u: sscore[u] for u in rest}
    else:
        ce = dict(hx.rerank(tq, rest)) if rest else {}
    for u in sorted(rest, key=lambda x: -ce.get(x, -99.0)):
        if head:
            h = hx.turn_units(hx.units[u].turn_id)[0]
            if h not in out:
                out.append(h)
        if u not in out:
            out.append(u)
    return out


def build(q, hx, **params):
    from core.pack import build_pack
    from core.types import EvidencePack

    p = {**VARIANTS[DEFAULT], **params}
    ranking, scored, feats, info = rank_units(q, hx, **p)
    sess = session_scores(ranking, scored, hx)
    keep = [sid for sid, _ in sess[: p["n_sess"]]]
    if p["mode"] == "trim":
        chosen = trim_units(q, hx, ranking, keep, head=p.get("head", True), prev_user=p.get("prev_user", True),
                            trim_query=p.get("trim_query", "raw"), unit_score=p.get("unit_score", "unit"),
                            best=p.get("best", "rank"))
    else:
        chosen = select_units(hx, ranking, scored, keep, mode=p["mode"], top_budget=p["top_budget"],
                              extra=p.get("extra", 1))
    abstain_p = abstain_probability(ranking, scored, feats, info["rec"])
    hint = make_hint(feats, chosen, hx, info["rec"], none_found=abstain_p >= p.get("abstain_hint", 9.0)) \
        if p["hint"] else ""
    passages, included, chars = build_pack(q, hx, chosen, budget=p["budget"], window=0, order="chrono",
                                           hint=hint, max_sessions=p["n_sess"])
    cite = keep[:1]
    return EvidencePack(question_id=q.question_id, variant=p.get("_variant", ""), route="ssa", passages=passages,
                        hint=hint, cite_plan=cite, abstain_p=abstain_p, included_turns=included, computed=None,
                        debug={"ranking": ranking, "query": info, "sessions": sess[:5], "chars": chars})


# ----------------------------------------------------------------------------------------------------------------
# Self-test (pure python): `.venv/bin/python -m lenses.ssa.b_semantic`
# ----------------------------------------------------------------------------------------------------------------
def _selftest():
    qs = [
        "I'm going back to our previous conversation about the children's book on dinosaurs. Can you remind me what "
        "color was the scaly body of the Plesiosaur in the image?",
        "Which Python library did you recommend for my data cleaning project?",
        "What was the name of that coworking space you recommended to me in Lisbon?",
        "Which code editor did you recommend I use for my Python practice scripts?",
        "What value did you recommend for the audio coupling capacitor in my Panasonic RF-2200?",
        "What is the name of the dentist you recommended for my check-up?",
        "Back in Bandung soon. Which restaurant in Cihampelas Walk does the great Nasi Goreng?",
    ]
    for q in qs:
        print("Q:", q)
        print("   clean:", clean_question(q))
        print("   topic/detail:", split_topic(q))
        print("   rec:", parse_rec(q))
        print("   decl:", declarative(q))
    units = [
        "For your Python data cleaning project, I recommend using pandas because it is great for cleaning messy data.",
        "Nice. For your data visualization project, I recommend using matplotlib for simple, clear charts.",
        "Morning quiet is a good filter. I recommended Café Rosa's Neukölln branch to a friend last month.",
        "The beekeeping association recommended De Zwarte Bij for ordering queen bees.",
        "Oh, and for the banking side, I'd avoid TD for now because their newcomer bank offer changed last month.",
        "Oh, For the audio coupling capacitor in your Panasonic RF-2200, I'd use a 0.022 µF film capacitor.",
    ]
    rp = parse_rec("Which Python library did you recommend for my data cleaning project?")

    class _U:  # tiny fake index: generic words frequent, distinctive words rare
        def __init__(self, t): self.text = t

    fake = [_U(t) for t in units] + [_U("data project plan for your project data") for _ in range(30)]

    class _H:
        pass
    hx = _H()
    hx.units = fake
    idf = history_idf(hx)
    for t in units:
        f = rec_features(t, rp["purpose"], rp["cat"], idf=idf)
        print(f"   rec={f['rec']:+.1f} purpose={f['purpose']:.2f} cat={f['cat']:.0f} | {t[:70]}")
    f0, f1 = (rec_features(units[i], rp["purpose"], rp["cat"], idf=idf) for i in (0, 1))
    assert f0["purpose"] > 0.9 and f1["purpose"] < 0.5, (f0, f1)
    assert parse_rec("What is the name of the dentist you recommended for my check-up?")["cat"] == "dentist"
    assert rec_features(units[2])["rec"] < 0 and rec_features(units[3])["rec"] < 0.5 and rec_features(units[4])["rec"] <= 0.5
    print("selftest ok")


if __name__ == "__main__":
    _selftest()
