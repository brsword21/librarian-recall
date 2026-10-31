"""SSA lens A, "recsent": structural lens for single-session-assistant questions.

Idea: the answer is one assistant sentence ("For your remote work days, I recommend Cafe Blau in Mitte."). Find it with
rules on top of core retrieval: boilerplate-free query, recommendation cues, purpose-phrase match with my->your flip,
penalties for recommendations made to other people / hedged or rejected options, a role prior from the question wording
("you recommended" -> assistant, "I mentioned" -> user). Then pack that turn whole (+ the user turn before it), cite
ONE session, and tell Reactor what is NOT the answer. Ordinal questions ("the 7th job on the list") get the list item
computed in code. See lenses/ssa/a_design.md.

Round 1 best variant: `recqs_cf7` (BM25 + role-BM25 + structural scan, no model server; quoted key line; shape boost;
cue floor 0.7). dev S 93.3 / dev200 S 93.3, PA 1.0, CF1 1.0, ~5.5k chars.
Round 2 best variant: `r2_conf2` = recqs_cf7 + one session only (pack == cite set, Sall = S) + confidence-tiered budget
(best-turn margin >= 1.3 -> 3000, >= 1.2 -> 4500, else 6000) + kind gate on abstain_p. dev 93.6 / dev200 93.7, PA 1.0,
CF1 = CF1all = 1.0, 4.5k / 4.2k chars. Also new: `role_features` / `role_guess` (is the answer in an assistant turn?),
used by lenses/ssa/a_handoff.py on the ssu route. See lenses/ssa/a_report.md.

Self-test (pure python, no core.index / models needed):  .venv/bin/python -m lenses.ssa.a_recsent
"""
from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter

from core.types import EvidencePack

_BASE = {"pool_k": 60, "rerank_n": 30, "w_struct": 1.0, "w_struct_other": 0.3, "budget": 7000, "max_sessions": 2,
         "tie": 0.85, "whole_turn": True, "fill_session": True, "hint_mode": "generic", "cite": 1}
VARIANTS = {
    # round 0 (hybrid + rerank; not swept in round 1, model server saturated): structural rerank + best turn whole +
    # rest of the best session + ONE citation + hint saying what is NOT the answer
    "rec": dict(_BASE),
    # same, plus the hint quotes the winning recommendation sentence when the structure is confident (margin >= 1.3x)
    "rec_quote": dict(_BASE, hint_mode="quote"),
    # cheap: one session only, 4k budget (chars penalty), no cross-encoder; tests whether precision alone suffices
    "rec_tight": dict(_BASE, pool_k=40, rerank_n=0, budget=4000, max_sessions=1, tie=1.01),
    # control: same packing, no structural features, no hint -> isolates what the rules add
    "topic": dict(_BASE, w_struct=0.0, w_struct_other=0.0, hint_mode="none"),
    # round 1: lexical channels only (no model server): BM25 + role-BM25 + structure
    "rec_bm25": dict(_BASE, channels="bm25", rerank_n=0),
    "recq_bm25": dict(_BASE, channels="bm25", rerank_n=0, hint_mode="quote"),
    # + shape boost: a recommendation-shaped sentence (cue + name + kind/purpose) outranks a lexically closer unit
    "recqs_bm25": dict(_BASE, channels="bm25", rerank_n=0, hint_mode="quote", shape_boost=1.0),
    # ROUND-1 BEST: + cue floor 0.7 (cue-less list items are not crushed on dev "you recommended" questions)
    "recqs_cf7": dict(_BASE, channels="bm25", rerank_n=0, hint_mode="quote", shape_boost=1.0, cue_floor=0.7),
    # budget cuts: lose PA (best turn != gold turn in ~1/3 of questions; the session fill saves them)
    "recqs_b5": dict(_BASE, channels="bm25", rerank_n=0, hint_mode="quote", shape_boost=1.0, budget=5000),
    "recqs_b4": dict(_BASE, channels="bm25", rerank_n=0, hint_mode="quote", shape_boost=1.0, budget=4000),
    # ---- round 2 (all on top of recqs_cf7; kind_gate only changes abstain_p) ----
    # no runner-up session (near-miss insurance off): the pack is exactly the cite set (Sall = S)
    "r2_solo": dict(_BASE, channels="bm25", rerank_n=0, hint_mode="quote", shape_boost=1.0, cue_floor=0.7,
                    tie=1.01, kind_gate=True),
    # budget 6000 (dev turn_rank: lens order keeps the gold turn at 6k, 5.0k chars on average)
    "r2_b6": dict(_BASE, channels="bm25", rerank_n=0, hint_mode="quote", shape_boost=1.0, cue_floor=0.7,
                  budget=6000, kind_gate=True),
    "r2_solo_b6": dict(_BASE, channels="bm25", rerank_n=0, hint_mode="quote", shape_boost=1.0, cue_floor=0.7,
                       tie=1.01, budget=6000, kind_gate=True),
    # + budget 5000 only when the best TURN is clearly ahead inside its session (turn_margin >= 1.3: best turn ==
    #   gold turn 16/16 dev, 4/4 dev200), else 6000
    "r2_conf": dict(_BASE, channels="bm25", rerank_n=0, hint_mode="quote", shape_boost=1.0, cue_floor=0.7,
                    tie=1.01, budget=6000, kind_gate=True, conf_budget=(1.3, 3500)),
    # two tiers: margin >= 1.3 -> 3000, >= 1.2 -> 4500, else 6000
    "r2_conf2": dict(_BASE, channels="bm25", rerank_n=0, hint_mode="quote", shape_boost=1.0, cue_floor=0.7,
                     tie=1.01, budget=6000, kind_gate=True, conf_budget=((1.3, 3000), (1.2, 4500))),
    # three tiers: >= 1.3 -> 2500, >= 1.2 -> 4000, >= 1.1 -> 5000, else 6000
    "r2_conf3": dict(_BASE, channels="bm25", rerank_n=0, hint_mode="quote", shape_boost=1.0, cue_floor=0.7,
                     tie=1.01, budget=6000, kind_gate=True, conf_budget=((1.3, 2500), (1.2, 4000), (1.1, 5000))),
    "r2_s55": dict(_BASE, channels="bm25", rerank_n=0, hint_mode="quote", shape_boost=1.0, cue_floor=0.7,
                   tie=1.01, budget=5500, kind_gate=True),
}

# ---------------------------------------------------------------------------------------------------------------------
# text normalisation
# ---------------------------------------------------------------------------------------------------------------------
STOP = set("""a an the and or but if of to in on at for from by with about into over after before than then so as is are
was were be been being do does did done have has had having i me my mine we our us you your yours he she it its they them
their this that these those what which who whom whose when where why how can could would should will shall may might must
again also just very really some any all each every much many more most other another such no not only own same too there
here up down out off once ever s t d ll re ve m o y name called remind remember recall forgot forget tell told know
please thanks thank wondering wonder wanted want trying try going go back looking look checking check thinking think
follow confirm clarify earlier previous last time chat conversation discussion discussed talked talk mentioned mention
said say recommended recommend suggested suggest gave give provided provide one thing something again""".split())

_TOK = re.compile(r"[a-z0-9][a-z0-9\-\.']*[a-z0-9]|[a-z0-9]")


def fold(s: str) -> str:
    """lowercase + strip accents (cafe/café, Bjørn)."""
    s = unicodedata.normalize("NFKD", s.replace("ø", "o").replace("Ø", "O"))
    return "".join(c for c in s if not unicodedata.combining(c)).lower().replace("’", "'")


def stem(w: str) -> str:
    w = w.strip(".'-")
    if w.endswith("'s"):
        w = w[:-2]
    if len(w) > 4 and w.endswith(("ches", "shes", "sses", "xes", "zes")):
        return w[:-2]
    for suf, keep in (("ies", "y"), ("ing", ""), ("ed", ""), ("s", "")):
        if len(w) > len(suf) + (2 if suf == "s" else 3) and w.endswith(suf) and not w.endswith(("ss", "us", "is")):
            return w[: -len(suf)] + keep
    return w


def toks(s: str) -> list[str]:
    return _TOK.findall(fold(s))


def content_stems(s: str) -> list[str]:
    out = []
    for t in toks(s):
        if t in STOP or len(t) < 2:
            continue
        out.append(stem(t))
        if "-" in t:  # time-tracking -> time, track too
            out += [stem(p) for p in t.split("-") if p and p not in STOP and len(p) > 1]
    return out


_SENT = re.compile(r"(?<=[.!?])\s+|\n+")
_ABBR = re.compile(r"(?:\b(?:Dr|Mr|Mrs|Ms|Prof|St|Mt|Jr|Sr|vs|No|approx|e\.g|i\.e|etc|Inc|Ltd|Co)\.|\b[A-Z]\.)$")


def sentences(text: str) -> list[str]:
    out: list[str] = []
    for s in _SENT.split(text):
        s = s.strip() if s else ""
        if not s:
            continue
        if out and _ABBR.search(out[-1]):      # "I recommend Dr." + "Vogel; the practice is ..." -> one sentence
            out[-1] = f"{out[-1]} {s}"
        else:
            out.append(s)
    return [s for s in out if len(s) > 2]


# ---------------------------------------------------------------------------------------------------------------------
# question parsing (rule-based query rewriting)
# ---------------------------------------------------------------------------------------------------------------------
# boilerplate that carries no topic: "I'm checking our previous chat about", "Can you remind me", "I forgot the name."
_BOILER = [
    r"\b(?:i'?m|i am|i was|i've been|i have been|i wanted to|i want to|i'd like to)\s+(?:just\s+)?(?:checking|going|looking|"
    r"thinking|following|trying to recall|trying to remember|wondering|planning to|planning)\b(?:\s+(?:back|up|through|"
    r"over|on|about|at))*(?:\s+(?:to|on|at|about))?",
    r"\b(?:in|from|on)?\s*(?:our|my|the)\s+(?:previous|last|earlier|past|prior)\s+(?:chat|conversation|discussion|"
    r"exchange|session|talk)s?(?:\s+(?:about|on|regarding|where|when))?",
    r"\b(?:can|could) you (?:please )?(?:remind me|tell me|help me remember|recall)(?: of| about| what| which)?",
    r"\bi (?:was|am) wondering(?: if you could remind me(?: of)?| if you remember)?",
    r"\bif you could remind me(?: of)?",
    r"\bdo you (?:remember|recall)",
    r"\bi (?:forgot|forget|can't remember|don't remember)(?: the name| the title| what it was called)?\.?",
    r"\b(?:we )?(?:talked|chatted|spoke) about (?:last time|earlier|before)",
    r"\b(?:last time|earlier|before|again)\b",
    r"\bi wanted to (?:confirm|clarify|check)(?: something)?",
    r"\bremind me\b",
]
_BOILER_RX = [re.compile(p, re.I) for p in _BOILER]
_TOPIC_RX = re.compile(r"(?:previous|last|earlier)\s+(?:chat|conversation|discussion)\s+(?:about|on|regarding)\s+"
                       r"([^.?!,]{3,120})", re.I)
_KIND_RX = [
    re.compile(r"(?:name|title) of (?:that|the|this|a)\s+(.+?)\s+(?:you|that you|which you|we)\b", re.I),
    re.compile(r"\b(?:which|what)\s+(.+?)\s+(?:did|do|would|should|have|has)\s+you\b", re.I),
    re.compile(r"\b(?:that|the)\s+([\w\- ]{2,40}?)\s+you\s+(?:recommended|suggested|mentioned)\b", re.I),
]
_PURPOSE_RX = re.compile(r"\b(?:recommend(?:ed)?|suggest(?:ed)?|use|try)\b(?:\s+(?:to me|i use|i try|me))?\s*"
                         r"((?:for|about|against|in|to)\s+[^?!]{2,80}?)(?=[?!]|\.\s|\.$|$)", re.I)
_ITEM_NOUNS = {"item", "bottle", "option", "step", "tip", "job", "parameter", "one", "idea", "point", "suggestion",
               "recommendation", "entry", "venue", "place", "book", "app", "tool", "website", "resource", "name"}
_ORD_WORDS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8,
              "ninth": 9, "tenth": 10, "eleventh": 11, "twelfth": 12}
_ORD_RX = re.compile(r"\b(?:(\d{1,3})(?:st|nd|rd|th)|number\s+(\d{1,3})|#(\d{1,3})|(" + "|".join(_ORD_WORDS) +
                     r")|(last|final))\b\s*([\w\-]+)?", re.I)

# rule-based synonyms for recommendation "kinds" (question head noun -> words the answer sentence may use instead)
SYN = {
    "coworking": ["cowork", "day pass", "hot desk", "workspace", "desk", "quiet place"],
    "cafe": ["cafe", "coffee", "coffee shop", "espresso"], "coffee": ["cafe", "coffee"],
    "shop": ["store", "supply", "shop"], "store": ["shop", "store", "supply"],
    "app": ["app", "tool", "tracker", "software"], "software": ["app", "tool", "software", "program"],
    "tool": ["app", "tool"], "library": ["library", "package", "module"], "editor": ["editor", "ide", "code"],
    "plugin": ["plugin", "plug-in", "vst"], "microphone": ["microphone", "mic"], "headphone": ["headphone", "cans"],
    "opening": ["opening", "defense", "defence", "gambit", "variation"], "book": ["book", "title", "read"],
    "dentist": ["dentist", "dental", "dr"], "doctor": ["doctor", "dr"], "adhesive": ["adhesive", "glue"],
    "restaurant": ["restaurant", "dinner", "eatery", "bistro"], "eatery": ["restaurant", "eatery", "cafe"],
    "hostel": ["hostel", "hotel", "stay"], "trail": ["trail", "route", "hike", "path"],
    "supplement": ["supplement"], "illustrator": ["illustrator", "artist"], "bank": ["bank", "account"],
    "club": ["club"], "coach": ["coach", "trainer"], "park": ["park"], "brand": ["brand"], "flour": ["flour"],
    "website": ["website", "site", "www", ".com", ".org"], "video": ["video", "youtube"],
}

# question says who said it
_ROLE_ASSIST = re.compile(r"\byou\s+(?:recommended|suggested|said|mentioned|told|provided|gave|wrote|created|made|"
                          r"listed|outlined|described|explained|named|proposed|shared|recommend|suggest)\b|\byour\s+"
                          r"(?:list|recommendation|suggestion|answer)", re.I)
_ROLE_USER = re.compile(r"\bi\s+(?:mentioned|told you|said|used|shared|gave you|described|wrote)\b|\bmy example\b",
                        re.I)
_REC_Q = re.compile(r"\b(?:recommend|suggest)", re.I)
_NAME_Q = re.compile(r"\b(?:name|title|called|named|brand|handle)\b|^\s*which\b|\bwhich\s+\w+(?:\s+\w+)?\s+(?:did|do|"
                     r"was|is|has|had)\b", re.I)
_NUM_Q = re.compile(r"\b(?:how many|how much|what value|which value|what ratio|how long|what year|phone number|"
                    r"what percentage|what (?:was|is) the (?:average|number))", re.I)


def parse_question(text: str) -> dict:
    t = text.strip()
    topic = ""
    m = _TOPIC_RX.search(t)
    if m:
        topic = m.group(1).strip()
    clean = t
    for rx in _BOILER_RX:
        clean = rx.sub(" ", clean)
    clean = re.sub(r"\s+", " ", re.sub(r"\s+([?.!,])", r"\1", clean)).strip(" ,.?") + "?"
    kind = ""
    for rx in _KIND_RX:
        km = rx.search(t)
        if km:
            kind = km.group(1).strip()
            kind = re.sub(r"^(?:name of|the|that|a)\s+", "", kind, flags=re.I)
            break
    purpose = ""
    pm = _PURPOSE_RX.search(t)
    if pm:
        purpose = pm.group(1).strip()
    ordinal = None
    has_list = bool(re.search(r"\blists?\b|\bnumber\s+\d", t, re.I))
    for om in _ORD_RX.finditer(t):
        word = om.group(0).split()[0]
        noun = fold(om.group(6) or "")
        item_noun = stem(noun) in _ITEM_NOUNS
        if word[:1].isupper() and om.start() > 0:          # "the Fifth Album" is a title, not an ordinal
            continue
        if om.group(1) or om.group(2) or om.group(3):
            if has_list or item_noun:
                ordinal = int(om.group(1) or om.group(2) or om.group(3))
        elif om.group(4) and (has_list or item_noun):
            ordinal = _ORD_WORDS[om.group(4).lower()]
        elif om.group(5) and has_list:
            ordinal = -1
        if ordinal is not None:
            break
    role = "assistant" if _ROLE_ASSIST.search(t) else ("user" if _ROLE_USER.search(t) else None)
    kind_stems = content_stems(kind)
    syn = []
    syn_map: dict[str, list[str]] = {}
    for w in toks(kind):
        xs = SYN.get(w, []) + SYN.get(stem(w), [])
        if xs:
            syn_map[stem(w)] = list(dict.fromkeys(fold(x) for x in xs))
            syn += xs
    purpose_stems = content_stems(purpose)
    return {
        "text": t, "clean": clean, "topic": topic, "kind": kind, "purpose": purpose, "ordinal": ordinal,
        "role": role, "rec_style": bool(_REC_Q.search(t)), "wants_number": bool(_NUM_Q.search(t)),
        "wants_name": bool(_NAME_Q.search(t)) and not _NUM_Q.search(t),
        "terms": list(dict.fromkeys(content_stems(clean))), "kind_stems": list(dict.fromkeys(kind_stems)),
        "head": kind_stems[-1] if kind_stems else "",
        "syn": list(dict.fromkeys(fold(s) for s in syn)), "syn_map": syn_map,
        "purpose_stems": list(dict.fromkeys(purpose_stems)),
        # my -> your flip: "for my remote work days" is said back as "for your remote work days"
        "purpose_said": re.sub(r"\bmy\b", "your", re.sub(r"\bme\b", "you", fold(purpose))),
    }


def retrieval_query(qp: dict) -> str:
    """Boilerplate-free query; the "previous conversation about X" topic is repeated (it names the session)."""
    q = qp["clean"]
    if qp["topic"] and fold(qp["topic"]) not in fold(q):
        q = f"{qp['topic']}. {q}"
    elif qp["topic"]:
        q = f"{qp['topic']}. {q}"
    return q


# ---------------------------------------------------------------------------------------------------------------------
# sentence-level structure: recommendation cues, beneficiary, contrast
# ---------------------------------------------------------------------------------------------------------------------
CUE_STRONG = re.compile(r"\b(?:i(?:'d| would)?\s+(?:highly\s+|strongly\s+|definitely\s+)?(?:recommend|suggest)|"
                        r"i(?:'d| would)\s+(?:use|go with|pick|choose|start with|try)|my\s+(?:top\s+)?(?:pick|"
                        r"recommendation|suggestion)\s+(?:is|would be)|i'?d say)\b", re.I)
CUE_MED = re.compile(r"\b(?:you should (?:check out|try|go with|use|look at|visit|get)|you (?:might|may|could|would) "
                     r"(?:like|enjoy|love|try|consider|want to try)|check out|go with|has a (?:good|great|nice)|"
                     r"(?:is|are) (?:a )?(?:great|good|solid|excellent|perfect|ideal) (?:choice|option|pick|place|"
                     r"spot|fit|bet|for)|worth (?:a visit|a look|checking|trying)|best bet|look into|try (?=[A-Z\d'\"])|"
                     r"consider (?=[A-Z\d'\"]))", re.I)
# recommendation for somebody else ("I recommended Cafe Rosa to a friend", "for your neighbor Bjorn's horse")
THIRD = re.compile(r"\b(?:to|for)\s+(?:a|my|your|his|her|their)\s+(?:friend|neighbou?r|colleague|coworker|co-worker|"
                   r"sister|brother|cousin|client|mom|dad|mother|father|son|daughter|wife|husband|partner|boss|student|"
                   r"aunt|uncle|niece|nephew)\b|\bfor\s+(?:your\s+\w+\s+)?[A-Z][a-z]+'s\b|\bsometimes sends\b|"
                   r"\b(?:another|other) (?:client|runner|customer|student)s?\b", re.I)
PAST_REC = re.compile(r"\bi\s+(?:recommended|suggested)\b", re.I)
CONTRAST = re.compile(r"\bbut\s+(?:that|it|this|they|i)\s+(?:is|was|would|are|were|haven'?t|have not|hasn'?t|didn'?t|"
                      r"don'?t|wouldn'?t)\b|\b(?:more about|not (?:ideal|a good|great|recommended|the best)|"
                      r"would let in|too (?:much|expensive|noisy|loud|heavy|pricey)|instead of|rather than|avoid)\b",
                      re.I)
ENTITY = re.compile(r"(?<![.!?]\s)(?<!^)\b(?:[A-Z][\w'&\-]+(?:\s+(?:[A-Z][\w'&\-]+|de|du|di|of|the|la|le|von|van|"
                    r"\d+))*)|\b\d+(?:[.,]\d+)?\s*(?:µ|u|m|k)?(?:f|v|ohm|w|mm|cm|kg|g|ml|%|°)\b|'[^']{3,60}'",
                    re.UNICODE)


_CAP = re.compile(r"\b(?!I\b|I'|I’)[A-Z][\w'&\-]*[A-Za-z0-9]")
_VAL = re.compile(r"\b\d+(?:[.,:]\d+)?\s*(?:µ|u|m|k)?(?:f|v|ohm|ω|w|mm|cm|kg|g|ml|%|°)?\b|['\"“‘][^'\"”’]{3,60}['\"”’]",
                  re.I)


def has_entity(sent: str) -> bool:
    """A proper name, a quoted title or a number somewhere after the first word (markdown stripped)."""
    body = re.sub(r"^[\W_]+", "", sent)
    body = re.sub(r"[*_`#]+", "", body)
    first, _, rest = body.partition(" ")
    rest = re.sub(r"^[^A-Za-z0-9]*", "", rest) if first.endswith(":") else rest
    return bool(_CAP.search(rest) or _VAL.search(rest))


def sent_struct(sent: str, qp: dict, idf: dict) -> tuple[float, dict]:
    """Structural score of one sentence for this question. Overlap is idf-weighted over the question's terms."""
    st = set(content_stems(sent))
    fs = fold(sent)
    terms = qp["terms"]
    # a synonym of a head-noun word ("day pass" for "coworking") counts as a match of that word only
    syn_ok = {k for k, xs in qp.get("syn_map", {}).items()
              if any(re.search(r"(?<![a-z])" + re.escape(x) + r"(?![a-z])", fs) for x in xs)}
    kind_hit = any(k in st for k in qp["kind_stems"]) or bool(syn_ok)
    head = qp.get("head", "")
    head_hit = bool(head) and (head in st or bool(syn_ok))   # synonyms exist only for head-like kind words
    wsum = sum(idf.get(t, 1.0) for t in terms) or 1.0
    # compound parts count 3/4 ("homebrew" ~ "Brew Supply", "space" ~ "workspace")
    part = {t for t in terms if t not in st and t not in syn_ok and len(t) >= 5
            and any(len(x) >= 4 and x != t and (t.startswith(x) or t.endswith(x) or x.endswith(t)) for x in st)}
    ov = (sum(idf.get(t, 1.0) for t in terms if t in st or t in syn_ok) + 0.75 * sum(idf.get(t, 1.0) for t in part)) / wsum
    purp = qp["purpose_stems"]
    p_ov = (sum(1 for p in purp if p in st or p in part) / len(purp)) if purp else 0.0
    kind_hit = kind_hit or any(k in part for k in qp["kind_stems"])
    head_hit = head_hit or head in part
    said = qp["purpose_said"]
    rest = said.split(" ", 1)[-1] if said else ""
    p_exact = len(rest.split()) >= 2 and rest in fs
    cue = 1.0 if CUE_STRONG.search(sent) else (0.6 if CUE_MED.search(sent) else 0.0)
    third = bool(THIRD.search(sent))
    past = bool(PAST_REC.search(sent)) and not re.search(r"\b(?:you|your)\b", fs.split("recommended")[-1][:40])
    contra = bool(CONTRAST.search(sent))
    ent = has_entity(sent)
    f = {"ov": round(ov, 3), "kind": kind_hit, "head": head_hit, "p_ov": round(p_ov, 2), "p_exact": p_exact, "cue": cue, "third": third,
         "past": past, "contra": contra, "ent": ent}
    if ov == 0 and not kind_hit:
        return 0.0, f
    s = ov + 0.25 * kind_hit + 0.5 * p_ov + 0.4 * p_exact
    if qp["rec_style"]:
        s *= qp.get("cue_floor", 0.4) + cue   # a "you recommended" question wants a sentence that recommends
        s += 0.15 * ent
        if qp["wants_name"] and not ent:   # "what was the name of ..." needs a proper name in the sentence
            s *= 0.5
        s -= 0.9 * third + 0.4 * past + 0.6 * contra
    else:
        s *= 1.0 + 0.2 * cue
        s -= 0.3 * contra * (cue > 0)
    return s, f


# ---------------------------------------------------------------------------------------------------------------------
# ordinal questions: compute the n-th list item in code
# ---------------------------------------------------------------------------------------------------------------------
_ITEM = re.compile(r"^\s*(?:(\d{1,3})[.)]|[-*•])\s+(.+)$")


def list_items(text: str) -> list[str]:
    """Items of the longest numbered (or bulleted) list in a turn."""
    blocks, cur, last_num = [], [], None
    for line in text.splitlines():
        m = _ITEM.match(line)
        if m:
            num = int(m.group(1)) if m.group(1) else None
            if num is not None and last_num is not None and num != last_num + 1 and num == 1:
                blocks.append(cur)
                cur = []
            cur.append(m.group(2).strip())
            last_num = num
        elif line.strip() and cur and not line.startswith((" ", "\t")):
            # a non-list line after a list closes the block unless it is a sub-line
            if len(cur) >= 2:
                blocks.append(cur)
                cur = []
                last_num = None
    if cur:
        blocks.append(cur)
    return max(blocks, key=len) if blocks else []


def nth_item(text: str, n: int) -> str | None:
    items = list_items(text)
    if not items:
        return None
    if n == -1:
        return items[-1]
    return items[n - 1] if 0 < n <= len(items) else None


# ---------------------------------------------------------------------------------------------------------------------
# hints and abstention
# ---------------------------------------------------------------------------------------------------------------------
def build_hint(qp: dict, best: dict | None, mode: str) -> str:
    if mode == "none":
        return ""
    parts = []
    if qp["role"] == "user":
        parts.append("The answer is something I (the user) said myself, not something the assistant said.")
    elif qp["rec_style"]:
        # the question's own words (my/me kept: the hint is appended to the user's question)
        what = f" {qp['purpose']}" if qp["purpose"] else ""
        kind = qp["kind"] if qp["kind"] and len(qp["kind"]) <= 40 else "item"
        parts.append(f"Answer with the {kind} the assistant itself recommended to me{what}. Not the answer: things "
                     "I mentioned, tried or heard from other people; things the assistant recommended to someone else "
                     "or for a different purpose; options it only listed, compared or advised against.")
    if best and best.get("item"):
        parts.append(f"Item {best['ord_label']} of the assistant's list ({best['date']}) reads: \"{best['item'][:200]}\".")
    elif qp["ordinal"]:
        parts.append("Count the items of the assistant's list carefully, in order.")
    if mode == "quote" and best and best.get("confident") and best.get("sentence"):
        sent = re.sub(r"[*_`#]+", "", best["sentence"]).strip()
        parts.append(f"Likely the line you need (assistant, session of {best['date']}): \"{sent[:260]}\"")
    return " ".join(parts)


def rec_shape(f: dict, qp: dict) -> float:
    """How much a sentence looks like THE answer to a "you recommended" question (0, 0.25, 0.5, 1)."""
    if not f.get("cue") or f.get("third") or f.get("past") or f.get("contra"):
        return 0.0
    fits = f.get("head") or (qp["purpose"] and f.get("p_ov", 0) >= 0.5)
    if f.get("ent") and fits and f.get("ov", 0) >= 0.5:
        return 1.0
    if f.get("ent") and fits and f.get("ov", 0) >= 0.3:
        return 0.5
    return 0.25


def abstain_estimate(qp: dict, best: dict | None) -> float:
    """SSA questions are almost always answerable; flag only when nothing in the history looks like the answer.
    Rec-style: is there ANY assistant sentence shaped like the asked recommendation (first-person cue + a name +
    the asked kind/purpose)? Other questions: weak term coverage of the best unit."""
    if not best:
        return 0.8
    f = best.get("feat", {})
    if qp["rec_style"]:
        p = {1.0: 0.05, 0.5: 0.35, 0.25: 0.6}.get(best.get("shape", 0.0), 0.8)
        if not f.get("cue"):
            p += 0.1
        # round 2 gate: a weak shape is evidence of absence only when the asked category (head noun or a synonym)
        # is also missing from the top session; if the session talks about that kind of thing, cap at 0.35
        if best.get("kind_in_top") is True and p > 0.35:
            p = 0.35
        return round(min(p, 0.95), 3)
    p = 0.05 + (0.25 if f.get("ov", 0) < 0.25 else 0.0)
    return round(min(p, 0.95), 3)


# ---------------------------------------------------------------------------------------------------------------------
# idf over the history's units (cached per history)
# ---------------------------------------------------------------------------------------------------------------------
_IDF: dict[tuple, dict] = {}


def history_idf(hx) -> dict:
    key = (getattr(hx, "split", ""), getattr(hx, "history_id", id(hx)))
    if key not in _IDF:
        df = Counter()
        for u in hx.units:
            df.update(set(content_stems(u.text)))
        n = max(1, len(hx.units))
        _IDF.clear() if len(_IDF) > 8 else None
        _IDF[key] = {t: math.log(1 + n / (1 + c)) for t, c in df.items()}
    return _IDF[key]


def unit_struct(hx, uid: int, qp: dict, idf: dict) -> tuple[float, dict]:
    u = hx.units[uid]
    best, bf, bs = 0.0, {}, ""
    for s in sentences(u.text):
        sc, f = sent_struct(s, qp, idf)
        if sc > best:
            best, bf, bs = sc, f, s
    if qp["role"] and u.role != qp["role"]:
        best *= 0.5 if qp["role"] == "assistant" else 0.7
    bf["sentence"] = bs
    return best, bf


# ---------------------------------------------------------------------------------------------------------------------
# role guess (round 2): is the answer in an ASSISTANT turn? For routing cue-less questions ("Back in Bandung soon.
# Which restaurant in Cihampelas Walk does the great Nasi Goreng?") that the text router sends to ssu.
# ---------------------------------------------------------------------------------------------------------------------
_FIRST = re.compile(r"\b(?:i|my|me|mine|i'm|i've|i'd|myself)\b", re.I)


def role_features(q, hx, use_ce: bool = False) -> dict:
    """Evidence-based role features. Lexical by default (BM25 per role); use_ce adds a cross-encoder over the top
    hybrid units. All features grow with 'the answer is in an assistant turn'."""
    qp = parse_question(q.question)
    query = retrieval_query(qp)
    first = len(_FIRST.findall(fold(q.question).replace("remind me", "")))
    ba = hx.bm25(query, k=5, roles="assistant")
    bu = hx.bm25(query, k=5, roles="user")
    sa = ba[0][1] if ba else 0.0
    su = bu[0][1] if bu else 0.0
    top = hx.bm25(query, k=5)
    frac = sum(hx.units[u].role == "assistant" for u, _ in top) / max(1, len(top))
    ours = len(re.findall(r"\b(?:our|ours)\b", fold(q.question).replace("our previous", "").replace("our last", "")
                          .replace("our earlier", "")))
    advice = bool(re.search(r"\b(?:should i|should my|do i need to)\b", fold(q.question)))   # asks for the assistant's advice
    f = {"q_nofirst": -float(first), "q_ours": float(ours), "q_advice": float(advice), "bm_diff": math.log1p(sa) - math.log1p(su), "top5_a": frac,
         "top1_a": float(bool(top) and hx.units[top[0][0]].role == "assistant")}
    # within the best BM25 session: are the question terms said by the user at all?
    if top:
        sid = hx.units[top[0][0]].session_id
        terms = set(qp["terms"])
        su_t = set()
        sa_t = set()
        for t in hx.sess_by_id[sid]["turns"]:
            (sa_t if t["role"] == "assistant" else su_t).update(content_stems(t["content"]))
        f["sess_cov_diff"] = (len(terms & sa_t) - len(terms & su_t)) / max(1, len(terms))
    else:
        f["sess_cov_diff"] = 0.0
    if use_ce:
        cand = [u for u, _ in hx.hybrid(q.question, k=20)]
        rr = hx.rerank(q.question, cand) if cand else []
        ca = max((s for u, s in rr if hx.units[u].role == "assistant"), default=-12.0)
        cu = max((s for u, s in rr if hx.units[u].role == "user"), default=-12.0)
        f["ce_diff"] = ca - cu
        f["ce_top1_a"] = float(bool(rr) and hx.units[rr[0][0]].role == "assistant")
    z = 0.9 * f["bm_diff"] + 1.2 * f["sess_cov_diff"] - 0.6 * first + 0.8 * f["top5_a"]
    if use_ce:
        z += 0.25 * max(-6.0, min(6.0, f["ce_diff"]))
    f["p_assistant"] = 1.0 / (1.0 + math.exp(-z))
    return f


def role_guess(q, hx, use_ce: bool = False) -> float:
    """P(the answer is in an assistant turn) for a question the text router could not place (see role_eval.py)."""
    return round(role_features(q, hx, use_ce)["p_assistant"], 3)


# ---------------------------------------------------------------------------------------------------------------------
# the lens
# ---------------------------------------------------------------------------------------------------------------------
def build(q, hx, **params) -> EvidencePack:
    from core.pack import build_pack  # lazy: the self-test runs without the core

    pool_k = params.get("pool_k", 60)
    rerank_n = params.get("rerank_n", 30)
    budget = params.get("budget", 7000)
    max_sessions = params.get("max_sessions", 2)
    tie = params.get("tie", 0.85)
    hint_mode = params.get("hint_mode", "generic")
    rrf_k = params.get("rrf_k", 20)

    qp = parse_question(q.question)
    qp["cue_floor"] = params.get("cue_floor", 0.4)
    query = retrieval_query(qp)
    idf = history_idf(hx)
    # rules are trusted fully on "you recommended" questions; on detail questions (dev style) they only nudge
    w_struct = params.get("w_struct", 1.0) if qp["rec_style"] else params.get("w_struct_other", 0.3)

    # 1) retrieval channels (rank lists)
    if params.get("channels", "hybrid") == "bm25":
        lists = [[u for u, _ in hx.bm25(query, k=pool_k)]]
    else:
        lists = [[u for u, _ in hx.hybrid(query, k=pool_k)]]
    if qp["role"]:
        lists.append([u for u, _ in hx.bm25(query, k=pool_k // 2, roles=qp["role"])])
    cand = list(dict.fromkeys(u for lst in lists for u in lst))
    if rerank_n:
        lists.append([u for u, _ in hx.rerank(query, cand[:rerank_n])])

    # 2) structural scan over every unit of the preferred role (cheap regex; catches one-line asides the retrievers
    #    bury, e.g. a recommendation dropped inside a conversation about something else)
    struct: dict[int, tuple[float, dict]] = {}
    if w_struct:
        scan = {u.uid for u in hx.units if qp["role"] is None or u.role == qp["role"]} | set(cand)
        for uid in scan:
            sc, f = unit_struct(hx, uid, qp, idf)
            if sc > 0:
                struct[uid] = (sc, f)
        cand = list(dict.fromkeys(cand + sorted(struct, key=lambda i: -struct[i][0])[: pool_k // 2]))

    # 3) fuse: rank-RRF of the retrieval lists (normalised to 1) + w_struct * structural score (normalised to 1)
    rrf: Counter = Counter()
    for lst in lists:
        for r, uid in enumerate(lst):
            rrf[uid] += 1.0 / (rrf_k + r + 1)
    mr = max(rrf.values()) if rrf else 1.0
    ms = max((v[0] for v in struct.values()), default=0.0) or 1.0
    # on "you recommended" questions a sentence SHAPED like the asked recommendation (cue + name + kind/purpose) gets
    # its structural score multiplied by (1 + shape): it can beat a lexically closer unit that recommends nothing
    boost = params.get("shape_boost", 0.0) if qp["rec_style"] else 0.0
    def st_part(uid):
        sc, f = struct.get(uid, (0.0, {}))
        shp = rec_shape(f, qp) if boost else 0.0
        return sc / ms * (1.0 + boost * (shp if shp >= 0.5 else 0.0)) if sc else 0.0
    final = {uid: rrf.get(uid, 0.0) / mr + w_struct * st_part(uid) for uid in cand}
    ranking = sorted(final, key=lambda i: -final[i])
    if not ranking:
        return EvidencePack(question_id=q.question_id, variant=params.get("_variant", ""), route="ssa", passages=[],
                            hint=build_hint(qp, None, hint_mode), abstain_p=0.9, debug={"ranking": []})

    # 4) best turn / session; a runner-up session only when it is nearly tied (near-miss insurance)
    b = ranking[0]
    bu = hx.units[b]
    best_sid = bu.session_id
    second = next((i for i in ranking if hx.units[i].session_id != best_sid), None)
    sids = [best_sid]
    if second is not None and max_sessions > 1 and final[second] >= tie * final[b]:
        sids.append(hx.units[second].session_id)

    # 5) units in priority order (build_pack stops at the budget): best turn whole, the user turn before it, the
    #    runner-up's best turn, the rest of the ranking inside the chosen sessions, then the rest of the best session
    order: list[int] = list(hx.turn_units(bu.turn_id)) if params.get("whole_turn", True) else [b]
    turns = hx.sess_by_id[best_sid]["turns"]
    if bu.turn_idx > 0:
        order += hx.turn_units(turns[bu.turn_idx - 1]["turn_id"])[-2:]
    if len(sids) > 1:
        su = hx.units[second]
        order += hx.turn_units(su.turn_id)[:3]
        if su.turn_idx > 0:
            order += hx.turn_units(hx.sess_by_id[su.session_id]["turns"][su.turn_idx - 1]["turn_id"])[-1:]
    order += [i for i in ranking if hx.units[i].session_id in sids]
    if params.get("fill_session", True):
        order += [i for t in turns for i in hx.turn_units(t["turn_id"])]
    order = list(dict.fromkeys(order))

    # 6) best-sentence info, code-computed list item for ordinal questions ("the 7th job", "the last venue")
    feat = struct.get(b, (0.0, {}))[1]
    best_info = {"date": bu.date_str, "session_id": best_sid, "sentence": feat.get("sentence", ""),
                 "feat": {k: v for k, v in feat.items() if k != "sentence"}}
    sc_b = struct.get(b, (0.0, {}))[0]
    sc_2 = max((struct[i][0] for i in struct if hx.units[i].session_id != best_sid), default=0.0)
    best_info["margin"] = round(sc_b / sc_2, 2) if sc_2 > 0 else 99.0
    # quote only a sentence that looks like THE answer: a first-person recommendation with a name/value, matching the
    # asked kind or purpose, well covered by the question terms, clearly ahead of every other session's best sentence
    best_info["confident"] = bool(qp["rec_style"] and feat.get("cue") and feat.get("ent")
                                  and not (feat.get("third") or feat.get("past") or feat.get("contra"))
                                  and (feat.get("p_exact") or (feat.get("p_ov", 0) >= 0.5 if qp["purpose"]
                                                                else feat.get("head")))
                                  and feat.get("ov", 0) >= params.get("quote_ov", 0.4)
                                  and best_info["margin"] >= params.get("quote_margin", 1.3))
    best_info["shape"] = max((rec_shape(v[1], qp) for i, v in struct.items() if hx.units[i].role == "assistant"),
                             default=0.0) if qp["rec_style"] else None
    if params.get("kind_gate") and qp["rec_style"] and (qp["kind_stems"] or qp["syn"]):
        a_text = " ".join(t["content"] for t in turns if t["role"] == "assistant")
        a_st = set(content_stems(a_text))
        fa = fold(a_text)
        best_info["kind_in_top"] = bool((qp["head"] and qp["head"] in a_st) or any(
            re.search(r"(?<![a-z])" + re.escape(x) + r"(?![a-z])", fa) for x in qp["syn"]))
    computed = None
    if qp["ordinal"]:
        # the list turn: first turn of the best session, in ranking order, whose list is long enough for n
        need = max(qp["ordinal"], 2)
        texts = {t["turn_id"]: t["content"] for t in turns}
        item = None
        for tid in dict.fromkeys([bu.turn_id] + [hx.units[i].turn_id for i in ranking[:40]
                                                 if hx.units[i].session_id == best_sid]):
            if hx.units[hx.turn_units(tid)[0]].role != "assistant" or len(list_items(texts[tid])) < need:
                continue
            item = nth_item(texts[tid], qp["ordinal"])
            break
        if item:
            best_info["item"] = item
            best_info["ord_label"] = "last" if qp["ordinal"] == -1 else f"#{qp['ordinal']}"
            computed = item

    # confidence of the best TURN inside the best session: fused score of the best unit vs the best unit of any other
    # turn of that session (debug + optional confidence-gated budget)
    t2 = next((i for i in ranking if hx.units[i].session_id == best_sid and hx.units[i].turn_id != bu.turn_id), None)
    best_info["turn_margin"] = round(final[b] / final[t2], 3) if t2 is not None and final[t2] > 0 else 99.0
    cb = params.get("conf_budget")
    tiers = [cb] if cb and not isinstance(cb[0], (tuple, list)) else (cb or [])
    for thr_m, bud in tiers:              # tiers sorted by threshold, highest first
        if best_info["turn_margin"] >= thr_m:
            budget = min(budget, bud)
            break
    hint = build_hint(qp, best_info, hint_mode)
    passages, included, chars = build_pack(q, hx, order, budget=budget, window=params.get("window", 0),
                                           order="chrono", hint=hint, max_sessions=len(sids), whole_turns=False)
    return EvidencePack(question_id=q.question_id, variant=params.get("_variant", ""), route="ssa", passages=passages,
                        hint=hint, cite_plan=sids[: params.get("cite", 1)], abstain_p=abstain_estimate(qp, best_info),
                        included_turns=included, computed=computed,
                        debug={"ranking": ranking, "query": query, "w_struct": w_struct,
                               "qp": {k: qp[k] for k in ("kind", "purpose", "topic", "role", "ordinal", "rec_style")},
                               "best": best_info, "chars": chars})


# ---------------------------------------------------------------------------------------------------------------------
# self-test on sample data (pure python)
# ---------------------------------------------------------------------------------------------------------------------
def _selftest() -> None:
    import json
    from pathlib import Path

    qs = [
        ("What was the name of that coworking space you recommended to me in Lisbon?", "coworking space", "assistant"),
        ("Which code editor did you recommend I use for my Python practice scripts?", "code editor", "assistant"),
        ("What was the name of the restaurant you recommended for dinner with my sons?", "restaurant", "assistant"),
        ("Which show was my example of Netflix only having the last season available?", "", "user"),
        ("I think we discussed work from home jobs for seniors earlier. Can you remind me what was the 7th job in the "
         "list you provided?", "", "assistant"),
        ("What was the last venue on the list of Portland indie music venues?", "", None),
        ("What opening did you recommend I try against 1.e4? I forgot the name.", "opening", "assistant"),
    ]
    for text, kind, role in qs:
        qp = parse_question(text)
        print(f"- {text[:70]}\n    clean={qp['clean']!r} kind={qp['kind']!r} purpose={qp['purpose']!r} "
              f"said={qp['purpose_said']!r} role={qp['role']} ord={qp['ordinal']} terms={qp['terms']}")
        assert kind == "" or qp["kind"].lower().endswith(kind.split()[-1]), qp["kind"]
        assert qp["role"] == role, (qp["role"], role)
    assert parse_question(qs[4][0])["ordinal"] == 7
    assert parse_question(qs[5][0])["ordinal"] == -1

    qp = parse_question("What was the name of the café you recommended for my remote work days?")
    idf: dict = {}
    sents = [
        "For your remote work days, I recommend Café Blau in Mitte; it has stable wifi and large tables.",
        "I recommended Café Rosa's Neukölln branch to a friend last month because it is quiet in the mornings.",
        "Want to meet at the café near Volkspark instead? They have that quiet back room on Saturday afternoons.",
        "My coworker Lena told me Café Rosa has better pastries, but I haven't been there yet.",
    ]
    scores = [sent_struct(s, qp, idf)[0] for s in sents]
    print("cafe scores", [round(x, 2) for x in scores])
    assert scores[0] == max(scores) and scores[0] > 1.5 * scores[1]

    qp = parse_question("What was the name of the farrier you recommended for my Icelandic horses?")
    s_good = sent_struct("I recommend a farrier named Ola Hansen for your Icelandic horses.", qp, idf)[0]
    s_bad = sent_struct("Oh, before I forget: I recommended a saddle fitter named Kari Lund for your neighbor Bjørn's "
                        "horse..", qp, idf)[0]
    print("farrier", round(s_good, 2), round(s_bad, 2))
    assert s_good > 2 * s_bad

    qp = parse_question("What was the name of the coach you recommended for my speedwork?")
    a = sent_struct("For structured speedwork now that your legs are feeling better, I recommend Coach Miriam at the "
                    "Toronto Lakeshore Running Club.", qp, idf)[0]
    b = sent_struct("For strength and mobility work, I recommend Coach Elena at the Toronto Lakeshore Running Club.",
                    qp, idf)[0]
    print("coach", round(a, 2), round(b, 2))
    assert a > b

    jobs = ("Here are some jobs:\n\n1. Customer service representative\n2. Virtual assistant\n3. Online tutor\n"
            "4. Freelance writer\n5. Bookkeeper\n6. Data entry\n7. Transcriptionist\n8. Consultant\n\nGood luck!")
    assert nth_item(jobs, 7) == "Transcriptionist", nth_item(jobs, 7)
    assert nth_item(jobs, -1) == "Consultant"

    # sample file: does the gold turn contain a cue sentence / does the ordinal extraction find the answer?
    p = Path(__file__).resolve().parents[2] / "runs" / "samples" / "ssa.jsonl"
    if p.exists():
        n = hit = 0
        for line in p.open():
            d = json.loads(line)
            if d["split"] != "dev":
                continue
            qp = parse_question(d["question"])
            if qp["ordinal"] and d["gold_turns"]:
                item = nth_item(d["gold_turns"][0]["text"], qp["ordinal"])
                n += 1
                hit += bool(item and fold(d["answer"]).strip(" .")[:12] in fold(item))
                print(f"  ordinal q: {d['question'][:60]!r} -> {item!r} (gold {d['answer'][:40]!r})")
        print(f"ordinal extraction on samples: {hit}/{n}")
    print("self-test ok")


if __name__ == "__main__":
    _selftest()
