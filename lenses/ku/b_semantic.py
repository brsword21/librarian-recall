"""KU lens B ("Semantyka"): hybrid retrieval -> sentence-level cross-encoder -> ownership/value features ->
old+new session PAIR selection -> chronological "attribute timeline" hint.

Why (see b_design.md): KU gold is always exactly 2 user turns in 2 sessions (old value, new value). Dev retrieval is
easy (BM25 top-2 sessions = gold 85-90%), so the score is decided by citing exactly the pair and telling Reactor which
value is current. Test adds same-persona near-misses: other people's values ("My sister Amina is still on a 15GB
plan"), other scopes ("The UX team in the Munich office has eight people"), same number / other attribute ("My UX
team uses six different research tools") and paraphrased updates ("The studio hired Ana Ferreira to teach my
Wednesday ceramics class after Miguel moved to Porto") that often name the OLD value -> value-anchored second hop.

Self-test (pure python, no index, no models):  .venv/bin/python lenses/ku/b_semantic.py --selftest
"""
from __future__ import annotations

import math
import re
from collections import defaultdict
from itertools import combinations

try:  # the lens runs inside the harness; the self-test runs without core
    from core.types import EvidencePack
except Exception:  # pragma: no cover
    EvidencePack = None

_BASE = {"pool": 60, "rr_depth": 40, "sent_rr": True, "alpha": 0.5, "own_w": 6.0, "val_w": 0.75, "cov_w": 3.0,
         "distinct_w": 1.0, "same_pen": 1.0, "decay": 0.0, "top_m": 5, "extra": 2, "prf": False,
         "budget": 6000, "window": 0, "units_per_sess": 2}
VARIANTS = {
    # V1: hybrid pool -> unit + sentence rerank -> ownership/value/coverage features -> pair; cite the pair,
    #     show 2 extra sessions for robustness, timeline hint
    "pair": dict(_BASE),
    # V2: V1 + value-anchored second hop (query = salient value tokens of the best sentences + question terms)
    "pair_prf": dict(_BASE, prf=True, top_m=6, link_w=5.0),
    # V3: V2 + mild recency prior on the NEW member of the pair (LiCoMemory-style time decay, linear in sess_idx)
    "pair_decay": dict(_BASE, prf=True, top_m=6, link_w=5.0, decay=1.5),
    # V4: V2 with only the cited pair in the pack (precision / chars ablation)
    "tight": dict(_BASE, prf=True, top_m=6, link_w=5.0, extra=0, budget=4000),
    # control: recall-first; unit rerank only, top-6 sessions, +-1 neighbours, cite top-2 by score, same hint
    "wide": dict(_BASE, sent_rr=False, alpha=0.0, own_w=0.0, val_w=0.0, cov_w=0.0, distinct_w=0.0, same_pen=0.0,
                 top_m=2, extra=4, budget=12000, window=1),
    # R1: V2 + assistant recap sentences ("You mentioned you increased your evening walk to five km"), rewritten to
    #     first person and scored like user sentences minus asst_pen
    "pair_asst": dict(_BASE, prf=True, top_m=6, link_w=5.0, asst=True, asst_pen=1.0),
    # R1: V2 with the Snowball stemmer for coverage / candidate sentences ('waking' ~ 'wake')
    "pair_stem": dict(_BASE, prf=True, top_m=6, link_w=5.0, snowball=True),
    # R1: V2 + 'before X' chronology (the session naming X must be the newer member of an old-value pair)
    "pair_before": dict(_BASE, prf=True, top_m=6, link_w=5.0, before_chrono=True),
    # R1 combined: V2 + Snowball + assistant recaps + 'before X' chronology
    "pair_r1": dict(_BASE, prf=True, top_m=6, link_w=5.0, snowball=True, asst=True, asst_pen=1.0, before_chrono=True),
    # R1b: pair_r1 + reverse value link (earlier sentence sharing a name with an update that names old+new) +
    #      timeline hygiene (no off-attribute or bare-question quotes in the hint)
    "pair_r1b": dict(_BASE, prf=True, top_m=6, link_w=5.0, snowball=True, asst=True, asst_pen=1.0, before_chrono=True,
                     rev_link=3.0, tl_min_cov=0.2),
    # R1c: pair_r1b + sentence window -1 when the attribute sentence carries no value
    "pair_r1c": dict(_BASE, prf=True, top_m=6, link_w=5.0, snowball=True, asst=True, asst_pen=1.0, before_chrono=True,
                     rev_link=3.0, tl_min_cov=0.2, prev_merge=True),
}


# ---------------------------------------------------------------------------------------------------------------
# Pure-python helpers (question parsing, sentence features, values, hint). No core imports.
# ---------------------------------------------------------------------------------------------------------------
_WH = re.compile(r"^\s*(what|which|who|whom|where|when|how\s+(?:many|much|often|long|far|old)|is|are|was|were|do|does|"
                 r"did|have|has|had|am|can|could)\b\s*", re.I)
_CUE = re.compile(r"\b(right now|these days|so far|at the moment|at present|now|currently|current|still|"
                  r"most recently|recently|lately|anymore|in total|total)\b", re.I)
_AUX_I = re.compile(r"\b(do|does|did|have|has|am|was|were|can|could)\s+(i|we)\b", re.I)
_NAME_OF = re.compile(r"^(is|was)?\s*(the\s+)?(name|title|number|amount|type|kind|brand)\s+of\s+", re.I)
OLD_Q = re.compile(r"\b(previous(?:ly)?|initial(?:ly)?|originally|earlier|at first|used to|before (?:i|my|getting|the|"
                   r"i\s+\w+ed)|first time|when i (?:just )?started)\b", re.I)
COMPARE_Q = re.compile(r"\b(?:more|less|fewer|increase|decrease|switch(?:ed)? to|change)\b.*\b(?:than|or)\b", re.I)
NOW_Q = re.compile(r"\b(now|currently|current|right now|these days|latest|most recent(?:ly)?|anymore|still)\b", re.I)

STOP = set("""a an the of in on at to for from by with and or but is are was were be been being do does did have has had
i me my mine we our you your he she it they them his her its their this that these those what which who whom whose where
when why how many much often there here so far now currently current right still ever yet just very also can could
would should will shall may might must am about into than then as if not no any some all each every more most own get got
go going went use using used up out over again since last first previous previously recently recent total number amount
kind type name called day days week weeks time times i'm i've i'd i'll""".split())
_TOK = re.compile(r"[A-Za-z][A-Za-z0-9'\-]*|\d[\d,.:/-]*")


try:  # Snowball (same family as the core's BM25 stemmer): 'waking' ~ 'wake', 'mornings' ~ 'morning'
    import Stemmer as _Stemmer
    _SNOW = _Stemmer.Stemmer("english")
except Exception:  # pragma: no cover
    _SNOW = None
USE_SNOWBALL = [False]
PREV_MERGE = [False]   # set per build() from the variant params (module-level so the pure helpers stay simple)
OWN_NAME = [False]     # r2 (lenses/ku/b_r2.py): bare-name subject without first person = someone else's value


def _stem(w: str) -> str:
    """Tiny suffix stripper (self-contained), or Snowball when the variant asks for it."""
    w = w.lower().strip("'-").replace("'s", "")
    if USE_SNOWBALL[0] and _SNOW is not None:
        return _SNOW.stemWord(w)
    if w.endswith("ss"):
        return w
    for suf in ("ies", "es", "s", "ing", "ed"):
        if len(w) > len(suf) + (2 if suf == "s" else 3) and w.endswith(suf):
            return w[: -len(suf)] + ("y" if suf == "ies" else "")
    return w


def content_terms(text: str) -> list[str]:
    out = []
    for w in _TOK.findall(text):
        wl = w.lower().strip("'")
        if wl.replace("'s", "") in STOP or len(wl) < 3:
            continue
        out.append(_stem(wl))
    return list(dict.fromkeys(out))


def declarative(question: str) -> str:
    """LLM-free reformulation: drop the wh-word, auxiliaries, temporal cue words and 'the name of'.
    'What is the name of my ceramics class instructor now?' -> 'my ceramics class instructor'."""
    s = question.strip().rstrip("?").strip()
    s = s.split("?")[0]                     # multi-part questions: keep the first clause
    s = _WH.sub("", s)
    s = _WH.sub("", s) if not re.match(r"(?i)(i|we)\b", s) else s     # 'What is my ...' -> 'my ...'
    s = _NAME_OF.sub("", s)
    s = _AUX_I.sub(r"\2", s)
    s = _CUE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip(" ,.")
    return s or question


def strip_cues(question: str) -> str:
    """Question form for dense / cross-encoder: keep the wh-form, drop 'now/current/these days' (they pull in any
    sentence with 'current' in it, e.g. 'my current job as a data analyst')."""
    s = _CUE.sub(" ", question)
    return re.sub(r"\s+", " ", s).strip()


def question_mode(question: str) -> str:
    """'both' (two questions: then + now), 'compare' (more/less, increase/decrease), 'old' (asks for the earlier
    value), else 'new'. A wrong 'both' makes Reactor give two values (= wrong), so it needs two question marks."""
    old, now = bool(OLD_Q.search(question)), bool(NOW_Q.search(question))
    if question.count("?") >= 2 and old:
        return "both"
    if COMPARE_Q.search(question):
        return "compare"
    if old and now:   # 'Rachel, an old colleague from my previous company, currently ...' -> new
        return "old" if re.search(r"\bbefore\b", question, re.I) else "new"
    return "old" if old else "new"


_SENT = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[A-Z0-9])|\n+")


def sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT.split(text) if s and len(s.strip()) > 2]


# --- ownership: whose value is it? ----------------------------------------------------------------------------
_REL = (r"friend|sister|brother|neighbou?r|coworker|co-worker|colleague|mom|mum|dad|mother|father|daughter|son|partner|"
        r"husband|wife|cousin|aunt|uncle|boss|manager|roommate|flatmate|classmate|teammate|niece|nephew|grandma|"
        r"grandpa|grandmother|grandfather|kid|child|wife's|husband's|landlord|coach|instructor|teacher|tutor|"
        r"running partner|training partner|old \w+|former \w+")
OTHER_POSS = re.compile(rf"\b(?:[Mm]y|[Oo]ur)\s+(?:{_REL})(?:\s+[A-Z][\wÀ-ſ]+)?(?:'s|’s)\s|"     # my sister Amina's X
                        rf"\b[A-Z][\wÀ-ſ]+(?:'s|’s)\s+(?!own\b)|"                       # Clara's UX team
                        rf"\b(?:[Hh]e|[Ss]he|[Tt]hey)\s+(?:has|have|had|drinks|pays|owns|uses|is|was|takes|does|keeps|swims|walks|"
                        rf"runs|buys|gets)\b|"
                        rf"\b(?:[Mm]y|[Oo]ur)\s+(?:{_REL})(?:\s+[A-Z][\wÀ-ſ]+)?\s+(?:has|have|had|is|was|used|pays|"
                        rf"drinks|owns|uses|takes|does|keeps|swims|walks|runs|buys|gets|told|says|said|mentioned)\b")
OTHER_SCOPE = re.compile(r"\b(?:the|their|his|her)\s+[\w\s-]{0,30}?\s+(?:in|at)\s+the\s+(?P<place>[A-Z][\w-]+)\s+"
                         r"(?:office|branch|team|studio|club|store)\b", re.I)
FIRST_PERSON = re.compile(r"\b(?:I|I'm|I've|I'd|I'll|my|My|we|We|our|Our|me)\b")
PAST_OTHER = re.compile(r"\b(?:used to|last (?:autumn|fall|spring|summer|winter)|years ago|"
                        r"when I was|hypothetical(?:ly)?|if I (?:had|were))\b", re.I)
NAME_SUBJ = re.compile(r"(?:^|[.;:]\s+|\b(?:and|but|while|because)\s+)\W*([A-Z][a-zÀ-ſ]+(?:\s+[A-Z][a-zÀ-ſ]+)?)\s+"
                       r"(?:also\s+|still\s+|now\s+|usually\s+)?(?:has|have|had|drinks|pays|owns|uses|takes|keeps|swims|"
                       r"walks|runs|buys|gets|earns|eats|lives|works|plays|reads|does|teaches)\b")
_PERSON_IN_Q = re.compile(rf"\b(?:{_REL})\b|\b[A-Z][a-zÀ-ſ]+\b", re.I)


def ownership(sent: str, question: str) -> float:
    """+1: the user states their own value; -1: someone else's value / other scope; 0: unclear.
    No penalty when the question itself is about that other person ('my neighbor Bjørn and I', 'Rachel')."""
    q_low = question.lower()
    m = OTHER_POSS.search(sent) or OTHER_SCOPE.search(sent)
    if m:
        # who/where the statement is about: relation nouns + capitalised names inside the match
        if m.groupdict().get("place"):
            who = [m.group("place")]
        else:
            who = re.findall(rf"\b(?:{_REL})\b", m.group(0), re.I)
            who += [w for w in re.findall(r"\b[A-Z][a-zÀ-ſ]+", m.group(0))
                    if w not in {"My", "Our", "The", "Their", "His", "Her", "He", "She", "They"}]
        if any(re.search(rf"\b{re.escape(w.lower())}\b", q_low) for w in who):
            return 0.5
        return -1.0
    if OWN_NAME[0] and not FIRST_PERSON.search(sent) and re.match(r"\s*how\s+(?:many|much|often|long|far)\b", question, re.I):
        # quantity questions only: in 'who teaches / what is the name' questions the named person IS the value
        # 'Carlos drinks three cups of coffee every morning.' (a named person's habit / possession, no I/my)
        nm = NAME_SUBJ.search(sent)
        if nm and not any(re.search(rf"\b{re.escape(w.lower())}\b", q_low) for w in nm.group(1).split()):
            return -1.0
    if PAST_OTHER.search(sent):
        return -0.3
    return 1.0 if FIRST_PERSON.search(sent) else 0.0


# --- values ---------------------------------------------------------------------------------------------------
_NUMW = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
         "seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety hundred thousand dozen "
         "half once twice").split()
_NUMW_RX = re.compile(r"\b(" + "|".join(_NUMW) + r")\b", re.I)
_NUM_RX = re.compile(r"(?<![\w/])\$?\d[\d,]*(?:\.\d+)?(?:\s?(?:k|K|GB|MB|km|%|am|pm|a\.m\.|p\.m\.))?|\b\d{1,2}:\d{2}\b")
_QUOTE_RX = re.compile(r"[\"'‘“]([^\"'’”]{2,40})[\"'’”]")
_CAP_RX = re.compile(r"(?<![.!?]\s)(?<!^)\b[A-Z][a-zÀ-ſ]+(?:\s+[A-Z][a-zÀ-ſ]+)*")
_W2N = {w: i for i, w in enumerate(_NUMW[:21])}
_W2N.update({"thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
             "dozen": 12, "once": 1, "twice": 2})


def values(sent: str, question: str = "") -> set[str]:
    """Salient value tokens of a sentence: numbers (word numbers normalised), times, quoted names, capitalised names
    that are not in the question. Used to tell 'two different values' (an update) from 'the same value repeated'."""
    q_low = question.lower()
    out = set()
    for m in _NUM_RX.finditer(sent):
        v = m.group(0).replace(",", "").replace("$", "").strip().lower()
        if re.fullmatch(r"20\d\d", v):     # a year is not the attribute value
            continue
        out.add(v)
    for m in _NUMW_RX.finditer(sent):
        out.add(str(_W2N.get(m.group(1).lower(), m.group(1).lower())))
    for m in _QUOTE_RX.finditer(sent):
        out.add(m.group(1).lower())
    for m in _CAP_RX.finditer(sent):
        w = m.group(0)
        if w.lower() not in q_low and w not in {"I", "I'm", "I've", "By", "Oh", "Also", "Anyway", "Actually", "So",
                                                 "But", "And", "Random", "Speaking", "Plus", "Maybe", "Monday",
                                                 "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"}:
            out.add(w.lower())
    return out


def has_value(sent: str, question: str = "") -> bool:
    return bool(values(sent, question))


# --- hint -----------------------------------------------------------------------------------------------------
def _short(s: str, n: int = 220) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    return s if len(s) <= n else s[: n - 1].rsplit(" ", 1)[0] + "…"


def build_hint(timeline: list[tuple[str, str]], mode: str, abstain_terms: list[str] | None = None) -> str:
    """timeline: [(date 'YYYY/MM/DD', sentence)], oldest first."""
    lines = []
    if timeline:
        lines.append("Statements about this in the history, oldest first:")
        for d, s in timeline:
            lines.append(f'- {d[:10]}: "{_short(s)}"')
    if mode == "old":
        lines.append("The value changed over time; the question asks for the EARLIER value (before the change).")
    elif mode == "both":
        lines.append("The value changed over time; give the earlier value and the current (most recent) value.")
    elif mode == "compare":
        lines.append("The value changed over time; answer by comparing the earliest statement with the most recent one.")
    else:
        lines.append("If the value changed over time, answer with the MOST RECENT value the user states about their "
                     "own situation; ignore values about other people, other places or past/hypothetical ones.")
    lines.append("Cite both the session with the earlier value and the session with the latest value.")
    if abstain_terms:
        lines.append(f"Note: {', '.join(repr(t) for t in abstain_terms)} never appears in the history; if no passage "
                     f"is about exactly what is asked, answer INSUFFICIENT_EVIDENCE.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------------------------
# Retrieval (uses the documented core API: hx.bm25 / hx.dense / hx.rerank / build_pack)
# ---------------------------------------------------------------------------------------------------------------
def _rrf(lists: list[list[tuple[int, float]]], weights: list[float], k: int = 60) -> list[int]:
    sc: dict[int, float] = defaultdict(float)
    for lst, w in zip(lists, weights):
        for r, (uid, _) in enumerate(lst):
            sc[uid] += w / (k + r + 1)
    return [u for u, _ in sorted(sc.items(), key=lambda x: -x[1])]


def _sent_rerank(query: str, texts: list[str]) -> list[float] | None:
    """Cross-encoder on raw sentences through the shared model server (same cache as hx.rerank). None if unavailable."""
    try:
        from core import models
        return [float(x) for x in models.rerank(query, texts)]
    except Exception:
        return None


def _hist_stats(hx) -> dict:
    """Per-history cache: document frequency of content terms over user units (for idf / missing-term checks)."""
    cache = hx.__dict__.setdefault("_ku_b_cache" + ("_snow" if USE_SNOWBALL[0] else ""), {})
    if "df" not in cache:
        df: dict[str, int] = defaultdict(int)
        n = 0
        for u in hx.units:
            if u.role != "user":
                continue
            n += 1
            for t in set(content_terms(u.index_text)):
                df[t] += 1
        cache["df"], cache["n"] = df, max(n, 1)
    return cache


def candidate_sentences(hx, uids: list[int], question: str, per_unit: int = 4) -> list[dict]:
    """Split candidate user units into sentences; keep those sharing a content term with the question or carrying a
    value (plus the first sentence of short answer turns, where the topic is in Unit.ctx)."""
    qt = set(content_terms(question))
    out = []
    for uid in uids:
        u = hx.units[uid]
        ss = sentences(u.text) or [u.text]
        scored = []
        for i, s in enumerate(ss):
            ov = len(qt & set(content_terms(s)))
            if ov or (len(ss) <= 2 and u.ctx):
                scored.append((ov, i, s))
        scored.sort(key=lambda x: (-x[0], x[1]))
        for ov, i, s in scored[:per_unit]:
            if not has_value(s, question) and i + 1 < len(ss):   # 'tried any good Korean restaurants?' + 'I've tried three'
                s = f"{s} {ss[i + 1]}"
            # value in the PREVIOUS sentence ("I named the new deck 'Decaf' yesterday. So now it's officially part
            # coffee, part skateboard"): sentence window -1 (variant param prev_merge)
            if PREV_MERGE[0] and not has_value(s, question) and i > 0 and has_value(ss[i - 1], question):
                s = f"{ss[i - 1]} {s}"
            text = f"{u.ctx} {s}".strip() if (i == 0 and u.ctx and ov == 0) else s
            out.append({"uid": uid, "sid": u.session_id, "text": text, "raw": s, "ov": ov,
                        "own": ownership(s, question), "val": has_value(s, question)})
    return out


# --- assistant recaps: test KU updates can live only in an assistant turn ("You mentioned you increased your evening
#     walk to five kilometers", "Margaret has retired, so your Saturday morning watercolour class is now taught by Priya
#     Nair"; idea credited to lens A's design). Keep only sentences that restate the USER's situation and rewrite them
#     to the user's perspective, so the same ownership / value / coverage / cross-encoder features apply.
RECAP = re.compile(r"\byou (?:mentioned|said|told me|noted|shared|explained)\b|\byou(?:'ve| have) (?:been |now |just |"
                   r"recently )?(?:started|switched|moved|increased|decreased|upgraded|downgraded|changed|got|bought|"
                   r"adopted|renamed|joined|hired|added|reached|finished|raised|cut|dropped)\b|\byour\b[^.?!]{1,60}?\b"
                   r"(?:is|are|has|have|was)\s+now\b|\bis now (?:taught|run|led|managed|handled|coached)\b|"
                   r"\byou (?:now|recently|just) (?!asked|said|mentioned)\w+", re.I)
_YOU = [(re.compile(r"\byou (?:mentioned|said|told me|noted|shared|explained)(?: that)?\s*", re.I), ""),
        (re.compile(r"\byour\b", re.I), "my"), (re.compile(r"\byours\b", re.I), "mine"),
        (re.compile(r"\byou're\b", re.I), "I'm"), (re.compile(r"\byou've\b", re.I), "I've"),
        (re.compile(r"\byou'll\b", re.I), "I'll"), (re.compile(r"\byou'd\b", re.I), "I'd"),
        (re.compile(r"\byourself\b", re.I), "myself"), (re.compile(r"\byou are\b", re.I), "I am"),
        (re.compile(r"\byou were\b", re.I), "I was"), (re.compile(r"\byou\b", re.I), "I")]


def to_user_view(sent: str) -> str:
    """'You mentioned you increased your evening walk to five kilometers' -> 'I increased my evening walk to five
    kilometers'. Pure string rewrite, used only on assistant recap sentences."""
    s = sent
    for rx, rep in _YOU:
        s = rx.sub(rep, s)
    s = re.sub(r"\s+", " ", s).strip()
    return s[:1].upper() + s[1:] if s else s


def recap_sentences(hx, uids: list[int], question: str) -> list[dict]:
    """Assistant units -> recap sentences about the user (rewritten to first person) that share a question term."""
    qt = set(content_terms(question))
    out = []
    for uid in uids:
        u = hx.units[uid]
        for s in sentences(u.text) or [u.text]:
            if not RECAP.search(s) or "?" in s:
                continue
            v = to_user_view(s)
            if not (qt & set(content_terms(v))):
                continue
            out.append({"uid": uid, "sid": u.session_id, "text": v, "raw": v, "ov": len(qt & set(content_terms(v))),
                        "own": ownership(v, question), "val": has_value(v, question), "asst": True})
    return out


def _prf_query(best_sents: list[str], question: str) -> str:
    """Value-anchored second hop: the update often names the OLD value ('after Miguel moved to Porto', 'because 15GB
    was not enough', "'Decaf' just made me think of work"). Query = value tokens of the best sentences + question
    content terms."""
    vals = []
    for s in best_sents:
        vals.extend(sorted(values(s, question)))
    return " ".join(list(dict.fromkeys(vals))[:6] + content_terms(question)[:6])


def coverage(text: str, qterms: list[str], idf: dict[str, float]) -> float:
    """idf-weighted share of the question's content terms present in `text` (0..1): the attribute-match signal that
    keeps '9 Icelandic horses on my farm' out of 'How many beehives do I have on my farm right now?'."""
    if not qterms:
        return 0.0
    have = set(content_terms(text))
    tot = sum(idf[t] for t in qterms) or 1.0
    return sum(idf[t] for t in qterms if t in have) / tot


def analyze(q, hx, **p) -> dict:
    """Steps 1-6 + abstention; everything build() needs, also used by scratch_b/probe_pairs.py."""
    question = q.question
    USE_SNOWBALL[0] = bool(p.get("snowball"))
    PREV_MERGE[0] = bool(p.get("prev_merge"))
    OWN_NAME[0] = bool(p.get("own_name"))
    mode = question_mode(question)
    qd = declarative(question)
    qs = strip_cues(question)
    st = _hist_stats(hx)
    qterms = content_terms(qd)          # declarative form: 'how long' / 'what time' are not attribute words
    idf = {t: math.log((st["n"] + 1) / (st["df"].get(t, 0) + 0.5)) for t in qterms}

    # 1) candidate pool: BM25 on the declarative form + dense on the cue-stripped question, user turns only
    #    (100% of dev/dev200 KU gold turns are user turns)
    pool_n = p.get("pool", 60)
    lists = [hx.bm25(qd, k=pool_n, roles="user"), hx.dense(qs, k=pool_n, roles="user"),
             hx.bm25(question, k=20, roles="user")]
    pool = _rrf(lists, [1.0, p.get("w_dense", 1.0), 0.5], k=p.get("rrf_k", 60))

    # 2) unit-level cross-encoder
    rr_depth = p.get("rr_depth", 40)
    unit_rr = dict(hx.rerank(qs, pool[:rr_depth]))

    # 3) value-anchored second hop (adds paraphrased updates that name the old value)
    if p.get("prf"):
        qt = set(qterms)
        best_txt = []
        for uid, _ in sorted(unit_rr.items(), key=lambda x: -x[1])[:3]:
            ss = sentences(hx.units[uid].text)
            best_txt.append(max(ss, key=lambda x: len(qt & set(content_terms(x)))) if ss else hx.units[uid].text)
        pq = _prf_query(best_txt, question)
        extra = [u for u, _ in hx.bm25(pq, k=15, roles="user") if u not in unit_rr]
        extra += [u for u, _ in hx.dense(pq, k=10, roles="user") if u not in unit_rr and u not in extra]
        if extra:
            unit_rr.update(dict(hx.rerank(qs, extra)))
    ranked_units = [u for u, _ in sorted(unit_rr.items(), key=lambda x: -x[1])]

    # 4) sentence-level scores + ownership / value / attribute-coverage features
    sents = candidate_sentences(hx, ranked_units[: max(rr_depth, 30)], question)
    # 4a) assistant recaps about the user (test KU updates can live only there), rewritten to first person
    asst_rr: dict[int, float] = {}
    if p.get("asst"):
        ak = p.get("asst_k", 30)
        apool = _rrf([hx.bm25(qd, k=ak, roles="assistant"), hx.dense(qs, k=ak, roles="assistant")], [1.0, 1.0])
        apool = [u for u in apool[: p.get("asst_depth", 20)] if RECAP.search(hx.units[u].text)]
        if apool:
            asst_rr = dict(hx.rerank(qs, apool))
            sents += recap_sentences(hx, apool, question)
    srr = _sent_rerank(qs, [s["text"] for s in sents]) if (p.get("sent_rr") and sents) else None
    a = p.get("alpha", 0.5) if srr is not None else 0.0
    for i, s in enumerate(sents):
        s["unit_rr"] = (asst_rr if s.get("asst") else unit_rr).get(s["uid"], -10.0)
        s["sent_rr"] = srr[i] if srr is not None else None
        s["cov"] = coverage(s["text"], qterms, idf)
        base = (1 - a) * s["unit_rr"] + (a * srr[i] if srr is not None else 0.0)
        s["score"] = (base + p.get("own_w", 0.0) * min(s["own"], 0.0) + p.get("val_w", 0.0) * s["val"]
                      + p.get("cov_w", 0.0) * s["cov"] - (p.get("asst_pen", 1.0) if s.get("asst") else 0.0))
    seen_units = {s["uid"] for s in sents}
    for uid in ranked_units[:rr_depth]:          # units without a kept sentence still compete at unit level
        if uid not in seen_units:
            u = hx.units[uid]
            sents.append({"uid": uid, "sid": u.session_id, "text": u.text, "raw": u.text[:300], "ov": 0, "own": 0.0,
                          "val": False, "cov": 0.0, "unit_rr": unit_rr[uid], "sent_rr": None,
                          "score": unit_rr[uid] - 1.0})

    # 5) session aggregation: best sentence per session
    best_by_sess: dict[str, dict] = {}
    for s in sents:
        if s["sid"] not in best_by_sess or s["score"] > best_by_sess[s["sid"]]["score"]:
            best_by_sess[s["sid"]] = s
    sess_rank = sorted(best_by_sess, key=lambda sid: -best_by_sess[sid]["score"])

    # 5b) value link: an update usually names the OLD value ("Marta Nogueira went on maternity leave and referred me
    #     to her colleague Bruno Almeida"); MiniLM scores such sentences very low (-9), so reward a LATER own sentence
    #     that shares a value with the anchor (best sentence), adds a new value and matches the attribute (cov)
    if p.get("link_w") and sess_rank:
        anchor = best_by_sess[sess_rank[0]]
        a_idx = hx.sess_by_id[anchor["sid"]]["sess_idx"]
        # single digits ('one tractor' / 'split one of my beehives') are not salient enough to link two statements
        a_vals = {v for v in values(anchor["raw"], question) if not re.fullmatch(r"\d", v)}
        if a_vals:
            for s in sents:
                if s["sid"] == anchor["sid"] or hx.sess_by_id[s["sid"]]["sess_idx"] < a_idx or s["own"] < 0:
                    continue
                sv = values(s["raw"], question)
                if (sv & a_vals) and (sv - a_vals) and s["cov"] >= p.get("link_cov", 0.25):
                    s["score"] += p["link_w"]
                    s["linked"] = True
            # reverse link: the anchor is an update that names the OLD value too ("I renamed my skateboard 'Flat
            # White'; 'Decaf' just made me think of work") -> an EARLIER own sentence carrying one of those values
            # (a name / quoted name / number with unit) is the old statement even if it says 'deck' not 'skateboard'
            if p.get("rev_link") and len(a_vals) >= 2:
                for s in sents:
                    if s["sid"] == anchor["sid"] or hx.sess_by_id[s["sid"]]["sess_idx"] >= a_idx or s["own"] < 0:
                        continue
                    shared = {v for v in values(s["raw"], question) & a_vals if not re.fullmatch(r"\d+", v)}
                    if shared:
                        s["score"] += p["rev_link"]
                        s["linked"] = True
            best_by_sess = {}
            for s in sents:
                if s["sid"] not in best_by_sess or s["score"] > best_by_sess[s["sid"]]["score"]:
                    best_by_sess[s["sid"]] = s
            sess_rank = sorted(best_by_sess, key=lambda sid: -best_by_sess[sid]["score"])
    n_sess = max(len(hx.sessions), 1)

    # 6) pair selection over the top-M sessions: both strong + two DIFFERENT values + (optional) recent new member
    top_m = sess_rank[: p.get("top_m", 5)]
    pair = top_m[:2]
    # 'before X' questions ("What gadget did I invest in before getting the Air Fryer?"): X marks the NEWER member;
    # the session that best covers the before-clause terms must be the later one of the pair
    before_anchor = None
    bm = re.search(r"\bbefore\s+(.+?)[?.]?$", question, re.I) if (p.get("before_chrono") and mode == "old") else None
    if bm and len(top_m) >= 2:
        bterms = content_terms(bm.group(1))
        if bterms:
            bcov = {sid: coverage(best_by_sess[sid]["text"], bterms, {t: 1.0 for t in bterms}) for sid in top_m}
            cand = max(top_m, key=lambda sid: bcov[sid])
            if bcov[cand] >= 0.5:
                before_anchor = cand
    if len(top_m) >= 2:
        best_val = -1e9
        for x, y in combinations(top_m, 2):
            old, new = sorted((x, y), key=lambda sid: hx.sess_by_id[sid]["sess_idx"])
            so, sn = best_by_sess[old], best_by_sess[new]
            v = so["score"] + sn["score"]
            vo, vn = values(so["raw"], question), values(sn["raw"], question)
            if vo and vn:
                v += p.get("distinct_w", 0.0) if vo != vn else -p.get("same_pen", 0.0)
            v += p.get("decay", 0.0) * hx.sess_by_id[new]["sess_idx"] / n_sess
            if before_anchor is not None and before_anchor in (old, new) and before_anchor != new:
                v -= p.get("before_pen", 5.0)
            if v > best_val:
                best_val, pair = v, [old, new]
    pair = sorted(pair, key=lambda sid: hx.sess_by_id[sid]["sess_idx"])

    # 8) abstention: idf-weighted share of question terms missing from the best 3 sentences + terms absent from
    #    the whole history (dev KU: idf-miss AUC .82, dev200 .90 with BM25 top-3; see scratch_b/abstain_signal.py)
    top3 = [best_by_sess[sid] for sid in sess_rank[:3]]
    miss = (1.0 - max(coverage(s["text"], qterms, idf) for s in top3)) if (qterms and top3) else 1.0
    absent = [t for t in qterms if st["df"].get(t, 0) == 0 and not t[0].isdigit()]
    # a capitalised question word that never occurs in the history (entity swap: 'Dr. Johnson' vs Dr. Smith,
    # 'Shinjuku' vs Harajuku, 'Italian' vs Korean): 3/6 dev KU unanswerable, 0/72 answerable
    caps = [w for w in re.findall(r"[A-Za-z][\w'-]+", question)[1:] if w[0].isupper()]
    cap_absent = sum(1 for w in caps if _stem(w) in absent)
    z = (p.get("ab_bias", -3.0) + p.get("ab_miss", 3.0) * miss + p.get("ab_absent", 0.8) * len(absent)
         + p.get("ab_cap", 2.5) * cap_absent)
    abstain_p = 1 / (1 + math.exp(-z))
    return {"mode": mode, "qd": qd, "qs": qs, "ranked_units": ranked_units, "sents": sents,
            "best_by_sess": best_by_sess, "sess_rank": sess_rank, "pair": pair, "miss": miss, "absent": absent,
            "abstain_p": abstain_p, "qterms": qterms, "idf": idf}


def build(q, hx, **p):
    from core.pack import build_pack  # imported lazily so the self-test runs without core

    r = analyze(q, hx, **p)
    pair, sess_rank, sents, best_by_sess = r["pair"], r["sess_rank"], r["sents"], r["best_by_sess"]

    # 7) pack: the pair + `extra` next-best sessions, best 2 units per session, chronological
    keep = list(pair) + [sid for sid in sess_rank if sid not in pair][: p.get("extra", 2)]
    unit_best: dict[int, float] = {}
    for s in sents:
        unit_best[s["uid"]] = max(unit_best.get(s["uid"], -1e9), s["score"])
    pack_uids = []
    for sid in keep:
        su = sorted({s["uid"] for s in sents if s["sid"] == sid and not s.get("asst")}, key=lambda uid: -unit_best[uid])
        pack_uids.extend(su[: p.get("units_per_sess", 2)])
        if best_by_sess[sid].get("asst"):        # the session's best statement is an assistant recap: add it too
            pack_uids.append(best_by_sess[sid]["uid"])

    # 9) hint: chronological timeline of the pair + same-attribute own statements in the shown sessions
    tl = []
    for sid in sorted(keep, key=lambda x: hx.sess_by_id[x]["sess_idx"]):
        s = best_by_sess[sid]
        # hint hygiene (tl_min_cov): never quote an off-attribute sentence ('This press release will also announce
        # a new employee' for 'my job title') or a bare question; Reactor would read it as a stated value
        if p.get("tl_min_cov") is not None and not s.get("linked") and (
                s["cov"] < p["tl_min_cov"] or (s["raw"].rstrip().endswith("?") and not s["val"])):
            continue
        if sid in pair or (s["own"] > 0 and s["val"] and s["cov"] >= p.get("tl_cov", 0.6)):
            tl.append((hx.sess_by_id[sid]["date_str"], s["raw"]))
    # name only proper nouns in the hint ('Johnson', 'Shinjuku', 'Italian'): a missing common word ('frequently')
    # is weak evidence and would push Reactor into a false abstention
    raw_absent = [w for w in re.findall(r"[A-Za-z][\w'-]+", q.question)[1:] if w[0].isupper() and _stem(w) in r["absent"]]
    hint = build_hint(tl[:4], r["mode"], raw_absent if r["abstain_p"] > p.get("ab_hint", 0.6) else None) \
        if p.get("hint", True) else ""

    passages, included_turns, _chars = build_pack(q, hx, pack_uids, budget=p.get("budget", 6000),
                                                  window=p.get("window", 0), order="chrono", hint=hint,
                                                  max_sessions=len(keep), whole_turns=False)
    pack_ids = {ps.id for ps in passages if ps.id}
    cite = [sid for sid in pair if sid in pack_ids]
    return EvidencePack(question_id=q.question_id, variant=p.get("_variant", ""), route="ku", passages=passages,
                        hint=hint, cite_plan=cite or None, abstain_p=round(r["abstain_p"], 4),
                        included_turns=included_turns, computed=None,
                        debug={"ranking": r["ranked_units"], "pair": pair, "sess_rank": sess_rank[:8],
                               "mode": r["mode"], "qd": r["qd"], "miss": round(r["miss"], 3), "absent": r["absent"]})


# ---------------------------------------------------------------------------------------------------------------
# Self-test on runs/samples/ku.jsonl (gold turn texts): parsing, ownership, values, pair logic, hint
# ---------------------------------------------------------------------------------------------------------------
def _selftest() -> None:
    import json
    from pathlib import Path

    ok = 0

    def check(cond, msg):
        nonlocal ok
        if not cond:
            raise AssertionError(msg)
        ok += 1

    check(declarative("What is the name of my ceramics class instructor now?") == "my ceramics class instructor",
          declarative("What is the name of my ceramics class instructor now?"))
    check(declarative("How many beehives do I have on my farm right now?") == "beehives I have on my farm",
          declarative("How many beehives do I have on my farm right now?"))
    check(declarative("What is my current monthly rent?") == "my monthly rent", declarative("What is my current monthly rent?"))
    check(question_mode("What was my previous personal best time for the charity 5K run?") == "old", "mode old")
    check(question_mode("Where do I initially keep my old sneakers?") == "old", "mode initially")
    check(question_mode("How many engineers do I lead when I just started my new role as Senior Software Engineer? "
                        "How many engineers do I lead now?") == "both", "mode both")
    check(question_mode("How many bikes do I currently own?") == "new", "mode new")
    check(question_mode("What was my personal best time in the charity 5K run?") == "new", "mode new (no cue)")

    q_ux = "How many people are on my UX team now?"
    check(ownership("My UX team now has eight people after the two new hires started today.", q_ux) == 1.0, "own ux")
    check(ownership("That reminds me: My friend Clara's UX team has six people.", q_ux) == -1.0, "friend Clara's")
    check(ownership("The UX team in the Munich office has eight people, but we only meet them quarterly.", q_ux) < 0,
          "munich scope")
    check(ownership("My sister Amina is still on a 15GB mobile data plan.", "How many gigabytes is my current mobile "
                                                                          "data plan?") < 0, "sister Amina")
    check(ownership("He drinks three cups of coffee every morning and says he won't give it up.",
                    "How many cups of coffee do I drink each morning these days?") < 0, "he drinks")
    check(ownership("Bjørn's daughter has 2 Icelandic horses.",
                    "How many Icelandic horses do my neighbor Bjørn and I have in total right now?") >= 0,
          "person named in question -> no penalty")
    check(ownership("I'm hoping to beat my personal best time of 25:50 this time around.",
                    "What was my personal best time in the charity 5K run?") == 1.0, "own pb")
    check(ownership("My friend Rachel actually just moved back to the suburbs again.",
                    "Where did Rachel move to after her recent relocation?") >= 0, "Rachel in question")

    check(values("I upgraded my mobile data plan to 25GB because 15GB was not enough.") >= {"25gb", "15gb"}, "GB")
    check(values("I've tried four different ones so far") == {"4"}, str(values("I've tried four different ones so far")))
    check("flat white" in values("I actually renamed my skateboard 'Flat White' this week; 'Decaf' just made me think"),
          "quoted names")
    check(values("I swam my usual 40 laps today") == values("I did 40 laps and my blood pressure"), "same value")

    # pair logic on the dev sample gold turns: the two gold sentences must carry different values (update signal)
    rows = [json.loads(l) for l in open(Path(__file__).resolve().parents[2] / "runs" / "samples" / "ku.jsonl")]
    n_pairs = n_distinct = 0
    for r in rows:
        if r["split"] != "dev" or r["abstain"] or len(r["gold_turns"]) != 2:
            continue
        a, b = r["gold_turns"]
        va, vb = values(a["text"], r["question"]), values(b["text"], r["question"])
        n_pairs += 1
        n_distinct += bool(va and vb and va != vb)
        qt = set(content_terms(r["question"]))
        best = max(sentences(b["text"]), key=lambda s: len(qt & set(content_terms(s))))
        check(ownership(best, r["question"]) >= 0, f"gold sentence judged as someone else's: {best[:120]}")
    check(n_distinct / max(n_pairs, 1) >= 0.7, f"distinct values only {n_distinct}/{n_pairs}")
    h = build_hint([("2023/04/11 (Tue) 10:00", "My ceramics class instructor is Miguel Silva."),
                    ("2023/08/03 (Thu) 09:00", "The studio hired Ana Ferreira to teach my Wednesday ceramics class "
                                               "after Miguel moved to Porto.")], "new")
    check("2023/08/03" in h and "MOST RECENT" in h and len(h) < 700, h)
    print(f"selftest ok: {ok} checks; gold pairs with distinct values {n_distinct}/{n_pairs}")
    print(h)


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        _selftest()
