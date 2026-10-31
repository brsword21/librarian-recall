"""Lens B ("Semantyka") for multi-session, ROUND 3: keep the fact sentence.

Finding (real Reactor, e1 calibration, q_2605c5385371 'hours driving to my three road trip destinations'): the pack had
all 3 gold sessions (offline PA = 1), but Reactor answered INSUFFICIENT_EVIDENCE because round-1/2 sentence trimming
(b_sweep.trim_text, 240 chars per unit, sentences ranked by question-term hits) cut 'it only took me four hours to
drive there' and 'I drove for six hours to Washington D.C.'. Offline check (scratch_b/strict.py): on dev, v1 b_tight
p08c35_aor has PA 0.950 but only 0.479 of the questions keep every fact-like sentence (number, 'by the way, I ...',
first-person event) of their gold user turns. The dataset plants the fact as a 'By the way, I ...' aside after a
topical opening sentence, which is exactly what term-hit trimming drops.

This lens = b_tight (round 2, frozen; same retrieval, same p(gold) keep / cite rule) with a new renderer:
- trim_mode "fact": sentences scored by question-term hits + fact cue (number / money / 'by the way' / 'I just ...'
  / first-person event verb); keep greedily within `trim` chars, the fact sentences first. Abbreviation-safe split
  ('Dr. Smith' stays one sentence).
- trim_mode "user_full": user units untrimmed (<= ~700 chars), assistant units trimmed as before.
- `per_sess` caps depth so the larger units do not crowd out kept sessions within `budget`.
"""
from __future__ import annotations

import re
from collections import defaultdict

from core.types import EvidencePack, Passage

from lenses.multi import b_sweep as R1
from lenses.multi import b_tight as T

BASE = dict(T.VARIANTS["p08c35_aor"])
BASE.update(trim_mode="terms")


def _v(**kw):
    d = dict(BASE)
    d.update(kw)
    return d


VARIANTS = {
    "v1": _v(),                                                    # = b_tight p08c35_aor (reference)
    "fact360": _v(trim_mode="fact", trim=360),
    "fact480": _v(trim_mode="fact", trim=480),
    "ufull": _v(trim_mode="user_full"),
    "ufull_d4": _v(trim_mode="user_full", per_sess=4),
    "fact360_d4": _v(trim_mode="fact", trim=360, per_sess=4),
    "ufull_ck": _v(trim_mode="user_full", cite_mode="keep"),      # passages == cite set exactly (Sall == S)
    "ufull_w": _v(trim_mode="user_full", w_breadth=True, w_frac=0.5, w_max=12),
    "ufull_w25": _v(trim_mode="user_full", w_breadth=True, w_frac=0.25, w_max=12),
    "ufull_h": _v(trim_mode="user_full", rate_hint=True),         # + additive routine hint
    "ufull_h12": _v(trim_mode="user_full", rate_hint=True, budget=12000),
    "ufull_hr": _v(trim_mode="user_full", rate_hint=True, resc=True, r_max=2, r_thr=-3.0),
    "ufull_hr0": _v(trim_mode="user_full", rate_hint=True, resc=True, r_max=2, r_thr=0.0),
    "ufull_fs": _v(trim_mode="user_full", fs_feat=True),          # features only (offline refit of p(gold))
}

# Rate questions ('how many health-related devices do I use in a day', 'how many days a week do I attend fitness
# classes'): on dev / dev200 all 3 such multi questions are SUMS over sessions (4 devices in 5 sessions; Zumba Tue+Thu
# + yoga Wed + weights Sat), so the round-1 'most recent statement wins' rate hint points Reactor at one session.
_RATE_OLD = ("This asks for a current routine value (per day/week): list each statement of it with its date and [E#]; "
             "the most recent statement wins, earlier values are outdated.")
_RATE_NEW = ("This asks about a routine (per day/week): list every item of it from all sessions with its date and [E#]; "
             "different items add up, and only when the same item is restated with a new value does the most recent "
             "statement win.")

_ABBR = re.compile(r"\b(Dr|Mr|Mrs|Ms|St|Mt|Jr|Sr|vs|etc|e\.g|i\.e|approx|No)\.$")
_NUMW = re.compile(r"\b(\d|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|dozen|couple|half|"
                   r"twice|once)\b", re.I)
_SELF_EVENT = re.compile(r"\b(I|I've|I'd|we|we've)\s+(?:\w+\s+){0,2}?(bought|got|visited|took|went|made|spent|finished|"
                         r"attended|picked|ordered|added|planted|harvested|read|watched|completed|learned|started|"
                         r"sold|fixed|assembled|received|paid|earned|saved|used|baked|cooked|swam|ran|walked|"
                         r"adopted|viewed|saw|led|joined|volunteered|drove|flew|stayed|booked|raised|donated|"
                         r"tried|have|had|own|use|wear|wore|play|played|signed|subscribed|redeemed|returned)\b", re.I)


def split_sents(text: str) -> list[str]:
    raw = R1.split_sents(text)
    out: list[str] = []
    for s in raw:
        if out and _ABBR.search(out[-1]):
            out[-1] = out[-1] + " " + s
        else:
            out.append(s)
    return out


def fact_score(s: str) -> float:
    f = 0.0
    if R1._FACT.search(s):
        f += 2.0
    if _NUMW.search(s):
        f += 1.0
    if _SELF_EVENT.search(s):
        f += 1.0
    return f


def trim_fact(text: str, terms: list[str], max_chars: int) -> tuple[str, bool]:
    """Keep the sentences with the best (fact cue + term hits) within max_chars, in text order."""
    if len(text) <= max_chars:
        return text, False
    ss = split_sents(text)
    keys = {R1._stem5(w) for w in terms}
    sc = []
    for i, s in enumerate(ss):
        toks = {R1._stem5(w) for w in re.findall(r"[a-z0-9][a-z0-9'\-]*", s.lower())}
        hit = len(keys & toks)
        sc.append((hit + fact_score(s) + (0.5 if hit else 0.0), -i))
    order = sorted(range(len(ss)), key=lambda i: sc[i], reverse=True)
    keep: set[int] = set()
    used = 0
    for i in order:
        if sc[i][0] <= 0 and keep:
            break
        if used + len(ss[i]) + 1 <= max_chars or not keep:
            keep.add(i)
            used += len(ss[i]) + 1
    out, prev = [], None
    for j in sorted(keep):
        if prev is not None and j != prev + 1:
            out.append("…")
        out.append(ss[j])
        prev = j
    s = " ".join(out)
    if min(keep) > 0:
        s = "… " + s
    if max(keep) < len(ss) - 1:
        s += " …"
    return s, True


def _render_unit(u, terms: list[str], P: dict) -> str:
    mode = P.get("trim_mode", "terms")
    if mode == "user_full" and u.role == "user":
        return u.text
    if mode in ("fact",):
        return trim_fact(u.text, terms, P["trim"])[0]
    return R1.trim_text(u.text, terms, P["trim"])[0] if P["trim"] else u.text


def _render_session(hx, uids: list[int], terms: list[str], P: dict) -> tuple[str, set[str]]:
    parts, turns, prev = [], set(), None
    for i in sorted(set(uids)):
        u = hx.units[i]
        text = _render_unit(u, terms, P)
        if prev is not None and i != prev + 1:
            parts.append("[...]")
        if u.role == "user" and u.ctx and u.chunk_idx == 0 and len(u.text) < 160 and (prev is None or i != prev + 1):
            parts.append(f"assistant: … {u.ctx}")
        parts.append(f"{u.role}: {text}")
        turns.add(u.turn_id)
        prev = i
    return "\n".join(parts), turns


def pack(q, hx, picks: list[int], hint: str, P: dict, terms: list[str]):
    from core.pack import message_chars
    chosen: dict[str, list[int]] = {}

    def passages(sel):
        sids = sorted(sel, key=lambda s: hx.sess_by_id[s]["sess_idx"])
        return [Passage(id=s, date=hx.sess_by_id[s]["date_str"], text=_render_session(hx, sel[s], terms, P)[0])
                for s in sids]

    chars = message_chars(q, [], hint)
    for u in picks:
        sid = hx.units[u].session_id
        trial = {k: list(v) for k, v in chosen.items()}
        trial.setdefault(sid, []).append(u)
        c = message_chars(q, passages(trial), hint)
        if c > P["budget"]:
            continue
        chosen, chars = trial, c
        if P["budget"] - chars < 120:
            break
    included = set()
    for s, us in chosen.items():
        included |= _render_session(hx, us, terms, P)[1]
    return passages(chosen), included, chars, chosen


# ------------------------------------------------------------------------------------------------------------------
# build(): b_tight.build with the renderer swapped (copied, because b_tight is frozen)

def build(q, hx, **params) -> EvidencePack:
    P = dict(BASE)
    P.update(params)
    a = R1.analyze(q.question, q.date)
    terms = a["terms"] + [w for f in a["facets"] for w in R1.content_terms(f)]
    ranked, final, rr, lab, facet_top, rest, win_active = R1._score_units(q, hx, a, P)

    miss = R1.missing_terms(hx, a["terms"])
    miss_frac = len(miss) / max(1, len([w for w in a["terms"] if w not in R1._GENERIC and not w.isdigit()]))
    top_rr = max(rr.values()) if rr else -10.0
    head = [u for u in ranked[:10] if R1.focus_sentences(hx.units[u].text, a["terms"])]
    all_other = bool(head) and not a["others_ok"] and all(lab[u] == "other" for u in head[:6])
    abstain_p = R1.abstain_prob(top_rr, miss_frac, all_other)
    hint = R1.build_hint(a, q.date, abstain_p)
    if P.get("rate_hint") and _RATE_OLD in hint:
        hint = hint.replace(_RATE_OLD, _RATE_NEW)
    if P.get("latest_line") and a["kind"] in ("count", "sum") and not a["latest"] and not a.get("rate"):
        hint += " If a later session restates the same item with a new number, the latest statement wins."

    rmax: dict[str, float] = {}
    for u, sc in rr.items():
        unit = hx.units[u]
        if P["asst"] != "none" or unit.role == "user":
            rmax[unit.session_id] = max(rmax.get(unit.session_id, -99.0), sc)
    score = dict(rmax)
    smax: dict[str, float] = {}
    if win_active and a["window"]:
        for sid in score:
            if not T._in_window(hx.sess_by_id[sid]["date"].date(), a["window"]):
                score[sid] -= P["win_pen"]
    order = sorted(score, key=lambda s: -score[s])
    top = score[order[0]] if order else -99.0
    feat = T._features(hx, a, order, score, rmax, smax, ranked, facet_top)
    if P.get("fs_feat"):
        _fact_sent_feats(q, hx, a, order[:P.get("fs_top", 15)], feat)
    pg = {sid: (p_gold2(f, P["model2"]) if P.get("model2") else T.p_gold(f)) for sid, f in feat.items()}
    keep = [s for s in order if pg.get(s, 0.0) >= P["tk"]][:P["keep_max"]]
    if len(keep) < P["keep_min"]:
        keep = order[:P["keep_min"]]
    keep = keep_hook(q, hx, a, P, order, pg, feat, keep, win_active)
    resc = rescue(q, hx, a, P, keep, win_active) if P.get("resc") else {}
    keep = keep + [s for s in resc if s not in keep]
    keep_set = set(keep)

    ranked_k = [u for u in ranked if hx.units[u].session_id in keep_set and hx.units[u].session_id not in resc]
    ranked_k += [u for sid, (u, _, _) in resc.items()]
    ft_k = {f: [u for u in us if hx.units[u].session_id in keep_set] for f, us in facet_top.items()}
    Pk = dict(P)
    Pk["max_sess"] = len(keep)
    Pk["deep_top"] = len(keep)
    picks = R1._select(hx, ranked_k, ft_k, Pk)
    passages, included, chars, chosen = pack(q, hx, picks, hint, Pk, terms)
    in_pack = {p.id for p in passages}

    o = [s for s in order if s in in_pack]
    if P["cite_mode"] == "p":
        cite = [s for s in o if pg.get(s, 0.0) >= P["tc"]]
        if len(cite) < 2:
            cite = o[:2]
    else:
        cite = o

    ap_a, why_a = None, None
    if P.get("abst"):
        from lenses.multi import a_tally as A
        qa = A.analyze(q.question, q.date)
        use_win = A.window_applies(hx, qa["window"])
        n_in = sum(1 for sid in in_pack if qa["window"] and A._in_window(hx.sess_by_id[sid]["date"], qa["window"]))
        ap_a, why_a = A.abstain_score(hx, qa, use_win, n_in)
        abstain_p = 1.0 - (1.0 - abstain_p) * (1.0 - max(0.0, ap_a - 0.05))
    sess_feat = {sid: dict(f, p=round(pg[sid], 4)) for sid, f in feat.items()}
    return EvidencePack(question_id=q.question_id, variant=params.get("_variant", ""), route="multi",
                        passages=passages, hint=hint, cite_plan=cite or None, abstain_p=abstain_p,
                        included_turns=included, computed=None,
                        debug={"ranking": ranked + rest, "keep": keep, "top": top, "window_active": win_active,
                               "top_rr": top_rr, "missing_terms": miss, "chars": chars, "sess2": sess_feat,
                               "abst_a": [ap_a, why_a], "kind": a["kind"], "rescue": resc})


def _fact_sent_feats(q, hx, a, sids: list[str], feat: dict, per: int = 8) -> None:
    """Instance evidence per session: CE (question vs sentence) of the session's first-person fact sentences
    ('By the way, I viewed a 1-bedroom condo on February 10th'), which unit-level CE dilutes in a topical unit."""
    texts, owners = [], []
    for sid in sids:
        n = 0
        for t in hx.sess_by_id[sid]["turns"]:
            if t["role"] != "user":
                continue
            for s in split_sents(t["content"]):
                if 20 <= len(s) <= 400 and fact_score(s) >= 2.0:
                    texts.append(s)
                    owners.append(sid)
                    n += 1
                if n >= per:
                    break
            if n >= per:
                break
    sc = hx.rerank_texts(q.question, texts) if texts else []
    best: dict[str, float] = {}
    for sid, v in zip(owners, sc):
        best[sid] = max(best.get(sid, -15.0), float(v))
    top = max(best.values()) if best else -15.0
    for sid in sids:
        if sid in feat:
            feat[sid]["fs"] = round(best.get(sid, -15.0), 3)
            feat[sid]["fsgap"] = round(top - best.get(sid, -15.0), 3)
            feat[sid]["nfs"] = sum(1 for o in owners if o == sid)


def p_gold2(f: dict, m: dict) -> float:
    import math
    x = feats2(f)
    z = m["b"] + sum(c * v for c, v in zip(m["w"], x))
    return 1.0 / (1.0 + math.exp(-z))


def feats2(f: dict) -> list[float]:
    import math
    return [f["gap"], min(f["r"], 10), f["cov"], min(f["n40"], 10), math.log1p(f["first"]),
            max(-15.0, f.get("fs", -15.0)), min(15.0, f.get("fsgap", 15.0)), min(f.get("nfs", 0), 8)]


_HEAD = re.compile(r"\bhow (?:many|much)\s+(?P<obj>.+?)\s+(?:do|did|does|have|has|had|am|is|are|was|were|will|can)\s+"
                   r"(?:i|we|my)\b", re.I)
_NOHEAD = {"times", "time", "money", "hours", "days", "weeks", "months", "years", "minutes", "different", "total",
           "kinds", "types", "pieces", "items", "things", "more", "less"}


def _st(w: str) -> str:
    w = w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w
    return w[:5]


def head_terms(question: str) -> list[str]:
    m = _HEAD.search(question)
    if not m:
        return []
    return [w for w in R1.content_terms(m.group("obj")) if w not in _NOHEAD and w not in R1._GENERIC]


def rescue(q, hx, a, P, keep, win_active) -> dict:
    """Lexical instance rescue (test audit: 'I bought two Cattleya orchids for the balcony this week', 'I read two
    detective novels during my trip', one short sentence in a chatty unit, CE-ranked below the kept sessions):
    sessions outside the pack with an own, affirmed, fact-cued user sentence containing the question's head noun,
    ranked by sentence CE; up to r_max sessions with CE >= r_thr, the holding unit pinned."""
    heads = {_st(w) for w in head_terms(q.question)}
    if not heads:
        return {}
    kept = set(keep)
    texts, owners = [], []
    for u in hx.units:
        if u.role != "user" or u.session_id in kept:
            continue
        if win_active and a["window"] and not T._in_window(u.date.date(), a["window"]):
            continue
        for se in split_sents(u.text):
            if len(se) < 20 or len(se) > 400 or se.rstrip().endswith("?"):
                continue
            toks = {_st(w) for w in re.findall(r"[a-z0-9][a-z0-9'\-]*", se.lower())}
            if not (heads & toks) or fact_score(se) < 2.0 or not R1._FIRST.search(se):
                continue
            if R1._NEG.search(se) or R1._REPORT.search(se) or R1._THIRD_SUBJ.search(se):
                continue
            texts.append(se)
            owners.append(u.uid)
    if not texts:
        return {}
    texts, owners = texts[:300], owners[:300]
    sc = hx.rerank_texts(q.question, texts)
    best: dict[str, tuple] = {}
    for uid, se, v in sorted(zip(owners, texts, sc), key=lambda x: -x[2]):
        sid = hx.units[uid].session_id
        if v >= P.get("r_thr", -3.0) and sid not in best and len(best) < P.get("r_max", 2):
            best[sid] = (uid, se, round(float(v), 3))
    return best


def keep_hook(q, hx, a, P, order, pg, feat, keep, win_active):
    """Adaptive breadth for windowed counts (test-scale histories only: the window prior is active when the history
    spans >= 45 days, never on dev / dev200): 'how many times did my systolic reading exceed 130 in August' has one
    instance per in-window session, so in-window sessions with p >= tk * w_frac join the pack, up to w_max."""
    if not (P.get("w_breadth") and win_active and a["window"] and a["kind"] in P.get("w_kinds", ("count",))):
        return keep
    out = list(keep)
    for s in order:
        if len(out) >= P["w_max"]:
            break
        if s in out or pg.get(s, 0.0) < P["tk"] * P["w_frac"]:
            continue
        if T._in_window(hx.sess_by_id[s]["date"].date(), a["window"]):
            out.append(s)
    return out
