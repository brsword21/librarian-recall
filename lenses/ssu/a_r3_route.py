"""SSU-A round 3: the ssu ROUTE lens (wrapper), building on the frozen lenses/ssu/merged_r2.py role8_r3rn.

BEST: `kg3crga` (lenses/ssu/a_report.md). Routed ssu dev S 86.1 -> 87.7, dev200 82.3 -> 85.0; gold-type ssu unchanged;
1 of 45 test ssu packs changes (q_f14893711b1a "What is my move date?" -> ku pair Sept 2 / Sept 16).
Rejected: `kg3crgt` (tense fix) moves 3 test R-77 cites to wrong sessions.

Order of decisions (debug: route_final, p_assistant, kugate):
  1. role_guess (SSA-B, lenses/ssa/b_role.py) >= 0.8 -> ssa lens (unchanged from v1).
  2. NEW typed value-disagreement gate (round-3 brief item 1) -> ku lens (lenses/ku/b_r2.py r2b).
     KU-B's ku_pair_p fires when its two best own-attribute sessions carry different value TOKENS; on test that is
     mostly planted decoys and drift (friend's starter "Bubbles", Francesca's cafe, a "briefly tried 68 uF" capacitor,
     a "12-volt lamp ... too dim", the friend's restaurant). The gate keeps KU-B's pair and adds:
       a. both pair sentences are affirmed own statements under SSU-A's classifier (a_owner.classify: owner is not
          mismatch, polarity is not neg / plan; REJECT_RX turns 'briefly tried ... removed it' into plan);
       b. both sentences carry a value of the SAME type (clock time, weekday, calendar date, money, number+unit) and
          the two value sets of that type are DISJOINT (an update, not a repetition);
       c. names never count as a value type (name drift and sibling decoys are names: Fig/Bica/Sombra, Bubbles/Masa).
  3. otherwise the ssu lens (a_struct2 r3rn), optionally with a compositional runner-up cite (see `ru2`).
"""
from __future__ import annotations

import re

from core.loader import load_lens

_loaded: dict = {}


def _lens(path: str):
    if path not in _loaded:
        _loaded[path] = load_lens(path)
    return _loaded[path]


# ---- typed values -------------------------------------------------------------------------------------------
_W2N = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
        "a couple of": 2, "a few": 3, "half": 0.5}
_NUMW = r"(?:\d+(?:[.,]\d+)?|" + "|".join(sorted((re.escape(k) for k in _W2N), key=len, reverse=True)) + r")"
_MONTHS = "january|february|march|april|may|june|july|august|september|october|november|december"
_DAYS = "monday|tuesday|wednesday|thursday|friday|saturday|sunday"
TIME_RX = re.compile(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm|a\.m\.|p\.m\.)|\b(\d{1,2}):(\d{2})\b", re.I)
DATE_RX = re.compile(rf"\b(?:({_MONTHS})\s+(\d{{1,2}})(?:st|nd|rd|th)?|(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?({_MONTHS}))\b", re.I)
DAY_RX = re.compile(rf"\b({_DAYS})s?\b", re.I)
MONEY_RX = re.compile(r"\$\s?(\d[\d,]*(?:\.\d+)?)\s*(k|thousand|million)?", re.I)
UNIT_RX = re.compile(rf"\b({_NUMW})\s*-?\s*(minutes?|mins?|hours?|hrs?|days?|weeks?|months?|years?|miles?|km|"
                     r"kilomet(?:er|re)s?|k\b|lbs?|pounds?|kg|kilos?|ohms?|k-?ohms?|volts?|v\b|µf|uf|mf|watts?|w\b|"
                     r"%|percent|times|laps|pages|books|reps|sets|cups?|tablespoons?|ounces?|oz|inches|feet|ft|"
                     r"degrees?|seconds?|secs?|people|guests|members|followers|episodes|chapters|kilometers)", re.I)


def _num(s: str) -> float | None:
    s = s.lower().replace(",", "")
    if s in _W2N:
        return float(_W2N[s])
    try:
        return float(s)
    except ValueError:
        return None


def _unit(u: str) -> str:
    u = u.lower().rstrip("s")
    return {"min": "minute", "hr": "hour", "lb": "pound", "kilo": "kg", "kilometer": "km", "kilometre": "km",
            "k-ohm": "kohm", "kohm": "kohm", "uf": "µf", "mf": "µf", "v": "volt", "w": "watt", "percent": "%",
            "sec": "second", "ft": "feet", "foot": "feet", "oz": "ounce"}.get(u, u)


def typed_values(sent: str, question: str = "") -> dict[str, set]:
    """{type: set(values)} for clock times, calendar dates, weekdays, money and number+unit. Values that appear in the
    question are dropped (the question's own anchor: 'Saturday mornings', 'charity 5K', 'R-77')."""
    ql = question.lower()
    out: dict[str, set] = {}

    def add(t, v):
        out.setdefault(t, set()).add(v)
    for m in TIME_RX.finditer(sent):
        if m.group(1):
            h, mi, ap = int(m.group(1)), int(m.group(2) or 0), m.group(3).lower().replace(".", "")
            h = h % 12 + (12 if ap == "pm" else 0)
            add("time", (h, mi))
        else:   # no am/pm: a clock time or a duration such as a 27:12 run time; compared only with its own kind
            h, mi = int(m.group(4)), int(m.group(5))
            if mi <= 59:
                add("hms", (h, mi))
    for m in DATE_RX.finditer(sent):
        mon = (m.group(1) or m.group(4)).lower()
        add("date", (mon, int(m.group(2) or m.group(3))))
    for m in DAY_RX.finditer(sent):
        d = m.group(1).lower()
        if d not in ql:
            add("weekday", d)
    for m in MONEY_RX.finditer(sent):
        v = _num(m.group(1))
        if v is not None:
            mult = {"k": 1e3, "thousand": 1e3, "million": 1e6}.get((m.group(2) or "").lower(), 1)
            add("money", v * mult)
    for m in UNIT_RX.finditer(sent):
        if re.search(re.escape(m.group(0).lower()), ql):
            continue
        v = _num(m.group(1))
        if v is not None:
            add("n:" + _unit(m.group(2)), v)
    return out


GIVEN_RX = re.compile(r"\bconsidering (?=(?:that |i |i'm |my |we |how |the fact))", re.I)


def own_view(sent: str) -> str:
    """'..., considering I like to wake up at 7:30 am' states a fact ('given that'), it is not a plan; PLAN_RX in
    a_owner treats every 'considering' as a plan marker, so it is rewritten before the own / polarity check."""
    return GIVEN_RX.sub("given ", sent)


def disagree(ta: dict, tb: dict) -> list[str]:
    """Types present in both sentences whose value sets are disjoint."""
    return [t for t in ta if t in tb and not (ta[t] & tb[t])]


def ku_gate(q, hx, min_score: float = 6.0, min_cov: float = 0.5, need_own: bool = True,
            gv: bool = False) -> tuple[bool, dict]:
    kb = _lens("lenses/ku/b_r2.py")
    from lenses.ssu.a_owner import classify, parse_question
    p = dict(kb.VARIANTS["r2b"])
    r = kb.B.analyze(q, hx, **p)
    ok, f = kb.confidence(r, hx, q.question, dict(p, gate_v=2))
    info: dict = {}
    pair = r.get("pair") or []
    if len(pair) < 2 or "pair_sc" not in f:
        return False, {"why": "no_pair"}
    info["pair_sc"], info["cov_min"] = f["pair_sc"], f["cov_min"]
    if min(f["pair_sc"]) < min_score or f["cov_min"] < min_cov:
        info["why"] = "weak"
        return False, info
    bbs = r["best_by_sess"]
    raws = [bbs[s]["raw"] for s in pair]
    qp = parse_question(q.question)
    if need_own:
        for s in raws:
            sc = classify(own_view(s), qp) if gv else classify(s, qp)
            if sc.owner == "mismatch" or sc.polarity in ("neg", "plan"):
                info["why"] = f"not_own:{sc.owner}/{sc.polarity}"
                return False, info
    tv = [typed_values(s, q.question) for s in raws]
    d = disagree(tv[0], tv[1])
    info["types"] = d
    if not d:
        info["why"] = "no_typed_disagreement"
        return False, info
    info["why"] = "fire"
    return True, info


def runner_gate(q, hx, pack) -> tuple[bool, dict]:
    """a_struct2's primary (first cite) and outlier runner-up: best units are affirmed own head statements whose
    typed values disagree -> an update pair (or a second instance) -> cite both."""
    from lenses.ssu.a_owner import parse_question, unit_class
    d = pack.debug or {}
    run = d.get("runner")
    prim = (pack.cite_plan or [None])[0]
    if not run or not prim or run == prim:
        return False, {"why": "no_runner"}
    best: dict = {}
    for uid in d.get("ranking", []):
        best.setdefault(hx.units[uid].session_id, uid)
    if prim not in best or run not in best:
        return False, {"why": "no_units"}
    qp = parse_question(q.question)
    scs = [unit_class(hx.units[best[s]], qp) for s in (prim, run)]
    for sc in scs:
        if sc.owner == "mismatch" or sc.polarity in ("neg", "plan") or not sc.head_hit:
            return False, {"why": f"not_own_head:{sc.owner}/{sc.polarity}/{sc.head_hit}"}
    dis = disagree(typed_values(scs[0].sent, q.question), typed_values(scs[1].sent, q.question))
    return bool(dis), {"why": "fire" if dis else "no_typed_disagreement", "types": dis}


# ---- route lens ---------------------------------------------------------------------------------------------
_BASE = {"ssu_lens": "lenses/ssu/a_struct2.py", "ssu_variant": "r3rn",
         "ssa_lens": "lenses/ssa/a_recsent.py", "ssa_variant": "r2_conf2",
         "ku_lens": "lenses/ku/b_r2.py", "ku_variant": "r2b",
         "role_module": "lenses/ssa/b_role.py", "thr": 0.8, "kugate": None}
VARIANTS = {
    "v1": dict(_BASE),                                                     # = merged_r2 role8_r3rn
    "kg6": dict(_BASE, kugate=dict(min_score=6.0, min_cov=0.5)),           # KU-B thresholds + typed own gate
    "kg4": dict(_BASE, kugate=dict(min_score=4.0, min_cov=0.4)),           # relaxed scores (typed gate carries it)
    "kg3": dict(_BASE, kugate=dict(min_score=3.0, min_cov=0.4)),
    "kg6_any": dict(_BASE, kugate=dict(min_score=6.0, min_cov=0.5, need_own=False)),   # ablation: no own check
    # + compositional misroutes (multi / temporal -> ssu: 'how old was I when', 'the two novels', 'X and the Y',
    # 'increase'): the outlier runner-up is gold on 9/10 such dev questions -> cite it too (ru_cite all).
    # 0 test ssu-route questions match COMP_RX (checked on the 45 test packs), so this is dev / robustness only.
    "kg4c": dict(_BASE, kugate=dict(min_score=4.0, min_cov=0.4), comp=dict(ru_cite="all")),
    "kg3c": dict(_BASE, kugate=dict(min_score=3.0, min_cov=0.4), comp=dict(ru_cite="all")),
    # + runner gate: a_struct2's outlier runner-up is cited too when its best unit and the primary's best unit are
    # affirmed own statements with a head hit and a typed value disagreement (same gate as ku_gate, on SSU-A's pair).
    # Offline (scratch_a/r3_rugate.py, without the head check): dev 7 firings (6 ku + 1 multi, runner gold in 7),
    # dev200 3 (1 gold; 'daily commute 45 min' vs 'journaling 15-30 min' -> the head check), test 0 of 45.
    "kg3r": dict(_BASE, kugate=dict(min_score=3.0, min_cov=0.4), rgate=True),
    "kg4r": dict(_BASE, kugate=dict(min_score=4.0, min_cov=0.4), rgate=True),
    "kg3cr": dict(_BASE, kugate=dict(min_score=3.0, min_cov=0.4), comp=dict(ru_cite="all"), rgate=True),
    # + 'considering I ...' read as 'given that' in the gate's own / polarity check
    "kg3crg": dict(_BASE, kugate=dict(min_score=3.0, min_cov=0.4, gv=True), comp=dict(ru_cite="all"), rgate=True),
    # + a gate firing is answerability evidence (two affirmed own typed statements): cap abstain_p of the ku pack
    # (the ku lens' abstain_p is calibrated on ku-routed questions and lowered the routed AUC 0.931 -> 0.914)
    "kg3crga": dict(_BASE, kugate=dict(min_score=3.0, min_cov=0.4, gv=True), comp=dict(ru_cite="all"), rgate=True,
                    ab_fire=0.05),
    # + tense fix (brief item 3): in 'did I / was I' questions a unit with a plan clause gets the future penalty even
    # when it also carries a past marker ('I'm thinking of trying X ... last week'); before, the past bonus won.
    "kg3crgt": dict(_BASE, kugate=dict(min_score=3.0, min_cov=0.4, gv=True), comp=dict(ru_cite="all"), rgate=True,
                    tense="plan_first"),
}


def _tense_plan_first(u, sc, q_past: bool, p) -> float:
    """a_struct.tense_adjust with the plan clause outranking the past marker."""
    from lenses.ssu.a_struct import FUT_RX, PAST_MARK_RX, WH_ASK_RX, norm
    d = 0.0
    tn = norm(u.text)
    if p["pen_fut"] and q_past:
        fut, past = bool(FUT_RX.search(tn)), bool(PAST_MARK_RX.search(tn))
        if fut:
            d -= p["pen_fut"]
        elif past:
            d += p["b_past"]
    if p["pen_ask"] and sc is not None and WH_ASK_RX.match(sc.sent) and not re.search(r"\d|\b[A-Z][a-z]+\b",
                                                                                    sc.sent[1:]):
        d -= p["pen_ask"]
    return d
COMP_RX = re.compile(r"\b(how old (?:was|will|am) i when|when (?:i|my|\w+) (?:was|were|got|get|moved|started)|the two|"
                     r"both|\w+ and the \w+|increase|decrease|difference|between)\b", re.I)


def choose(q, hx, thr: float = 0.8, kugate: dict | None = None,
           role_module: str = "lenses/ssa/b_role.py") -> tuple[str, dict]:
    info: dict = {}
    try:
        p = _lens(role_module).role_guess(q, hx)
        info["p_assistant"] = round(p, 4)
        if p >= thr:
            return "ssa", info
    except Exception as exc:
        info["role_error"] = repr(exc)[:200]
    if kugate is not None:
        try:
            hit, f = ku_gate(q, hx, **kugate)
            info["kugate"] = f
            if hit:
                return "ku", info
        except Exception as exc:
            info["kugate_error"] = repr(exc)[:200]
    return "ssu", info


def build(q, hx, _variant: str = "", ssu_lens: str = "", ssu_variant: str = "", ssa_lens: str = "",
          ssa_variant: str = "", ku_lens: str = "", ku_variant: str = "", role_module: str = "lenses/ssa/b_role.py",
          thr: float = 0.8, kugate: dict | None = None, comp: dict | None = None, rgate: bool = False,
          tense: str | None = None, ab_fire: float | None = None, **_):
    r, info = choose(q, hx, thr=thr, kugate=kugate, role_module=role_module)
    path, v = {"ssa": (ssa_lens, ssa_variant), "ku": (ku_lens, ku_variant)}.get(r, (ssu_lens, ssu_variant))
    lens = _lens(path)
    params = dict(lens.VARIANTS[v])
    if r == "ssu" and comp and COMP_RX.search(q.question):
        params.update(comp)
        info["comp"] = True
    # the tense fix patches THIS wrapper's private copy of a_struct2 (load_lens gives every caller its own module
    # object) for the duration of the call only; the frozen v1 lenses never see it
    orig = getattr(lens, "tense_adjust", None)
    if r == "ssu" and tense == "plan_first" and orig is not None:
        lens.tense_adjust = _tense_plan_first
    try:
        pack = lens.build(q, hx, _variant=v, **params)
    finally:
        if orig is not None:
            lens.tense_adjust = orig
    if r == "ssu" and rgate and params.get("ru_cite", "no") == "no":
        try:
            hit, rg = runner_gate(q, hx, pack)
            info["rgate"] = rg
            if hit:
                pack = lens.build(q, hx, _variant=v, **dict(params, ru_cite="all"))
        except Exception as exc:
            info["rgate_error"] = repr(exc)[:200]
    if r == "ku" and ab_fire is not None and pack.abstain_p is not None:
        info["abstain_p_inner"] = pack.abstain_p
        pack.abstain_p = min(pack.abstain_p, ab_fire)
    pack.route = r
    pack.debug = {**(pack.debug or {}), "route_final": r, "inner_lens": path, "inner_variant": v, **info}
    return pack
