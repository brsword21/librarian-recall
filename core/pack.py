"""Ranked units -> passages under a character budget, and the prompt any LLM can answer from."""
from __future__ import annotations

from collections import OrderedDict

from core.types import Passage, Question

DEFAULT_BUDGET = 12000
# Budgets were tuned on the Reactive AI challenge message, which wraps the same passages in a longer fixed frame than
# render_prompt. Adding the difference keeps every pack identical to the submitted system.
_FRAME_EXTRA = {True: 374, False: 446}


def question_text(q: Question, hint: str = "") -> str:
    return f"{q.question}\n{hint.strip()}" if hint and hint.strip() else q.question


def render_prompt(question: str, passages: list[Passage], hint: str = "", question_date: str = "") -> str:
    """The pack as one prompt: question (+ the code's notes), its date, labelled passages, answering rules."""
    text = f"{question}\n{hint.strip()}" if hint and hint.strip() else question
    lines = [f"Question: {text.strip()}"]
    if question_date:
        lines.append(f"Today is {question_date}: answer as of that moment.")
    if passages:
        lines += ["", "Evidence:"]
        for n, p in enumerate(passages, 1):
            about = ([f"source: {p.id}"] if p.id else []) + ([f"date: {p.date}"] if p.date else [])
            source = f" ({', '.join(about)})" if about else ""
            lines.append(f"[E{n}]{source} {p.text}")
        lines += ["", "Answer only from the evidence above and cite the [E#] label of every passage you use. If the "
                      "evidence does not answer the question, say that you do not know."]
    return "\n".join(lines)


def message_chars(q: Question, passages: list[Passage], hint: str = "") -> int:
    return len(render_prompt(q.question, passages, hint, q.date_str)) + _FRAME_EXTRA[bool(passages)]


def render(hx, uids: list[int], whole_turns: bool = False) -> tuple[str, set[str]]:
    """Units of ONE session -> text: 'user: ...' / 'assistant: ...', '[...]' marks skipped material."""
    if whole_turns:
        turns = list(dict.fromkeys(hx.units[i].turn_id for i in sorted(uids)))
        uids = [j for t in turns for j in hx.turn_units(t)]
    uids = sorted(set(uids))
    parts, turns_seen = [], set()
    prev = None
    for i in uids:
        u = hx.units[i]
        turns_seen.add(u.turn_id)
        if prev is not None and i == prev + 1 and hx.units[prev].turn_id == u.turn_id:
            parts[-1] += " " + u.text
        else:
            if prev is not None and i != prev + 1:
                parts.append("[...]")
            parts.append(f"{u.role}: {u.text}")
        prev = i
    return "\n".join(parts), turns_seen


def build_pack(q: Question, hx, ranked, budget: int = DEFAULT_BUDGET, window: int = 1, order: str = "chrono",
               hint: str = "", max_sessions: int | None = None, whole_turns: bool = False, max_units: int | None = None,
               pinned=None, max_chars_per_session: int | None = None,
               ) -> tuple[list[Passage], set[str], int]:
    """Greedy: walk `ranked` (uids or (uid, score)), add each unit with +/-window neighbours from its session while the
    full message stays within `budget` chars. Returns (passages, included_turn_ids, message_chars).
    order='chrono' sorts passages by session date (LongMemEval reading tip); 'rank' keeps first-hit order.
    Priority is the order of `ranked`: a unit that does not fit is skipped and later (smaller) ones may still enter.
    pinned: uids added first, without neighbours (e.g. the cited pair's sentences), before `ranked` is walked.
    max_chars_per_session: cap on one session's rendered text, so a chatty session cannot eat the budget.
    window=0 with explicit neighbour uids in `ranked` is the supported way to hand-pick context."""
    chosen: "OrderedDict[str, set[int]]" = OrderedDict()   # session_id -> uids, insertion = first-hit order
    used = 0

    def passages_for(sel) -> list[Passage]:
        sids = list(sel)
        if order == "chrono":
            sids.sort(key=lambda s: hx.sess_by_id[s]["sess_idx"])
        out = []
        for sid in sids:
            text, _ = render(hx, sorted(sel[sid]), whole_turns)
            out.append(Passage(id=sid, date=hx.sess_by_id[sid]["date_str"], text=text))
        return out

    chars = message_chars(q, [], hint)
    n_pin = 0
    if pinned:
        pinned = [p[0] if isinstance(p, (tuple, list)) else p for p in pinned]
        n_pin = len(pinned)
        ranked = list(pinned) + list(ranked)
    for pos, item in enumerate(ranked):
        uid = item[0] if isinstance(item, (tuple, list)) else item
        sid = hx.units[uid].session_id
        if max_sessions is not None and sid not in chosen and len(chosen) >= max_sessions:
            continue
        add = set(hx.neighbors(uid, window, window)) if (window and pos >= n_pin) else {uid}
        if max_chars_per_session is not None:
            have = chosen.get(sid, set())
            if len(render(hx, sorted(have | add), whole_turns)[0]) > max_chars_per_session:
                add = {uid}
                if have and len(render(hx, sorted(have | add), whole_turns)[0]) > max_chars_per_session:
                    continue
        trial = OrderedDict((k, set(v)) for k, v in chosen.items())
        trial.setdefault(sid, set()).update(add)
        c = message_chars(q, passages_for(trial), hint)
        if c > budget:
            if len(add) > 1:  # retry without neighbours
                trial = OrderedDict((k, set(v)) for k, v in chosen.items())
                trial.setdefault(sid, set()).add(uid)
                c = message_chars(q, passages_for(trial), hint)
            if c > budget:
                continue
        chosen, chars = trial, c
        used += 1
        if (max_units is not None and used >= max_units) or budget - chars < 150:
            break
    passages = passages_for(chosen)
    included: set[str] = set()
    for sid, uids in chosen.items():
        included |= render(hx, sorted(uids), whole_turns)[1]
    return passages, included, chars
