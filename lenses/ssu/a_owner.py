"""Lens A "Struktura" for single-session-user (ssu): owner-aware structural reranking.

Idea: retrieval finds the topic; rules decide WHOSE fact it is and whether it actually happened.
Test near-misses are planted as possessor swaps ("My friend Ines booked her birthday dinner at ..."),
negations / plans ("..., but I haven't tried it yet", "I'm thinking about kegging ...") and update
chains ("I've renamed my starter ...", "I returned the X and bought a Y instead"). A cross-encoder
scores all of those as perfect matches, so we re-weight candidates with:

  owner     possessor chain of the head noun vs the question's chain (my cat / my brother's dog)
  polarity  negated / planned / stale ("used to", "old", "still says") vs affirmed / new
  recency   "end up", "last ... I mentioned", "now", "current", "these days" -> latest valid mention
  reply     user turn that answers the assistant's question ("What's her name?" -> "Sombra. ...")
  aside     "By the way / Oh, and / Also," disclosure cue (lift ~3x for gold turns on dev)

Pure-python helpers (question grammar, sentence classifier, value extraction, hint builder) have a
self-test that runs without core.index:   .venv/bin/python lenses/ssu/a_owner.py
"""
# (no `from __future__ import annotations`: dataclasses then need the module in sys.modules, which spec-loading skips)

import math
import re
from dataclasses import dataclass, field

from core.types import EvidencePack

try:  # same sentence splitter as the core chunker
    from core.units import sentences as _core_sentences
except Exception:  # pragma: no cover
    _core_sentences = None


def sentences(text: str) -> list[str]:
    if _core_sentences is not None:
        return _core_sentences(text)
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


# ----------------------------------------------------------------------------------------------------------------
# lexicon
# ----------------------------------------------------------------------------------------------------------------
_SPELL = [(r"colour", "color"), (r"favourite", "favorite"), (r"neighbour", "neighbor"), (r"theatre", "theater"),
          (r"centre", "center"), (r"organis", "organiz"), (r"jewellery", "jewelry"), (r"\bgrey\b", "gray"),
          (r"\bmum\b", "mom"), (r"\bpyjama", "pajama"), (r"travell", "travel")]

_IRREG = {"bought": "buy", "got": "get", "gotten": "get", "took": "take", "taken": "take", "went": "go", "gone": "go",
          "ate": "eat", "eaten": "eat", "gave": "give", "given": "give", "made": "make", "met": "meet", "paid": "pay",
          "sold": "sell", "wrote": "write", "written": "write", "ran": "run", "swam": "swim", "drove": "drive",
          "flew": "fly", "left": "leave", "found": "find", "began": "begin", "begun": "begin", "saw": "see",
          "seen": "see", "brought": "bring", "chose": "choose", "chosen": "choose", "taught": "teach",
          "caught": "catch", "thought": "think", "spent": "spend", "won": "win", "held": "hold", "kept": "keep",
          "lost": "lose", "told": "tell", "said": "say", "sent": "send", "built": "build", "did": "do", "done": "do",
          "had": "have", "has": "have", "was": "be", "were": "be", "is": "be", "are": "be", "am": "be", "been": "be",
          "felt": "feel", "heard": "hear", "knew": "know", "known": "know", "learnt": "learn", "read": "read",
          "rode": "ride", "ridden": "ride", "sang": "sing", "sung": "sing", "slept": "sleep", "stood": "stand",
          "understood": "understand", "wore": "wear", "worn": "wear", "baked": "bake", "named": "name"}

# relations that can own things ("my friend's dog", "My brother adopted ...")
REL = {"friend", "friends", "neighbor", "neighbors", "brother", "sister", "mom", "mother", "dad", "father", "wife",
       "husband", "boyfriend", "girlfriend", "fiance", "fiancé", "fiancee", "fiancée", "son", "sons", "daughter",
       "daughters", "coworker", "co-worker", "coworkers", "colleague", "colleagues", "boss", "manager", "cousin",
       "uncle", "aunt", "niece", "nephew", "grandma", "grandmother", "grandpa", "grandfather", "roommate", "flatmate",
       "housemate", "classmate", "teammate", "partner", "kid", "kids", "child", "children", "teen", "teens",
       "sibling", "parents", "parent", "in-laws", "mentor", "student", "client", "landlord", "landlady", "tutor",
       "teacher", "instructor", "coach", "doctor", "therapist", "boyfriend's", "neighbour"}

# head-noun synonyms (rule-based query expansion); keys are normalised (US spelling, lower case)
SYN = {
    "cat": ["cat", "kitten", "kitty"], "kitten": ["kitten", "cat"],
    "dog": ["dog", "puppy", "pup"], "puppy": ["puppy", "dog", "pup"],
    "toddler": ["toddler", "little one"],
    "partner": ["partner", "husband", "wife", "boyfriend", "girlfriend", "spouse", "fiancé", "fiance", "fiancée"],
    "husband": ["husband", "partner", "spouse"], "wife": ["wife", "partner", "spouse"],
    "restaurant": ["restaurant", "bistro", "trattoria", "diner", "eatery", "pizzeria", "steakhouse", "brasserie",
                   "taqueria", "izakaya", "bar-b-q", "bbq joint"],
    "café": ["café", "cafe", "coffee shop", "coffeehouse", "coffee place"],
    "cafe": ["cafe", "café", "coffee shop", "coffeehouse", "coffee place"],
    "shop": ["shop", "store"], "store": ["store", "shop"],
    "neighborhood": ["neighborhood", "district", "area", "barrio", "quarter"],
    "instructor": ["instructor", "teacher", "coach", "tutor", "taught by", "teaches"],
    "teacher": ["teacher", "instructor", "tutor", "taught by", "teaches"],
    "tutor": ["tutor", "teacher", "instructor"],
    "apartment": ["apartment", "flat"], "flat": ["flat", "apartment"],
    "bicycle": ["bicycle", "bike"], "bike": ["bike", "bicycle"],
    "mom": ["mom", "mother"], "mother": ["mother", "mom"], "dad": ["dad", "father"], "father": ["father", "dad"],
    "grandma": ["grandma", "grandmother", "nana"], "grandmother": ["grandmother", "grandma", "nana"],
    "kids": ["kids", "children"], "children": ["children", "kids"],
    "film": ["film", "movie"], "films": ["films", "movies"], "movie": ["movie", "film"],
    "job title": ["job title", "title", "position", "role", "promoted to"],
    "job": ["job", "position", "role", "work as"],
    "physiotherapist": ["physiotherapist", "physio", "physical therapist"],
    "doctor": ["doctor", "dr."], "dentist": ["dentist", "dental"],
    "podcast": ["podcast"], "playlist": ["playlist"],
    "rent": ["rent"], "salary": ["salary", "pay"],
    "beer": ["beer", "ale", "lager", "ipa", "stout", "saison", "porter", "pilsner", "brew"],
    "book": ["book", "novel", "reading"],
    "grill": ["grill", "smoker", "kamado"],
    "car": ["car", "sedan", "hatchback", "suv"],
    "firm": ["firm", "practice", "office"], "company": ["company", "firm", "contractor", "builders"],
    "group": ["group", "club", "crew"], "club": ["club", "group"],
    "museum": ["museum", "gallery"], "shelter": ["shelter", "rescue"],
    "commute": ["commute", "drive to work"],
    "last-name": ["last name", "surname", "maiden name", "old name"], "first-name": ["first name"],
    "maiden-name": ["maiden name", "last name"], "nick-name": ["nickname"],
}

ATTR = {"name", "brand", "breed", "color", "type", "kind", "size", "speed", "model", "style", "title", "value",
        "ratio", "price", "cost", "age", "date", "time", "make", "flavor", "genre", "species", "variety", "number",
        "percentage", "discount", "amount", "voltage", "impedance", "frequency", "resistance", "capacity", "wattage",
        "capacitance", "rating", "length", "weight", "height", "distance"}

STOP = set("""a an the of to in on at for from with by about as into my me i i'm i've i'd we our us you your yours he
she it they them what which who whom whose where when why how is are was were be been being do does did done have has
had having can could will would should shall may might must that this these those there here and or but if so not no
than then too very just also any some all many much more most less few long often usually ever again ago recently
lately per every each around up end ended last first now currently current still these days told tell mention
mentioned say said remind forgot forget""".split())

TEMPORAL_MODS = {"new", "old", "current", "latest", "previous", "recent", "former", "first", "last", "now"}

NEG_WORDS = {"not", "never", "didn't", "don't", "doesn't", "haven't", "hasn't", "wasn't", "weren't", "won't",
             "couldn't", "can't", "cannot", "isn't", "aren't", "hadn't", "wouldn't", "no", "nor", "without"}
PLAN_RX = re.compile(r"\b(thinking (?:about|of)|planning(?: to| on)?|plan(?:s)? to|want(?:s|ed)? to|wanna|going to|"
                     r"gonna|might|considering|consider|hop(?:e|ing) to|tempted(?: to)?|would like to|'d like to|"
                     r"looking (?:for|into|at)|wondering (?:if|whether|about)|should i|maybe i'?ll|if i|someday|"
                     r"eventually|on my list|wish(?:list)?|debating|deciding)\b")
STALE_RX = re.compile(r"\b(used to|still says|old|former|formerly|previous(?:ly)?|originally|at first|ex-|"
                      r"no longer|anymore|returned|switched from|got rid of|sold my|gave away)\b")
NEW_RX = re.compile(r"\b(renamed|instead|switched to|replaced (?:\w+ ){0,4}with|now|these days|nowadays|"
                    r"upgraded to|changed (?:it )?to|ended up|finally)\b")
ASIDE_RX = re.compile(r"\b(by the way|oh,? and|also,|speaking of|quick aside|before i forget|"
                      r"on a (?:different|related|separate|happier|brighter|side|completely different) note|"
                      r"completely unrelated|totally unrelated|unrelated:|random (?:\w+ )?(?:note|news|update|thing)|"
                      r"\w+ update:|fun fact|side note)", re.I)
RECENCY_RX = re.compile(r"\b(end(?:ed)? up|now|current(?:ly)?|these days|nowadays|latest|most recent(?:ly)?|"
                        r"anymore|still|at the moment|right now|i last mentioned|last (?:\w+ ){0,4}(?:i |i'd |that i )"
                        r"(?:\w+ )?mention)", re.I)
PAST_RX = re.compile(r"\b(previous(?:ly)?|before i|used to|former|originally|initially|at first|old (?:name|job|"
                     r"address|car|phone|apartment|place|stance|instructor|teacher)|maiden)\b", re.I)
ASK_ASSISTANT_RX = re.compile(r"\byou (?:recommend|suggest|told|mention|gave|said|propos|advis|list)|\bdid you\b",
                              re.I)
GENERIC_OTHER_RX = re.compile(r"^(?:a|an|the|some|one|this|that)\s+(?:[\w-]+\s+){0,2}?(nurse|father|mother|guy|woman|"
                              r"man|person|lady|friend|coworker|colleague|neighbor|kid|parent|dad|mom|girl|boy|"
                              r"couple|family|member|stranger|customer|client|student|teacher|someone)\b")
PERSON_VERB = ("says|said|told|tells|took|made|used|bought|adopted|adopts|named|booked|tried|calls|called|went|ran|"
               "attended|submitted|registered|visited|cooked|baked|kegged|brewed|planted|renovated|recommended|"
               "mentioned|swears|is taking|was taking|just adopted|just bought|just booked")
NAME_SUBJ_RX = re.compile(r"^([A-Z][a-zà-ÿ'’]+)(?: [A-Z][a-zà-ÿ'’]+)? (?:" + PERSON_VERB + r")\b")
DISCOURSE_RX = re.compile(r"^(?:(?:oh|and|also|actually|honestly|anyway|well|so|plus|btw|ok|okay|yeah|yes|no|"
                          r"hey|hi|ugh|ha|haha|lol|right|sure|cool|great|thanks|thank you|funny enough|"
                          r"by the way|speaking of [^,:]+|on a [^,:]+ note|before i forget|quick aside|"
                          r"random [^,:]+|[\w ]+ update|completely unrelated|totally unrelated|one more thing|"
                          r"it's funny how time goes|that reminds me)[,:!.—–-]*\s+)+", re.I)
CLAUSE_RX = re.compile(r"\s*(?:[;:—–]|\s-\s|,\s*(?:but|and|so|while|whereas|although|though|because)\s|\bbut\b)\s*")
CAP_STOP = {"I", "The", "A", "An", "My", "Our", "We", "He", "She", "It", "They", "Yes", "No", "Oh", "Well", "So",
            "And", "But", "Also", "Hey", "Hi", "Thanks", "Thank", "Okay", "Ok", "Like", "Just", "That", "This",
            "Then", "Sure", "Great", "Ugh", "Haha", "Honestly", "Actually", "Anyway", "Maybe", "Probably",
            "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday", "January", "February",
            "March", "April", "May", "June", "July", "August", "September", "October", "November", "December",
            "Not", "Kind", "Sort", "Definitely", "Totally", "Exactly", "Perfect", "Good", "Right", "Wow", "Ha", "True",
            "Absolutely", "Yeah", "Yep", "Nope", "Same", "Agreed", "Fair", "Nice", "Cool", "Awesome", "Mostly",
            "Both", "Neither", "Either", "Only", "Probably", "Possibly", "Hmm", "Hm", "Lol", "Wait", "Fine"}


SLOTS = [{"monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "weekday", "weekend"},
         {"morning", "afternoon", "evening", "night", "midday", "lunchtime"},
         {"younger", "older", "elder", "eldest", "youngest", "oldest", "middle", "big", "little", "baby"},
         {"first", "second", "third", "fourth", "last", "final"},
         {"spring", "summer", "autumn", "fall", "winter"},
         {"front", "back", "upstairs", "downstairs"}]
REJECT_RX = re.compile(r"\b(?:briefly|temporarily) (?:tried|used|fitted|put|installed|swapped)|"
                       r"\bbut (?:it|they|the \w+|that|this|i|we)\b[^.]{0,80}?\b(?:not|n't|too|lost|weak|worse|worried|"
                       r"failed|hum|noisy|dim|wrong|back|returned|swapped|instead|gave up)\b|\b(?:swapped|went|switched) "
                       r"back\b|\breverted\b")
PERCEIVE_RX = re.compile(r"\b(?:saw|seen|see|heard|read|noticed|came across|watched|met|spotted|found out about|"
                         r"walked past|passed)\s+(?:about\s+)?(?:a|an|some|this)\s+(?:[\w-]+\s+){0,3}$")


def _slot_conflict(np_mods: list, qp) -> bool:
    qm = set(qp.mods)
    for g in SLOTS:
        want = qm & g
        have = set(np_mods) & g
        if want and have and not (have & want):
            return True
    return False


def norm(s: str) -> str:
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"').lower()
    for a, b in _SPELL:
        s = re.sub(a, b, s)
    return s


def stem(w: str) -> str:
    w = w.lower().strip("'\".,!?;:()")
    if w.endswith("'s"):
        w = w[:-2]
    if w in _IRREG:
        return _IRREG[w]
    for suf, rep in (("ying", "y"), ("ies", "y"), ("ied", "y"), ("ing", ""), ("ed", "")):
        if w.endswith(suf) and len(w) - len(suf) >= 2:
            w = w[: -len(suf)] + rep
            break
    else:
        if w.endswith("es") and len(w) > 4 and (w[-3] in "sxz" or w[-4:-2] in ("ch", "sh")):
            w = w[:-2]
        elif w.endswith("s") and not w.endswith("ss") and len(w) > 3:
            w = w[:-1]
    if len(w) > 3 and w[-1] == w[-2] and w[-1] not in "aeiouls":
        w = w[:-1]
    if len(w) > 3 and w.endswith("e"):
        w = w[:-1]
    return w


def toks(s: str) -> list[str]:
    return re.findall(r"[a-z0-9][a-z0-9'.-]*[a-z0-9]|[a-z0-9]", norm(s))


# ----------------------------------------------------------------------------------------------------------------
# question grammar
# ----------------------------------------------------------------------------------------------------------------
@dataclass
class QParse:
    text: str
    wh: str = ""
    attr: str = ""
    anchor: str = ""            # e.g. "sourdough starter"
    head: str = ""              # e.g. "starter"
    mods: list = field(default_factory=list)      # content modifiers of the head, e.g. ["sourdough"]
    chain: list = field(default_factory=list)     # possessor relations, e.g. ["brother"] for "my brother's dog"
    verbs: list = field(default_factory=list)     # stems of the question's content verbs, e.g. ["camp"]
    context: list = field(default_factory=list)   # other content terms (relative clause, PPs)
    recency: bool = False
    past_wanted: bool = False
    first_person: bool = True   # the question's subject is "I" (enables subject-based owner checks)
    ask_assistant: bool = False
    strong: bool = False        # anchor came from a reliable pattern (used for the abstention gate)
    source: str = ""

    def head_forms(self) -> list[str]:
        forms = list(SYN.get(self.anchor, [])) + list(SYN.get(self.head, [])) + [self.anchor, self.head]
        out = []
        for f in forms:
            if f and f not in out:
                out.append(f)
        return out


_NP = r"((?:[a-z][a-z0-9-]* ){0,4}?[a-z][a-z0-9-]*)"
_STOPNP = (r"(?= (?:i|i'm|i've|i'd|i'll|we|where|that|which|who|you|for|from|in|at|on|with|before|after|now|is|was|"
           r"were|are|did|do|does|every|each|per|these|this|to|when|while|since|during|lately|recently|currently|"
           r"right|so|by|and|or|gave|give|gives|got|bought|made|said|told|sent|took|has|had|lives|live|works|teaches|"
           r"usually|always|still)\b|[?.!,]|$)")


def _split_anchor(phrase: str) -> tuple[str, str, list[str]]:
    words = [w for w in phrase.split() if w not in ("the", "a", "an", "my", "our", "that", "this")]
    # "type of cocktail recipe" -> "cocktail recipe"
    while len(words) >= 3 and words[0] in ATTR and words[1] == "of":
        words = words[2:]
    words = [w for w in words if w not in TEMPORAL_MODS] or words
    if not words:
        return "", "", []
    anchor = " ".join(words)
    head = words[-1]
    mods = [w for w in words[:-1] if w not in STOP]
    return anchor, head, mods


def parse_question(question: str) -> QParse:
    raw = re.sub(r"\s*(?:I (?:forgot|lost track|can't remember|don't remember)[^.?!]*[.?!]?\s*)+$", "", question.strip())
    q = norm(raw)
    q = re.sub(r"\b(last|first|maiden|middle|full|nick) name\b", r"\1-name", q)
    qp = QParse(text=raw)
    m = re.match(r"\s*(how many|how much|how long|how old|how often|what time|what|which|where|who|whom|when|how)\b",
                 q)
    qp.wh = m.group(1) if m else ""
    qp.recency = bool(RECENCY_RX.search(q))
    qp.past_wanted = bool(PAST_RX.search(q))
    qp.ask_assistant = bool(ASK_ASSISTANT_RX.search(q))
    third = re.search(r"\b(?:did|does|do|has|is|was|had) ([A-Z][a-z]+)\b", raw[1:])
    qp.first_person = qp.wh not in ("who", "whom") and not third
    am = re.search(r"\b(name|brand|breed|color|type|kind|size|speed|model|style|title|value|ratio|price|age|"
                   r"date|make|flavor|genre)\b", q)
    qp.attr = am.group(1) if am else {"where": "place", "when": "date", "who": "person", "how many": "count",
                                      "how much": "amount", "how long": "duration", "what time": "time",
                                      "how old": "age"}.get(qp.wh, "")
    phrase, chain, src = "", [], ""
    m = re.search(r"\bmy " + _NP + r"'s " + _NP + r"'s (?:" + "|".join(sorted(ATTR)) + r")\b", q)
    if m:
        chain, phrase, src = [m.group(1).split()[-1]], m.group(2), "a:chain"
    # (a) "my X's Y" -- X's attribute (my cat's name) or X owns Y (my brother's new dog)
    for m in ([] if phrase else re.finditer(r"\bmy " + _NP + r"'s " + _NP + _STOPNP, q)):
        x, y = m.group(1), m.group(2)
        y_words = y.split()
        if y_words and (y_words[-1] in ATTR or y_words[-1] in TEMPORAL_MODS):
            phrase, src = x, "a:attr"
        else:
            xh = x.split()[-1]
            if xh in REL:
                chain = [xh]
                phrase = y
            else:  # "my bachelor's degree" -> anchor "bachelor degree"
                phrase = x + " " + y
            src = "a:owner"
        break
    # (b) "name of the X", "brand of the X"
    if not phrase:
        m = re.search(r"\b(?:name|brand|title|breed|style|type|kind|model) of (?:the |my |that |this |our )?" + _NP
                      + _STOPNP, q)
        if m:
            phrase, src = m.group(1), "b:of"
    # (c) "last X I mentioned", "the X I last mentioned"
    if not phrase:
        m = re.search(r"\blast " + _NP + r" (?:i|i'd|that i) (?:last )?mention", q) or \
            re.search(r"\b(?:the|that) " + _NP + r" i last mention", q)
        if m:
            phrase, src = m.group(1), "c:last"
    # (h) "how many X do/did I ..."
    if not phrase:
        m = re.search(r"\bhow (?:many|much) " + _NP + r" (?:do|did|does|have|has|am|was|were|is|are) (?:i|my|we)\b", q)
        if m and m.group(1).split()[-1] not in ("time", "money"):
            phrase, src = m.group(1), "h:howmany"
    # (d) "what/which X did I ..."
    if not phrase:
        m = re.search(r"\b(?:what|which) " + _NP + r" (?:did|do|does|am|was|were|is|are|have|has|had) (?:i|my|we)\b",
                      q)
        if m and m.group(1).split()[-1] not in ATTR:
            phrase, src = m.group(1), "d:wh"
        elif m and len(m.group(1).split()) >= 2:  # "what dial lamp voltage did I ..." -> "dial lamp", attr voltage
            phrase, src = " ".join(m.group(1).split()[:-1]), "d:wh"
            qp.attr = m.group(1).split()[-1]
    # (e) "my X" (first one that is not an attribute)
    if not phrase:
        for m in re.finditer(r"\bmy " + _NP + _STOPNP, q):
            cand = m.group(1)
            cw = [w for w in cand.split() if w not in TEMPORAL_MODS]
            if cw and (cw[-1] not in ATTR or len(cw) >= 2):
                phrase, src = cand, "e:my"
                break
    # (g) object of the question verb: "did I redeem a $5 coupon", "do I take yoga classes"
    if not phrase:
        m = re.search(r"\b(?:did|do|have|had) i (?:\w+ ){1,2}?(?:a |an |the |some )?(?:\$?[\d.,]+ )?" + _NP + _STOPNP, q)
        if m and m.group(1).split()[0] not in STOP:
            phrase, src = m.group(1), "g:object"
    # (f) fallback: last content run
    if not phrase:
        words = [w for w in re.findall(r"[a-z][a-z0-9-]*", q) if w not in STOP and w not in ATTR]
        phrase, src = " ".join(words[-2:]), "f:fallback"
    mrel = re.search(r"\bmy (\w+) ([A-Z][a-z]+)\b", raw)
    if mrel and mrel.group(1).lower() in REL and src in ("e:my", "f:fallback", "g:object"):
        phrase, src = mrel.group(1).lower(), "e:my"
    qp.anchor, qp.head, qp.mods = _split_anchor(phrase)
    qp.chain = chain
    qp.source = src
    qp.strong = src in ("a:attr", "a:owner", "a:chain", "b:of", "c:last", "e:my", "h:howmany") or bool(re.search(r"\b[A-Z][a-z]{2,}",
                                                                                           raw[1:]))
    # "my sister Emily" -> keep the proper name as a modifier
    for m in re.finditer(r"\bmy (\w+) ([A-Z][a-z]+)\b", raw):
        if m.group(1).lower() in REL and m.group(2).lower() not in qp.mods:
            qp.mods.append(m.group(2).lower())
    # verbs: first content word after "did I / do I / I'd / mentioned / I"
    verbs = []
    qt = re.findall(r"[a-z][a-z'-]*", q)
    skip = {"just", "last", "first", "ever", "finally", "recently", "usually", "always", "really", "been", "actually",
            "still", "currently", "now", "once", "also", "up"}
    for i, t in enumerate(qt):
        if t in ("i", "i'd", "i've", "i'm", "we", "mentioned", "up"):
            j = i + 1
            while j < len(qt) and qt[j] in skip:
                j += 1
            if j < len(qt):
                v = qt[j]
                if v not in STOP and v not in ATTR and v not in ("mention", "mentioned", "told", "tell", "say",
                                                                 "said", "using", "use"):
                    verbs.append(stem(v))
    qp.verbs = [v for v in dict.fromkeys(verbs) if v and v != stem(qp.head)]
    content = [w for w in re.findall(r"[a-z][a-z0-9-]*", q) if w not in STOP and w not in ATTR and len(w) > 2]
    used = set(qp.anchor.split()) | set(qp.chain)
    qp.context = [w for w in content if w not in used and stem(w) not in qp.verbs]
    return qp


# ----------------------------------------------------------------------------------------------------------------
# sentence classifier
# ----------------------------------------------------------------------------------------------------------------
@dataclass
class SentClass:
    sent: str
    head_hit: bool = False
    mods_hit: int = 0
    verb_hit: bool = False
    owner: str = "unknown"       # match | mismatch | unknown
    owner_why: str = ""
    polarity: str = "aff"        # aff | neg | plan | stale | new
    aside: bool = False
    reply: bool = False
    values: list = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return self.owner != "mismatch" and self.polarity not in ("neg", "plan") and (self.head_hit or self.verb_hit
                                                                                      or self.reply)


def _form_rx(form: str) -> str:
    f = re.escape(norm(form))
    return r"(?<![a-z])" + f + r"(?:s|es)?(?![a-z])"


def _strip_discourse(s: str) -> str:
    prev = None
    while prev != s:
        prev = s
        s = DISCOURSE_RX.sub("", s, count=1)
    return s


def _subject_chain(clause_raw: str, qp: QParse) -> tuple[str, list[str]]:
    """Owner implied by the clause subject: ('self', []) | ('rel', [rels]) | ('other', []) | ('', [])."""
    c = _strip_discourse(clause_raw.strip())
    cl = norm(c)
    if re.match(r"(?:i|i'm|i've|i'd|i'll|we|we've|we're|we'd|we'll|me and|my family and i)\b", cl) or \
            re.match(r"(?:my|our) [\w\s'-]{0,30}? and i\b", cl):
        return "self", []
    m = re.match(r"(?:my|our) ((?:[\w-]+ ){0,2}?)([\w-]+)(?: [a-z]+)?(?:'s ((?:[\w-]+ ){0,2}?)([\w-]+))?", cl)
    if m:
        rel1 = m.group(2)
        rel2 = m.group(4)
        heads = {norm(f) for f in qp.head_forms()}
        if rel1 in REL and rel1 not in heads:
            chain = [rel1]
            if rel2 and rel2 in REL and rel2 not in heads:
                chain.append(rel2)
            if m.group(1).strip() and qp.head in REL:  # "my running partner" while asking about "my partner"
                chain = chain or [m.group(1).strip()]
            return "rel", chain
        if rel1 in heads and m.group(1).strip() and qp.head in REL and \
                m.group(1).strip().split()[-1] not in qp.mods + ["younger", "older", "new"]:
            return "rel", [m.group(1).strip()]       # "my running partner" vs "my partner"
        return "self", []
    if GENERIC_OTHER_RX.match(cl) or re.match(r"(?:someone|somebody|everyone|people|a lot of people)\b", cl):
        return "other", []
    if NAME_SUBJ_RX.match(c):
        return "other", []
    return "", []


_NP_BREAK = (STOP - {"my", "our", "his", "her", "their", "new", "old", "first", "last", "favorite"}) | {
    "has", "had", "adopted", "got", "bought", "named", "called", "took", "takes", "uses", "used", "booked", "tried",
    "went", "says", "said", "told", "loves", "picked", "found", "made", "visited", "with", "about", "like"}
_SOFT_MODS = {"new", "younger", "older", "little", "big", "dear", "lovely", "current", "wonderful", "amazing",
              "rescue", "sweet", "baby", "eldest", "youngest", "oldest", "adopted", "newly", "own", "best"}


def _possessor_before(left_norm: str, qp: QParse) -> tuple[str, list[str], list[str]]:
    """Possessor of the noun phrase that ends at the head: walk back over <= 3 modifiers, then read the determiner.
    Returns (kind, chain, modifiers) with kind in self | rel | pron | ''."""
    t = left_norm.split()
    mods = []
    i = len(t) - 1
    while i >= 0 and len(mods) < 3 and t[i] not in _NP_BREAK and not t[i].endswith("'s") and \
            t[i] not in ("my", "our", "his", "her", "their"):
        mods.append(t[i])
        i -= 1
    if i < 0:
        return "", [], mods
    d = t[i]
    if d in ("my", "our"):
        return "self", [], mods
    if d in ("his", "their") or (d == "her" and not qp.chain):
        return "pron", [], mods
    if d.endswith("'s"):
        owner = d[:-2]
        if owner in ("it", "that", "what", "there", "here", "who", "one", "today", "yesterday", "life", "let"):
            return "", [], mods
        # "my friend helen's", "my brother brian's", "my neighbor's", "helen's"
        j, rels = i - 1, []
        while j >= 0 and j >= i - 3 and t[j] not in ("my", "our"):
            if t[j] in REL:
                rels.append(t[j])
            j -= 1
        if owner in REL:
            rels.append(owner)
        return "rel", (rels[-1:] or [owner]), mods
    return "", [], mods


def classify(sent: str, qp: QParse, ctx: str = "", is_first: bool = False, unit_head: bool = False) -> SentClass:
    sc = SentClass(sent=sent)
    sn = norm(sent)
    sc.aside = bool(ASIDE_RX.search(sent))
    # head mention
    hit_pos = None
    for f in qp.head_forms():
        m = re.search(_form_rx(f), sn)
        if m:
            hit_pos = m.start() if hit_pos is None else min(hit_pos, m.start())
    sc.head_hit = hit_pos is not None
    sc.mods_hit = sum(1 for w in qp.mods if re.search(_form_rx(w), sn))
    stoks = [stem(t) for t in toks(sent)]
    sc.verb_hit = any(v in stoks for v in qp.verbs)
    # reply to the assistant's question ("What's her name?" -> "Sombra. She's ...")
    if is_first and ctx.strip().endswith("?"):
        cn = norm(ctx)
        head_in_ctx = any(re.search(_form_rx(f), cn) for f in qp.head_forms())
        attr_in_ctx = bool(qp.attr) and qp.attr in cn and bool(re.search(r"\b(what|which|where|who|how)\b", cn))
        if head_in_ctx or (attr_in_ctx and (unit_head or sc.head_hit)):
            sc.reply = True
            if not sc.head_hit and any(re.search(_form_rx(f), cn) for f in qp.head_forms()):
                sc.head_hit = True
    # clause that holds the head (or the verb) decides owner / polarity
    clauses = [c for c in CLAUSE_RX.split(sent) if c and c.strip()]
    target = None
    for c in clauses:
        cn = norm(c)
        if any(re.search(_form_rx(f), cn) for f in qp.head_forms()):
            target = c
            break
    if target is None:
        for c in clauses:
            if any(v in [stem(t) for t in toks(c)] for v in qp.verbs):
                target = c
                break
    target = target or sent
    tn = norm(target)
    # --- owner
    kind, chain = "", []
    probe = qp.head_forms() if sc.head_hit else list(qp.mods)   # no head: a matched modifier ("birthday") stands in
    for f in probe:
        m = re.search(_form_rx(f), tn)
        if m:
            kind, chain, npmods = _possessor_before(tn[: m.start()], qp)
            after = tn[m.end():]
            if sc.head_hit and _slot_conflict(npmods + tn[: m.start()].split()[-4:], qp):
                kind, chain = "rel", ["other " + " ".join(sorted(set(npmods) - set(qp.mods))) or "slot"]
            elif not kind and PERCEIVE_RX.search(tn[: m.start()]):
                kind, chain = "other", []
            if qp.head in REL and sc.head_hit:
                # "my running partner" is not "my partner"; "my partner's husband" is about the husband
                extra = [w for w in npmods if w not in qp.mods and w not in _SOFT_MODS]
                if kind == "self" and extra:
                    kind, chain = "rel", [" ".join(reversed(extra)) + " " + qp.head]
                elif re.match(r"'s (?:[\w-]+ )?(" + "|".join(sorted(REL)) + r")\b", after):
                    kind, chain = "rel", [qp.head + "'s " + re.match(r"'s (?:[\w-]+ )?([\w-]+)", after).group(1)]
            break
    if not kind and qp.first_person:
        kind, chain = _subject_chain(target, qp)
        if not kind and target is not sent:
            kind, chain = _subject_chain(sent, qp)
    qchain = [norm(c) for c in qp.chain]
    if kind == "self":
        sc.owner, sc.owner_why = ("match", "my") if not qchain else ("unknown", "my(no rel)")
        if qchain and any(re.search(_form_rx(r), sn) for r in qchain):
            sc.owner = "match"
    elif kind == "rel":
        heads = {norm(f) for f in qp.head_forms()}
        anchor_words = set(norm(qp.anchor).split())
        # the question itself names this possessor ("my favorite artist's debut album") -> not someone else's
        ch = [c for c in chain if c not in heads and c not in anchor_words]
        if not ch:
            sc.owner, sc.owner_why = ("match", "rel=head") if not qchain else ("unknown", "rel=head")
        elif qchain and ch[: len(qchain)] == qchain[: len(ch)] and len(ch) == len(qchain):
            sc.owner, sc.owner_why = "match", "rel=" + "/".join(ch)
        elif qchain and len(ch) == 1 and ch[0] in qchain:
            sc.owner, sc.owner_why = "match", "rel=" + ch[0]
        else:
            sc.owner, sc.owner_why = "mismatch", "rel=" + "/".join(ch)
    elif kind in ("pron", "other"):
        if qchain and kind == "pron":
            sc.owner, sc.owner_why = "unknown", kind
        else:
            sc.owner, sc.owner_why = "mismatch", kind
    # --- polarity (round 1): judged per occurrence of a question verb; one affirmed occurrence wins
    # ("I'm thinking of going back to Hawaii, I loved it when I went with my family" is affirmed for "go").
    # Without a question verb only a negated head ("we don't have a dog") or, for name questions, a planned
    # naming act ("considered calling my podcast X") counts. The round-0 rule "I'm thinking ... anywhere in the
    # clause" demoted 5/64 dev gold sentences ("I'm thinking of visiting my sister Emily in Denver").
    all_t = toks(sent)
    occ = []
    for i, t in enumerate(all_t):
        if stem(t) in qp.verbs:
            win4 = all_t[max(0, i - 4): i]
            win_txt = " ".join(all_t[max(0, i - 6): i])
            if any(w in NEG_WORDS or w.endswith("n't") for w in win4):
                occ.append("neg")
            elif PLAN_RX.search(win_txt):
                occ.append("plan")
            else:
                occ.append("aff")
    if occ:
        sc.polarity = "aff" if "aff" in occ else ("neg" if "neg" in occ else "plan")
    else:
        hm = None
        for f in qp.head_forms():
            hm = re.search(_form_rx(f), tn)
            if hm:
                break
        if hm:
            left = tn[: hm.start()].split()[-3:]
            if any(w in NEG_WORDS or w.endswith("n't") for w in left) and "not only" not in tn:
                sc.polarity = "neg"
        if sc.polarity == "aff" and qp.attr == "name" and re.search(
                r"\b(?:thinking (?:about|of)|considering|considered|might|may|planning(?: to| on)?|want(?:ed)? to|"
                r"debating|tempted to|toying with)\s+(?:\w+\s+)?(?:nam|call|renam|dub)\w*", tn):
            sc.polarity = "plan"
    if sc.polarity in ("aff",) and qp.recency and REJECT_RX.search(sn):
        sc.polarity = "plan"   # tried and rejected: not the value the user ended up with
    if sc.polarity == "aff":
        if STALE_RX.search(tn) and not NEW_RX.search(tn):
            sc.polarity = "stale"
        elif NEW_RX.search(tn):
            sc.polarity = "new"
        elif STALE_RX.search(sn) and not NEW_RX.search(sn):
            sc.polarity = "stale"
    # --- values (names) for name questions
    if qp.attr in ("name", "") or qp.wh in ("who", "which"):
        sc.values = extract_names(sent, qp, reply=sc.reply)
    return sc


def extract_names(sent: str, qp: QParse, reply: bool = False) -> list[str]:
    out = []
    s = sent.replace("’", "'")
    pats = [r"\b(?:named|called|name is|name's|names? (?:it|him|her|them)|calls? (?:it|him|her|them)|"
            r"renamed(?: \w+){0,4}?|christened|dubbed)\s+['\"“]?([A-Z][\w'-]+(?: [A-Z][\w'-]+){0,3})"]
    forms = "|".join(re.escape(f) for f in qp.head_forms() if f)
    if forms:
        pats.append(r"\b(?:my|our|the|a) (?:new |little |rescue |younger |older |youngest |oldest |adopted |)"
                    r"(?:" + forms + r")s?,? ([A-Z][\w'-]+(?: [A-Z][\w'-]+)?)")
        pats.append(r"\b([A-Z][\w'-]+),? (?:the|my|our) (?:new |little |)(?:" + forms + r")\b")
    for p in pats:
        for m in re.finditer(p, s, flags=0):
            v = m.group(1).strip(" '\".,")
            first = v.split()[0]
            if first in CAP_STOP:
                continue
            if v not in out:
                out.append(v)
    if reply:
        m = re.match(r"\s*([A-Z][\w'-]+(?: [A-Z][\w'-]+){0,3})[.!,]", s)
        if m and m.group(1).split()[0] not in CAP_STOP and m.group(1) not in out:
            out.insert(0, m.group(1))
    return out


# ----------------------------------------------------------------------------------------------------------------
# scoring
# ----------------------------------------------------------------------------------------------------------------
DEFAULTS = dict(pool=60, bm25_k=40, scan=True, scan_cap=80, rerank_cap=140, max_sessions=2, window=1, budget=7000,
                drop_decoys=True, recency=True, rel_thr=0.55, cite_prev=False, notes=False, hint="owner",
                w_owner=0.15, w_head=0.10, w_mod=0.04, w_verb=0.04, w_aside=0.04, w_reply=0.08, w_value=0.05,
                pen_other=0.25, pen_neg=0.35, pen_plan=0.45, pen_stale=0.6, structure=True, whole_turns=False)


def _sig(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def unit_class(u, qp: QParse) -> SentClass:
    """Best sentence of a unit (prefers a head-hit sentence); ctx used for the reply pattern."""
    sents = sentences(u.text) or [u.text]
    un = norm(u.text)
    unit_head = any(re.search(_form_rx(f), un) for f in qp.head_forms())
    best, best_key = None, None
    for i, s in enumerate(sents):
        sc = classify(s, qp, ctx=getattr(u, "ctx", "") if (i == 0 and getattr(u, "chunk_idx", 0) == 0) else "",
                      is_first=(i == 0 and getattr(u, "chunk_idx", 0) == 0), unit_head=unit_head)
        key = (sc.head_hit or sc.reply, sc.valid, sc.mods_hit + sc.verb_hit + bool(sc.values), sc.reply)
        if best is None or key > best_key:
            best, best_key = sc, key
    return best


def struct_adjust(sc: SentClass, qp: QParse, p: dict) -> tuple[float, float]:
    f, b = 1.0, 0.0
    if sc.owner == "match":
        b += p["w_owner"]
    elif sc.owner == "mismatch":
        f *= p["pen_other"]
    if sc.polarity == "neg":
        f *= p["pen_neg"]
    elif sc.polarity == "plan":
        f *= p["pen_plan"]
    elif sc.polarity == "stale" and not qp.past_wanted:
        f *= p["pen_stale"] if (qp.recency or qp.attr == "name") else 0.85
    elif sc.polarity == "new" and qp.recency:
        b += 0.05
    if sc.head_hit:
        b += p["w_head"]
    b += min(sc.mods_hit, 2) * p["w_mod"]
    if sc.verb_hit:
        b += p["w_verb"]
    if sc.aside:
        b += p["w_aside"]
    if sc.reply:
        b += p["w_reply"]
    if sc.values and qp.attr == "name":
        b += p["w_value"]
    return f, b


def head_presence(hx, qp: QParse) -> dict:
    rxs = [re.compile(_form_rx(f)) for f in qp.head_forms() if f]
    user_hit = any_hit = full_hit = 0
    for u in hx.units:
        tn = norm(u.text)
        if any(r.search(tn) for r in rxs):
            any_hit += 1
            if u.role == "user":
                user_hit += 1
                if not qp.mods or all(re.search(_form_rx(m), tn) for m in qp.mods[:2]):
                    full_hit += 1
    chain_user = 1
    if qp.chain:
        crx = [re.compile(_form_rx(f)) for c in qp.chain for f in SYN.get(c, [c])]
        chain_user = sum(1 for u in hx.units if u.role == "user" and any(r.search(norm(u.text)) for r in crx))
    return {"user": user_hit, "any": any_hit, "full": full_hit, "chain": chain_user}


def abstain_prob(qp: QParse, pres: dict, top: SentClass | None, top_rr: float) -> float:
    if qp.strong and pres["any"] == 0:
        return 0.9
    if qp.strong and pres["user"] == 0:
        return 0.75
    if qp.chain and pres.get("chain", 1) == 0:
        return 0.85
    if qp.strong and qp.mods and pres["full"] == 0:
        return 0.3
    if top is None:
        return 0.6
    if not top.valid:
        return 0.6 if top.owner == "mismatch" else 0.45
    if top_rr < 0.15:
        return 0.35
    return 0.05 if top.owner == "match" else 0.12


def build_hint(qp: QParse, p: dict, decoys: list, abstain_p: float, pres: dict, notes: list) -> str:
    if p.get("hint") == "none":
        return ""
    if qp.chain:
        owner = "'s ".join(qp.chain)
        parts = [f"Answer from what the user said about their {owner}'s {qp.anchor or 'item'} (not anyone else's). "
                 f"Plans or things the user did not actually do are not the answer."]
    else:
        parts = [f"Answer from what the user said about their own {qp.anchor or 'item'}. Mentions of other people's "
                 f"{qp.head or 'items'} (friends, relatives, neighbors, coworkers), plans, or things the user did not "
                 f"actually do are not the answer."]
    if qp.recency and p.get("recency", True):
        parts.append("If the user mentioned it more than once, the most recent dated statement wins.")
    for d in decoys[:2]:
        parts.append(f"Note: the {d['date']} line \"{d['snip']}\" is {d['why']}, not the user's own.")
    if abstain_p >= 0.7 and qp.strong:
        if pres["user"] == 0:
            parts.append(f"The user never mentions '{qp.head}' in their own messages; if no passage states it, "
                         f"answer INSUFFICIENT_EVIDENCE.")
        else:
            parts.append("If no passage states it for the user, answer INSUFFICIENT_EVIDENCE.")
    if notes:
        parts.append("Candidate statements (oldest first): " + " | ".join(notes))
    return " ".join(parts)


def _snip(s: str, n: int = 110) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    return s if len(s) <= n else s[: n - 1].rsplit(" ", 1)[0] + "…"


def _why(sc: SentClass) -> str:
    if sc.owner == "mismatch":
        w = sc.owner_why.split("=", 1)[-1]
        return {"pron": "about someone else", "other": "about someone else"}.get(w, f"about the user's {w}")
    return {"neg": "something the user did not do", "plan": "only a plan"}.get(sc.polarity, "not the answer")


# ----------------------------------------------------------------------------------------------------------------
# lens entry point
# ----------------------------------------------------------------------------------------------------------------
VARIANTS = {
    # rank 1 for test: structure on, decoy sessions dropped, short owner/recency hint, cite the primary session
    "owner": dict(),
    # rank 2: + code-written chain-of-note (candidate statements with dates and labels) inside the hint
    "notes": dict(notes=True),
    # rank 3: precision pack, a single session (CF1 and chars), same structure
    "owner1": dict(max_sessions=1, budget=4500),
    # control: same pool + reranker, no structure, no hint (measures what the rules add)
    "plain": dict(structure=False, hint="none", drop_decoys=False, recency=False),
}


def build(q, hx, **params) -> EvidencePack:
    p = {**DEFAULTS, **{k: v for k, v in params.items() if not k.startswith("_")}}
    from core.pack import build_pack  # lazy: the core lands in round 1

    qp = parse_question(q.question)
    # ---- candidate pool: hybrid on user units + bm25 on synonym-expanded query + lexical head scan
    pool, seen = [], set()

    def add(uids):
        for uid in uids:
            if uid not in seen:
                seen.add(uid)
                pool.append(uid)

    add(uid for uid, _ in hx.hybrid(q.question, k=p["pool"], roles=["user"]))
    forms = qp.head_forms()
    if len(forms) > 2 or qp.mods:
        add(uid for uid, _ in hx.bm25(" ".join([q.question] + forms + qp.mods), k=p["bm25_k"], roles=["user"]))
    if p["scan"] and qp.head:
        rxs = [re.compile(_form_rx(f)) for f in forms if f]
        hits = [u.uid for u in hx.units if u.role == "user" and any(r.search(norm(u.text)) for r in rxs)]
        if qp.mods:  # head + modifier first
            hits.sort(key=lambda uid: -sum(bool(re.search(_form_rx(m), norm(hx.units[uid].text))) for m in qp.mods))
        add(hits[: p["scan_cap"]])
    pool = pool[: p["rerank_cap"]]
    rr = dict(hx.rerank(q.question, pool)) if pool else {}

    # ---- structural re-weighting
    cls, score = {}, {}
    for uid in pool:
        u = hx.units[uid]
        base = _sig(rr.get(uid, -10.0))
        if p["structure"]:
            sc = unit_class(u, qp)
            f, b = struct_adjust(sc, qp, p)
            cls[uid] = sc
            score[uid] = (base + b) * f
        else:
            score[uid] = base
    ranked = sorted(pool, key=lambda x: -score[x])

    # ---- primary session (recency-aware)
    primary_uid = ranked[0] if ranked else None
    prev_uid = None
    if ranked and p["structure"] and p["recency"] and qp.recency:
        best = score[ranked[0]]
        valid = [uid for uid in ranked if score[uid] >= p["rel_thr"] * best and cls[uid].valid]
        if valid:
            valid.sort(key=lambda uid: (hx.units[uid].date, score[uid]), reverse=True)
            primary_uid = valid[0]
            older = [uid for uid in valid if hx.units[uid].session_id != hx.units[primary_uid].session_id]
            prev_uid = older[0] if older else None

    # ---- order for the pack: primary session, then the rest; decoy-only sessions dropped
    decoys, order = [], []
    if primary_uid is not None:
        psid = hx.units[primary_uid].session_id
        order = [primary_uid] + [uid for uid in ranked if hx.units[uid].session_id == psid and uid != primary_uid]
        if prev_uid is not None:
            order.append(prev_uid)
        for uid in ranked:
            if uid in order:
                continue
            sc = cls.get(uid)
            if p["drop_decoys"] and sc is not None and (sc.head_hit or sc.verb_hit) and not sc.valid:
                if len(decoys) < 3 and score[uid] >= 0.25 * score[primary_uid]:
                    u = hx.units[uid]
                    decoys.append({"uid": uid, "date": u.date_str[:10], "snip": _snip(sc.sent, 90), "why": _why(sc)})
                continue
            order.append(uid)

    top_sc = cls.get(primary_uid) if primary_uid is not None else None
    top_rr = _sig(rr.get(primary_uid, -10.0)) if primary_uid is not None else 0.0
    pres = head_presence(hx, qp) if qp.head else {"user": 1, "any": 1, "full": 1, "chain": 1}
    abstain_p = abstain_prob(qp, pres, top_sc, top_rr) if p["structure"] else (0.5 if top_rr < 0.15 else 0.1)

    notes = []
    if p["notes"] and p["structure"]:
        cand, seen_s = [], set()
        for uid in order[:12]:
            sid = hx.units[uid].session_id
            if cls.get(uid) and (cls[uid].head_hit or cls[uid].reply) and sid not in seen_s:
                seen_s.add(sid)
                cand.append(uid)
        cand = cand[:4]
        cand.sort(key=lambda uid: hx.units[uid].date)
        for uid in cand:
            sc = cls[uid]
            tag = "" if sc.valid else f" [{_why(sc)}]"
            if sc.polarity == "stale" and not qp.past_wanted:
                tag = " [old value]"
            notes.append(f"{hx.units[uid].date_str[:10]}: \"{_snip(sc.sent, 120)}\"{tag}")
    shown_decoys = decoys if p["drop_decoys"] else []
    hint = build_hint(qp, p, [] if p["notes"] else shown_decoys, abstain_p, pres, notes) if p["structure"] else ""

    passages, included, chars = build_pack(q, hx, order, budget=p["budget"], window=p["window"], order="chrono",
                                           hint=hint, max_sessions=p["max_sessions"], whole_turns=p["whole_turns"])
    cite = []
    if primary_uid is not None:
        cite.append(hx.units[primary_uid].session_id)
        if p["cite_prev"] and prev_uid is not None:
            cite.append(hx.units[prev_uid].session_id)
    cite = [c for c in cite if any(ps.id == c for ps in passages)] or None
    computed = None
    if top_sc is not None and qp.attr == "name" and top_sc.values and top_sc.valid:
        computed = top_sc.values[0]
    dbg = {"ranking": ranked, "qparse": {k: getattr(qp, k) for k in ("anchor", "head", "mods", "chain", "verbs",
                                                                       "attr", "recency", "past_wanted", "strong",
                                                                       "source")},
           "presence": pres, "decoys": [d["uid"] for d in decoys],
           "top": [(uid, round(score[uid], 3), cls[uid].owner, cls[uid].polarity, cls[uid].head_hit)
                   for uid in ranked[:8]] if p["structure"] else ranked[:8]}
    return EvidencePack(question_id=q.question_id, variant=params.get("_variant", ""), route="ssu",
                        passages=passages, hint=hint, cite_plan=cite, abstain_p=abstain_p,
                        included_turns=included, computed=computed, debug=dbg)


# ----------------------------------------------------------------------------------------------------------------
# self-test (no core.index needed):  .venv/bin/python lenses/ssu/a_owner.py
# ----------------------------------------------------------------------------------------------------------------
def _selftest() -> None:
    import json
    from pathlib import Path

    # 1) question grammar on the sample file (dev + test styles)
    rows = [json.loads(line) for line in open(Path(__file__).resolve().parents[2] / "runs/samples/ssu.jsonl")]
    print("== question grammar")
    for r in rows:
        qp = parse_question(r["question"])
        print(f"  {r['split']:5} {qp.source:10} head={qp.head!r:16} mods={qp.mods} chain={qp.chain} verbs={qp.verbs} "
              f"attr={qp.attr} rec={int(qp.recency)} past={int(qp.past_wanted)} | {r['question']}")

    # 2) gold user sentences must not be demoted (dev)
    print("== gold sentences (dev): head hit / owner / polarity")
    demoted = hit = n = 0
    for r in rows:
        if r["split"] != "dev" or r["abstain"]:
            continue
        qp = parse_question(r["question"])
        for g in r["gold_turns"]:
            if g["role"] != "user":
                continue
            ss = sentences(g["text"])
            scs = [classify(s, qp, ctx=g.get("prev_turn_tail", "")[-200:], is_first=(i == 0)) for i, s in enumerate(ss)]
            ans = r["answer"].lower().split()[0].strip(".,")
            best = next((sc for sc in scs if ans in sc.sent.lower()), scs[0])
            n += 1
            hit += best.head_hit
            bad = best.owner == "mismatch" or best.polarity in ("neg", "plan")
            demoted += bad
            if bad or not best.head_hit:
                print(f"  {'DEMOTED' if bad else 'no-head'} owner={best.owner}({best.owner_why}) pol={best.polarity} "
                      f"head={qp.head!r} | {_snip(best.sent, 140)}")
    print(f"  gold sentences: {n}, head hit {hit}, demoted {demoted}")

    # 3) synthetic near-miss shapes (made-up sentences that mirror the planted patterns)
    print("== synthetic near-miss checks")
    cases = [
        ("What is my dog's name?", "My friend Rosa just adopted a beagle puppy named Pepper.", "mismatch", None),
        ("What is my dog's name?", "Our rescue dog Biscuit is scared of the vacuum.", "match", "aff"),
        ("What is my brother's new dog's name?", "My brother adopted a dog named Rocket last month.", "match", None),
        ("What is the name of my brother's new dog?", "My brother's neighbor has a dog named Comet.", "mismatch",
         None),
        ("What is the name of my brother's new dog?", "My brother adopted a dog named Rocket recently.", "match",
         None),
        ("What is my favorite color?", "By the way, My cousin Ana's favorite color is mint.", "mismatch", None),
        ("What is my favorite color?", "My favorite color is mint.", "match", "aff"),
        ("What is my favorite color?", "My favorite color used to be red when I was a kid.", "match", "stale"),
        ("What was the last restaurant I mentioned trying?", "My wife tried a new ramen restaurant downtown.",
         "mismatch", None),
        ("What was the last restaurant I mentioned trying?",
         "My uncle said Smoky Joe's is the best barbecue restaurant in town, but I haven't tried it yet.", None, "neg"),
        ("What was the last state park I mentioned camping at?",
         "My son's team had a picnic at Pine Hill State Park, but we didn't camp overnight.", None, "neg"),
        ("What was the last state park I mentioned camping at?", "We camped at Eagle Rock State Park last weekend.",
         "match", "aff"),
        ("What was the style of the last beer I mentioned I'd kegged?",
         "Oh, and I'm thinking about kegging a red ale for the barbecue.", None, "plan"),
        ("What was the style of the last beer I mentioned I'd kegged?", "I finally kegged my amber lager last weekend.",
         "match", None),
        ("What did I end up naming my sourdough starter?", "My friend Lucia named her sourdough starter Doughy too.",
         "mismatch", None),
        ("What did I end up naming my sourdough starter?", "I've renamed my sourdough starter Levi.", "match", "new"),
        ("What is my skateboard's name now?", "My coworker says he calls his skateboard 'Latte'.", "mismatch", None),
        ("What's my partner's name?", "My sister's partner Tom came over to see the kitchen.", "mismatch", None),
        ("What's my partner's name?", "My partner Sam is handling bedtime tonight.", "match", None),
        ("What's my partner's name?", "My running partner's husband Ben recommended a builder.", "mismatch", None),
        ("What is my toddler's name?", "My neighbor's toddler is named Max, born the same month as mine.", "mismatch",
         None),
        ("How many grandchildren do I have?", "By the way, My neighbor has one grandchild.", "mismatch", None),
        ("What is my move date?", "My neighbor's move date is May 3, so the elevator will be busy.", "mismatch", None),
        ("Who gave me a new stand mixer as a birthday gift?", "My aunt gave me a stand mixer for my birthday.", None,
         "aff"),
        ("What did I buy for my sister's birthday gift?", "For my sister's birthday, I got her a red scarf.", "match",
         None),
        ("Who teaches my Saturday morning pottery class now?", "Oh, and My Tuesday evening pottery class is taught by "
         "Ines.", "mismatch", None),
        ("Who teaches my Saturday morning pottery class now?", "My Saturday morning pottery class is taught by Rui.",
         "match", None),
        ("What did I end up naming my sourdough starter?", "I saw a sourdough starter named Gus in a baking forum.",
         "mismatch", None),
        ("What capacitor value did I end up using in my old radio?",
         "I briefly tried a 47 uF capacitor in my old radio, but the hum got worse and I swapped back.", None, "plan"),
    ]
    bad = 0
    for qtext, s, want_owner, want_pol in cases:
        qp = parse_question(qtext)
        sc = classify(s, qp)
        ok = (want_owner is None or sc.owner == want_owner) and (want_pol is None or sc.polarity == want_pol)
        bad += not ok
        print(f"  {'ok ' if ok else 'BAD'} owner={sc.owner}({sc.owner_why}) pol={sc.polarity} head={sc.head_hit} "
              f"vals={sc.values} | {s}")
    # reply pattern
    qp = parse_question("What is my cat's name?")
    sc = classify("Pixie. She's a tiny gray cat with a loud purr.", qp, ctx="That sounds adorable. What's her name?",
                  is_first=True)
    print(f"  {'ok ' if sc.reply and sc.values[:1] == ['Pixie'] else 'BAD'} reply={sc.reply} vals={sc.values}")
    bad += not (sc.reply and sc.values[:1] == ["Pixie"])
    print(f"synthetic failures: {bad}")


if __name__ == "__main__":
    _selftest()
