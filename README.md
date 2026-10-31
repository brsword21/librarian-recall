# Librarian Recall

**Memory recall for AI assistants: from months of chat history, the few passages that answer a question.**

An assistant that remembers a user for months has a history of hundreds of sessions, often over a million characters.
Putting all of it in the prompt is slow and expensive, and models miss facts buried in long contexts. Plain vector
search is cheap but often misses the right passage, especially for questions about dates, counts or values that
changed over time.

Librarian sits between the memory and the model. It takes a history and a question and returns an **evidence pack**:
- the 1–9 passages that answer the question, with their dates (typically about 0.2% of the archive);
- notes computed by code: day gaps between dated events, tallies across sessions, the latest value of a changed fact;
- a citation plan: which sessions support the answer;
- an estimate that the history does not answer the question at all.

Any LLM can answer from the pack. The `prompt` field holds it ready to send.

Why it is useful:
- **Faster:** the model reads about 3,000 characters instead of 1.6 million. A pack takes milliseconds to about a
  second to build on a laptop once the archive is indexed.
- **Cheaper:** a few hundred input tokens per question instead of hundreds of thousands. No hosted model is needed:
  retrieval runs on two small local models.
- **More accurate:** on our benchmark the pack contains what a reader needs for 94% of answerable questions. Its
  planned citations score 0.92 F1, and every answer comes with its source sessions.

Librarian was built for the Reactive AI Memory Retrieval Challenge at the START Warsaw x Poland Tech Week Hackathon
(October 2026). There, every answer had to come from the organisers' model, so all of our work went into choosing
what that model reads.

## Results

The offline score is `S = 100 · mean(0.65·PA + 0.30·CF1 − 0.05·chars/16000)`:
- **PA:** the pack lets a reader answer;
- **CF1:** citation F1 against the gold sessions;
- **chars:** pack length.

| Dataset | Naive keyword baseline | Generic hybrid retrieval | **Librarian** |
|---|---|---|---|
| LongMemEval-derived dev (526 questions, 5 types) | 42.9 | 73.4 (80-question sample) | **87.1** |
| dev200: dev questions at full 200-session scale | | | **87.0** |
| LoCoMo (never seen while building; 1,535 questions) | 31.3 | 56.8 (same pack size) | **66.4** |

On the challenge test set (146 questions, ~1.6M characters of history each), the organisers' model answered all
146 from our packs, with a median pack of 3,170 characters. On LoCoMo, a different dataset of conversations between
two people, Librarian scores about 2x the baseline, the same ratio as on dev. S is not comparable 1:1 across the two
datasets, because LoCoMo needs its own definition of PA.

## How it works

```
history ─► units (sentence-aligned chunks, dated) ─► BM25 + bge-small embeddings + cross-encoder rerank
question ─► router (5 question types) + evidence hand-offs ─► one lens per type ─► evidence pack ─► any LLM
```

| Question type | What the lens does |
|---|---|
| a fact the user stated ("What is my dog's name?") | precise retrieval of the user's own statement, owner checks against decoys |
| something the assistant said ("Which app did you recommend?") | retrieval over assistant turns, with the user's request as context |
| facts spread over sessions ("How many trips did I take?") | gathers every instance and asks the reader to list, then count |
| a value that changed ("What is my current price?") | a timeline of the attribute: oldest to newest statement, latest wins |
| dates and gaps ("How many days between X and Y?") | an event calendar; the day gap is computed by code and passed as a note |

The text router is wrong on about a third of dev questions, so routing is soft. After the router, two detectors look
at what was retrieved: is the question about something the assistant said, and do the best sessions state two
different values of the same attribute? Either can move the question to another lens. A routing error therefore
costs a few points on that question rather than the whole answer.

## Quick start

Requirements: Python 3.12+, about 1 GB for models, macOS or Linux.

```sh
git clone https://github.com/brsword21/librarian-recall.git && cd librarian-recall
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m core.model_server &        # embeddings + reranking on 127.0.0.1:8765 (downloads both models once)
```

Python:
```python
from librarian import Librarian
from ingest import to_history

history, info = to_history(open("my_chat_export.txt").read())   # or build the dict yourself, see below
lib = Librarian()
hx = lib.index(history)
pack = lib.pack(hx, "How many days passed between the demo day and our first paying customer?")
print(pack["route"], pack["computed"])   # temporal 16
print(pack["prompt"])                    # send this to your LLM
```

HTTP API (standard library only):
```sh
.venv/bin/python -m server --port 8080          # LIBRARIAN_API_KEY=... to require a bearer token
curl -s localhost:8080/v1/histories -H 'Content-Type: application/json' \
     -d "$(python3 -c 'import json; print(json.dumps({"name": "notes.md", "content": open("notes.md").read()}))')"
curl -s localhost:8080/v1/pack -H 'Content-Type: application/json' \
     -d '{"history_id": "u_…", "question": "Who is our CTO now?"}'
```
The endpoints are documented at the top of `server.py`:
- `GET /health`;
- `POST /v1/histories` (indexing continues in the background);
- `GET /v1/histories`;
- `POST /v1/pack`.

## Input format

A history is a list of dated sessions of turns:
```json
{"history_id": "h_1", "sessions": [
  {"session_id": "s_1", "date": "2023/05/12 (Fri) 05:49",
   "turns": [{"turn_id": "t_1", "role": "user", "content": "My dog Grace turns three next week."},
             {"turn_id": "t_2", "role": "assistant", "content": "Happy early birthday to Grace!"}]}]}
```

`ingest.to_history` also accepts:
- a ChatGPT export (`conversations.json`);
- a list of sessions or of `{role, content}` messages;
- a WhatsApp-style chat export (one session per day; every human turn is `user`, prefixed with the speaker's name);
- Markdown or plain-text notes (a date heading starts a new session).

## Deployment

The model server is a separate HTTP service (`/embed`, `/rerank`, cached in `cache/models.sqlite`), and
`LIBRARIAN_MODELS` points the library at it. Run `server.py` and `core.model_server` together in one container or VM.
An edge layer, such as a Cloudflare Worker or an MCP server, then calls `/v1/pack` and passes `prompt` to whichever
model answers. Workers cannot run PyTorch themselves. `LIBRARIAN_CACHE` moves the cache, which holds stored histories
and model results.

## Limits

- **English only:** the BM25 stemmer, both models and the notes are English. Other languages are untested.
- **Tuned on LongMemEval-style questions.** Out of distribution, single facts and dates hold up (LoCoMo single-hop 81
  S, temporal 72), but questions that join many sessions without a counting cue are weak (multi-hop 26). Questions
  that name a calendar window ("the first weekend of October 2023") are not filtered by date yet.
- **The "not in the history" estimate is a hint, not a decision.** It separates unanswerable questions well on
  LongMemEval-style data (AUC 0.81) but not on LoCoMo's person-swap questions (0.56).
- **The first question on a new archive embeds all of it.** Call `Librarian.warm()`, or let the API index in the
  background. A 1.7M-character history is a few thousand units.

## Repository

```
librarian.py      Python API: Librarian.index / warm / pack
server.py         HTTP API
ingest.py         file formats -> history
core/             units, BM25 + dense index, rerank, dates, router, pack budgeting, model client and server
lenses/           one lens per question type, plus the routed pipeline (lenses/pipeline.py)
configs/          best.json: the routed system; best_v2.json: the frozen v2 routes it builds on
```

Each lens file's docstring records what it changes and what it measured. These are the files of the submitted
system, unchanged except for import paths. Rebuilding the challenge test packs with this repository reproduces all
146 submitted packs exactly: same passages, notes, routes and citation plans.

## License

**PolyForm Noncommercial 1.0.0** ([LICENSE](LICENSE)). The license allows:
- research, evaluation, personal projects, teaching;
- use by noncommercial organisations.

**Commercial use needs a commercial license from us.** That covers using Librarian in a product or a paid service, or
inside a company. To get one, open an issue titled "Commercial license" in this repository. Third-party components
and their licenses are listed in [NOTICE.md](NOTICE.md).
