# Third-party components

Librarian Recall's own code is under the MIT license (see LICENSE). It uses, but does not include, the following
components. Each keeps its own license.

| Component | Use | License |
|---|---|---|
| [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) | embeddings (downloaded at first run) | MIT |
| [cross-encoder/ms-marco-MiniLM-L-6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2) | reranking (downloaded at first run) | Apache-2.0 (see the model card) |
| PyTorch | model server | BSD-style / Apache-2.0 |
| sentence-transformers, transformers | model server | Apache-2.0 |
| bm25s | lexical retrieval | MIT |
| PyStemmer (Snowball) | stemming | MIT / BSD |
| NumPy, SciPy | numerics | BSD |
| python-dateutil | date parsing in `ingest.py` | Apache-2.0 / BSD |

Development data: the lenses were tuned on questions converted from LongMemEval (Wu et al., 2024, MIT,
https://github.com/xiaowu0162/LongMemEval). The out-of-distribution check used LoCoMo (Maharana et al., 2024,
CC BY-NC 4.0). No dataset is included in this repository.
