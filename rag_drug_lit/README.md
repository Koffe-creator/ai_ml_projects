# RAG over Atopic Dermatitis Literature

A retrieval-augmented generation (RAG) system that answers questions from a
corpus of **real PubMed abstracts on atopic dermatitis**, with **inline
citations** and a **hallucination guard** — plus a two-tier evaluation of both
retrieval and answer quality.

## What it does

1. **Retrieve** — a TF-IDF + cosine-similarity retriever (`scikit-learn`) finds
   the most relevant passages for a question (`rag/retriever.py`).
2. **Generate** — Claude answers using *only* the retrieved passages and cites
   the source document id(s); if the answer isn't in the context, it says so
   rather than guessing (`rag/generate.py`).
3. **Evaluate** — two layers:
   - retrieval quality with standard IR metrics, Recall@k and MRR (`eval/eval_retrieval.py`)
   - answer quality with **DeepEval** (Faithfulness + Answer Relevancy, judged by
     a GPT model) (`eval/test_rag_deepeval.py`)

The retriever is intentionally transparent (no opaque vector DB) so every
retrieval score can be inspected and explained.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env     # add ANTHROPIC_API_KEY (answers) and OPENAI_API_KEY (DeepEval judge)
```

## Usage

Ask a question (retrieve → cited answer):

```bash
python -m rag.ask "Which receptor does dupilumab target?"
python -m rag.ask --k 4 "What treatments exist for moderate-to-severe atopic dermatitis?"
python -m rag.ask --eval "Which receptor does dupilumab target?"   # also print DeepEval scores
```

`--k` sets how many passages to retrieve (default 3). `--eval` additionally
scores the answer with DeepEval faithfulness + answer relevancy (needs
`OPENAI_API_KEY`).

Evaluate retrieval quality (free, no LLM):

```bash
python -m eval.eval_retrieval --k 3
# Recall@3: 1.000   MRR: 0.938   (on the bundled 8-question set)
```

Evaluate answer quality with DeepEval (GPT judge, a few cents):

```bash
python -m pytest eval/test_rag_deepeval.py -v
```

## The corpus — real PubMed abstracts

`corpus/` holds real abstracts pulled from PubMed via NCBI's E-utilities API,
each stored with its title and PMID. To rebuild or extend it:

```bash
python fetch_pubmed.py          # edit the TOPICS dict to change subjects
```

### Adding full papers as PDF

Abstracts are short; full papers arrive as PDF. `fetch_pdfs.py` downloads a few
open-access atopic-dermatitis papers from Europe PMC into `raw_pdfs/`, and
`ingest_pdfs.py` extracts their plain text into `corpus/` (one `.txt` per PDF).
The retriever only reads `.txt`, so nothing downstream changes.

```bash
python fetch_pdfs.py     # open-access PDFs -> raw_pdfs/  (edit pmc_ids to change)
python ingest_pdfs.py    # PDFs -> corpus/*.txt
```

Only plain text is extracted (no formatting or images) because TF-IDF retrieval
just needs words — no need to convert PDF to docx first. The PDFs themselves and
the extracted `corpus/PMC*.txt` are gitignored (regenerate them with the two
commands above); the committed corpus is the curated set of abstracts.

## Design notes (talking points)

- **Grounded + cited:** the system prompt forces answers to come only from
  retrieved context and to cite `[doc_id]`, making every claim traceable.
- **Hallucination guard:** out-of-corpus questions (e.g. "ibuprofen dose for a
  headache") return low similarity and the model declines to answer.
- **Two-tier evaluation:** retrieval is scored independently of the LLM
  (Recall@k / MRR) — the usual first place a RAG system fails — and answer
  quality is scored separately with DeepEval's faithfulness/relevancy metrics.
- **Real, messy data:** MRR is 0.938 (not a suspicious 1.000) because two
  general atopic-dermatitis reviews overlap — exactly the kind of near-miss real
  literature produces.

## Structure

- `corpus/` — one `.txt` per abstract (real PubMed records, title + PMID)
- `fetch_pubmed.py` — downloads abstracts from PubMed into the corpus
- `ingest_pdfs.py` — extracts text from PDFs in `raw_pdfs/` into the corpus
- `rag/retriever.py` — TF-IDF chunk retriever
- `rag/generate.py` — Claude generation with citation + grounding rules
- `rag/ask.py` — CLI: retrieve → generate
- `eval/eval_retrieval.py` + `eval/questions.json` — retrieval benchmark
- `eval/test_rag_deepeval.py` — DeepEval answer-quality tests (pytest)
- `rag_walkthrough.ipynb` — step-by-step notebook
- `docs/flow.md` — diagram of how the files connect
