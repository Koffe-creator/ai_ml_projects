# RAG Pipeline — Flow & File Map

How the pieces connect, and which file does each step.

## (1) Main RAG flow — `python -m rag.ask "<question>"`

```
   a question comes in
        |
        |   $ .venv/bin/python -m rag.ask "Which receptor does dupilumab target?"
        v
+-----------------------------------------------------------+
|  rag/ask.py            (orchestrator / command line)      |
|    reads the question + options (--k, --model)            |
+-----------------------------------------------------------+
        |
        |  STEP 1: RETRIEVE
        v
+-----------------------------------------------------------+
|  rag/retriever.py                                         |
|    - loads corpus/*.txt        <-- corpus/ad_*.txt (PubMed)|
|    - splits into chunks                                   |
|    - TF-IDF + cosine similarity vs. the question          |
|    - returns top-k passages: [{source, text, score}, ...] |
+-----------------------------------------------------------+
        |
        |  passages (the context)
        v
+-----------------------------------------------------------+
|  rag/generate.py                                          |
|    - build_prompt(): passages + question -> one prompt    |
|    - INSTRUCTIONS: use ONLY context, cite [doc_id], refuse|
|    - answer(): calls Claude   <-- ANTHROPIC_API_KEY (.env)|
+-----------------------------------------------------------+
        |
        |  STEP 2: GENERATE
        v
   CITED ANSWER printed to the screen
   (e.g. "...targets the IL-4 receptor alpha chain [ad_dupilumab.txt].")
```

## (2) Retrieval evaluation — `python -m eval.eval_retrieval --k 3`  (free, no API)

```
+-----------------------------------------------------------+
|  eval/eval_retrieval.py                                   |
|    - loads eval/questions.json   <-- 8 Q's + correct doc  |
|    - for each Q: calls retriever.search()  --> retriever  |
|    - compares retrieved sources vs. the correct doc       |
+-----------------------------------------------------------+
        |
        v
   Recall@3 = 1.000   MRR = 1.000      (did we fetch the right doc?)
```

## (3) Answer evaluation — `pytest eval/test_rag_deepeval.py`  (DeepEval + GPT judge)

```
+-----------------------------------------------------------+
|  eval/test_rag_deepeval.py                                |
|    for each test question:                                |
|      1. retriever.search()   ----------> rag/retriever.py |
|      2. answer()             ----------> rag/generate.py  |
|                                          (ANTHROPIC_API_KEY)|
|      3. wrap in LLMTestCase(input, actual_output, context)|
|      4. assert_test(case, [Faithfulness, AnswerRelevancy])|
+-----------------------------------------------------------+
        |
        |  DeepEval sends case to the judge
        v
   GPT judge  <-- OPENAI_API_KEY (.env)
        |
        v
   PASS / FAIL  (is the answer grounded + on-topic?)
```

## One-glance summary

| Step | File | Reads / calls | Needs key? |
|---|---|---|---|
| Entry (CLI) | `rag/ask.py` | the question | - |
| Retrieve | `rag/retriever.py` | `corpus/*.txt` | no |
| Generate | `rag/generate.py` | Claude | `ANTHROPIC_API_KEY` |
| Eval retrieval | `eval/eval_retrieval.py` | `questions.json` + retriever | no |
| Eval answer | `eval/test_rag_deepeval.py` | retriever + generate + DeepEval | both keys |

`retriever.py` and `generate.py` are the engine; `ask.py` and both eval files
just call them in different ways. That separation is what makes the engine
testable in isolation.
