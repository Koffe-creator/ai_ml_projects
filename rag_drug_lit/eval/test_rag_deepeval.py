"""Evaluate ANSWER quality with DeepEval (an LLM-as-judge framework).

The retrieval eval (eval_retrieval.py) checks if we fetched the right
documents. This file goes one step further: it checks the actual answer
our RAG system writes, using two DeepEval metrics scored by a GPT judge.

  Faithfulness     - is every claim in the answer backed by the retrieved
                     context? (i.e. did it avoid hallucinating?)
  Answer Relevancy - does the answer actually address the question?

DeepEval runs on top of pytest, so it can be run either way:
    deepeval test run eval/test_rag_deepeval.py
    pytest eval/test_rag_deepeval.py

Needs OPENAI_API_KEY in .env (the judge) and ANTHROPIC_API_KEY (our answer).
"""

import pytest
from dotenv import load_dotenv

from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric

from rag.retriever import Retriever
from rag.generate import answer

load_dotenv()

# A few questions we expect our system to answer well.
QUESTIONS = [
    "Which receptor chain does dupilumab target?",
    "Which oral JAK inhibitors are used for moderate-to-severe atopic dermatitis?",
    "What is the first-line topical treatment for atopic dermatitis flare-ups?",
]


# Build the retriever once and reuse it across all test cases.
@pytest.fixture(scope="module")
def retriever():
    return Retriever("corpus")


@pytest.mark.parametrize("question", QUESTIONS)
def test_answer_is_faithful_and_relevant(retriever, question):
    # Run our real RAG pipeline to get the answer and the context it used.
    passages = retriever.search(question, k=3)
    output = answer(question, passages)
    context = [p["text"] for p in passages]

    test_case = LLMTestCase(
        input=question,
        actual_output=output,
        retrieval_context=context,
    )

    # The GPT judge scores each metric from 0 to 1; we require at least 0.7.
    faithfulness = FaithfulnessMetric(threshold=0.7)
    relevancy = AnswerRelevancyMetric(threshold=0.7)

    assert_test(test_case, [faithfulness, relevancy])
