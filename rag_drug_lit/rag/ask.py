"""Command-line tool: ask a question and get a cited answer.

Add --eval to also score the answer with DeepEval (faithfulness + answer
relevancy, judged by a GPT model). That needs OPENAI_API_KEY and costs a
few cents per run.

Example:
    python -m rag.ask "Which receptor does dupilumab target?"
    python -m rag.ask --k 4 "What treatments exist for moderate-to-severe atopic dermatitis?"
    python -m rag.ask --eval "Which receptor does dupilumab target?"
"""

import argparse

from rag.retriever import Retriever
from rag.generate import answer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--corpus", default="corpus")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--model", default="claude-haiku-4-5")
    parser.add_argument("--eval", action="store_true",
                        help="also score the answer with DeepEval (needs OPENAI_API_KEY)")
    args = parser.parse_args()

    # Step 1: find the most relevant passages.
    retriever = Retriever(args.corpus)
    passages = retriever.search(args.question, k=args.k)

    print("Retrieved passages:")
    for p in passages:
        print("  [" + p["source"] + "]  score=" + format(p["score"], ".3f"))
    print()

    # Step 2: let Claude answer using only those passages.
    answer_text = answer(args.question, passages, model=args.model)
    print("Answer:")
    print(answer_text)

    # Step 3 (optional): grade the answer with DeepEval.
    if args.eval:
        from deepeval.test_case import LLMTestCase
        from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric

        test_case = LLMTestCase(
            input=args.question,
            actual_output=answer_text,
            retrieval_context=[p["text"] for p in passages],
        )

        faithfulness = FaithfulnessMetric(threshold=0.7)
        faithfulness.measure(test_case)
        relevancy = AnswerRelevancyMetric(threshold=0.7)
        relevancy.measure(test_case)

        print()
        print("DeepEval scores (GPT judge):")
        print("  faithfulness     %.2f  -  %s" % (faithfulness.score, faithfulness.reason))
        print("  answer relevancy %.2f  -  %s" % (relevancy.score, relevancy.reason))


if __name__ == "__main__":
    main()
