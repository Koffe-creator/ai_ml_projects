"""Checks how good our retrieval is, using questions with known answers.

For each question we already know which document SHOULD be retrieved
(stored in questions.json). We measure two standard numbers:

  Recall@k : did the correct document show up in the top-k results?
  MRR      : how high did the correct document rank? (1.0 = always first)

This tests retrieval on its own, without calling the LLM - because if the
wrong passages are retrieved, no LLM can give a good answer.

Run it with:
    python -m eval.eval_retrieval --k 3
"""

import argparse
import json

from rag.retriever import Retriever


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="corpus")
    parser.add_argument("--questions", default="eval/questions.json")
    parser.add_argument("--k", type=int, default=3)
    args = parser.parse_args()

    retriever = Retriever(args.corpus)

    with open(args.questions) as f:
        questions = json.load(f)

    hits = 0
    reciprocal_ranks = []

    for item in questions:
        passages = retriever.search(item["question"], k=args.k)
        sources = [p["source"] for p in passages]
        correct = item["relevant_doc"]

        if correct in sources:
            hits = hits + 1
            rank = sources.index(correct) + 1   # position 0 -> rank 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            rank = None
            reciprocal_ranks.append(0.0)

        print("  rank=" + str(rank) + "  " + item["question"][:55])

    total = len(questions)
    recall = hits / total
    mrr = sum(reciprocal_ranks) / total

    print()
    print("Recall@" + str(args.k) + ": " + format(recall, ".3f"))
    print("MRR:       " + format(mrr, ".3f"))


if __name__ == "__main__":
    main()
