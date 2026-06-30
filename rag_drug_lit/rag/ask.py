"""Command-line tool: ask a question and get a cited answer.

Example:
    python -m rag.ask "Which receptor does dupilumab target?"
    python -m rag.ask --k 4 "What treatments exist for moderate-to-severe atopic dermatitis?"
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
    args = parser.parse_args()

    # Step 1: find the most relevant passages.
    retriever = Retriever(args.corpus)
    passages = retriever.search(args.question, k=args.k)

    print("Retrieved passages:")
    for p in passages:
        print("  [" + p["source"] + "]  score=" + format(p["score"], ".3f"))
    print()

    # Step 2: let Claude answer using only those passages.
    print("Answer:")
    print(answer(args.question, passages, model=args.model))


if __name__ == "__main__":
    main()
