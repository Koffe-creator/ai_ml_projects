"""Phase 2: LLM-as-evaluator pipeline.

For each cluster in a dataset:
  1. extract its top marker genes
  2. ask an LLM to predict the cell type from those genes
  3. compare the LLM's prediction against the ground-truth label

Usage:
    python -m llm_judge.run_llm_judge --dataset pbmc --out results/llm_judge.csv
"""

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

from data.load_dataset import load
from llm_judge.llm_annotator import annotate_clusters, judge_match
from llm_judge.marker_genes import top_markers_per_cluster


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="pbmc")
    parser.add_argument("--cluster-key", default="cell_type",
                        help="obs column holding cluster / ground-truth labels")
    parser.add_argument("--model", default="claude-haiku-4-5")
    parser.add_argument("--n-genes", type=int, default=15)
    parser.add_argument("--out", default="results/llm_judge.csv")
    parser.add_argument("--log", default="results/accuracy_log.txt",
                        help="file to append each run's accuracy to")
    args = parser.parse_args()

    adata = load(args.dataset)

    # 1. top marker genes per cluster
    markers = top_markers_per_cluster(adata, cluster_key=args.cluster_key, n_genes=args.n_genes)

    # 2. LLM predicts a cell type from each cluster's marker genes
    print(f"Querying {args.model} for {len(markers)} clusters...")
    llm_pred = annotate_clusters(markers, model=args.model)

    # 3. ground truth = the cluster label itself (curated cell-type names);
    #    a second LLM call judges semantic equivalence (handles synonyms/plurals).
    rows = []
    for cluster, pred in llm_pred.items():
        match = judge_match(truth=cluster, pred=pred, model=args.model)
        rows.append({"cluster": cluster, "llm_prediction": pred, "match": match})
    table = pd.DataFrame(rows)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_path, index=False)

    accuracy = table["match"].mean()
    print(table.to_string(index=False))
    print(f"\nLLM accuracy vs ground truth (semantic match): {accuracy:.3f}")
    print(f"Saved -> {out_path}")

    # append this run's accuracy to the log
    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, "a") as f:
        f.write(f"{timestamp}  dataset={args.dataset}  model={args.model}  Accuracy = {accuracy:.3f}\n")
    print(f"Appended -> {log_path}")


if __name__ == "__main__":
    main()
