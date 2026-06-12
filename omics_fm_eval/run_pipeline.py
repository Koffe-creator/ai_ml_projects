"""Full pipeline: Phase 1 (foundation-model benchmark) + Phase 2 (LLM judge).

Phase 1 — embed each model, linear-probe cell-type classification accuracy.
Phase 2 — LLM annotates clusters from marker genes; a second LLM judges
          semantic equivalence against the ground-truth labels.

Usage:
    python run_pipeline.py --dataset pbmc --models pca geneformer
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from data.gene_id_mapping import symbols_to_ensembl
from data.load_dataset import load
from eval.run_benchmark import MODEL_REGISTRY
from eval.tasks import batch_integration_asw, cell_type_classification
from llm_judge.llm_annotator import annotate_clusters, judge_match
from llm_judge.marker_genes import top_markers_per_cluster


def run_phase1(adata, model_names):
    results = {}
    for name in model_names:
        model = MODEL_REGISTRY[name]()
        model_input = symbols_to_ensembl(adata) if name == "geneformer" else adata
        embedding = model.embed(model_input)
        scores = cell_type_classification(embedding, adata)
        if "batch" in adata.obs:
            scores.update(batch_integration_asw(embedding, adata))
        results[name] = scores
        print(f"  {name}: {scores}")
    return results


def run_phase2(adata, cluster_key, model, n_genes):
    markers = top_markers_per_cluster(adata, cluster_key=cluster_key, n_genes=n_genes)
    print(f"  querying {model} for {len(markers)} clusters...")
    llm_pred = annotate_clusters(markers, model=model)
    rows = []
    for cluster, pred in llm_pred.items():
        match = judge_match(truth=cluster, pred=pred, model=model)
        rows.append({"cluster": cluster, "llm_prediction": pred, "match": match})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="pbmc")
    parser.add_argument("--models", nargs="+", default=["pca", "geneformer"],
                        choices=list(MODEL_REGISTRY))
    parser.add_argument("--cluster-key", default="cell_type")
    parser.add_argument("--llm-model", default="claude-haiku-4-5")
    parser.add_argument("--n-genes", type=int, default=15)
    parser.add_argument("--outdir", default="results")
    args = parser.parse_args()

    adata = load(args.dataset)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("=== PHASE 1: foundation-model benchmark ===")
    phase1 = run_phase1(adata, args.models)

    print("\n=== PHASE 2: LLM-as-evaluator ===")
    phase2 = run_phase2(adata, args.cluster_key, args.llm_model, args.n_genes)
    phase2.to_csv(outdir / "llm_judge.csv", index=False)
    llm_acc = phase2["match"].mean()
    print(phase2.to_string(index=False))

    # consolidate all accuracies (Phase 1 + Phase 2) into one results file
    combined = dict(phase1)
    combined["llm_judge"] = {"accuracy": float(llm_acc)}
    with open(outdir / "benchmark.json", "w") as f:
        json.dump(combined, f, indent=2)

    print("\n=== SUMMARY ===")
    for name, scores in phase1.items():
        print(f"  Phase 1  {name:12s}  Accuracy = {scores['accuracy']:.3f}")
    print(f"  Phase 2  {'LLM judge':12s}  Accuracy = {llm_acc:.3f}")

    # append every method's accuracy from this run to the shared log
    log_path = outdir / "accuracy_log.txt"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, "a") as f:
        for name, scores in phase1.items():
            f.write(f"{timestamp}  dataset={args.dataset}  method={name}  Accuracy = {scores['accuracy']:.3f}\n")
        f.write(f"{timestamp}  dataset={args.dataset}  method=llm_judge  Accuracy = {llm_acc:.3f}\n")
    print(f"\nAppended -> {log_path}")


if __name__ == "__main__":
    main()
