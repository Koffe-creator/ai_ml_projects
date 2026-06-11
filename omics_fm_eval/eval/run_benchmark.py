"""Run the Phase 1 benchmark: embed each model on each dataset and score tasks.

Usage:
    python run_benchmark.py --dataset pbmc --models pca geneformer
"""

import argparse
import json
from pathlib import Path

from data.gene_id_mapping import symbols_to_ensembl
from data.load_dataset import load
from eval.tasks import batch_integration_asw, cell_type_classification
from models.baselines import PCABaseline
from models.foundation import GeneformerModel

MODEL_REGISTRY = {
    "pca": PCABaseline,
    "geneformer": lambda: GeneformerModel(model_dir="models/checkpoints/geneformer-V2-104M"),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--models", nargs="+", required=True, choices=list(MODEL_REGISTRY))
    parser.add_argument("--out", default="results/benchmark.json")
    args = parser.parse_args()

    adata = load(args.dataset)
    results = {}

    for model_name in args.models:
        model = MODEL_REGISTRY[model_name]()
        model_input = symbols_to_ensembl(adata) if model_name == "geneformer" else adata
        embedding = model.embed(model_input)

        scores = cell_type_classification(embedding, adata)
        if "batch" in adata.obs:
            scores.update(batch_integration_asw(embedding, adata))

        results[model_name] = scores
        print(f"{model_name}: {scores}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
