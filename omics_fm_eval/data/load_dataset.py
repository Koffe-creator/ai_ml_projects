"""Load and preprocess scRNA-seq datasets for benchmarking.

Each dataset loader returns an AnnData object with:
  - .X: raw or normalized counts
  - .obs['cell_type']: ground-truth cell type labels
  - .obs['batch']: batch identifier (for integration tasks)
"""

import scanpy as sc


def load_pbmc():
    adata = sc.datasets.pbmc3k_processed()
    adata.obs["cell_type"] = adata.obs["louvain"]
    return adata


DATASETS = {
    "pbmc": load_pbmc,
}


def load(name: str):
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Available: {list(DATASETS)}")
    return DATASETS[name]()
