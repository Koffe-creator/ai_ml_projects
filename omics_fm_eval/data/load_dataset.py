"""Load and preprocess scRNA-seq datasets for benchmarking.

Each dataset loader returns an AnnData object with:
  - .X: raw or normalized counts
  - .obs['cell_type']: ground-truth cell type labels
  - .obs['batch']: batch identifier (for integration tasks)
"""

import scanpy as sc


def load_pbmc():
    """PBMC 3k (10x Genomics) — 2638 cells, 8 curated immune cell types."""
    adata = sc.datasets.pbmc3k_processed()
    adata.obs["cell_type"] = adata.obs["louvain"]
    return adata


def load_pbmc68k():
    """PBMC 68k (reduced) — 700 cells, 10 finer-grained immune cell types."""
    adata = sc.datasets.pbmc68k_reduced()
    adata.obs["cell_type"] = adata.obs["bulk_labels"]
    return adata


DATASETS = {
    "pbmc": load_pbmc,
    "pbmc68k": load_pbmc68k,
}


def load(name: str):
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Available: {list(DATASETS)}")
    return DATASETS[name]()
