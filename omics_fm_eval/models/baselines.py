"""Classical baseline embeddings."""

import numpy as np
import scanpy as sc
from anndata import AnnData

from .base import EmbeddingModel


class PCABaseline(EmbeddingModel):
    name = "pca"

    def __init__(self, n_components: int = 50):
        self.n_components = n_components

    def embed(self, adata: AnnData) -> np.ndarray:
        ad = adata.copy()
        sc.pp.pca(ad, n_comps=self.n_components)
        return ad.obsm["X_pca"]
