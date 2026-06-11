"""Common interface for foundation models and classical baselines.

Each model wrapper implements `embed(adata) -> np.ndarray` returning a
cell x latent_dim embedding matrix, used downstream for classification
and integration metrics.
"""

from abc import ABC, abstractmethod

import numpy as np
from anndata import AnnData


class EmbeddingModel(ABC):
    name: str

    @abstractmethod
    def embed(self, adata: AnnData) -> np.ndarray:
        """Return a (n_cells, n_latent) embedding matrix."""
        raise NotImplementedError
