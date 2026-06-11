"""Benchmark tasks evaluated on model embeddings."""

import numpy as np
from anndata import AnnData
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


def cell_type_classification(embedding: np.ndarray, adata: AnnData, label_key: str = "cell_type", seed: int = 0):
    """Linear-probe cell type classification accuracy/F1."""
    y = adata.obs[label_key].values
    X_train, X_test, y_train, y_test = train_test_split(
        embedding, y, test_size=0.2, random_state=seed, stratify=y
    )
    clf = LogisticRegression(max_iter=1000, n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "macro_f1": f1_score(y_test, y_pred, average="macro"),
    }


def batch_integration_asw(embedding: np.ndarray, adata: AnnData, batch_key: str = "batch", label_key: str = "cell_type"):
    """Average silhouette width for batch mixing (lower batch ASW = better mixing)
    and cell type separation (higher label ASW = better)."""
    from sklearn.metrics import silhouette_score

    return {
        "batch_asw": silhouette_score(embedding, adata.obs[batch_key].values),
        "label_asw": silhouette_score(embedding, adata.obs[label_key].values),
    }
