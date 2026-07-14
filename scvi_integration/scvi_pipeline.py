"""Compare batch-integration methods on the heart-cell atlas.

The atlas mixes four cell-isolation protocols (cell_source), which creates a
technical batch effect. We compare three representations - plain PCA (baseline),
Harmony (fast linear correction of the PCA), and scVI (a deep generative model
on raw counts) - with a batch-mixing entropy and a cell-type silhouette, plus
before/after UMAPs.

    python scvi_pipeline.py
"""

import numpy as np
import scanpy as sc
import scvi
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BATCH = "cell_source"
CELLTYPE = "cell_type"


def batch_mixing_entropy(embedding, batch_labels, k=30):
    """Mean entropy of batch labels among each cell's k nearest neighbours,
    scaled to 0-1. Higher = batches better mixed locally."""
    codes = batch_labels.astype("category").cat.codes.values
    n_batches = codes.max() + 1
    neighbors = NearestNeighbors(n_neighbors=k).fit(embedding).kneighbors(
        embedding, return_distance=False)
    entropies = []
    for row in neighbors:
        p = np.bincount(codes[row], minlength=n_batches)
        p = p[p > 0] / len(row)
        entropies.append(-(p * np.log(p)).sum())
    return float(np.mean(entropies)) / np.log(n_batches)


# --- load; subsample for CPU speed; keep raw counts for scVI ---
adata = scvi.data.heart_cell_atlas_subsampled()
sc.pp.subsample(adata, n_obs=12000, random_state=0)
adata.layers["counts"] = adata.X.copy()
print("cells:", adata.n_obs, " protocols:", adata.obs[BATCH].nunique(),
      " cell types:", adata.obs[CELLTYPE].nunique())

sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat_v3",
                            layer="counts", subset=True)
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)
sc.pp.pca(adata)

# --- Harmony: fast linear correction of the PCA ---
# call harmonypy directly; it already returns cells-by-components, so no transpose
import harmonypy
ho = harmonypy.run_harmony(adata.obsm["X_pca"], adata.obs, [BATCH])
adata.obsm["X_pca_harmony"] = np.asarray(ho.Z_corr)

# --- scVI: deep generative model on raw counts ---
scvi.model.SCVI.setup_anndata(adata, layer="counts", batch_key=BATCH)
model = scvi.model.SCVI(adata)
model.train(max_epochs=40)
adata.obsm["X_scVI"] = model.get_latent_representation()

# --- score and UMAP each representation ---
methods = [("PCA (baseline)", "X_pca"),
           ("Harmony", "X_pca_harmony"),
           ("scVI", "X_scVI")]

summary = []
for name, rep in methods:
    mix = batch_mixing_entropy(adata.obsm[rep], adata.obs[BATCH])
    bio = silhouette_score(adata.obsm[rep], adata.obs[CELLTYPE])
    summary.append({"Method": name,
                    "batch-mixing": round(mix, 3),
                    "cell-type silhouette": round(bio, 3)})
    sc.pp.neighbors(adata, use_rep=rep)
    sc.tl.umap(adata)
    adata.obsm["umap_" + rep] = adata.obsm["X_umap"].copy()

import pandas as pd
print()
print(pd.DataFrame(summary).to_string(index=False))

# --- UMAPs: one row per method, coloured by protocol and by cell type ---
fig, axes = plt.subplots(len(methods), 2, figsize=(13, 15))
for row, (name, rep) in enumerate(methods):
    for col, color in enumerate([BATCH, CELLTYPE]):
        ax = axes[row][col]
        coords = adata.obsm["umap_" + rep]
        for value in adata.obs[color].unique():
            mask = (adata.obs[color] == value).values
            ax.scatter(coords[mask, 0], coords[mask, 1], s=2, label=str(value))
        ax.set_title("%s - by %s" % (name, color))
        ax.set_xticks([]); ax.set_yticks([])
        ax.legend(markerscale=4, fontsize=6, loc="best")

plt.tight_layout()
import os
os.makedirs("results", exist_ok=True)
plt.savefig("results/integration_comparison.png", dpi=130)
print()
print("saved comparison UMAPs to results/integration_comparison.png")
