"""Cell-type annotation with scANVI (label transfer on top of scVI).

scVI gave the best integrated latent space, so we build on it. scANVI is its
semi-supervised extension: given cell-type labels for some cells, it predicts
the labels for the rest.

To test it honestly we hide 30% of the labels ("query"), train scANVI on the
remaining 70% ("reference"), predict the hidden cells, and score the predictions
against the truth.

    python scanvi_celltyping.py
"""

import numpy as np
import pandas as pd
import scanpy as sc
import scvi
from sklearn.metrics import accuracy_score, classification_report
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BATCH = "cell_source"
CELLTYPE = "cell_type"

# --- load; keep raw counts ---
adata = scvi.data.heart_cell_atlas_subsampled()
sc.pp.subsample(adata, n_obs=12000, random_state=0)
adata.layers["counts"] = adata.X.copy()
sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat_v3",
                            layer="counts", subset=True)

# --- hide 30% of labels to act as the unlabelled query ---
rng = np.random.default_rng(0)
is_query = rng.random(adata.n_obs) < 0.30
adata.obs["labels"] = adata.obs[CELLTYPE].astype(str)
adata.obs.loc[is_query, "labels"] = "Unknown"
print("reference cells:", int((~is_query).sum()), " query (hidden):", int(is_query.sum()))

# --- scVI first (integrated latent space) ---
scvi.model.SCVI.setup_anndata(adata, layer="counts", batch_key=BATCH)
scvi_model = scvi.model.SCVI(adata)
scvi_model.train(max_epochs=40)

# --- scANVI on top: learns to predict the labels ---
scanvi_model = scvi.model.SCANVI.from_scvi_model(
    scvi_model, labels_key="labels", unlabeled_category="Unknown")
scanvi_model.train(max_epochs=20)

adata.obs["predicted"] = scanvi_model.predict()

# --- score the predictions on the hidden query cells only ---
truth = adata.obs[CELLTYPE].astype(str)[is_query]
pred = adata.obs["predicted"][is_query]
print()
print("accuracy on the hidden query cells: %.3f" % accuracy_score(truth, pred))
print()
print(classification_report(truth, pred, zero_division=0))

# --- UMAP of the scANVI latent, coloured by true and predicted labels ---
adata.obsm["X_scANVI"] = scanvi_model.get_latent_representation()
sc.pp.neighbors(adata, use_rep="X_scANVI")
sc.tl.umap(adata)

# one fixed colour per cell type, shared across both panels
cell_types = sorted(set(adata.obs[CELLTYPE]) | set(adata.obs["predicted"]))
colors = plt.cm.tab20(np.linspace(0, 1, len(cell_types)))
color_map = dict(zip(cell_types, colors))

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, color in zip(axes, [CELLTYPE, "predicted"]):
    coords = adata.obsm["X_umap"]
    for value in cell_types:
        mask = (adata.obs[color] == value).values
        ax.scatter(coords[mask, 0], coords[mask, 1], s=2, color=color_map[value], label=value)
    ax.set_title("scANVI latent - %s" % ("true labels" if color == CELLTYPE else "predicted"))
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(markerscale=4, fontsize=6, loc="best")
plt.tight_layout()

import os
os.makedirs("results", exist_ok=True)
plt.savefig("results/scanvi_celltyping.png", dpi=130)
print("saved results/scanvi_celltyping.png")
