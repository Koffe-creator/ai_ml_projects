"""Perturb-seq analysis of the Papalexi 2021 ECCITE-seq CRISPR screen.

Steps:
  1. load the data and attach the perturbation labels to the RNA modality
  2. standard preprocessing (normalize, log, highly-variable genes, scale, PCA)
  3. Mixscape - separate cells that truly responded to a perturbation (KO) from
     "escapers" that carry a guide but look unperturbed (NP)
  4. E-distance - quantify how strong each perturbation's effect is vs. control
  5. save a ranked bar plot of perturbation strength

Run it with:
    python perturbseq_pipeline.py
"""

import os

import scanpy as sc
import pertpy as pt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- 1. load + attach labels ---
mdata = pt.data.papalexi_2021()
rna = mdata.mod["rna"].copy()
for col in ["perturbation", "gene_target", "NT"]:
    rna.obs[col] = mdata.obs[col].values
print("cells:", rna.n_obs, " genes:", rna.n_vars, " targets:", rna.obs["gene_target"].nunique())

# --- 2. preprocessing ---
sc.pp.normalize_total(rna)
sc.pp.log1p(rna)
sc.pp.highly_variable_genes(rna, subset=True)
sc.pp.scale(rna)
sc.pp.pca(rna)

# --- 3. Mixscape ---
ms = pt.tl.Mixscape()
ms.perturbation_signature(rna, pert_key="perturbation", control="NT")
ms.mixscape(rna, pert_key="gene_target", control="NT", layer="X_pert")

print("\nMixscape global classification (did cells actually respond?):")
print(rna.obs["mixscape_class_global"].value_counts())

# --- 4. E-distance: how far is each perturbation from the control? ---
distance = pt.tl.Distance(metric="edistance", obsm_key="X_pca")
edist = distance.onesided_distances(rna, groupby="gene_target", selected_group="NT")
edist = edist.drop("NT").sort_values(ascending=False)

print("\nStrongest perturbations by E-distance from control:")
print(edist.head(10))

# --- 5. plot ---
os.makedirs("results", exist_ok=True)
top = edist.head(15)
plt.figure(figsize=(7, 6))
plt.barh(top.index[::-1], top.values[::-1])
plt.xlabel("E-distance from non-targeting control")
plt.title("Perturbation strength (Papalexi 2021)")
plt.tight_layout()
plt.savefig("results/perturbation_strength.png", dpi=150)
print("\nsaved results/perturbation_strength.png")
