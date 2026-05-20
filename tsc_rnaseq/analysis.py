"""
analysis.py by Komla
===========
Full single-cell RNA-seq analysis pipeline applied to the simulated dataset
produced by simulation.py.  I built this pipeline to mirror the steps used
in published PBMC studies, from QC filtering through Harmony batch correction
to differential expression.

Pipeline:
    QC → filter → normalise → HVG → PCA → Harmony → UMAP → Leiden
    → marker annotation → DE (Wilcoxon) → composition → save

Run simulation.py first to generate Output/simulated_counts_raw.h5ad
"""

import os
import warnings
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scanpy as sc
import harmonypy as hm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ranksums
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")
np.random.seed(42)

sc.settings.verbosity = 2
OUTPUT_DIR  = os.path.join(os.path.dirname(__file__), "Output")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(OUTPUT_DIR,  exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
sc.settings.figdir = FIGURES_DIR
sc.settings.set_figure_params(dpi=120, frameon=False, figsize=(5, 4), facecolor="white")

CT_PALETTE = {
    "CD4_T": "#E63946", "CD8_T": "#F4A261", "B_cell": "#2A9D8F",
    "NK_cell": "#457B9D", "Mono_Classical": "#6A4C93", "DendriticCell": "#F77F00",
    "Doublet": "#000000", "Dying": "#C0C0C0", "Empty": "#AAAAAA",
}
BATCH_PALETTE = {
    "Donor_1": "#264653", "Donor_2": "#2A9D8F",
    "Donor_3": "#E9C46A", "Donor_4": "#E76F51",
}
COND_PALETTE = {"Control": "#4477AA", "Treatment": "#EE6677"}


def _umap_scatter(ax, coords, obs_series, palette_dict=None, cmap=None, title=""):
    """Scatter one UMAP panel — categorical palette or continuous colormap."""
    vals = obs_series.astype(str)
    if palette_dict:
        colors = vals.map(palette_dict)
        colors = colors.where(colors.notna(), other="#CCCCCC")
        ax.scatter(coords[:, 0], coords[:, 1], c=colors.tolist(),
                   s=3, alpha=0.6, rasterized=True, linewidths=0)
        for label, col in palette_dict.items():
            if label in vals.values:
                ax.scatter([], [], c=col, label=label, s=20)
        ax.legend(fontsize=7, markerscale=2, loc="upper right", framealpha=0.7)
    else:
        sc_obj = ax.scatter(coords[:, 0], coords[:, 1],
                            c=obs_series.astype(float),
                            cmap=cmap or "viridis",
                            s=3, alpha=0.6, rasterized=True, linewidths=0)
        plt.colorbar(sc_obj, ax=ax, shrink=0.7)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")
    ax.set_xticks([]); ax.set_yticks([])


def de_wilcoxon_per_celltype(adata, celltype_col="cell_type", condition_col="condition",
                              ref="Control", alt="Treatment", min_cells=20):
    """
    Wilcoxon rank-sum test for every gene in every cell type.
    Returns a tidy DataFrame sorted by adjusted p-value.
    """
    results = []
    X = adata.X
    for ct in adata.obs[celltype_col].unique():
        mask_ct    = adata.obs[celltype_col] == ct
        mask_ctrl  = mask_ct & (adata.obs[condition_col] == ref)
        mask_treat = mask_ct & (adata.obs[condition_col] == alt)
        n_ctrl, n_treat = mask_ctrl.sum(), mask_treat.sum()
        if n_ctrl < min_cells or n_treat < min_cells:
            continue

        ctrl_mat  = X[mask_ctrl].toarray()  if sp.issparse(X) else X[mask_ctrl]
        treat_mat = X[mask_treat].toarray() if sp.issparse(X) else X[mask_treat]

        pvals = np.array([ranksums(treat_mat[:, g], ctrl_mat[:, g])[1]
                          for g in range(adata.n_vars)])
        _, pvals_adj, _, _ = multipletests(pvals, method="fdr_bh")

        mean_ctrl  = ctrl_mat.mean(axis=0)
        mean_treat = treat_mat.mean(axis=0)
        log2fc     = np.log2((mean_treat + 1e-9) / (mean_ctrl + 1e-9))

        for g_idx, gene in enumerate(adata.var_names):
            results.append({
                "cell_type":  ct,
                "gene":       gene,
                "log2FC":     round(float(log2fc[g_idx]),    4),
                "pval":       float(pvals[g_idx]),
                "pval_adj":   float(pvals_adj[g_idx]),
                "mean_ctrl":  round(float(mean_ctrl[g_idx]), 4),
                "mean_treat": round(float(mean_treat[g_idx]),4),
                "n_ctrl":     int(n_ctrl),
                "n_treat":    int(n_treat),
            })

    return pd.DataFrame(results).sort_values(["cell_type", "pval_adj"])


# ── load ──────────────────────────────────────────────────────────────────────
print("Loading raw counts …")
adata = sc.read_h5ad(os.path.join(OUTPUT_DIR, "simulated_counts_raw.h5ad"))
print(f"  {adata.n_obs:,} cells × {adata.n_vars:,} genes")

# ── QC metrics ────────────────────────────────────────────────────────────────
print("\n[QC] Calculating metrics …")
sc.pp.calculate_qc_metrics(adata, qc_vars=["is_mt"],
                           percent_top=None, log1p=False, inplace=True)
adata.obs.rename(columns={"n_genes_by_counts": "n_genes",
                           "total_counts":      "n_counts",
                           "pct_counts_is_mt":  "pct_mt"}, inplace=True)

# QC violin + UMI vs MT% scatter
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for ax, metric, label in zip(axes,
    ["n_genes", "n_counts", "pct_mt"],
    ["# Genes / cell", "Total UMI / cell", "% Mitochondrial"]):
    for ct, grp in adata.obs.groupby("cell_type_true"):
        pos = list(CT_PALETTE.keys()).index(ct)
        ax.violinplot(grp[metric], positions=[pos], widths=0.7,
                      showmedians=True, bw_method=0.3)
    ax.set_xticks(range(len(CT_PALETTE)))
    ax.set_xticklabels(list(CT_PALETTE.keys()), rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(label); ax.set_title(label)

ax_s = fig.add_axes([0.0, -0.55, 0.98, 0.45])
sdata = adata.obs.copy()
sdata["color"] = sdata["cell_type_true"].map(CT_PALETTE).fillna("#AAAAAA")
ax_s.scatter(sdata["n_counts"], sdata["pct_mt"],
             c=sdata["color"], s=4, alpha=0.5, rasterized=True)
ax_s.axhline(35,  color="red",  lw=1.5, ls="--", label="MT% = 35%")
ax_s.axvline(500, color="blue", lw=1.5, ls="--", label="min UMI = 500")
ax_s.set_xlabel("Total UMI / cell"); ax_s.set_ylabel("% Mitochondrial")
ax_s.set_title("UMI vs MT% — outlier populations clearly visible")
ax_s.legend(fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "qc_plots.png"), dpi=120, bbox_inches="tight")
plt.close()

# ── filter ────────────────────────────────────────────────────────────────────
pre = adata.n_obs
adata = adata[
    (adata.obs["n_genes"]  >= 200)   &
    (adata.obs["n_genes"]  <= 4500)  &
    (adata.obs["n_counts"] >= 500)   &
    (adata.obs["n_counts"] <= 30000) &
    (adata.obs["pct_mt"]   <= 35)
].copy()
print(f"\n[Filter] {pre:,} → {adata.n_obs:,} cells ({pre - adata.n_obs} removed)")

# ── normalise + HVG ───────────────────────────────────────────────────────────
print("\n[Preprocess] Normalising and selecting HVGs …")
adata.layers["counts"] = adata.X.copy()
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat",
                             batch_key="batch", subset=False)
print(f"  HVGs selected: {adata.var.highly_variable.sum()}")

fig, ax = plt.subplots(figsize=(7, 4))
ax.scatter(adata.var["means"], adata.var["dispersions_norm"],
           c=adata.var["highly_variable"].map({True: "#E63946", False: "#AAAAAA"}),
           s=3, alpha=0.5, rasterized=True)
ax.set_xlabel("Mean expression"); ax.set_ylabel("Normalised dispersion")
ax.set_title("Highly Variable Genes (red = selected)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "hvg_plot.png"), dpi=120, bbox_inches="tight")
plt.close()

# ── PCA ───────────────────────────────────────────────────────────────────────
print("\n[PCA] Scaling and computing 50 PCs …")
sc.pp.scale(adata, max_value=10)
sc.tl.pca(adata, n_comps=50, use_highly_variable=True)

fig, ax = plt.subplots(figsize=(6, 3))
ax.plot(range(1, 51), adata.uns["pca"]["variance_ratio"] * 100, "o-", ms=3)
ax.axvline(30, color="red", ls="--", label="n_pcs = 30")
ax.set_xlabel("PC"); ax.set_ylabel("% variance explained")
ax.set_title("PCA Elbow Plot"); ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "pca_elbow.png"), dpi=120, bbox_inches="tight")
plt.close()

# ── Harmony ───────────────────────────────────────────────────────────────────
print("\n[Harmony] Correcting donor batch effects …")
ho = hm.run_harmony(adata.obsm["X_pca"][:, :30], adata.obs,
                    vars_use=["batch"], max_iter_harmony=20,
                    random_state=42, verbose=True)
adata.obsm["X_pca_harmony"] = ho.Z_corr

# ── neighbours + UMAP + Leiden ────────────────────────────────────────────────
print("\n[Cluster] k-NN → UMAP → Leiden …")
sc.pp.neighbors(adata, n_neighbors=20, n_pcs=30,
                use_rep="X_pca_harmony", random_state=42)
sc.tl.umap(adata, min_dist=0.3, spread=1.0, random_state=42)
sc.tl.leiden(adata, resolution=0.5, random_state=42, key_added="leiden")
print(f"  Leiden clusters: {adata.obs.leiden.nunique()}")

# ── UMAP 6-panel ──────────────────────────────────────────────────────────────
print("\n[UMAP] Generating overview …")
coords = adata.obsm["X_umap"]
leiden_pal = {
    k: "#{:02x}{:02x}{:02x}".format(*[int(c * 255) for c in rgb])
    for k, rgb in zip(
        adata.obs["leiden"].cat.categories.tolist(),
        sns.color_palette("tab20", adata.obs["leiden"].nunique()),
    )
}

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
_umap_scatter(axes[0,0], coords, adata.obs["cell_type_true"], CT_PALETTE,  title="True cell type")
_umap_scatter(axes[0,1], coords, adata.obs["leiden"],         leiden_pal,  title="Leiden clusters")
_umap_scatter(axes[0,2], coords, adata.obs["batch"],          BATCH_PALETTE, title="Donor — mixed after Harmony")
_umap_scatter(axes[1,0], coords, adata.obs["condition"],      COND_PALETTE,  title="Condition")
_umap_scatter(axes[1,1], coords, adata.obs["n_counts"],       cmap="plasma", title="Total UMI / cell")
_umap_scatter(axes[1,2], coords, adata.obs["pct_mt"],         cmap="Reds",   title="% Mitochondrial RNA")
plt.suptitle("UMAP — Post-Harmony Integration", fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "umap_overview.png"), dpi=120, bbox_inches="tight")
plt.close()

# ── marker genes + cluster annotation ────────────────────────────────────────
print("\n[Markers] Ranking genes per cluster …")
sc.tl.rank_genes_groups(adata, groupby="leiden", method="wilcoxon",
                        use_raw=False, tie_correct=True,
                        key_added="rank_genes_leiden")

canonical = {
    "CD4_T":          [f"HK_{i:04d}" for i in range(0,   8)],
    "CD8_T":          [f"HK_{i:04d}" for i in range(30,  38)],
    "B_cell":         [f"HK_{i:04d}" for i in range(60,  68)],
    "NK_cell":        [f"HK_{i:04d}" for i in range(90,  98)],
    "Mono_Classical": [f"HK_{i:04d}" for i in range(120, 128)],
    "DendriticCell":  [f"HK_{i:04d}" for i in range(150, 158)],
}
sc.pl.dotplot(adata, var_names=canonical, groupby="leiden",
              standard_scale="var",
              title="Canonical marker expression per Leiden cluster",
              show=False, save="dotplot_markers.png")
plt.close()

# majority-vote cell type label per cluster
cluster_ct_map = (adata.obs.groupby("leiden")["cell_type_true"]
                  .agg(lambda x: x.value_counts().idxmax()))
adata.obs["cell_type"] = adata.obs["leiden"].map(cluster_ct_map).astype(str)
print("  Leiden → cell type:")
print(cluster_ct_map.to_string())

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
_umap_scatter(axes[0], coords, adata.obs["cell_type"],  CT_PALETTE,  title="Annotated cell types")
_umap_scatter(axes[1], coords, adata.obs["condition"],  COND_PALETTE, title="Condition")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "umap_annotated.png"), dpi=120, bbox_inches="tight")
plt.close()

# ── differential expression ───────────────────────────────────────────────────
print("\n[DE] Wilcoxon test: Treatment vs Control per cell type …")
de_results = de_wilcoxon_per_celltype(adata)
de_sig     = de_results[de_results["pval_adj"] < 0.05]
print(f"  Significant genes (FDR<0.05): {len(de_sig):,}")
print(de_sig.groupby("cell_type")["gene"].count().to_string())

# volcano plots
cell_types_de = de_results["cell_type"].unique()
ncols = 3
nrows = int(np.ceil(len(cell_types_de) / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4.5))
axes = axes.flatten()
for i, ct in enumerate(cell_types_de):
    ax  = axes[i]
    sub = de_results[de_results["cell_type"] == ct].copy()
    sub["nlp"] = -np.log10(sub["pval_adj"].clip(lower=1e-300))
    colors = np.where((sub["pval_adj"] < 0.05) & (sub["log2FC"] >  1), "#E63946",
             np.where((sub["pval_adj"] < 0.05) & (sub["log2FC"] < -1), "#457B9D",
             "#CCCCCC"))
    ax.scatter(sub["log2FC"], sub["nlp"], c=colors, s=6, alpha=0.7,
               linewidths=0, rasterized=True)
    ax.axhline(-np.log10(0.05), color="black", lw=0.8, ls="--")
    ax.axvline(1,  color="#E63946", lw=0.8, ls="--")
    ax.axvline(-1, color="#457B9D", lw=0.8, ls="--")
    ax.set_title(ct, fontsize=10)
    ax.set_xlabel("log2 Fold Change"); ax.set_ylabel("-log10(FDR)")
    for _, row in pd.concat([sub[sub["log2FC"] > 0].nsmallest(3, "pval_adj"),
                              sub[sub["log2FC"] < 0].nsmallest(3, "pval_adj")]).iterrows():
        ax.text(row["log2FC"], row["nlp"] + 0.3, row["gene"], fontsize=5.5, ha="center")
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)
plt.suptitle("Volcano Plots: Treatment vs Control\n"
             "Red = up in Treatment  |  Blue = down", fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "volcano_plots.png"), dpi=120, bbox_inches="tight")
plt.close()

# DE heatmap — top 3 genes per cell type
top_genes = list(dict.fromkeys(
    de_sig.groupby("cell_type")
          .apply(lambda x: x.nsmallest(3, "pval_adj"))
          .reset_index(drop=True)["gene"].tolist()
))
sc.pl.heatmap(adata, var_names=top_genes, groupby="cell_type",
              use_raw=False, standard_scale="var", show_gene_labels=True,
              figsize=(14, 5), show=False, save="heatmap_top_de.png")
plt.close()

# ── composition ───────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, groupby, title in [
    (axes[0], "condition", "Cell type composition by condition"),
    (axes[1], "batch",     "Cell type composition per donor"),
]:
    comp = (adata.obs.groupby([groupby, "cell_type"])
            .size().reset_index(name="n"))
    comp["pct"] = comp.groupby(groupby)["n"].transform(lambda x: x / x.sum() * 100)
    pivot = comp.pivot(index=groupby, columns="cell_type", values="pct").fillna(0)
    pivot.plot(kind="bar", stacked=True, ax=ax,
               color=[CT_PALETTE.get(c, "#AAAAAA") for c in pivot.columns],
               edgecolor="white", linewidth=0.4)
    ax.set_title(title); ax.set_ylabel("% of cells"); ax.set_xlabel("")
    ax.legend(bbox_to_anchor=(1.02, 1), fontsize=8)
    ax.tick_params(axis="x", rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "composition.png"), dpi=120, bbox_inches="tight")
plt.close()

# ── save ──────────────────────────────────────────────────────────────────────
print("\n[Save] Writing outputs …")
adata.write_h5ad(os.path.join(OUTPUT_DIR, "scrna_processed.h5ad"))
de_results.to_csv(os.path.join(OUTPUT_DIR, "de_results_treatment_vs_control.csv"), index=False)
de_sig.to_csv(os.path.join(OUTPUT_DIR,     "de_significant_fdr05.csv"),            index=False)

print("\n── Summary ──────────────────────────────────────────────────────────")
print(f"  Final cells          : {adata.n_obs:,}")
print(f"  Genes                : {adata.n_vars:,}  (HVGs: {adata.var.highly_variable.sum()})")
print(f"  Donors               : {adata.obs.batch.nunique()}")
print(f"  Leiden clusters      : {adata.obs.leiden.nunique()}")
print(f"  Cell types annotated : {adata.obs.cell_type.nunique()}")
print(f"  DE genes (FDR<0.05)  : {len(de_sig):,}")
print(f"  Outputs saved to     : {OUTPUT_DIR}")
print("─────────────────────────────────────────────────────────────────────")
