"""
simulation.py
=============
Simulates a realistic single-cell RNA-seq count matrix modelled after
10x Chromium PBMC data.  I designed this to produce a dataset that is
statistically close to real scRNA-seq data while remaining fully
reproducible from a fixed random seed.

Design choices
--------------
- 6 immune cell types with biologically motivated proportions
- Counts drawn from a Negative Binomial distribution (dispersion = 5)
  to match the overdispersion observed in real 10x libraries
- 4 donor batches with multiplicative library-size and gene-level noise
- 2 conditions (Control / Treatment) with a simulated treatment effect
  on 25 genes
- Three classes of outliers are intentionally injected so that QC
  filtering in the downstream analysis script can demonstrate their
  removal:
      * Doublets      — two cells merged in one droplet (high UMI, high genes)
      * Dying cells   — high mitochondrial fraction, low cytoplasmic counts
      * Empty droplets — ambient RNA only (very low UMI)

Output
------
All files are written to  ./Output/
    simulated_counts_raw.h5ad   — raw AnnData object (pre-QC)
    simulation_metadata.csv     — per-cell ground-truth annotations
"""

import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad
from scipy.stats import nbinom

# ── reproducibility ───────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "Output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── simulation parameters ─────────────────────────────────────────────────────
N_GENES             = 3_000
N_MT_GENES          = 50       # last 50 genes labelled MT-*
N_CELLS_BASE        = 1_500    # true cells per donor batch
N_BATCHES           = 4        # donors; batches 0-1 → Control, 2-3 → Treatment
N_OUTLIER_PER_BATCH = 40       # ~13 of each outlier class per batch

CELL_TYPES = ["CD4_T", "CD8_T", "B_cell", "NK_cell", "Mono_Classical", "DendriticCell"]
CT_PROPS   = [0.28,    0.18,    0.20,     0.10,      0.18,             0.06]

# Gene-index blocks that define each cell type's marker programme
CT_MARKER_BLOCKS = {
    "CD4_T":          (0,   30),
    "CD8_T":          (30,  60),
    "B_cell":         (60,  90),
    "NK_cell":        (90,  120),
    "Mono_Classical": (120, 150),
    "DendriticCell":  (150, 180),
}

# Treatment effect: genes 200-214 up-regulated, 215-224 down-regulated
TREATMENT_UP_IDX   = list(range(200, 215))
TREATMENT_DOWN_IDX = list(range(215, 225))


# ── helpers ───────────────────────────────────────────────────────────────────

def _build_gene_names():
    names  = [f"HK_{i:04d}" for i in range(N_GENES - N_MT_GENES)]
    names += [f"MT-{i:02d}"  for i in range(N_MT_GENES)]
    return names


def _make_ct_mean(ct_name, base_low=0.05, base_high=1.5, marker_high=8.0):
    """Return a per-gene mean expression vector for one cell type."""
    mu = np.random.uniform(base_low, base_high, N_GENES).astype(np.float32)
    s, e = CT_MARKER_BLOCKS[ct_name]
    mu[s:e] = np.random.uniform(marker_high * 0.7, marker_high * 1.3, e - s)
    mu[-N_MT_GENES:] = np.random.uniform(0.02, 0.15, N_MT_GENES)
    return mu


def _nb_sample(mu, dispersion=5.0, size=1):
    """Sample from NB(mu, dispersion) — shape (size, n_genes)."""
    mu = np.clip(mu, 1e-6, None)
    p  = dispersion / (dispersion + mu)
    return nbinom.rvs(dispersion, p, size=(size, len(mu))).astype(np.float32)


def _batch_scalars(batch_id):
    """Donor-specific library-size scalar and gene-level noise vector."""
    rng        = np.random.RandomState(batch_id * 7 + 13)
    lib_scale  = rng.uniform(0.7, 1.4)
    gene_scale = rng.lognormal(0.0, 0.25, N_GENES).astype(np.float32)
    return lib_scale, gene_scale


def _apply_treatment(mu, condition):
    mu = mu.copy()
    if condition == "Treatment":
        mu[TREATMENT_UP_IDX]   *= np.random.uniform(2.5, 4.0, len(TREATMENT_UP_IDX))
        mu[TREATMENT_DOWN_IDX] *= np.random.uniform(0.2, 0.4, len(TREATMENT_DOWN_IDX))
    return mu


# ── per-batch simulation ──────────────────────────────────────────────────────

def simulate_batch(batch_id, ct_mu):
    rng       = np.random.RandomState(batch_id * 99 + 1)
    condition = "Control" if batch_id < 2 else "Treatment"
    lib_scale, gene_scale = _batch_scalars(batch_id)

    ct_labels  = rng.choice(CELL_TYPES, size=N_CELLS_BASE, p=CT_PROPS)
    true_counts = []
    for ct in ct_labels:
        mu   = ct_mu[ct] * lib_scale * gene_scale
        mu   = _apply_treatment(mu, condition)
        true_counts.append(_nb_sample(mu, dispersion=5.0)[0])
    counts = np.vstack(true_counts)

    n_each = N_OUTLIER_PER_BATCH // 3

    # Doublets: sum two randomly chosen real cells
    idx1     = rng.choice(N_CELLS_BASE, n_each)
    idx2     = rng.choice(N_CELLS_BASE, n_each)
    doublets = (counts[idx1] + counts[idx2]).astype(np.float32)

    # Dying cells: cytoplasmic RNA suppressed, MT RNA elevated
    dying = np.zeros((n_each, N_GENES), dtype=np.float32)
    for i in range(n_each):
        base          = np.ones(N_GENES, dtype=np.float32) * 0.3
        base[-N_MT_GENES:] = rng.uniform(4.0, 10.0, N_MT_GENES)
        dying[i]      = _nb_sample(base, dispersion=3.0)[0]

    # Empty droplets: ambient RNA only
    empty = np.zeros((n_each, N_GENES), dtype=np.float32)
    ambient = np.ones(N_GENES, dtype=np.float32) * 0.01
    for i in range(n_each):
        empty[i] = _nb_sample(ambient, dispersion=2.0)[0]

    all_counts = np.vstack([counts, doublets, dying, empty])
    n_outliers = n_each * 3
    labels = np.concatenate([
        ct_labels,
        ["Doublet"] * n_each,
        ["Dying"]   * n_each,
        ["Empty"]   * n_each,
    ])
    is_outlier = np.concatenate([
        np.zeros(N_CELLS_BASE, dtype=bool),
        np.ones(n_outliers,    dtype=bool),
    ])

    meta = pd.DataFrame({
        "cell_type_true": labels,
        "batch":          f"Donor_{batch_id + 1}",
        "condition":      condition,
        "is_outlier":     is_outlier,
    })
    return all_counts, meta


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("Building cell-type mean expression programmes …")
    gene_names = _build_gene_names()
    ct_mu      = {ct: _make_ct_mean(ct) for ct in CELL_TYPES}

    print(f"Simulating {N_BATCHES} donor batches ({N_CELLS_BASE} cells + "
          f"{N_OUTLIER_PER_BATCH} outliers each) …")
    batches = [simulate_batch(i, ct_mu) for i in range(N_BATCHES)]

    counts_all = np.vstack([b[0] for b in batches])
    meta_all   = pd.concat([b[1] for b in batches], ignore_index=True)

    adata = ad.AnnData(
        X   = sp.csr_matrix(counts_all),
        obs = meta_all,
        var = pd.DataFrame(index=gene_names),
    )
    adata.var["is_mt"]    = adata.var_names.str.startswith("MT-")
    adata.obs_names       = [f"cell_{i}" for i in range(adata.n_obs)]
    adata.uns["sim_seed"] = SEED

    print(f"\nRaw matrix : {adata.n_obs:,} cells × {adata.n_vars:,} genes")
    print(f"Outliers   : {meta_all.is_outlier.sum()} "
          f"({100 * meta_all.is_outlier.mean():.1f} %)")
    print(f"\nCell-type distribution (including outliers):")
    print(meta_all.cell_type_true.value_counts().to_string())

    # ── save ─────────────────────────────────────────────────────────────────
    h5ad_path = os.path.join(OUTPUT_DIR, "simulated_counts_raw.h5ad")
    csv_path  = os.path.join(OUTPUT_DIR, "simulation_metadata.csv")

    adata.write_h5ad(h5ad_path)
    meta_all.to_csv(csv_path, index=False)

    print(f"\nSaved → {h5ad_path}")
    print(f"Saved → {csv_path}")
    print("\nSimulation complete.  Run analysis.py to proceed.")


if __name__ == "__main__":
    main()
