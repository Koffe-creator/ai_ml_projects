"""Extract top marker genes per cluster from an AnnData object."""

import scanpy as sc
from anndata import AnnData


def top_markers_per_cluster(adata: AnnData, cluster_key: str = "leiden", n_genes: int = 15) -> dict[str, list[str]]:
    ad = adata.copy()
    sc.tl.rank_genes_groups(ad, groupby=cluster_key, method="wilcoxon")
    groups = ad.uns["rank_genes_groups"]["names"].dtype.names
    return {group: list(ad.uns["rank_genes_groups"]["names"][group][:n_genes]) for group in groups}
