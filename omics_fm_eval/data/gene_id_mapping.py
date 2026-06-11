"""Convert AnnData var_names between gene symbols and Ensembl gene IDs.

Geneformer requires Ensembl gene IDs (e.g. "ENSG00000139618"). Datasets like
PBMC ship with gene symbols (e.g. "BRCA2"), so this module maps symbols to
Ensembl IDs via pybiomart, dropping genes that can't be mapped.
"""

from anndata import AnnData


def symbols_to_ensembl(adata: AnnData, species: str = "hsapiens") -> AnnData:
    """Return a copy of `adata` with var_names replaced by Ensembl gene IDs.

    Genes with no mapping are dropped. Requires `pip install pybiomart`.
    """
    from pybiomart import Server

    server = Server(host="http://www.ensembl.org")
    dataset = server.marts["ENSEMBL_MART_ENSEMBL"].datasets[f"{species}_gene_ensembl"]
    mapping_df = dataset.query(attributes=["external_gene_name", "ensembl_gene_id"])
    mapping = dict(zip(mapping_df["Gene name"], mapping_df["Gene stable ID"]))

    ensembl_ids = adata.var_names.map(mapping)
    keep = ~ensembl_ids.isna()

    ad = adata[:, keep].copy()
    ad.var["gene_symbol"] = ad.var_names
    ad.var_names = ensembl_ids[keep]
    ad.var_names_make_unique()
    return ad
