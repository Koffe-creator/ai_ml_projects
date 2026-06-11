"""Wrappers for single-cell foundation models."""

import numpy as np
from anndata import AnnData

from .base import EmbeddingModel


class GeneformerModel(EmbeddingModel):
    """Wrapper around Geneformer (HuggingFace transformers, runs on CPU).

    `model_dir` should point to a downloaded Geneformer checkpoint, e.g.
    `Geneformer-V2-104M` from https://huggingface.co/ctheodoris/Geneformer

    `adata.var_names` must be Ensembl gene IDs (the tokenizer maps gene IDs
    to tokens via its internal dictionary) - see `data/gene_id_mapping.py`.
    """

    name = "geneformer"

    def __init__(self, model_dir: str, emb_layer: int = -1, model_version: str = "V2"):
        self.model_dir = model_dir
        self.emb_layer = emb_layer
        self.model_version = model_version

    def embed(self, adata: AnnData) -> np.ndarray:
        import os
        import tempfile

        import geneformer
        from geneformer import EmbExtractor, TranscriptomeTokenizer

        # V1 checkpoints (e.g. 12L-30M) use the gc30M gene dictionaries,
        # not the package default gc104M (V2) dictionaries.
        if self.model_version == "V1":
            dict_dir = os.path.join(os.path.dirname(geneformer.__file__), "gene_dictionaries_30m")
            token_dictionary_file = os.path.join(dict_dir, "token_dictionary_gc30M.pkl")
            gene_median_file = os.path.join(dict_dir, "gene_median_dictionary_gc30M.pkl")
            gene_mapping_file = os.path.join(dict_dir, "ensembl_mapping_dict_gc30M.pkl")
        else:
            pkg_dir = os.path.dirname(geneformer.__file__)
            token_dictionary_file = os.path.join(pkg_dir, "token_dictionary_gc104M.pkl")
            gene_median_file = os.path.join(pkg_dir, "gene_median_dictionary_gc104M.pkl")
            gene_mapping_file = os.path.join(pkg_dir, "ensembl_mapping_dict_gc104M.pkl")

        with tempfile.TemporaryDirectory() as tmpdir:
            h5ad_dir = os.path.join(tmpdir, "h5ad")
            tok_dir = os.path.join(tmpdir, "tokenized")
            os.makedirs(h5ad_dir, exist_ok=True)
            os.makedirs(tok_dir, exist_ok=True)

            ad = adata.copy()
            ad.obs["n_counts"] = np.asarray(ad.X.sum(axis=1)).flatten()
            ad.write_h5ad(os.path.join(h5ad_dir, "data.h5ad"))

            tokenizer = TranscriptomeTokenizer(
                nproc=4,
                model_version=self.model_version,
                token_dictionary_file=token_dictionary_file,
                gene_median_file=gene_median_file,
                gene_mapping_file=gene_mapping_file,
                use_h5ad_index=True,
            )
            tokenizer.tokenize_data(h5ad_dir, tok_dir, "tokenized", file_format="h5ad")

            extractor = EmbExtractor(
                model_type="Pretrained",
                emb_layer=self.emb_layer,
                emb_mode="cell",
                cell_emb_style="mean_pool",
                forward_batch_size=8,
                max_ncells=adata.n_obs,
                nproc=4,
                model_version=self.model_version,
                token_dictionary_file=token_dictionary_file,
            )
            embs = extractor.extract_embs(
                self.model_dir,
                os.path.join(tok_dir, "tokenized.dataset"),
                tmpdir,
                "embs",
            )
        return embs.values
