# do_moable — Chemical Structure-Based Pathway Analysis

A Jupyter notebook pipeline that takes a compound's **SMILES string** as input and returns **enriched biological pathways** using a pre-trained drug encoder model.

## How it works

1. **SMILES → Morgan Fingerprint** — converts the input chemical structure into a 2048-bit Morgan fingerprint using RDKit.
2. **Fingerprint → Drug Embedding** — passes the fingerprint through a pre-trained `DrugEncoder` neural network (`moable.pth`) to produce a latent chemical embedding.
3. **Cosine Similarity** — computes similarity between the drug embedding and a library of gene perturbation (GP) embeddings (`GP_embedding_dict`).
4. **Pathway Enrichment (GSEA)** — ranks genes by connectivity score and runs gene set enrichment analysis against the **KEGG 2016** pathway database using [GSEApy](https://gseapy.readthedocs.io/).

## Example

Input SMILES (Kavain):
```
COC1=CC(=O)OC(C1)/C=C/c2ccccc2
```

Top enriched pathways:
| Pathway | NES | NOM p-val |
|---|---|---|
| p53 signaling pathway | 2.46 | 0.0 |
| MicroRNAs in cancer | 2.34 | 0.0 |
| Chronic myeloid leukemia | 2.33 | 0.0 |
| Acute myeloid leukemia | 2.27 | 0.0 |
| Neurotrophin signaling pathway | 2.25 | 0.0 |

## Requirements

```bash
pip install torch rdkit gseapy scikit-learn pandas numpy
```

## Data files (not included in repo — too large for GitHub)

| File | Description |
|---|---|
| `moable.pth` | Pre-trained DrugEncoder model weights |
| `data/GP_embedding_dict.zip` | Gene perturbation embedding library |
| `data/GP_sig_df.pkl` | Gene perturbation signature metadata |

## Usage

Open `final_myscript.ipynb` and update the `drug_dict` with your compound:

```python
drug_dict = {
    "your_compound": "YOUR_SMILES_STRING_HERE"
}
```

Then run all cells. Results are saved to `data/output/`.
