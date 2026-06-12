# Omics Foundation Model Evaluation

A two-phase evaluation suite for single-cell transcriptomics. It asks one
question from two angles: **how well do different methods identify cell types
in scRNA-seq data, and is an expensive foundation model actually better than
simple baselines or a general-purpose LLM?**

## The two phases

### Phase 1 — Foundation-model benchmark (embeddings + linear probe)

Each method turns every cell into a vector (an *embedding*), then we test how
well that embedding captures cell-type identity:

1. Turn each cell's gene-expression profile into a vector
   - **PCA** (baseline): 1838 genes → 50 principal components
   - **Geneformer** (foundation model): genes → 768-dim transformer embedding
2. Train a logistic-regression *linear probe* on 80% of the cells to predict
   the known cell type from the embedding
3. Score accuracy / macro-F1 on the held-out 20%

Everything is held identical except the embedding, so the accuracy numbers are
a fair head-to-head: *does the 104M-parameter foundation model produce a more
cell-type-separable embedding than a 2-second PCA?*

### Phase 2 — LLM-as-evaluator (zero-shot annotation from marker genes)

A completely different approach that never touches embeddings:

1. For each cluster, extract its top marker genes (most differentially expressed)
2. Hand that list of **gene names** to a general-purpose LLM and ask it to name
   the cell type — zero-shot, no training, no expression values
3. A *second* LLM call judges whether the prediction is semantically equivalent
   to the ground-truth label (so "B cell" ≈ "B cells", "NK" ≈ "Natural Killer")

This tests whether an LLM's biological knowledge alone can annotate cell types,
and — by running it repeatedly — how *reliable* that annotation is. (Ambiguous
clusters like NK vs. cytotoxic T cells flip between runs; clear ones are stable.)

### Why both

Phase 1 and Phase 2 are different tools answering the same question. Run them
together and you can compare, on the same dataset: a classical baseline, a
transcriptomics foundation model, and a general-purpose LLM.

## Structure

- `data/` — dataset loading and gene ID mapping (symbol → Ensembl)
- `models/` — embedding models: classical baselines + foundation models
- `eval/` — Phase 1 tasks (linear-probe classification) and runner
- `llm_judge/` — Phase 2 pipeline (marker genes → LLM annotation → semantic judge)
- `results/` — outputs: `benchmark.json`, `llm_judge.csv`, `accuracy_log.txt`
- `scripts/` — one-off setup utilities
- `run_pipeline.py` — runs both phases end-to-end

## Setup

```bash
pip install -r requirements.txt
```

### Geneformer (Phase 1 foundation model)

Geneformer isn't on PyPI and must be installed from its HuggingFace repo:

```bash
git clone https://huggingface.co/ctheodoris/Geneformer
pip install ./Geneformer
pip install "transformers==4.46.0" loompy
```

Download the V2-104M checkpoint and place it at `models/checkpoints/geneformer-V2-104M/`
(needs `config.json` and `model.safetensors`):

```bash
cd Geneformer && git lfs pull --include="Geneformer-V2-104M/*"
cp -r Geneformer-V2-104M ../omics_fm_eval/models/checkpoints/geneformer-V2-104M
```

If you're on macOS / a CPU-only machine, the published package hardcodes
`device="cuda"` in a couple of places, which crashes on CPU. Patch it once:

```bash
python scripts/patch_geneformer_cpu.py
```

### API key (Phase 2 LLM judge)

The LLM judge needs an Anthropic API key. Copy the template and paste your key:

```bash
cp .env.example .env
# then edit .env and set ANTHROPIC_API_KEY=sk-ant-...
```

`.env` is gitignored and must never be committed. Get a key at
[console.anthropic.com](https://console.anthropic.com/settings/keys).

## Datasets

Pick with `--dataset`:

| Name | Cells | Cell types | Notes |
|---|---|---|---|
| `pbmc` | 2638 | 8 (coarse: B, CD4 T, CD8 T, NK, monocytes…) | PBMC 3k, the easy case |
| `pbmc68k` | 700 | 10 (fine-grained: T reg, naive vs memory T…) | PBMC 68k reduced, harder subtype resolution |

Both standardize their ground-truth labels into `adata.obs["cell_type"]`, so
the same commands work for either.

## Running

### Full pipeline (both phases)

```bash
python run_pipeline.py --dataset pbmc    --models pca geneformer
python run_pipeline.py --dataset pbmc68k --models pca geneformer
```

Prints a summary and writes:
- `results/benchmark.json` — accuracy for every method (Phase 1 + Phase 2)
- `results/llm_judge.csv` — per-cluster LLM prediction + match
- `results/accuracy_log.txt` — appends one `Accuracy = xxx` line per method, per run

> Run in the foreground so the log persists to your working tree.
> Geneformer on CPU takes ~25 min; drop it (`--models pca`) for a fast run.

### Phases individually

```bash
python -m eval.run_benchmark --dataset pbmc --models pca geneformer   # Phase 1 only
python -m llm_judge.run_llm_judge --dataset pbmc                      # Phase 2 only
```

## Example results

| Method | pbmc (coarse, 8 types) | pbmc68k (fine, 10 types) |
|---|---|---|
| PCA (embedding + linear probe) | 0.945 | 0.779 |
| Geneformer V2-104M | 0.778 | — |
| LLM judge (zero-shot from marker genes) | 0.75–0.88 | 0.500 |

Two findings worth a write-up:

1. **On coarse cell types, a general-purpose LLM — given only gene *names* —
   roughly matches the dedicated foundation model, and plain PCA beats both.**
2. **Granularity matters enormously.** On fine-grained subtypes (T reg, naive
   vs. memory T) the LLM collapses to generic categories ("T cell") and accuracy
   halves. The easy-dataset numbers oversell zero-shot LLM annotation.

The LLM's range reflects run-to-run variance on biologically ambiguous clusters
(see `accuracy_log.txt`).
