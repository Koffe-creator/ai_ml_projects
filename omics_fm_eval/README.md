# Omics Foundation Model Evaluation

Benchmark suite for single-cell transcriptomics foundation models against
classical baselines, plus an LLM-as-evaluator layer that uses general-purpose
LLMs to judge cell-type predictions and marker-gene biological plausibility.

## Structure

- `data/` — dataset loading and gene ID mapping (symbol -> Ensembl)
- `models/` — wrapper interfaces for foundation models + classical baselines
- `eval/` — benchmark tasks (cell type classification, batch integration) and runner
- `llm_judge/` — LLM-as-evaluator pipeline (marker gene -> LLM cell type prediction, agreement scoring)
- `results/` — benchmark output (JSON)
- `scripts/` — one-off setup utilities

## Setup

```bash
pip install -r requirements.txt
```

### Geneformer

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

## Run the benchmark

```bash
python -m eval.run_benchmark --dataset pbmc --models pca geneformer
```

Outputs a leaderboard JSON to `results/benchmark.json`, e.g.:

```json
{
  "pca": {"accuracy": 0.945, "macro_f1": 0.944},
  "geneformer": {"accuracy": 0.778, "macro_f1": 0.663}
}
```

`cell_type_classification` is a linear-probe (logistic regression) on the
model's cell embeddings, evaluated on an 80/20 stratified split.

## Phase 2: LLM-as-evaluator (in progress)

For each model's predicted clusters, extract top marker genes (`llm_judge/marker_genes.py`),
query an LLM for cell-type prediction (`llm_judge/llm_annotator.py`, requires
`ANTHROPIC_API_KEY`), and compare against ground truth and the foundation
model's own prediction (`llm_judge/agreement.py`).
