"""Query an LLM to predict cell type from a marker gene list.

Requires ANTHROPIC_API_KEY env var.
"""

import os

from anthropic import Anthropic

PROMPT_TEMPLATE = """You are an expert in single-cell transcriptomics.
Given the following top marker genes for a cell cluster, predict the most
likely cell type. Respond with only the cell type name, nothing else.

Marker genes: {genes}
"""


def predict_cell_type(genes: list[str], model: str = "claude-sonnet-4-6") -> str:
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    prompt = PROMPT_TEMPLATE.format(genes=", ".join(genes))
    response = client.messages.create(
        model=model,
        max_tokens=50,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()


def annotate_clusters(marker_dict: dict[str, list[str]], model: str = "claude-sonnet-4-6") -> dict[str, str]:
    return {cluster: predict_cell_type(genes, model=model) for cluster, genes in marker_dict.items()}
