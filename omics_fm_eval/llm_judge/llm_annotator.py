"""Query an LLM to predict cell type from a marker gene list.

Reads ANTHROPIC_API_KEY from the environment, loading a local `.env` file
if present (see `.env.example`).
"""

import os

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()  # loads ANTHROPIC_API_KEY from a .env file if one exists

PROMPT_TEMPLATE = """You are an expert in single-cell transcriptomics.
Given the following top marker genes for a cell cluster, predict the most
likely cell type. Respond with only the cell type name, nothing else.

Marker genes: {genes}
"""


def predict_cell_type(genes: list[str], model: str = "claude-haiku-4-5") -> str:
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    prompt = PROMPT_TEMPLATE.format(genes=", ".join(genes))
    response = client.messages.create(
        model=model,
        max_tokens=50,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()


def annotate_clusters(marker_dict: dict[str, list[str]], model: str = "claude-haiku-4-5") -> dict[str, str]:
    return {cluster: predict_cell_type(genes, model=model) for cluster, genes in marker_dict.items()}


JUDGE_TEMPLATE = """You are evaluating single-cell annotation results.
Two cell-type labels are given. Answer "yes" if they refer to the same cell
type (allowing for synonyms, plurals, and differences in granularity such as
"T cell" vs "CD4 T cell"), otherwise "no". Respond with only "yes" or "no".

Label A (ground truth): {truth}
Label B (prediction): {pred}
"""


def judge_match(truth: str, pred: str, model: str = "claude-haiku-4-5") -> bool:
    """Use an LLM to decide whether two free-text cell-type labels are equivalent."""
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    prompt = JUDGE_TEMPLATE.format(truth=truth, pred=pred)
    response = client.messages.create(
        model=model,
        max_tokens=5,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip().lower().startswith("y")
