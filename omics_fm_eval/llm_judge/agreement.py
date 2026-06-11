"""Compare LLM cell type predictions against ground truth and model predictions."""

import pandas as pd


def agreement_table(
    ground_truth: dict[str, str],
    foundation_model_pred: dict[str, str],
    llm_pred: dict[str, str],
) -> pd.DataFrame:
    clusters = sorted(ground_truth.keys())
    return pd.DataFrame(
        {
            "cluster": clusters,
            "ground_truth": [ground_truth[c] for c in clusters],
            "foundation_model": [foundation_model_pred[c] for c in clusters],
            "llm": [llm_pred[c] for c in clusters],
            "llm_matches_truth": [llm_pred[c].lower() == ground_truth[c].lower() for c in clusters],
            "llm_matches_model": [llm_pred[c].lower() == foundation_model_pred[c].lower() for c in clusters],
        }
    )
