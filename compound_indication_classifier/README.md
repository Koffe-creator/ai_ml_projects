# Compound Indication Classifier

Predict whether a small molecule is active for **mental health** directly from
its chemical structure, using RDKit features and a scikit-learn model.

This is a compound-indication / mechanism-of-action classification task: given a
molecule's SMILES string, decide if it belongs to the mental-health indication.

## How it works

1. Each molecule's SMILES is turned into numbers with RDKit, two ways:
   - **7 physicochemical descriptors** (molecular weight, logP, TPSA, H-bond
     donors/acceptors, rotatable bonds, aromatic rings) — interpretable
   - **2048-bit Morgan fingerprint** — marks which substructures are present
2. A random forest is trained on each feature set. The label is imbalanced
   (~14% active), so we use `class_weight="balanced"` and report **ROC-AUC** and
   **average precision** instead of accuracy.
3. Performance is estimated with **10-fold cross-validation on the training set**
   (mean ± standard deviation), then confirmed once on a held-out test set.

> **Data note:** the full dataset is not included. A 300-row sample
> (`data/sample_compounds.csv`) is provided so the pipeline runs out of the box;
> the metrics below are from the full 15,116-compound dataset.

## Results (15,116 compounds, 2,156 mental-health-active)

| Features | 10-fold CV ROC-AUC | Test ROC-AUC | Test avg precision | Interpretable |
|---|---|---|---|---|
| 7 descriptors | 0.672 ± 0.017 | 0.673 | 0.253 | yes |
| Morgan fingerprint (2048-bit) | 0.703 ± 0.018 | 0.722 | 0.294 | no |

The cross-validation mean matches the held-out test score, and the small spread
(±0.017) shows the estimate is stable across folds rather than the luck of one
split.

Two takeaways:

- **The descriptor model's top features are logP, molecular weight, and TPSA** —
  exactly the properties that govern blood-brain barrier penetration. That makes
  chemical sense: a mental-health (CNS) drug has to cross the blood-brain barrier
  to reach the brain, so the same physicochemistry matters.
- **Fingerprints are more accurate but not interpretable.** They capture specific
  substructures, raising ROC-AUC by about 0.05, at the cost of being unable to
  explain any single feature — the classic interpretability-vs-accuracy tradeoff.

## Run it

```bash
pip install -r requirements.txt
python train.py
```

Expects `data/compound_names.csv` with `CanonicalSMILES` and a `mental_health`
column (TRUE/FALSE). Other indication columns (`inflammation`,
`multiple_sclerosis`) are present in the same format and can be swapped in by
changing the label column in `train.py`.
