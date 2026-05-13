# Compound Classifier

A multi-model classification pipeline in R that predicts compound class labels (case vs. control) from molecular features, using an ensemble of machine learning algorithms optimized by ROC-AUC.

## Pipeline

### 1 — Load & Partition Data
- `compound_features.csv` — molecular feature matrix (rows = compounds, columns = features + `Class`)
- `compound_names.csv` — compound annotation metadata
- Compounds with missing `Class` labels are held out for post-hoc prediction
- 80/20 stratified train/test split (`createDataPartition`)

### 2 — Train Models
10-fold cross-validation repeated 10×, optimized by ROC-AUC (`twoClassSummary`).

Supported algorithms (configure via `selected_methods`):

| Key | Method |
|---|---|
| `glmnet` | Elastic Net (L1/L2 regularization) |
| `rf` | Random Forest |
| `pls` | Partial Least Squares |
| `gbm` | Gradient Boosting |
| `svmLinear2` | Linear SVM |
| `nnet` | Neural Network |

### 3 — Ensemble
A GLM meta-model (`caretStack`) is stacked on top of the base learners using bootstrap resampling.

### 4 — Evaluate
- AUC per model and ensemble on the held-out test set
- ROC curves with Youden-optimal threshold
- Hard class calls per compound per model

### 5 — Output

| File | Description |
|---|---|
| `ROC_training_summary_table_*.csv` | CV ROC summary across models |
| `all_models_*_*.png` | Dot plots of CV metrics (ROC, Sens, Spec) |
| `variableImportance_*.csv` | Scaled feature importance per model |
| `all_test_ROCs_*.csv` | Test-set AUC + Youden cutoffs |
| `calls_*.csv` | Final hard calls + predicted probabilities |
| `rf_model_*.RDS` | Saved Random Forest model object |

## Configuration

Edit the top of `compound_classifier.R` before running:

```r
seed             <- 123
namesies         <- "annot_v5"          # label used in output filenames
selected_methods <- c("glmnet", "rf")   # pick any combination of supported methods
```

## Requirements

```r
install.packages(c("tidyverse", "caret", "caretEnsemble", "e1071",
                   "xgboost", "pls", "nnet", "kernlab", "klaR",
                   "pROC", "limma", "caTools"))
```

## Usage

```r
source("compound_classifier.R")
```

Results are saved to the `output/` folder.
