# Home Price Prediction

A machine learning project predicting residential sale prices using the [Kaggle House Prices dataset](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data), with an interactive Gradio deployment.

## Pipeline

### Step 1 — Load Data
`train.csv` is included in the repository — no download needed. Just run the notebook.

### Step 2 — Explore & Visualize
- Scatter plots of key features vs. `SalePrice`
- Correlation heatmap across all 80 features

### Step 3 — Preprocess
- Numeric columns: missing values filled with the column median
- Categorical columns: label-encoded with `sklearn.LabelEncoder`

### Step 4 — Train/Test Split
- 80/20 split (`random_state=42`)
- 1,168 training samples / 292 test samples

### Step 5 — Train Models
| Model | Library |
|---|---|
| Linear Regression | `sklearn` |
| XGBoost Regressor | `xgboost` |

### Step 6 — Evaluate
- R² score on the test set for each model
- Coefficient inspection for the linear model
- Random sample prediction check

### Step 7 — Deploy
Interactive Gradio app with sliders for 7 key features:

| Feature | Description |
|---|---|
| `GrLivArea` | Above-ground living area (sq ft) |
| `OverallQual` | Overall material & finish quality (1–10) |
| `TotalBsmtSF` | Basement area (sq ft) |
| `GarageCars` | Garage capacity (cars) |
| `FullBath` | Number of full bathrooms |
| `YearBuilt` | Year of construction |
| `LotArea` | Lot size (sq ft) |

All other features default to their training-set medians.

## Dataset

- **Source:** [Kaggle — House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data)
- **File:** `train.csv` — 1,460 rows × 81 columns
- **Target:** `SalePrice`

## Requirements

```bash
pip install pandas scikit-learn xgboost gradio matplotlib seaborn
```

## Usage

Open `Home_Price_Prediction.ipynb` in Google Colab or Jupyter and run all cells.
