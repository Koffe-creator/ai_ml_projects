# TensorFlow — Banknote Authentication

Binary classification of banknotes (authentic vs. fake) using a TensorFlow Deep Neural Network, compared against a Random Forest baseline.

## Dataset
`bank_note_data.csv` — 1,372 samples, 4 image-derived features (`Image.Var`, `Image.Skew`, `Image.Curt`, `Entropy`), binary target `Class`.

## Models
| Model | Test Accuracy |
|---|---|
| TensorFlow DNN [10, 20, 10] | ~97% |
| Random Forest (200 trees) | ~99% |

## Requirements
```bash
pip install tensorflow scikit-learn pandas seaborn
```
