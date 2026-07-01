# Skin-Lesion CNN (DermaMNIST)

A convolutional neural network in PyTorch that classifies dermatoscopic
skin-lesion images into 7 classes, using the DermaMNIST dataset (28x28 colour
images) from MedMNIST.

## How it works

1. Load DermaMNIST's official train / validation / test splits (via the
   `medmnist` package — it downloads automatically on first run).
2. Train a small CNN (two conv blocks + two fully-connected layers) with:
   - a **learning-rate schedule** (decay partway through) for stable training
   - **best-model checkpointing** — we keep the weights from the best validation
     epoch and score the test set with those, not the last epoch
3. Report **AUC** (the MedMNIST benchmark's headline metric) plus **balanced
   accuracy** and **macro-F1**.

The classes are very imbalanced (one class is ~two thirds of the data), so plain
accuracy is misleading — a model that always guesses the majority class already
scores ~0.67. AUC, balanced accuracy, and macro-F1 tell the real story.

## Results (20 epochs, CPU)

| Model | AUC | Accuracy | Balanced acc | Macro-F1 |
|---|---|---|---|---|
| This small CNN (~0.2M params) | 0.893 | 0.717 | 0.306 | 0.318 |
| MedMNIST benchmark ResNet-18 (28), 100 epochs | 0.917 | 0.735 | — | — |

Two takeaways:

- **A tiny CNN, properly trained, nearly matches the benchmark ResNet-18 on AUC**
  (0.893 vs 0.917) at a fraction of the size. Careful training (LR schedule,
  best-model selection) mattered more than raw model capacity here — an earlier
  under-trained ResNet-18 scored far worse.
- **High AUC does not mean per-class competence.** Balanced accuracy (0.306) and
  macro-F1 (0.318) are low, so the model is still weak on the rare lesion classes
  even at near-benchmark AUC. On imbalanced medical data the per-class metrics are
  what matter clinically.

## Run it

```bash
pip install -r requirements.txt
python train.py
```

Switch the architecture by editing `model_name` in `train.py` ("smallcnn" or
"resnet18"). Note: the MedMNIST benchmark trains for 100 epochs; this runs a
shorter schedule that is practical on CPU.
