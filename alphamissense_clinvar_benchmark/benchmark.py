"""Benchmark AlphaMissense against ClinVar clinical labels.

AlphaMissense gives every missense variant a pathogenicity score (0-1).
ClinVar gives expert clinical labels. Here we treat ClinVar as the ground
truth and ask: how well does the AlphaMissense score separate pathogenic
from benign variants?

We only keep variants with a definite ClinVar label (pathogenic or benign) -
the "uncertain significance" and "conflicting" ones have no ground truth to
grade against.

Run it with:
    python benchmark.py
"""

import os

import pandas as pd
import matplotlib
matplotlib.use("Agg")            # save to a file, no screen needed
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

# how ClinVar wording maps to a binary label
pathogenic_labels = ["Pathogenic", "Likely pathogenic", "Pathogenic/Likely pathogenic"]
benign_labels = ["Benign", "Likely benign", "Benign/Likely benign"]

df = pd.read_csv("data/variants.csv")

# keep only variants with a definite label and a numeric AlphaMissense score
labels = []
for sig in df["clinvar_significance"]:
    if sig in pathogenic_labels:
        labels.append(1)
    elif sig in benign_labels:
        labels.append(0)
    else:
        labels.append(None)
df["label"] = labels

df["am_score"] = pd.to_numeric(df["am_score"], errors="coerce")
df = df.dropna(subset=["label", "am_score"])
df["label"] = df["label"].astype(int)

print("variants with a definite ClinVar label:", len(df))
print("  pathogenic:", int(df["label"].sum()), " benign:", int((df["label"] == 0).sum()))
print()

# --- main benchmark: does the AlphaMissense score rank pathogenic above benign? ---
auc = roc_auc_score(df["label"], df["am_score"])
ap = average_precision_score(df["label"], df["am_score"])
print("AlphaMissense score vs ClinVar label:")
print("  ROC-AUC          : %.3f" % auc)
print("  Average precision: %.3f" % ap)
print()

# --- save a ROC curve ---
fpr, tpr, thresholds = roc_curve(df["label"], df["am_score"])

os.makedirs("results", exist_ok=True)
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label="AlphaMissense (AUC = %.3f)" % auc)
plt.plot([0, 1], [0, 1], linestyle="--", color="grey", label="random")
plt.xlabel("false positive rate")
plt.ylabel("true positive rate")
plt.title("AlphaMissense vs ClinVar - ROC curve")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("results/roc_curve.png", dpi=150)
print("saved ROC curve to results/roc_curve.png")
print()

# --- how does AlphaMissense's own call (pathogenic / benign) line up? ---
# pred codes: P = likely pathogenic, B = likely benign, A = ambiguous
called = df[df["am_pred"].isin(["P", "B"])]
correct = 0
for pred, label in zip(called["am_pred"], called["label"]):
    if (pred == "P" and label == 1) or (pred == "B" and label == 0):
        correct += 1
print("AlphaMissense's own class (ambiguous ones excluded):")
print("  variants it called P or B:", len(called))
print("  agreement with ClinVar   : %.3f" % (correct / len(called)))
print("  called ambiguous (no call):", int((df["am_pred"] == "A").sum()))
print()

# --- spotlight: the classic NPC1 Niemann-Pick mutation ---
spot = df[df["variant_id"] == "chr18:g.21116700A>G"]
if len(spot):
    row = spot.iloc[0]
    print("Spotlight - NPC1 I1061T (a.k.a. I1063T in the paper's numbering):")
    print("  ClinVar      :", row["clinvar_significance"])
    print("  AlphaMissense: score %.3f, call '%s'" % (row["am_score"], row["am_pred"]))
    print("  -> a known pathogenic variant that AlphaMissense scores as benign (a real miss)")
