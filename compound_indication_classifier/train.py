"""Predict whether a compound is active for mental health, from its structure.

We turn each molecule (a SMILES string) into numbers with RDKit, then train a
random forest. We try two kinds of features and compare them:
  - 7 simple descriptors (molecular weight, logP, TPSA, ...) - easy to explain
  - a 2048-bit Morgan fingerprint - usually more accurate, not interpretable

The label is imbalanced (~14% active), so we use class_weight="balanced" and
judge the model with ROC-AUC and average precision instead of accuracy.

Run it from the project folder with:
    python train.py
"""

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors, rdFingerprintGenerator
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score


# load the compounds (each row has a SMILES and a TRUE/FALSE mental_health label).
# A small sample is included so this runs out of the box; point this at the full
# dataset (data/compound_names.csv) to reproduce the results in the README.
df = pd.read_csv("data/sample_compounds.csv")

descriptor_names = [
    "mol_weight", "logp", "h_bond_donors", "h_bond_acceptors",
    "tpsa", "rotatable_bonds", "aromatic_rings",
]

# turn every molecule into both descriptor and fingerprint features
morgan = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

descriptor_rows = []
fingerprint_rows = []
labels = []

for smiles, active in zip(df["CanonicalSMILES"], df["mental_health"]):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        continue   # skip molecules RDKit cannot read

    descriptor_rows.append([
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Lipinski.NumHDonors(mol),
        Lipinski.NumHAcceptors(mol),
        rdMolDescriptors.CalcTPSA(mol),
        Lipinski.NumRotatableBonds(mol),
        rdMolDescriptors.CalcNumAromaticRings(mol),
    ])
    fingerprint_rows.append(list(morgan.GetFingerprint(mol)))
    labels.append(1 if active else 0)

print("molecules used:", len(labels), " positives:", sum(labels))

# train and score a model on each feature type
feature_sets = [
    ("descriptors", descriptor_rows),
    ("fingerprint", fingerprint_rows),
]

for name, rows in feature_sets:
    X = pd.DataFrame(rows)
    y = pd.Series(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=0, n_jobs=-1
    )
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]

    print()
    print("===", name, "===")
    print("ROC-AUC          :", round(roc_auc_score(y_test, probs), 3))
    print("Average precision:", round(average_precision_score(y_test, probs), 3))

    # the descriptor importances are interpretable; fingerprint bits are not
    if name == "descriptors":
        importances = list(zip(descriptor_names, model.feature_importances_))
        importances.sort(key=lambda pair: pair[1], reverse=True)
        print("most important descriptors:")
        for desc_name, score in importances:
            print("  ", desc_name, round(score, 3))
