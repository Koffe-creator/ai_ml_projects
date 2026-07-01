"""Train a CNN to classify skin-lesion images (DermaMNIST).

DermaMNIST is a set of 28x28 colour dermatoscopic images in 7 lesion classes.
The classes are very imbalanced (one class is about two thirds of the data),
so plain accuracy is misleading. We follow the MedMNIST benchmark and report
AUC (its headline metric), and we also report balanced accuracy and macro-F1,
which give the rare (medically important) classes equal weight.

Proper-training details:
  - a learning-rate schedule (decay partway through) for stable convergence
  - we keep the BEST-validation model and score the test set with that, rather
    than whatever the last epoch happened to look like

Pick the model by editing model_name below: "smallcnn" or "resnet18".

Run it from the project folder with:
    python train.py
"""

import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from medmnist import DermaMNIST
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score

torch.manual_seed(0)

model_name = "smallcnn"   # "smallcnn" or "resnet18"
epochs = 20

# --- data ---
to_tensor = transforms.ToTensor()   # pixels -> 0-1, shape (channels, height, width)

train_data = DermaMNIST(split="train", transform=to_tensor, download=True, size=28)
val_data = DermaMNIST(split="val", transform=to_tensor, download=True, size=28)
test_data = DermaMNIST(split="test", transform=to_tensor, download=True, size=28)

train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
val_loader = DataLoader(val_data, batch_size=128)
test_loader = DataLoader(test_data, batch_size=128)

print("model:", model_name, " epochs:", epochs, " train:", len(train_data), " test:", len(test_data))


# --- a small CNN (used when model_name == "smallcnn") ---
class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 7)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))    # 28x28 -> 14x14
        x = self.pool(self.relu(self.conv2(x)))     # 14x14 -> 7x7
        x = x.flatten(1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# --- build the chosen model ---
if model_name == "resnet18":
    model = models.resnet18(weights=None, num_classes=7)
    # adapt ResNet-18 for tiny 28x28 images: 3x3 stride-1 first conv, no early maxpool
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
else:
    model = SmallCNN()

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# drop the learning rate by 10x halfway and again near the end
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[epochs // 2, epochs * 3 // 4], gamma=0.1)

all_classes = [0, 1, 2, 3, 4, 5, 6]


# score a data split: returns accuracy, balanced accuracy, macro-F1, and AUC
def evaluate(loader):
    model.eval()
    true_labels = []
    prob_rows = []
    with torch.no_grad():
        for images, labels in loader:
            labels = labels.squeeze(1)              # DermaMNIST labels come as (batch, 1)
            probs = torch.softmax(model(images), dim=1)
            true_labels.extend(labels.tolist())
            prob_rows.extend(probs.tolist())

    preds = [row.index(max(row)) for row in prob_rows]   # class with the highest probability
    acc = accuracy_score(true_labels, preds)
    bal = balanced_accuracy_score(true_labels, preds)
    macro_f1 = f1_score(true_labels, preds, average="macro")
    auc = roc_auc_score(true_labels, prob_rows, multi_class="ovr", labels=all_classes)
    return acc, bal, macro_f1, auc


# --- training loop, keeping the best-validation model ---
best_auc = -1.0
best_state = None

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        labels = labels.squeeze(1)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    scheduler.step()
    acc, bal, macro_f1, auc = evaluate(val_loader)
    print("epoch %2d  loss %.3f  val_auc %.3f  val_acc %.3f  val_bal_acc %.3f  val_macro_f1 %.3f"
          % (epoch + 1, running_loss / len(train_loader), auc, acc, bal, macro_f1))

    if auc > best_auc:
        best_auc = auc
        best_state = copy.deepcopy(model.state_dict())

# --- score the test set with the best model ---
model.load_state_dict(best_state)
acc, bal, macro_f1, auc = evaluate(test_loader)
print()
print("TEST (best model)  AUC %.3f  accuracy %.3f  balanced_acc %.3f  macro_f1 %.3f"
      % (auc, acc, bal, macro_f1))
