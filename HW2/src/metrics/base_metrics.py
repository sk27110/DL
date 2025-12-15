import torch
from torchmetrics.classification import Accuracy, Precision, Recall, F1Score

class ClassificationMetrics:
    def __init__(self, num_classes: int, device: str = "cpu"):
        self.device = device
        self.num_classes = num_classes

        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes).to(device)
        self.precision = Precision(task="multiclass", num_classes=num_classes, average="macro").to(device)
        self.recall = Recall(task="multiclass", num_classes=num_classes, average="macro").to(device)
        self.f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro").to(device)

        self._metrics = {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1
        }