import torch
import torch.nn as nn


class MLPClassifier(nn.Module):
    def __init__(self, in_dim=768, num_classes=14) -> None:
        super().__init__()
        self.model=nn.Sequential(
            nn.Linear(in_dim,256),
            nn.LeakyReLU(),
            nn.Linear(256,num_classes)
        )
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, x):
        return self.model(x)

    def get_loss(self, x, multi_hot):
        y = self.forward(x)
        return self.loss(y, multi_hot)
