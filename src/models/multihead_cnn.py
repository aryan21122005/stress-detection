import torch
from torch import nn
from torchvision import models


class MultiHeadCNN(nn.Module):
    def __init__(self, num_emotions=7, num_engagement=4, num_stress=4, pretrained=True):
        super().__init__()
        if pretrained:
            try:
                base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
                _pretrained_src = "ResNet18_Weights.DEFAULT"
            except Exception:
                try:
                    base = models.resnet18(pretrained=True)
                    _pretrained_src = "pretrained=True"
                except Exception:
                    base = models.resnet18(pretrained=False)
                    _pretrained_src = "pretrained=False"
        else:
            try:
                base = models.resnet18(weights=None)
                _pretrained_src = "weights=None"
            except Exception:
                base = models.resnet18(pretrained=False)
                _pretrained_src = "pretrained=False"
        self.backbone = nn.Sequential(*list(base.children())[:-1])
        self.fc = nn.Linear(512, 256)
        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.2)
        self.emotion_head = nn.Linear(256, num_emotions)
        self.engagement_head = nn.Linear(256, num_engagement)
        self.stress_head = nn.Linear(256, num_stress)
        print(f"[MultiHeadCNN] ResNet18 base init: {_pretrained_src}")

    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        x = self.act(self.fc(self.dropout(x)))
        emo = self.emotion_head(x)
        eng = self.engagement_head(x)
        stress = self.stress_head(x)
        return emo, eng, stress

    def load_partial_state_dict(self, state_dict):
        self.load_state_dict(state_dict, strict=False)
