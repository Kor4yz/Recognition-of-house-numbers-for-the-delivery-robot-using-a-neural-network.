import torch, torch.nn as nn
from torchvision.models import resnet18

class SVHNResNet(nn.Module):
    def __init__(self, num_classes: int = 10, pretrained: bool = False):
        super().__init__()
        self.backbone = resnet18(weights=None if not pretrained else "IMAGENET1K_V1")
        self.backbone.conv1.stride = (1,1)
        self.backbone.maxpool = nn.Identity()
        in_feats = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_feats, num_classes)

    def forward(self, x): return self.backbone(x)
