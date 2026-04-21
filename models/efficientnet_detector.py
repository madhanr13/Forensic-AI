"""
ForensicAI — EfficientNet-B0 Detector Model

Binary classifier (Real vs. AI-Generated) built on EfficientNet-B0
with ImageNet transfer learning. Includes Grad-CAM support via
hook-compatible architecture.
"""

import torch
import torch.nn as nn
from torchvision import models


class ForensicNetB0(nn.Module):
    """
    EfficientNet-B0 customized for binary image forensics classification.

    Features:
        - Pretrained ImageNet backbone for robust low-level feature extraction
        - Custom classification head with dropout
        - Compatible with Grad-CAM (hooks on self.features[-1])
    """

    def __init__(self, num_classes: int = 2, pretrained: bool = True, dropout: float = 0.3):
        super().__init__()

        # Load backbone
        if pretrained:
            weights = models.EfficientNet_B0_Weights.DEFAULT
        else:
            weights = None

        base = models.efficientnet_b0(weights=weights)

        # Feature extractor (all conv layers)
        self.features = base.features

        # Adaptive pooling
        self.avgpool = base.avgpool

        # Custom classifier
        in_features = base.classifier[1].in_features  # 1280 for B0
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout / 2),
            nn.Linear(512, num_classes),
        )

        # Initialize classifier weights
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def freeze_backbone(self):
        """Freeze all backbone parameters (for initial fine-tuning)."""
        for param in self.features.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze all backbone parameters (for end-to-end fine-tuning)."""
        for param in self.features.parameters():
            param.requires_grad = True

    def get_gradcam_target_layer(self) -> nn.Module:
        """Return the target layer for Grad-CAM visualization."""
        return self.features[-1]
