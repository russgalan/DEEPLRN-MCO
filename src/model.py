"""
ResNet50 backbone + custom classification head (Section 3, steps 3-4),
with support for staged/progressive unfreezing:

  Stage 0: only the new head is trainable (backbone fully frozen).
  Stage 1: unfreeze layer4 (the last residual block) at a lower LR.
  Stage 2: unfreeze layer3 + layer4.
  Stage 3: unfreeze everything (full fine-tune).

This lets train.py implement "train the head first, then progressively
unfreeze deeper layers at a lower learning rate" instead of unfreezing
everything at once.
"""
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights

from utils import CLASS_NAMES

NUM_CLASSES = len(CLASS_NAMES)

# ResNet50's children in unfreezing order, deepest-first, matching the
# stages described above. layer4 is unfrozen before layer3, etc.
UNFREEZE_STAGES = {
    0: [],                                  # head only
    1: ["layer4"],
    2: ["layer4", "layer3"],
    3: ["layer4", "layer3", "layer2", "layer1", "conv1", "bn1"],  # full fine-tune
}


class ResNetWasteClassifier(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = 0.3):
        super().__init__()
        backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        in_features = backbone.fc.in_features

        # Drop the original ImageNet fc layer; keep everything up to
        # global average pooling (backbone.avgpool is already GAP).
        backbone.fc = nn.Identity()
        self.backbone = backbone

        # Classification head: GAP (already applied inside backbone) ->
        # Dropout -> Dense -> ReLU -> Dropout -> Dense (logits).
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes),
        )

        # Start fully frozen; train.py calls set_unfreeze_stage() to
        # progressively open up layers.
        self.freeze_backbone()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)  # (B, in_features), GAP already applied
        return self.head(features)  # (B, num_classes) logits (softmax via loss fn)

    def freeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False

    def set_unfreeze_stage(self, stage: int) -> None:
        """Freeze everything, then unfreeze only the named submodules for
        this stage. Stage 0 = head-only training."""
        assert stage in UNFREEZE_STAGES, f"stage must be one of {list(UNFREEZE_STAGES)}"
        self.freeze_backbone()
        for name in UNFREEZE_STAGES[stage]:
            module = getattr(self.backbone, name)
            for p in module.parameters():
                p.requires_grad = True

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]


def build_model(dropout: float = 0.3, device=None) -> ResNetWasteClassifier:
    model = ResNetWasteClassifier(dropout=dropout)
    if device is not None:
        model = model.to(device)
    return model
