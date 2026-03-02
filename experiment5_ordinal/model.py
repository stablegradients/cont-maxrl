"""ResNet-18 model for age estimation.

Two variants:
- Softmax head (L outputs) for classification/RL methods
- Regression head (1 output) for MSE regression baseline
"""

import torch
import torch.nn as nn
import torchvision.models as models


def create_model(
    L: int = 100,
    method: str = "ordinal_ce",
    pretrained: bool = True,
    seed: int = 42,
) -> nn.Module:
    """Create ResNet-18 with appropriate head.

    Args:
        L: Number of age bins (100 or 2)
        method: Training method name. "mse" gets regression head, all others get softmax head.
        pretrained: Use ImageNet pretrained weights
        seed: Seed for final FC layer initialization

    Returns:
        ResNet-18 model with replaced final layer
    """
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)

    # Set seed for reproducible FC layer init
    torch.manual_seed(seed)

    if method == "mse":
        # Regression head: single scalar output
        model.fc = nn.Linear(512, 1)
    else:
        # Classification/RL head: L output logits
        model.fc = nn.Linear(512, L)

    return model
