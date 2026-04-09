import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class DecoderBlock(nn.Module):
    """Upsample + skip concat + double conv block."""
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.has_skip = skip_ch > 0
        conv_in_ch = in_ch + skip_ch
        self.conv1 = nn.Conv2d(conv_in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        if skip is not None:
            # Handle possible 1-pixel misalignment from odd dims
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)
        return x


class DepthResNet18UNet(nn.Module):
    """Pretrained ResNet-18 encoder + U-Net decoder for dense depth prediction.

    Follows DORN's freezing recipe: conv1, layer1, and ALL encoder BatchNorms are frozen.
    """

    def __init__(self, K: int, method: str = "ordinal_ce", pretrained: bool = True, seed: int = 42):
        super().__init__()
        self.K = K
        self.method = method

        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)

        # Encoder components
        self.enc_conv1 = backbone.conv1
        self.enc_bn1 = backbone.bn1
        self.enc_relu = backbone.relu
        self.enc_maxpool = backbone.maxpool
        self.enc_layer1 = backbone.layer1
        self.enc_layer2 = backbone.layer2
        self.enc_layer3 = backbone.layer3
        self.enc_layer4 = backbone.layer4

        # Decoder
        self.dec1 = DecoderBlock(in_ch=512, skip_ch=256, out_ch=256)
        self.dec2 = DecoderBlock(in_ch=256, skip_ch=128, out_ch=128)
        self.dec3 = DecoderBlock(in_ch=128, skip_ch=64,  out_ch=64)
        self.dec4 = DecoderBlock(in_ch=64,  skip_ch=0,   out_ch=32)
        self.dec5 = DecoderBlock(in_ch=32,  skip_ch=0,   out_ch=32)

        # Output head
        torch.manual_seed(seed)
        if method == "mse":
            self.head = nn.Conv2d(32, 1, kernel_size=1)
        else:
            self.head = nn.Conv2d(32, K, kernel_size=1)

        # Apply DORN freezing
        self._apply_dorn_freezing()

    def _apply_dorn_freezing(self):
        """Freeze conv1, bn1, layer1, and ALL BatchNorm layers in encoder layers 2-4."""
        for p in self.enc_conv1.parameters():
            p.requires_grad_(False)
        for p in self.enc_bn1.parameters():
            p.requires_grad_(False)
        for p in self.enc_layer1.parameters():
            p.requires_grad_(False)
        for layer in [self.enc_layer2, self.enc_layer3, self.enc_layer4]:
            for m in layer.modules():
                if isinstance(m, nn.BatchNorm2d):
                    for p in m.parameters():
                        p.requires_grad_(False)

    def train(self, mode=True):
        """Override train() to keep frozen BN layers in eval mode always."""
        super().train(mode)
        self.enc_bn1.eval()
        for layer in [self.enc_layer1, self.enc_layer2, self.enc_layer3, self.enc_layer4]:
            for m in layer.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # Encoder
        x = self.enc_conv1(images)
        x = self.enc_bn1(x)
        x = self.enc_relu(x)
        x0 = self.enc_maxpool(x)
        x1 = self.enc_layer1(x0)
        x2 = self.enc_layer2(x1)
        x3 = self.enc_layer3(x2)
        x4 = self.enc_layer4(x3)

        # Decoder with skips
        d = self.dec1(x4, skip=x3)
        d = self.dec2(d, skip=x2)
        d = self.dec3(d, skip=x1)
        d = self.dec4(d, skip=None)
        d = self.dec5(d, skip=None)
        out = self.head(d)
        return out


def create_model(K: int, method: str, pretrained: bool = True, seed: int = 42) -> DepthResNet18UNet:
    return DepthResNet18UNet(K=K, method=method, pretrained=pretrained, seed=seed)


def convert_decoder_bns_to_sync(model):
    """Convert decoder BatchNorms to SyncBatchNorm for DDP.
    Encoder BNs are frozen and stay as plain BNs (in eval mode, no sync needed)."""
    for name in ["dec1", "dec2", "dec3", "dec4", "dec5"]:
        dec = getattr(model, name)
        sync_dec = nn.SyncBatchNorm.convert_sync_batchnorm(dec)
        setattr(model, name, sync_dec)
    return model
