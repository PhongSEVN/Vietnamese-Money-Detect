import torch
from torch import nn
from torchvision import models


class MobileNet(nn.Module):
    def __init__(self, num_classes=9, freeze_backbone=True):
        super().__init__()
        # MobileNetV3-Large: tốt hơn V2 (~75.2% vs ~71.9% ImageNet top-1)
        self.model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2)

        # Giữ nguyên classifier pretrained (Linear 960→1280 + Hardswish + Dropout),
        # chỉ thay layer cuối
        num_ftrs = self.model.classifier[-1].in_features  # 1280
        self.model.classifier[-1] = nn.Linear(num_ftrs, num_classes)

        if freeze_backbone:
            for param in self.model.features.parameters():
                param.requires_grad = False

    def unfreeze_backbone(self, num_layers=None):
        layers = list(self.model.features.children())
        if num_layers is None:
            target_layers = layers
        else:
            target_layers = layers[-num_layers:]
        for layer in target_layers:
            for param in layer.parameters():
                param.requires_grad = True

    def forward(self, x):
        return self.model(x)


if __name__ == '__main__':
    model = MobileNet()
    image = torch.randn(1, 3, 224, 224)
    output = model(image)
    print(output.shape)
