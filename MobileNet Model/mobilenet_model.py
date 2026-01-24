import torch
from torch import nn
from torchvision import models


class MobileNet(nn.Module):
    def __init__(self, num_classes=9, freeze_backbone=True):
        super().__init__()
        self.model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        
        # Thay classifier layer cuối
        num_ftrs = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(num_ftrs, num_classes)
        
        if freeze_backbone:
            for param in self.model.features.parameters():
                param.requires_grad = False

    def forward(self, x):
        return self.model(x)

if __name__ == '__main__':
    model = MobileNet()
    image = torch.randn(1, 3, 224, 224)
    output = model(image)
    print(output.shape)