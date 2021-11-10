import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class Resnet(nn.Module):
    """
    Embedding extraction using Resnet-50 backbone
    Parameters
    ----------
    embedding_size:
        Size of embedding vector.
    pretrained:
        Whether to use pretrained weight on ImageNet.
    """
    def __init__(self, pretrained=True):
        super().__init__()

        # transfer channel to 3
        self.conv0 = nn.Conv2d(22, 3, kernel_size=7, stride=2, padding=3, bias=False)

        model = models.resnet50(pretrained=pretrained)
        # for param in model.parameters():
        #     param.requires_grad = False

        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)

        self.model = model


    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = self.conv0(image)
        x = self.model(x)
        return x