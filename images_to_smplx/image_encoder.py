import torch
import torch.nn as nn
import torchvision.models as models


class ImageEncoder(nn.Module):
    def __init__(self, output_size=2048):
        super(ImageEncoder, self).__init__()
        # Use a pre-trained ResNet50 model
        base_model = models.resnet50(pretrained=True)
        # Remove the last classification layer
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])
        self.output_size = output_size

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(-1, self.output_size)
        return x