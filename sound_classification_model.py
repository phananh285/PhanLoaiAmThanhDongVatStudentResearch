import torch.nn.functional as F
from torch.nn import init
from torch import nn
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor, Normalize
import torchvision.models as models

# ----------------------------
# Audio Classification Model
# ----------------------------
class AudioClassifier (nn.Module):
    # ----------------------------
    # Build the model architecture
    # ----------------------------
    def __init__(self, pretrained_model):
        super(AudioClassifier, self).__init__()
        self.resnet = nn.hub.load('pytorch/vision', 'resnet50', pretrained=True)
        self.features = nn.Sequential(*list(pretrained_model.children())[:-1])
    # ----------------------------
    # Forward pass computations
    # ----------------------------
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return self.resnet(x)
       

