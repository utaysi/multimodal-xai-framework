import torch.nn as nn
import torchvision.models as models

def get_image_model():
    # Load pretrained ResNet50 model
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 2)  # Binary classification
    return model
