import os
import torch
import torch.nn as nn
import torchvision.models as models

def get_image_model():
    # Initialize model
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 2)  # Binary classification
    
    # Check for trained model in models directory
    model_path = os.path.join(os.path.dirname(__file__), 'best_tumor_model.pth')
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()  # Set to evaluation mode
        print(f"Loaded trained model with validation accuracy: {checkpoint['val_acc']:.2f}%")
    else:
        print("Using pretrained model without tumor-specific training")
    
    return model
