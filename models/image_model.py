def get_image_model():
    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 2)  # Binary classification
    return model