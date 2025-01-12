from typing import Optional, Union
import torch
import numpy as np
from captum.attr import LayerGradCam
from captum.attr._utils.visualization import visualize_image_attr

class GradCAM:
    def __init__(self, model, target_layer: Optional[torch.nn.Module] = None):
        """
        Initialize GradCAM explainer using Captum's implementation
        
        Args:
            model: PyTorch model to explain
            target_layer: Target convolutional layer for GradCAM. If None, will
                        automatically find the last convolutional layer.
        """
        self.model = model
        self.model.eval()
        
        # Find target layer if not provided
        if target_layer is None:
            for module in self.model.modules():
                if isinstance(module, torch.nn.Conv2d):
                    target_layer = module
            if target_layer is None:
                raise ValueError("Could not find convolutional layer in model")
        
        self.explainer = LayerGradCam(self.model, target_layer)
        
    def explain(self, image: torch.Tensor, target: Optional[int] = None) -> np.ndarray:
        """
        Generate GradCAM heatmap for the input image
        
        Args:
            image: Input image tensor of shape (C, H, W) or (1, C, H, W)
            target: Target class index to explain. If None, uses model's prediction
            
        Returns:
            Numpy array of shape (H, W) containing the GradCAM heatmap
        """
        # Add batch dimension if needed
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
            
        # Get model prediction if target not specified
        if target is None:
            with torch.no_grad():
                output = self.model(image)
                target = output.argmax(dim=1).item()
        
        # Generate GradCAM attribution
        attribution = self.explainer.attribute(
            image,
            target=target,
            relu_attributions=True
        )
        
        # Convert to numpy and remove batch dimension
        heatmap = attribution.squeeze().cpu().detach().numpy()
        
        # Ensure proper dimensions for interpolation
        if len(attribution.shape) == 2:
            attribution = attribution.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
        elif len(attribution.shape) == 3:
            attribution = attribution.unsqueeze(1)  # Add channel dim
            
        # Upsample heatmap to match input image size
        heatmap = torch.nn.functional.interpolate(
            attribution,
            size=image.shape[-2:],
            mode='bilinear',
            align_corners=False
        )
        
        # Remove batch and channel dimensions
        heatmap = heatmap.squeeze().cpu().detach().numpy()
        
        # Apply sigmoid normalization for better contrast
        heatmap = 1 / (1 + np.exp(-heatmap))
        
        # Clip and normalize to [0, 1] range
        heatmap = np.clip(heatmap, 0, 1)
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-7)
        
        return heatmap

    def generate(self, input_tensor: torch.Tensor) -> np.ndarray:
        """Alias for explain method for backward compatibility"""
        return self.explain(input_tensor)
