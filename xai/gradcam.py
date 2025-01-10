import torch
import torch.nn.functional as F
import numpy as np

class GradCAMExplainer:
    def __init__(self, model):
        self.model = model
        self.model.eval()
        
        # Get the last convolutional layer
        self.target_layer = None
        for module in self.model.modules():
            if isinstance(module, torch.nn.Conv2d):
                self.target_layer = module
        
        if self.target_layer is None:
            raise ValueError("Could not find convolutional layer in model")
        
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_backward_hook(self._save_gradient)
    
    def _save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def explain(self, image):
        """
        Generate GradCAM heatmap for the input image
        
        Args:
            image: Input image tensor of shape (C, H, W)
        
        Returns:
            Numpy array of shape (H, W) containing the GradCAM heatmap
        """
        # Add batch dimension if needed
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        
        # Forward pass
        output = self.model(image)
        
        # Get predicted class (maximum probability)
        pred_class = output.argmax(dim=1)
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass for predicted class
        output[0, pred_class].backward()
        
        # Global average pooling of gradients
        weights = torch.mean(self.gradients, dim=(2, 3))
        
        # Weight the activations by the gradients
        cam = torch.zeros(self.activations.shape[2:], dtype=torch.float32, device=self.activations.device)
        for i, w in enumerate(weights[0]):
            cam += w * self.activations[0, i]
        
        # Apply ReLU to focus on features that have a positive influence
        cam = F.relu(cam)
        
        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-7)
        
        # Resize to input image size
        cam = F.interpolate(
            cam.unsqueeze(0).unsqueeze(0),
            size=image.shape[2:],
            mode='bilinear',
            align_corners=False
        )
        
        return cam[0, 0].cpu().numpy()
