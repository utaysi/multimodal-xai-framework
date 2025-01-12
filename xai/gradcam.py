import torch
import torch.nn.functional as F
import numpy as np

class GradCAM:
    def __init__(self, model):
        self.model = model
        self.model.eval()
        
        # Get the last convolutional layer
        self.target_layer = None
        for module in self.model.modules():
            if isinstance(module, torch.nn.Conv2d):
                self.target_layer = module
                
        # For guided backpropagation
        self.hooks = []
        for module in self.model.modules():
            if isinstance(module, torch.nn.ReLU):
                self.hooks.append(module.register_backward_hook(self._relu_backward_hook))
        
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
        
    def _relu_backward_hook(self, module, grad_input, grad_output):
        # Guided backpropagation - only pass positive gradients
        if isinstance(module, torch.nn.ReLU):
            return (F.relu(grad_input[0]),)
    
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
        
        # GradCAM++ weight calculation
        gradients = self.gradients
        activations = self.activations
        
        # Calculate alpha coefficients
        numerator = gradients.pow(2)
        denominator = 2 * gradients.pow(2) + \
            activations * gradients.pow(3).sum(dim=(2, 3), keepdim=True)
        alpha = numerator / (denominator + 1e-7)
        
        # Calculate weights using weighted average
        weights = (alpha * F.relu(gradients)).sum(dim=(2, 3))
        
        # Weight the activations by the gradients
        cam = torch.zeros(activations.shape[2:], dtype=torch.float32, device=activations.device)
        for i, w in enumerate(weights[0]):
            cam += w * activations[0, i]
        
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
        
        # Generate guided GradCAM by multiplying with guided backprop
        guided_grads = self.gradients.mean(dim=1, keepdim=True)
        # Resize guided gradients to match CAM size
        guided_grads = F.interpolate(
            guided_grads,
            size=image.shape[2:],
            mode='bilinear',
            align_corners=False
        )
        guided_cam = cam * guided_grads
        
        # Normalize guided GradCAM
        guided_cam = guided_cam - guided_cam.min()
        guided_cam = guided_cam / (guided_cam.max() + 1e-7)
        
        return guided_cam[0, 0].cpu().numpy()

    def generate(self, input_tensor):
        """Generate GradCAM explanation (alias for explain)"""
        result = self.explain(input_tensor)
        if isinstance(result, torch.Tensor):
            result = result.cpu()
        return result
