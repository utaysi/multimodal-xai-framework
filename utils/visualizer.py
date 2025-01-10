import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

def visualize_explanations(image, gradcam_exp, shap_exp, lime_exp, output_path):
    """
    Visualize different XAI explanations for image data
    
    Args:
        image: Original input image
        gradcam_exp: GradCAM heatmap
        shap_exp: SHAP values
        lime_exp: LIME explanation
    """
    plt.figure(figsize=(15, 5))
    
    # Original image
    plt.subplot(141)
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy().transpose(1, 2, 0)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')
    
    # GradCAM
    plt.subplot(142)
    plt.imshow(gradcam_exp, cmap='jet')
    plt.title('GradCAM')
    plt.axis('off')
    
    # SHAP
    plt.subplot(143)
    # Overlay SHAP heatmap on original image
    overlaid_shap = overlay_heatmap(image, shap_exp)
    plt.imshow(overlaid_shap)
    plt.title('SHAP')
    plt.axis('off')
    
    # LIME
    plt.subplot(144)
    plt.imshow(lime_exp, cmap='viridis')
    plt.title('LIME')
    plt.axis('off')
    
    plt.tight_layout()
    # Save to file
    plt.savefig(output_path)
    plt.close()

def visualize_text_explanations(text, shap_exp, lime_exp, output_path):
    """
    Visualize different XAI explanations for text data
    
    Args:
        text: Original input text
        shap_exp: SHAP values for each token
        lime_exp: List of (word, importance) tuples from LIME
    """
    plt.figure(figsize=(12, 6))
    
    # Convert text to list of tokens
    tokens = text.split()
    
    # Process SHAP values
    shap_values = np.array(shap_exp).flatten() if hasattr(shap_exp, 'flatten') else np.array(shap_exp)
    
    # Process LIME values (convert from list of tuples)
    lime_dict = dict(lime_exp)  # Convert LIME tuples to dictionary
    lime_values = np.array([lime_dict.get(token, 0.0) for token in tokens])
    
    # SHAP visualization
    plt.subplot(211)
    y_pos = np.arange(len(tokens))
    plt.barh(y_pos, shap_values)
    plt.yticks(y_pos, tokens)
    plt.title('SHAP Token Importance')
    plt.xlabel('SHAP value')
    
    # LIME visualization
    plt.subplot(212)
    plt.barh(y_pos, lime_values)
    plt.yticks(y_pos, tokens)
    plt.title('LIME Token Importance')
    plt.xlabel('LIME score')
    
    plt.tight_layout()
    # Save to file
    plt.savefig(output_path)
    plt.close()

def overlay_heatmap(image, heatmap, alpha=0.5):
    """
    Utility function to overlay heatmap on image
    
    Args:
        image: Original image
        heatmap: Explanation heatmap
        alpha: Transparency of overlay
    
    Returns:
        Overlaid image
    """
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy().transpose(1, 2, 0)
    
    # Normalize heatmap
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    
    # Convert to RGB heatmap
    cmap = plt.cm.jet
    heatmap_rgb = cmap(heatmap)[..., :3]
    
    # Overlay
    overlaid = (1 - alpha) * image + alpha * heatmap_rgb
    
    # Ensure values are in valid range
    overlaid = np.clip(overlaid, 0, 1)
    
    return overlaid
