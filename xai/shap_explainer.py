import numpy as np
import torch
import shap
from transformers import AutoTokenizer

class ShapExplainer:
    def __init__(self, model, modality='image'):
        self.model = model
        self.modality = modality
        if modality == 'text':
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    
    def predict_fn_image(self, images):
        """Helper function for SHAP image prediction"""
        # Convert to torch tensor and move to correct device
        device = next(self.model.parameters()).device
        batch = torch.stack([torch.from_numpy(img.transpose(2, 0, 1)).float() for img in images]).to(device)
        
        with torch.no_grad():
            outputs = self.model(batch)
            probs = torch.softmax(outputs, dim=1)
        
        return probs.cpu().numpy()
    
    def predict_fn_text(self, texts):
        """Helper function for SHAP text prediction"""
        device = next(self.model.parameters()).device
        # Tokenize all texts in batch
        encodings = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        # Move encodings to correct device
        encodings = {k: v.to(device) for k, v in encodings.items()}
        
        with torch.no_grad():
            outputs = self.model(**encodings)
            probs = torch.softmax(outputs.logits, dim=1)
        
        return probs.cpu().numpy()
    
    def explain(self, data):
        """
        Generate SHAP values for the input data
        
        Args:
            data: Input data (image tensor or text string)
        
        Returns:
            For images: Numpy array of shape (H, W) containing the SHAP values
            For text: Numpy array of shape (num_tokens,) containing token importance scores
        """
        device = next(self.model.parameters()).device
        
        if self.modality == 'image':
            # Convert torch tensor to numpy if needed
            if isinstance(data, torch.Tensor):
                data = data.cpu().numpy().transpose(1, 2, 0)
            
            # Create background dataset (black image)
            background = np.zeros((1,) + data.shape)
            
            # Initialize SHAP explainer for PyTorch
            explainer = shap.GradientExplainer(
                model=self.model,
                data=torch.from_numpy(background).float().permute(0, 3, 1, 2).to(device)
            )
            
            # Get SHAP values
            input_tensor = torch.from_numpy(np.expand_dims(data, axis=0)).float().permute(0, 3, 1, 2).to(device)
            shap_values = explainer.shap_values(input_tensor)
            
            # Process SHAP values for visualization
            if isinstance(shap_values, list):
                # For multi-class, take values for predicted class
                pred_class = self.predict_fn_image(np.expand_dims(data, axis=0)).argmax()
                shap_values = shap_values[pred_class]
            
            # Process SHAP values for visualization
            attribution_map = np.abs(shap_values[0])  # Get first sample
            if len(attribution_map.shape) == 3:  # If shape is (C, H, W)
                attribution_map = attribution_map.mean(axis=0)  # Average over channels -> (H, W)
            elif len(attribution_map.shape) > 3:  # If shape is more complex
                attribution_map = attribution_map.mean(axis=tuple(range(attribution_map.ndim-2)))  # Average all but last 2 dims
            
            return attribution_map
            
        elif self.modality == 'text':
            # For text, use a simpler approach with word-level importance
            words = data.split()
            base_prediction = self.predict_fn_text([data])[0]
            
            # Calculate importance by removing each word
            token_importance = np.zeros(len(words))
            for i, word in enumerate(words):
                # Create text without this word
                modified_words = words.copy()
                modified_words[i] = '[UNK]'
                modified_text = ' '.join(modified_words)
                
                # Get prediction difference
                modified_prediction = self.predict_fn_text([modified_text])[0]
                token_importance[i] = np.abs(modified_prediction - base_prediction).mean()
            
            return token_importance
        
        else:
            raise ValueError(f"Unsupported modality: {self.modality}")
