import numpy as np
import torch
from lime import lime_image, lime_text
from lime.lime_text import LimeTextExplainer
from skimage.segmentation import mark_boundaries
from transformers import AutoTokenizer

class LimeExplainer:
    def __init__(self, model, modality='image'):
        self.model = model
        self.modality = modality
        if modality == 'text':
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            self.text_explainer = LimeTextExplainer(class_names=['negative', 'neutral', 'positive'])
    
    def predict_fn_image(self, images):
        """Helper function for LIME image prediction"""
        # Convert to torch tensor and normalize if needed
        batch = torch.stack([torch.from_numpy(img.transpose(2, 0, 1)).float() for img in images])
        
        with torch.no_grad():
            outputs = self.model(batch)
            probs = torch.softmax(outputs, dim=1)
        
        return probs.cpu().numpy()
    
    def predict_fn_text(self, texts):
        """Helper function for LIME text prediction"""
        # Tokenize all texts in batch
        encodings = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        
        with torch.no_grad():
            outputs = self.model(**encodings)
            probs = torch.softmax(outputs.logits, dim=1)
        
        return probs.cpu().numpy()
    
    def explain(self, data):
        """
        Generate LIME explanation for the input data
        
        Args:
            data: Input data (image tensor or text string)
        
        Returns:
            For images: Numpy array of shape (H, W, 3) containing the explanation visualization
            For text: List of (word, importance) tuples
        """
        if self.modality == 'image':
            # Convert torch tensor to numpy if needed
            if isinstance(data, torch.Tensor):
                data = data.cpu().numpy().transpose(1, 2, 0)
            
            # Initialize LIME image explainer
            explainer = lime_image.LimeImageExplainer()
            
            # Get explanation
            explanation = explainer.explain_instance(
                data,
                self.predict_fn_image,
                top_labels=1,
                hide_color=0,
                num_samples=1000
            )
            
            # Get image and mask for the top predicted label
            label = explanation.top_labels[0]
            mask = explanation.get_image_and_mask(
                label,
                positive_only=True,
                num_features=5,
                hide_rest=True
            )[1]
            
            # Create visualization
            visualization = mark_boundaries(data, mask)
            
            return visualization
            
        elif self.modality == 'text':
            # Get explanation
            exp = self.text_explainer.explain_instance(
                data,
                self.predict_fn_text,
                num_features=10,
                num_samples=1000
            )
            
            # Get feature importance scores
            word_importance = exp.as_list()
            
            return word_importance
        
        else:
            raise ValueError(f"Unsupported modality: {self.modality}")
