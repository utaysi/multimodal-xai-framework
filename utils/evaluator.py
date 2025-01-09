import numpy as np
from sklearn.metrics import accuracy_score
import torch
from transformers import AutoTokenizer

class XAIEvaluator:
    """Evaluator for XAI methods"""
    
    @staticmethod
    def evaluate_faithfulness(model, data, explanation, mode='text'):
        """
        Evaluate how faithful the explanation is to the model's decision
        by measuring prediction change when most important features are removed
        
        Args:
            model: The model being explained
            data: Original input data
            explanation: Explanation scores for features
            mode: 'text' or 'image'
            
        Returns:
            faithfulness_score: How well explanation identifies important features
        """
        if mode == 'text':
            # For text, remove top K% important words and check prediction change
            words = data.split()
            
            # Handle different explanation formats
            if isinstance(explanation, list) and len(explanation) > 0 and isinstance(explanation[0], tuple):
                # LIME format: list of (word, importance) tuples
                word_scores = dict(explanation)
                importance_scores = np.array([word_scores.get(word, 0.0) for word in words])
            else:
                # SHAP format: array of importance scores
                importance_scores = np.array(explanation)
            
            # Get indices of top K important words
            top_k = int(len(words) * 0.3)  # Remove top 30% important words
            top_indices = np.argsort(np.abs(importance_scores))[-top_k:]
            
            # Create modified text without these words
            modified_words = words.copy()
            for idx in top_indices:
                modified_words[idx] = '[UNK]'
            modified_text = ' '.join(modified_words)
            
            # Get predictions (handle BERT model input)
            if hasattr(model, 'config') and hasattr(model.config, 'model_type') and model.config.model_type == 'bert':
                # Tokenize inputs for BERT
                tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
                orig_encoding = tokenizer(data, return_tensors='pt', padding=True, truncation=True)
                mod_encoding = tokenizer(modified_text, return_tensors='pt', padding=True, truncation=True)
                
                with torch.no_grad():
                    orig_output = model(**orig_encoding)
                    mod_output = model(**mod_encoding)
                
                orig_pred = torch.softmax(orig_output.logits, dim=1).cpu().numpy()[0]
                mod_pred = torch.softmax(mod_output.logits, dim=1).cpu().numpy()[0]
            else:
                # For other models
                orig_pred = model(data)
                mod_pred = model(modified_text)
            
            # Faithfulness score = prediction difference
            return np.abs(orig_pred - mod_pred).mean()
            
        elif mode == 'image':
            # For images, mask out top K% important pixels
            if isinstance(explanation, torch.Tensor):
                explanation = explanation.cpu().numpy()
            
            # Get image dimensions
            if isinstance(data, torch.Tensor):
                h, w = data.shape[-2:]
            else:
                h, w = data.shape[-2:]
            
            # Ensure explanation has shape (H, W)
            if len(explanation.shape) > 2:
                if explanation.shape[-1] == 2:  # Special case for SHAP values
                    explanation = explanation.reshape(h, w)
                else:
                    explanation = explanation.mean(axis=0)  # Average across extra dimensions
            elif len(explanation.shape) < 2:
                explanation = explanation.reshape(h, w)  # Reshape to match image dimensions
            
            # Create mask
            mask = np.zeros((h, w))
            k = int(0.3 * h * w)  # Mask top 30% important pixels
            flat_exp = explanation.ravel()
            top_k_indices = np.argsort(np.abs(flat_exp))[-k:]  # Use absolute values for importance
            mask.ravel()[top_k_indices] = 1
            
            # Convert mask to tensor with proper shape and move to same device as data
            mask_tensor = torch.from_numpy(mask).to(data.device)
            
            # Ensure mask has correct shape (H, W)
            if len(mask_tensor.shape) > 2:
                mask_tensor = mask_tensor.mean(dim=0)  # Average across extra dimensions
            elif len(mask_tensor.shape) < 2:
                mask_tensor = mask_tensor.view(data.shape[-2:])  # Reshape to match image dimensions
            
            # Expand mask to match image channels (C, H, W)
            mask_tensor = mask_tensor.unsqueeze(0).expand(data.shape[0], -1, -1)
            
            # Apply mask to image
            masked_image = data.clone()
            masked_image[mask_tensor > 0] = 0
            
            # Get predictions
            with torch.no_grad():
                orig_output = model(data.unsqueeze(0))
                mask_output = model(masked_image.unsqueeze(0))
                
                orig_pred = torch.softmax(orig_output, dim=1).cpu().numpy()[0]
                mask_pred = torch.softmax(mask_output, dim=1).cpu().numpy()[0]
            
            # Faithfulness score = prediction difference
            return np.abs(orig_pred - mask_pred).mean()
    
    @staticmethod
    def evaluate_consistency(explanations, similar_inputs):
        """
        Evaluate how consistent explanations are across similar inputs
        
        Args:
            explanations: List of explanations for similar inputs
            similar_inputs: List of similar input samples
            
        Returns:
            consistency_score: How consistent explanations are
        """
        # Convert explanations to numpy arrays
        exp_arrays = [np.array(exp).flatten() for exp in explanations]
        
        # Calculate pairwise correlations
        n = len(exp_arrays)
        correlations = []
        
        for i in range(n):
            for j in range(i+1, n):
                corr = np.corrcoef(exp_arrays[i], exp_arrays[j])[0,1]
                correlations.append(corr)
        
        # Return mean correlation
        return np.mean(correlations)
    
    @staticmethod
    def evaluate_localization(explanation, ground_truth_mask):
        """
        Evaluate how well explanation localizes relevant regions
        (For image tasks where ground truth regions are available)
        
        Args:
            explanation: Generated explanation heatmap
            ground_truth_mask: Binary mask of ground truth important regions
            
        Returns:
            iou_score: Intersection over Union score
        """
        # Threshold explanation to create binary mask
        if isinstance(explanation, torch.Tensor):
            explanation = explanation.cpu().numpy()
        
        exp_mask = explanation > np.percentile(explanation, 70)  # Top 30% as important
        
        # Calculate IoU
        intersection = np.logical_and(exp_mask, ground_truth_mask).sum()
        union = np.logical_or(exp_mask, ground_truth_mask).sum()
        
        return intersection / (union + 1e-6)
    
    @staticmethod
    def compare_methods(results_dict):
        """
        Compare different XAI methods based on evaluation metrics
        
        Args:
            results_dict: Dictionary containing evaluation results for each method
            
        Returns:
            comparison_summary: Dictionary with comparative analysis
        """
        summary = {
            'best_faithfulness': None,
            'best_consistency': None,
            'best_localization': None,
            'overall_best': None,
            'recommendations': []
        }
        
        # Find best method for each metric
        metrics = ['faithfulness', 'consistency', 'localization']
        for metric in metrics:
            if any(metric in res for res in results_dict.values()):
                scores = {method: res[metric] 
                         for method, res in results_dict.items() 
                         if metric in res}
                best_method = max(scores.items(), key=lambda x: x[1])[0]
                summary[f'best_{metric}'] = best_method
        
        # Calculate overall best method
        method_scores = {}
        for method, results in results_dict.items():
            score = np.mean([v for v in results.values()])
            method_scores[method] = score
        
        summary['overall_best'] = max(method_scores.items(), key=lambda x: x[1])[0]
        
        # Generate recommendations
        for method, results in results_dict.items():
            if results.get('faithfulness', 1) < 0.5:
                summary['recommendations'].append(
                    f"Consider improving {method}'s faithfulness through better feature attribution")
            
            if results.get('consistency', 1) < 0.5:
                summary['recommendations'].append(
                    f"Investigate why {method} shows low consistency across similar inputs")
        
        return summary
