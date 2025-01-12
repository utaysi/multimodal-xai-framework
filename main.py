import os
import torch
import logging
import argparse
import numpy as np
import torchvision.transforms as transforms
from models.image_model import get_image_model
from models.text_model import get_text_model
from utils.evaluator import XAIEvaluator
from utils.data_loader import TumorDataset, TweetDataset
from utils.visualizer import visualize_explanations, visualize_text_explanations
from xai.gradcam import GradCAM
from xai.shap_explainer import ShapExplainer
from xai.lime_explainer import LimeExplainer
from transformers import AutoTokenizer
from datetime import datetime
import time

def setup_logging(output_dir):
    """Set up logging configuration"""
    log_file = os.path.join(output_dir, 'execution.log')
    
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_formatter = logging.Formatter('%(message)s')
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(file_formatter)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

def setup_output_dir():
    """Create and return output directory path"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(base_dir, 'results')
    output_dir = os.path.join(results_dir, f'xai_results_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    return output_dir, base_dir

def process_image_data(device, output_dir, base_dir, num_samples='3'):
    """Process tumor detection images with XAI methods"""
    logging.info("\nProcessing Image Dataset:")
    
    # Setup image processing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    data_dir = os.path.join(base_dir, 'data', 'tumor_detection')
    logging.info(f"Looking for tumor images in: {data_dir}")
    
    if os.path.exists(data_dir):
        subdirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        logging.info(f"Found subdirectories: {subdirs}")
    else:
        logging.error(f"Directory not found: {data_dir}")
        return
    
    # Load dataset and model
    tumor_dataset = TumorDataset(data_dir, transform=transform)
    image_model = get_image_model().to(device)
    logging.info(f"Loaded {len(tumor_dataset)} tumor images")
    
    # Initialize XAI methods
    gradcam = GradCAM(image_model)
    shap_image = ShapExplainer(image_model, 'image')
    lime_image = LimeExplainer(image_model, 'image')
    logging.info("Image XAI explainers initialized")
    
    # Generate explanations
    samples_to_process = len(tumor_dataset) if num_samples == 'all' else min(int(num_samples), len(tumor_dataset))
    for i in range(samples_to_process):
        start_time = time.time()
        sample = tumor_dataset[i]
        image = sample['image'].to(device)
        label = sample['label']
        mask = sample['mask'].to(device)
        similar_inputs = [x.to(device) for x in sample['similar_inputs']]
        
        logging.info(f"\nProcessing image {i+1}")
        logging.info(f"True label: {'Malignant' if label == 1 else 'Benign'}")
        
        # Get model prediction
        with torch.no_grad():
            output = image_model(image.unsqueeze(0))
            probs = torch.softmax(output, dim=1)
            pred_label_idx = torch.argmax(probs).item()
            pred_conf = probs[0][pred_label_idx].item()
            pred_label = 'Malignant' if pred_label_idx == 1 else 'Benign'
        logging.info(f"Model prediction: {pred_label} (confidence: {pred_conf:.3f})")
        
        # Generate explanations
        gradcam_exp = gradcam.explain(image)
        shap_exp = shap_image.explain(image)
        lime_exp = lime_image.explain(image)
        
        # Evaluate and compare XAI methods
        evaluator = XAIEvaluator()
        
        # Evaluate explanations using all metrics
        gradcam_eval = evaluator.evaluate_all(
            model=image_model,
            data=image,
            explanation=gradcam_exp,
            mode='image',
            ground_truth_mask=mask,
            similar_inputs=similar_inputs
        )
        
        shap_eval = evaluator.evaluate_all(
            model=image_model,
            data=image,
            explanation=shap_exp,
            mode='image',
            ground_truth_mask=None,
            similar_inputs=None
        )
        
        lime_eval = evaluator.evaluate_all(
            model=image_model,
            data=image,
            explanation=lime_exp,
            mode='image',
            ground_truth_mask=None,
            similar_inputs=None
        )
        
        # Log results
        logging.info("\nXAI Method Evaluation:")
        logging.info("GradCAM:")
        for metric, score in gradcam_eval.items():
            logging.info(f"  {metric.capitalize()}: {score:.3f}")
            
        logging.info("SHAP:")
        for metric, score in shap_eval.items():
            logging.info(f"  {metric.capitalize()}: {score:.3f}")
            
        if isinstance(lime_exp, np.ndarray):
            logging.info("LIME:")
            for metric, score in lime_eval.items():
                logging.info(f"  {metric.capitalize()}: {score:.3f}")
        
        # Compare methods using all evaluation metrics
        results = {
            'GradCAM': gradcam_eval,
            'SHAP': shap_eval,
            'LIME': lime_eval
        }
        comparison = evaluator.compare_methods(results)
        
        logging.info("\nMethod Comparison:")
        logging.info(f"Best performing method: {comparison['overall_best']}")
        if comparison['recommendations']:
            logging.info("Recommendations:")
            for rec in comparison['recommendations']:
                logging.info(f"- {rec}")
        
        output_path = os.path.join(output_dir, f'explanations_{i}.png')
        visualize_explanations(image, gradcam_exp, shap_exp, lime_exp, output_path)
        
        elapsed = time.time() - start_time
        logging.info(f"Time taken: {elapsed:.2f} seconds")

def process_text_data(device, output_dir, base_dir, num_samples='3'):
    """Process tweet sentiment data with XAI methods"""
    logging.info("\nProcessing Text Dataset:")
    
    # Setup and load text data
    tweet_data = os.path.join(base_dir, 'data', 'tweet_sentiment_extraction', 'train.csv')
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    tweet_dataset = TweetDataset(tweet_data, tokenizer)
    text_model = get_text_model().to(device)
    logging.info(f"Loaded {len(tweet_dataset)} tweets")
    
    # Initialize XAI methods
    shap_text = ShapExplainer(text_model, 'text')
    lime_text = LimeExplainer(text_model, 'text')
    logging.info("Text XAI explainers initialized")
    
    # Generate explanations
    samples_to_process = len(tweet_dataset) if num_samples == 'all' else min(int(num_samples), len(tweet_dataset))
    for i in range(samples_to_process):
        start_time = time.time()
        data = tweet_dataset[i]
        text = data['text']
        label = data['label']
        
        logging.info(f"\nProcessing tweet {i+1}")
        logging.info(f"Text: {text}")
        logging.info(f"True sentiment: {['Negative', 'Neutral', 'Positive'][label]}")
        
        # Get model prediction
        encoded = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            output = text_model(**encoded)
            probs = torch.softmax(output.logits, dim=1)
            pred_label = torch.argmax(probs).item()
            pred_conf = probs[0][pred_label].item()
        logging.info(f"Model prediction: {['Negative', 'Neutral', 'Positive'][pred_label]} (confidence: {pred_conf:.3f})")
        
        # Generate explanations
        shap_exp = shap_text.explain(text)
        lime_exp = lime_text.explain(text)
        
        # Evaluate and compare XAI methods
        evaluator = XAIEvaluator()
        
        # Calculate faithfulness scores
        shap_faith = evaluator.evaluate_faithfulness(text_model, text, shap_exp, mode='text')
        lime_faith = evaluator.evaluate_faithfulness(text_model, text, lime_exp, mode='text')
        
        # Log results
        logging.info("\nXAI Method Evaluation:")
        
        # Get tokens from text
        tokens = text.split()
        
        # SHAP results
        shap_values = np.array(shap_exp).flatten() if hasattr(shap_exp, 'flatten') else np.array(shap_exp)
        logging.info(f"\nSHAP - Faithfulness: {shap_faith:.3f}")
        logging.info("Token importance scores:")
        for token, importance in zip(tokens, shap_values):
            logging.info(f"  {token}: {importance:.3f}")
        
        # LIME results
        lime_dict = dict(lime_exp)
        logging.info(f"\nLIME - Faithfulness: {lime_faith:.3f}")
        logging.info("Token importance scores:")
        for token in tokens:
            importance = lime_dict.get(token, 0.0)
            logging.info(f"  {token}: {importance:.3f}")
        
        # Compare methods
        results = {
            'SHAP': {'faithfulness': shap_faith},
            'LIME': {'faithfulness': lime_faith}
        }
        comparison = evaluator.compare_methods(results)
        
        logging.info("\nMethod Comparison:")
        logging.info(f"Best performing method: {comparison['overall_best']}")
        if comparison['recommendations']:
            logging.info("Recommendations:")
            for rec in comparison['recommendations']:
                logging.info(f"- {rec}")
        
        output_path = os.path.join(output_dir, f'text_explanations_{i}.png')
        visualize_text_explanations(text, shap_exp, lime_exp, output_path)
        
        elapsed = time.time() - start_time
        logging.info(f"Time taken: {elapsed:.2f} seconds")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='XAI Analysis for Image and Text Data')
    parser.add_argument('--mode', choices=['all', 'image', 'text'], default='all',
                      help='Which dataset to process (default: all)')
    parser.add_argument('--samples', type=str, default='3',
                      help='Number of samples to process (default: 3) or "all" to process entire dataset')
    args = parser.parse_args()
    
    # Validate samples argument
    if args.samples != 'all' and not args.samples.isdigit():
        parser.error("--samples must be a positive number or 'all'")
    
    # Setup
    output_dir, base_dir = setup_output_dir()
    setup_logging(output_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    logging.info(f"Saving results to: {output_dir}")
    
    # Process data based on mode
    if args.mode in ['all', 'image']:
        process_image_data(device, output_dir, base_dir, args.samples)
    
    if args.mode in ['all', 'text']:
        process_text_data(device, output_dir, base_dir, args.samples)

if __name__ == '__main__':
    main()
