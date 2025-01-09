import os
import torch
import logging
import argparse
import torchvision.transforms as transforms
from models.image_model import get_image_model
from models.text_model import get_text_model
from utils.data_loader import TumorDataset, TweetDataset
from utils.visualizer import visualize_explanations, visualize_text_explanations
from xai.gradcam import GradCAMExplainer
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

def process_image_data(device, output_dir, base_dir, num_samples=3):
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
    gradcam = GradCAMExplainer(image_model)
    shap_image = ShapExplainer(image_model, 'image')
    lime_image = LimeExplainer(image_model, 'image')
    logging.info("Image XAI explainers initialized")
    
    # Generate explanations
    for i in range(min(num_samples, len(tumor_dataset))):
        start_time = time.time()
        image, label = tumor_dataset[i]
        image = image.to(device)
        
        logging.info(f"\nProcessing image {i+1}")
        logging.info(f"True label: {'Malignant' if label == 1 else 'Benign'}")
        
        gradcam_exp = gradcam.explain(image)
        shap_exp = shap_image.explain(image)
        lime_exp = lime_image.explain(image)
        
        output_path = os.path.join(output_dir, f'explanations_{i}.png')
        visualize_explanations(image, gradcam_exp, shap_exp, lime_exp, output_path)
        
        elapsed = time.time() - start_time
        logging.info(f"Time taken: {elapsed:.2f} seconds")

def process_text_data(device, output_dir, base_dir, num_samples=3):
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
    for i in range(min(num_samples, len(tweet_dataset))):
        start_time = time.time()
        data = tweet_dataset[i]
        text = data['text']
        label = data['label']
        
        logging.info(f"\nProcessing tweet {i+1}")
        logging.info(f"Text: {text}")
        logging.info(f"True sentiment: {['Negative', 'Neutral', 'Positive'][label]}")
        
        shap_exp = shap_text.explain(text)
        lime_exp = lime_text.explain(text)
        
        output_path = os.path.join(output_dir, f'text_explanations_{i}.png')
        visualize_text_explanations(text, shap_exp, lime_exp, output_path)
        
        elapsed = time.time() - start_time
        logging.info(f"Time taken: {elapsed:.2f} seconds")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='XAI Analysis for Image and Text Data')
    parser.add_argument('--mode', choices=['all', 'image', 'text'], default='all',
                      help='Which dataset to process (default: all)')
    parser.add_argument('--samples', type=int, default=3,
                      help='Number of samples to process (default: 3)')
    args = parser.parse_args()
    
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
