import os
import torch
import torchvision.transforms as transforms
from models.image_model import get_image_model
from models.text_model import get_text_model
from utils.data_loader import TumorDataset, TweetDataset
from utils.visualizer import visualize_explanations, visualize_text_explanations
from xai.gradcam import GradCAMExplainer
from xai.shap_explainer import ShapExplainer
from xai.lime_explainer import LimeExplainer
from transformers import AutoTokenizer

import os
import time
from datetime import datetime

def main():
    # Set up paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'data', 'tumor_detection')
    tweet_data = os.path.join(base_dir, 'data', 'tweet_sentiment_extraction', 'train.csv')
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(base_dir, 'results')
    output_dir = os.path.join(results_dir, f'xai_results_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Saving results to: {output_dir}")

    # 1. Load datasets
    # Image dataset
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # Load tumor dataset with debug info
    print(f"\nLooking for tumor images in: {data_dir}")
    print("Expecting either 'pos'/'neg' or 'malignant'/'benign' subdirectories")
    
    if os.path.exists(data_dir):
        subdirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        print(f"Found subdirectories: {subdirs}")
    else:
        print(f"Directory not found: {data_dir}")
        
    tumor_dataset = TumorDataset(data_dir, transform=transform)
    
    # Text dataset
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    tweet_dataset = TweetDataset(tweet_data, tokenizer)
    
    print(f"Loaded {len(tumor_dataset)} tumor images and {len(tweet_dataset)} tweets")
    
    # 2. Initialize models
    image_model = get_image_model().to(device)
    text_model = get_text_model().to(device)
    print("Models initialized")
    
    # 3. Initialize XAI methods
    gradcam = GradCAMExplainer(image_model)
    shap_image = ShapExplainer(image_model, 'image')
    shap_text = ShapExplainer(text_model, 'text')
    lime_image = LimeExplainer(image_model, 'image')
    lime_text = LimeExplainer(text_model, 'text')
    print("XAI explainers initialized")
    
    # 4. Generate explanations
    # For image data (process 3 images)
    print("\nGenerating explanations for tumor images...")
    num_images = 3
    for i in range(min(num_images, len(tumor_dataset))):
        start_time = time.time()
        image, label = tumor_dataset[i]
        image = image.to(device)
        
        print(f"\nProcessing image {i+1}")
        print(f"True label: {'Malignant' if label == 1 else 'Benign'}")
        
        # Generate explanations
        gradcam_exp = gradcam.explain(image)
        shap_exp = shap_image.explain(image)
        lime_exp = lime_image.explain(image)
        
        # Visualize
        # Save visualization
        output_path = os.path.join(output_dir, f'explanations_{i}.png')
        visualize_explanations(image, gradcam_exp, shap_exp, lime_exp, output_path)
        
        elapsed = time.time() - start_time
        print(f"Time taken: {elapsed:.2f} seconds")
    
    # For text data (process 3 tweets)
    print("\nGenerating explanations for tweets...")
    num_tweets = 3
    for i in range(min(num_tweets, len(tweet_dataset))):
        start_time = time.time()
        data = tweet_dataset[i]
        text = data['text']
        label = data['label']
        
        print(f"\nProcessing tweet {i+1}")
        print(f"Text: {text}")
        print(f"True sentiment: {['Negative', 'Neutral', 'Positive'][label]}")
        
        # Generate explanations
        shap_exp = shap_text.explain(text)
        lime_exp = lime_text.explain(text)
        
        # Visualize
        # Save visualization
        output_path = os.path.join(output_dir, f'text_explanations_{i}.png')
        visualize_text_explanations(text, shap_exp, lime_exp, output_path)
        
        elapsed = time.time() - start_time
        print(f"Time taken: {elapsed:.2f} seconds")

if __name__ == '__main__':
    main()
