import os
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from transformers import AutoTokenizer
import torchvision.transforms as transforms

class TumorDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Data augmentation transforms for similar inputs
        self.aug_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        ])
        
        # Load images and labels
        # Support both naming conventions: benign/malignant and neg/pos
        label_mapping = {
            'benign': 0, 'neg': 0,  # negative/benign class
            'malignant': 1, 'pos': 1  # positive/malignant class
        }
        
        # Check each subdirectory
        for label_dir in os.listdir(data_dir):
            label_dir_lower = label_dir.lower()
            if label_dir_lower in label_mapping:
                full_dir = os.path.join(data_dir, label_dir)
                label_idx = label_mapping[label_dir_lower]
                
                # Only process if it's a directory
                if os.path.isdir(full_dir):
                    for img_name in os.listdir(full_dir):
                        if img_name.endswith(('.png', '.jpg', '.jpeg')):
                            self.images.append(os.path.join(full_dir, img_name))
                            self.labels.append(label_idx)
        
        if len(self.images) == 0:
            raise ValueError(
                f"No images found in {data_dir}. "
                "Directory should contain 'pos'/'neg' or 'malignant'/'benign' subdirectories with images."
            )
    
    def __len__(self):
        return len(self.images)
    
    def _generate_mask(self, image_size):
        """Generate a synthetic binary mask for demonstration purposes"""
        # Create circular mask centered on image
        h, w = image_size
        y, x = np.ogrid[-h//2:h//2, -w//2:w//2]
        mask = x*x + y*y <= (min(h,w)//4)**2
        return mask.astype(np.float32)
    
    def _get_similar_inputs(self, image):
        """Generate 3 similar inputs using data augmentation"""
        similar_inputs = []
        for _ in range(3):
            # Apply augmentation transforms
            aug_image = self.aug_transform(image)
            # Apply same normalization as original transform
            if self.transform:
                aug_image = self.transform(aug_image)
            similar_inputs.append(aug_image)
        return similar_inputs
    
    def __getitem__(self, idx):
        # Load image
        image_path = self.images[idx]
        image = Image.open(image_path).convert('RGB')
        
        # Get original image size before transforms
        orig_size = image.size[::-1]  # (height, width)
        
        # Apply transformations if any
        if self.transform:
            image = self.transform(image)
        
        # Generate synthetic mask
        mask = self._generate_mask(orig_size)
        
        # Generate similar inputs
        similar_inputs = self._get_similar_inputs(Image.open(image_path).convert('RGB'))
        
        # Get label
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return {
            'image': image,
            'label': label,
            'mask': torch.from_numpy(mask),
            'similar_inputs': torch.stack(similar_inputs)
        }

class TweetDataset(Dataset):
    def __init__(self, csv_file, tokenizer=None):
        self.data = pd.read_csv(csv_file)
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        else:
            self.tokenizer = tokenizer
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        tweet = str(self.data.iloc[idx]['text'])
        sentiment = self.data.iloc[idx]['sentiment']
        
        # Convert sentiment to numeric
        sentiment_map = {'negative': 0, 'neutral': 1, 'positive': 2}
        label = torch.tensor(sentiment_map[sentiment], dtype=torch.long)
        
        # Tokenize tweet
        encoding = self.tokenizer(
            tweet,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': label,
            'text': tweet
        }
