import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import AutoTokenizer

class TumorDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        self.labels = []
        
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
    
    def __getitem__(self, idx):
        # Load image
        image_path = self.images[idx]
        image = Image.open(image_path).convert('RGB')
        
        # Apply transformations if any
        if self.transform:
            image = self.transform(image)
        
        # Get label
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return image, label

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
