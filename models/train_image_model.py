import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.image_model import get_image_model
from utils.data_loader import TumorDataset
import logging
from tqdm import tqdm

def train_model(data_dir, batch_size=32, num_epochs=50, learning_rate=0.0001, device='cuda', n_folds=5):
    """Train the tumor detection model using k-fold cross validation"""
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting {n_folds}-fold cross validation")
    
    # Data augmentation and normalization for training
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Just resize and normalize for validation
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load full dataset
    full_dataset = TumorDataset(data_dir, transform=train_transform)
    
    # Create k-fold cross validation splits
    from torch.utils.data import SubsetRandomSampler
    from sklearn.model_selection import KFold
    
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Track metrics across folds
    fold_val_accuracies = []
    best_model_state = None
    best_val_acc = 0.0
    
    # K-fold cross validation loop
    for fold, (train_idx, val_idx) in enumerate(kfold.split(full_dataset)):
        logger.info(f"\nFold {fold + 1}/{n_folds}")
        
        # Create data loaders for this fold
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        
        train_loader = DataLoader(
            full_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=2,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True
        )
        
        val_loader = DataLoader(
            full_dataset,
            batch_size=batch_size,
            sampler=val_sampler,
            num_workers=2,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True
        )
        
        logger.info(f"Training set size: {len(train_idx)}")
        logger.info(f"Validation set size: {len(val_idx)}")
    
        # Initialize model for this fold
        model = get_image_model().to(device)
        
        # Use different learning rates for pretrained layers and new layers
        optimizer = optim.AdamW([
            {'params': [p for n, p in model.named_parameters() if 'fc' not in n], 'lr': learning_rate/10},
            {'params': model.fc.parameters(), 'lr': learning_rate}
        ])
        
        # Cosine annealing scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=5, T_mult=2, eta_min=1e-6
        )
        
        criterion = nn.CrossEntropyLoss()
    
        # Early stopping parameters
        fold_best_val_acc = 0.0
        patience = 7
        patience_counter = 0
    
        # Training loop for this fold
        for epoch in range(num_epochs):
            logger.info(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for inputs, labels in tqdm(train_loader, desc="Training"):
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
            
            train_acc = 100. * train_correct / train_total
            train_loss = train_loss / len(train_loader)
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for inputs, labels in tqdm(val_loader, desc="Validation"):
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()
            
            val_acc = 100. * val_correct / val_total
            val_loss = val_loss / len(val_loader)
            
            # Log metrics
            logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Update learning rate
            scheduler.step()
            
            # Save best model for this fold
            if val_acc > fold_best_val_acc:
                fold_best_val_acc = val_acc
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model_state = model.state_dict()
                    # Save intermediate best model
                    model_path = os.path.join(os.path.dirname(__file__), 'best_tumor_model.pth')
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_acc': val_acc,
                    }, model_path)
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered after epoch {epoch+1}")
                break
        
        # Store this fold's best validation accuracy
        fold_val_accuracies.append(fold_best_val_acc)
        logger.info(f"Best validation accuracy for fold {fold + 1}: {fold_best_val_acc:.2f}%")
    
    # Calculate and log cross-validation results
    mean_val_acc = sum(fold_val_accuracies) / len(fold_val_accuracies)
    std_val_acc = (sum((x - mean_val_acc) ** 2 for x in fold_val_accuracies) / len(fold_val_accuracies)) ** 0.5
    
    logger.info("\nCross-validation Results:")
    logger.info(f"Mean validation accuracy: {mean_val_acc:.2f}% ± {std_val_acc:.2f}%")
    logger.info(f"Best validation accuracy: {best_val_acc:.2f}%")
    
    # Save the best model
    if best_model_state is not None:
        model_path = os.path.join(os.path.dirname(__file__), 'best_tumor_model.pth')
        torch.save({
            'model_state_dict': best_model_state,
            'val_acc': best_val_acc,
            'mean_val_acc': mean_val_acc,
            'std_val_acc': std_val_acc
        }, model_path)
        logger.info(f"Best model saved to {model_path}")
    
    return mean_val_acc, std_val_acc, best_val_acc

if __name__ == '__main__':
    import torch.backends.cudnn as cudnn
    
    # Enable cuDNN auto-tuner
    cudnn.benchmark = True
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set data directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data', 'tumor_detection')
    
    # Train model with cross-validation
    mean_acc, std_acc, best_acc = train_model(data_dir, device=device)
