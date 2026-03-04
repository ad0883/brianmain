"""
PyTorch Training Script for Brain Tumor Detection Model
GPU-accelerated training on Windows with RTX 4060
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model_pytorch import get_model, BrainTumorCNN
from src.config import (
    TRAIN_DIR, TEST_DIR, MODEL_DIR, IMG_SIZE, BATCH_SIZE, 
    EPOCHS, LEARNING_RATE, CLASS_NAMES
)


def get_device():
    """Get the best available device"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✓ GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device("cpu")
        print("⚠ No GPU detected. Training will run on CPU.")
    return device


def get_data_transforms():
    """Get data transforms for training and validation"""
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform


def create_data_loaders(batch_size=BATCH_SIZE):
    """Create data loaders for training and testing"""
    train_transform, val_transform = get_data_transforms()
    
    # Load datasets
    train_dataset = datasets.ImageFolder(str(TRAIN_DIR), transform=train_transform)
    test_dataset = datasets.ImageFolder(str(TEST_DIR), transform=val_transform)
    
    # Split training into train and validation
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Apply validation transform to validation split
    val_dataset.dataset.transform = val_transform
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, train_dataset, test_dataset


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, total_epochs):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs} [Train]", leave=False)
    
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            'loss': f'{running_loss/len(pbar):.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    return running_loss / len(train_loader), 100. * correct / total


def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return running_loss / len(val_loader), 100. * correct / total


def plot_training_history(history, save_path='training_history_pytorch.png'):
    """Plot and save training history"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss plot
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    axes[0].set_title('Model Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy plot
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Train Accuracy')
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Val Accuracy')
    axes[1].set_title('Model Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Training history saved to {save_path}")


def train_model(model_type='custom', epochs=EPOCHS, learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE):
    """
    Main training function
    
    Args:
        model_type: 'custom', 'resnet', or 'efficientnet'
        epochs: number of training epochs
        learning_rate: learning rate for optimizer
        batch_size: batch size for training
    """
    print("="*60)
    print("BRAIN TUMOR DETECTION - PyTorch GPU TRAINING")
    print("="*60)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model Type: {model_type}")
    print(f"Epochs: {epochs}")
    print(f"Batch Size: {batch_size}")
    print(f"Learning Rate: {learning_rate}")
    print("="*60)
    
    # Get device
    device = get_device()
    
    # Create data loaders
    print("\n[1/4] Loading and preprocessing data...")
    try:
        train_loader, val_loader, test_loader, train_dataset, test_dataset = create_data_loaders(batch_size)
        print(f"✓ Training samples: {len(train_dataset)}")
        print(f"✓ Validation samples: {len(val_loader.dataset)}")
        print(f"✓ Test samples: {len(test_dataset)}")
        print(f"✓ Classes: {CLASS_NAMES}")
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        print("\nPlease ensure you have downloaded the dataset!")
        print("Download from: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset")
        print("Place the data in the 'data/' directory with Training/ and Testing/ subdirectories")
        return None
    
    # Build model
    print("\n[2/4] Building model...")
    model = get_model(model_type, num_classes=len(CLASS_NAMES)).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Model built successfully")
    print(f"✓ Total parameters: {total_params:,}")
    print(f"✓ Trainable parameters: {trainable_params:,}")
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Training history
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    best_val_acc = 0.0
    best_model_path = MODEL_DIR / 'brain_tumor_model_pytorch_best.pth'
    
    # Ensure model directory exists
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    # Train model
    print("\n[3/4] Training model...")
    print("-"*60)
    
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, epochs
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}/{epochs}: "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, best_model_path)
            print(f"  ✓ New best model saved! (Val Acc: {val_acc:.2f}%)")
    
    print("-"*60)
    print("✓ Training completed")
    
    # Evaluate on test set
    print("\n[4/4] Evaluating model on test set...")
    
    # Load best model for evaluation
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2f}%")
    print("="*60)
    
    # Save final model
    final_model_path = MODEL_DIR / f"brain_tumor_model_pytorch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_names': CLASS_NAMES,
        'test_acc': test_acc,
    }, final_model_path)
    print(f"\n✓ Final model saved to: {final_model_path}")
    print(f"✓ Best model saved to: {best_model_path}")
    
    # Plot training history
    plot_training_history(history)
    
    return model, history


def evaluate_model(model_path, batch_size=BATCH_SIZE):
    """Evaluate a saved model on test data"""
    device = get_device()
    
    # Load model
    model = BrainTumorCNN(num_classes=len(CLASS_NAMES)).to(device)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load test data
    _, val_transform = get_data_transforms()
    test_dataset = datasets.ImageFolder(str(TEST_DIR), transform=val_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Evaluate
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2f}%")
    
    return test_loss, test_acc


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Brain Tumor Detection Model (PyTorch)')
    parser.add_argument('--model', type=str, default='custom',
                        choices=['custom', 'resnet', 'efficientnet'],
                        help='Model architecture to use')
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE,
                        help='Learning rate')
    parser.add_argument('--evaluate', type=str, default=None,
                        help='Path to model to evaluate')
    
    args = parser.parse_args()
    
    if args.evaluate:
        evaluate_model(args.evaluate, args.batch_size)
    else:
        train_model(
            model_type=args.model,
            epochs=args.epochs,
            learning_rate=args.lr,
            batch_size=args.batch_size
        )
