"""
Improved Training Script - Focus on Glioma Accuracy
Strategies:
1. Fine-tune ResNet backbone (unfreeze last layers)
2. Class-weighted loss to handle class imbalance
3. Stronger data augmentation
4. Cosine annealing learning rate
5. Label smoothing
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import Counter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model_pytorch import BrainTumorResNet
from src.config import TRAIN_DIR, TEST_DIR, MODEL_DIR, IMG_SIZE, CLASS_NAMES


def get_device():
    """Get the best available device"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✓ GPU detected: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")
        print("⚠ No GPU detected")
    return device


def get_improved_transforms():
    """Stronger data augmentation for better generalization"""
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),  # Slightly larger for random crop
        transforms.RandomCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(30),  # Increased rotation
        transforms.RandomAffine(
            degrees=0, 
            translate=(0.15, 0.15),  # Increased translation
            scale=(0.85, 1.15),  # Add scaling
            shear=10
        ),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),  # Cutout augmentation
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform


def create_weighted_sampler(dataset):
    """Create weighted sampler to balance classes"""
    targets = [dataset.targets[i] for i in range(len(dataset))]
    class_counts = Counter(targets)
    
    print(f"  Class distribution: {dict(class_counts)}")
    
    # Calculate weights - give more weight to underrepresented classes
    weights = []
    for t in targets:
        weights.append(1.0 / class_counts[t])
    
    return WeightedRandomSampler(weights, len(weights), replacement=True)


def get_class_weights(train_dataset, device):
    """Calculate class weights for loss function"""
    targets = train_dataset.targets
    class_counts = Counter(targets)
    total = len(targets)
    
    # Inverse frequency weighting
    weights = []
    for i in range(len(CLASS_NAMES)):
        count = class_counts.get(i, 1)
        weight = total / (len(CLASS_NAMES) * count)
        weights.append(weight)
    
    # Boost glioma weight further (index 0)
    weights[0] *= 1.5  # Extra boost for glioma
    
    weights = torch.FloatTensor(weights).to(device)
    print(f"  Class weights: {dict(zip(CLASS_NAMES, weights.cpu().numpy()))}")
    return weights


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross entropy with label smoothing for better generalization"""
    def __init__(self, smoothing=0.1, weight=None):
        super().__init__()
        self.smoothing = smoothing
        self.weight = weight
    
    def forward(self, pred, target):
        n_classes = pred.size(-1)
        
        # Create smoothed labels
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (n_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        
        log_probs = torch.log_softmax(pred, dim=-1)
        
        if self.weight is not None:
            log_probs = log_probs * self.weight.unsqueeze(0)
        
        return (-true_dist * log_probs).sum(dim=-1).mean()


def create_model_with_unfrozen_layers(num_classes=4, unfreeze_layers=20):
    """Create ResNet50 with some backbone layers unfrozen for fine-tuning"""
    model = BrainTumorResNet(num_classes=num_classes, pretrained=True)
    
    # Count total parameters
    all_params = list(model.base_model.parameters())
    total_params = len(all_params)
    
    # Freeze first layers, unfreeze last layers
    for i, param in enumerate(all_params):
        if i < total_params - unfreeze_layers:
            param.requires_grad = False
        else:
            param.requires_grad = True
    
    # Always unfreeze batch norm layers (important for domain adaptation)
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            for param in module.parameters():
                param.requires_grad = True
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total:,}")
    print(f"  Trainable parameters: {trainable:,} ({100*trainable/total:.1f}%)")
    
    return model


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, total_epochs):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Per-class tracking
    class_correct = [0] * 4
    class_total = [0] * 4
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs} [Train]", leave=False)
    
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Track per-class accuracy
        for i in range(len(labels)):
            label = labels[i].item()
            class_total[label] += 1
            if predicted[i] == labels[i]:
                class_correct[label] += 1
        
        pbar.set_postfix({
            'loss': f'{running_loss/len(pbar):.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    # Print per-class training accuracy
    print(f"    Per-class train acc: ", end="")
    for i, name in enumerate(CLASS_NAMES):
        if class_total[i] > 0:
            acc = 100 * class_correct[i] / class_total[i]
            print(f"{name[:3]}:{acc:.0f}% ", end="")
    print()
    
    return running_loss / len(train_loader), 100. * correct / total


def validate(model, val_loader, criterion, device):
    """Validate the model with per-class metrics"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Per-class tracking
    class_correct = [0] * 4
    class_total = [0] * 4
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Track per-class accuracy
            for i in range(len(labels)):
                label = labels[i].item()
                class_total[label] += 1
                if predicted[i] == labels[i]:
                    class_correct[label] += 1
    
    # Calculate per-class accuracy
    class_acc = {}
    for i, name in enumerate(CLASS_NAMES):
        if class_total[i] > 0:
            class_acc[name] = 100 * class_correct[i] / class_total[i]
        else:
            class_acc[name] = 0
    
    return running_loss / len(val_loader), 100. * correct / total, class_acc


def main():
    print("=" * 70)
    print("IMPROVED TRAINING - FOCUS ON GLIOMA ACCURACY")
    print("=" * 70)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Hyperparameters
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 0.0001  # Lower LR for fine-tuning
    UNFREEZE_LAYERS = 30    # Unfreeze more layers
    
    print(f"\nConfiguration:")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Learning Rate: {LEARNING_RATE}")
    print(f"  Unfrozen Layers: {UNFREEZE_LAYERS}")
    print("=" * 70)
    
    device = get_device()
    
    # 1. Load data with improved transforms
    print("\n[1/4] Loading data with improved augmentation...")
    train_transform, val_transform = get_improved_transforms()
    
    full_train_dataset = datasets.ImageFolder(str(TRAIN_DIR), transform=train_transform)
    test_dataset = datasets.ImageFolder(str(TEST_DIR), transform=val_transform)
    
    # Split training into train/val
    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Get class weights from full training set
    class_weights = get_class_weights(full_train_dataset, device)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True
    )
    
    print(f"  ✓ Training samples: {len(train_dataset)}")
    print(f"  ✓ Validation samples: {len(val_dataset)}")
    print(f"  ✓ Test samples: {len(test_dataset)}")
    
    # 2. Build model with unfrozen layers
    print("\n[2/4] Building model with fine-tuning enabled...")
    model = create_model_with_unfrozen_layers(
        num_classes=4, 
        unfreeze_layers=UNFREEZE_LAYERS
    ).to(device)
    
    # 3. Setup training
    print("\n[3/4] Setting up training...")
    
    # Label smoothing loss with class weights
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1, weight=class_weights)
    
    # AdamW optimizer with weight decay
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=0.01
    )
    
    # Cosine annealing scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    
    # Training loop
    print("\n[4/4] Training...")
    print("-" * 70)
    
    best_val_acc = 0
    best_glioma_acc = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'glioma_acc': []}
    
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, EPOCHS
        )
        
        # Use standard cross entropy for validation (no smoothing)
        val_criterion = nn.CrossEntropyLoss()
        val_loss, val_acc, class_acc = validate(model, val_loader, val_criterion, device)
        
        scheduler.step()
        
        # Log metrics
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['glioma_acc'].append(class_acc.get('glioma', 0))
        
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1}/{EPOCHS}: "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% | "
              f"LR: {current_lr:.6f}")
        print(f"    Val per-class: Gli:{class_acc['glioma']:.1f}% Men:{class_acc['meningioma']:.1f}% "
              f"NoT:{class_acc['notumor']:.1f}% Pit:{class_acc['pituitary']:.1f}%")
        
        # Save best model (prioritize glioma accuracy while maintaining overall accuracy)
        glioma_acc = class_acc.get('glioma', 0)
        
        # Combined score: overall accuracy + glioma bonus
        current_score = val_acc + 0.3 * glioma_acc
        best_score = best_val_acc + 0.3 * best_glioma_acc
        
        if current_score > best_score and val_acc > 85:  # Minimum threshold
            best_val_acc = val_acc
            best_glioma_acc = glioma_acc
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'class_acc': class_acc,
                'class_names': CLASS_NAMES
            }, MODEL_DIR / 'brain_tumor_model_pytorch_best.pth')
            
            print(f"  ✓ New best model saved! (Val Acc: {val_acc:.2f}%, Glioma: {glioma_acc:.1f}%)")
    
    print("-" * 70)
    print("✓ Training completed")
    
    # Final evaluation on test set
    print("\n" + "=" * 70)
    print("FINAL EVALUATION ON TEST SET")
    print("=" * 70)
    
    # Load best model
    checkpoint = torch.load(MODEL_DIR / 'brain_tumor_model_pytorch_best.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_acc, test_class_acc = validate(model, test_loader, nn.CrossEntropyLoss(), device)
    
    print(f"\nTest Results:")
    print(f"  Overall Accuracy: {test_acc:.2f}%")
    print(f"\nPer-Class Accuracy:")
    for name in CLASS_NAMES:
        print(f"  {name}: {test_class_acc[name]:.2f}%")
    
    print(f"\nBest Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Best Glioma Accuracy: {best_glioma_acc:.2f}%")
    
    # Plot training history
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes[0, 0].plot(history['train_loss'], label='Train')
    axes[0, 0].plot(history['val_loss'], label='Validation')
    axes[0, 0].set_title('Loss')
    axes[0, 0].legend()
    axes[0, 0].set_xlabel('Epoch')
    
    axes[0, 1].plot(history['train_acc'], label='Train')
    axes[0, 1].plot(history['val_acc'], label='Validation')
    axes[0, 1].set_title('Overall Accuracy')
    axes[0, 1].legend()
    axes[0, 1].set_xlabel('Epoch')
    
    axes[1, 0].plot(history['glioma_acc'], label='Glioma', color='red', linewidth=2)
    axes[1, 0].set_title('Glioma Accuracy (Validation)')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy %')
    axes[1, 0].legend()
    
    # Bar chart of final per-class accuracy
    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4']
    axes[1, 1].bar(CLASS_NAMES, [test_class_acc[n] for n in CLASS_NAMES], color=colors)
    axes[1, 1].set_title('Test Set Per-Class Accuracy')
    axes[1, 1].set_ylabel('Accuracy %')
    axes[1, 1].set_ylim(0, 100)
    for i, name in enumerate(CLASS_NAMES):
        axes[1, 1].text(i, test_class_acc[name] + 1, f'{test_class_acc[name]:.1f}%', 
                       ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('training_history_improved.png', dpi=150)
    print(f"\n✓ Training history saved to training_history_improved.png")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
