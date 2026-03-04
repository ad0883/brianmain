"""
PyTorch CNN Model Architecture for Brain Tumor Detection
Supports GPU acceleration on Windows
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class BrainTumorCNN(nn.Module):
    """Custom CNN for brain tumor classification"""
    
    def __init__(self, num_classes=4, dropout_rate=0.5):
        super(BrainTumorCNN, self).__init__()
        
        # First Convolutional Block
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25)
        )
        
        # Second Convolutional Block
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25)
        )
        
        # Third Convolutional Block
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25)
        )
        
        # Fourth Convolutional Block
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25)
        )
        
        # Fifth Convolutional Block
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25)
        )
        
        # Fully Connected Layers (for 224x224 input)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.classifier(x)
        return x


class BrainTumorResNet(nn.Module):
    """Transfer learning model using ResNet50"""
    
    def __init__(self, num_classes=4, pretrained=True, dropout_rate=0.5):
        super(BrainTumorResNet, self).__init__()
        
        # Load pretrained ResNet50
        self.base_model = models.resnet50(weights='IMAGENET1K_V1' if pretrained else None)
        
        # Freeze base model layers
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # Replace the final fully connected layer
        num_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.base_model(x)
    
    def unfreeze_layers(self, num_layers=10):
        """Unfreeze last n layers for fine-tuning"""
        layers = list(self.base_model.children())[:-1]
        for layer in layers[-num_layers:]:
            for param in layer.parameters():
                param.requires_grad = True


class BrainTumorEfficientNet(nn.Module):
    """Transfer learning model using EfficientNet-B0"""
    
    def __init__(self, num_classes=4, pretrained=True, dropout_rate=0.5):
        super(BrainTumorEfficientNet, self).__init__()
        
        # Load pretrained EfficientNet-B0
        self.base_model = models.efficientnet_b0(weights='IMAGENET1K_V1' if pretrained else None)
        
        # Freeze base model layers
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # Replace classifier
        num_features = self.base_model.classifier[1].in_features
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.base_model(x)


def get_model(model_type='custom', num_classes=4, pretrained=True):
    """
    Factory function to get the appropriate model
    
    Args:
        model_type: 'custom', 'resnet', or 'efficientnet'
        num_classes: number of output classes
        pretrained: whether to use pretrained weights for transfer learning
    
    Returns:
        PyTorch model
    """
    if model_type == 'custom':
        return BrainTumorCNN(num_classes=num_classes)
    elif model_type == 'resnet':
        return BrainTumorResNet(num_classes=num_classes, pretrained=pretrained)
    elif model_type == 'efficientnet':
        return BrainTumorEfficientNet(num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = BrainTumorCNN(num_classes=4).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    x = torch.randn(4, 3, 224, 224).to(device)
    output = model(x)
    print(f"Output shape: {output.shape}")
