"""
Training Script for Brain Tumor Detection Model
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Configure GPU before importing TensorFlow
import tensorflow as tf

# Enable GPU memory growth to prevent TensorFlow from allocating all GPU memory
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✓ GPU(s) detected: {len(gpus)}")
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu.name}")
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")
else:
    print("⚠ No GPU detected. Training will run on CPU.")
    print("  To use GPU, install CUDA and cuDNN, then reinstall tensorflow-gpu")

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    EPOCHS, MODEL_PATH, MODEL_DIR, CLASS_NAMES
)
from src.data_preprocessing import DataPreprocessor
from src.model import BrainTumorCNN, create_custom_cnn, compile_model, get_callbacks


def plot_training_history(history, save_path='training_history.png'):
    """Plot and save training history"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Accuracy plot
    axes[0, 0].plot(history.history['accuracy'], label='Train Accuracy')
    axes[0, 0].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[0, 0].set_title('Model Accuracy')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Loss plot
    axes[0, 1].plot(history.history['loss'], label='Train Loss')
    axes[0, 1].plot(history.history['val_loss'], label='Val Loss')
    axes[0, 1].set_title('Model Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Precision plot
    if 'precision' in history.history:
        axes[1, 0].plot(history.history['precision'], label='Train Precision')
        axes[1, 0].plot(history.history['val_precision'], label='Val Precision')
        axes[1, 0].set_title('Model Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    
    # Recall plot
    if 'recall' in history.history:
        axes[1, 1].plot(history.history['recall'], label='Train Recall')
        axes[1, 1].plot(history.history['val_recall'], label='Val Recall')
        axes[1, 1].set_title('Model Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Training history saved to {save_path}")


def train_model(model_type='custom', epochs=EPOCHS, use_transfer_learning=False):
    """
    Main training function
    
    Args:
        model_type: 'custom' or 'transfer' for transfer learning
        epochs: number of training epochs
        use_transfer_learning: whether to use pre-trained weights
    """
    print("="*60)
    print("BRAIN TUMOR DETECTION - MODEL TRAINING")
    print("="*60)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model Type: {model_type}")
    print(f"Epochs: {epochs}")
    print("="*60)
    
    # Initialize data preprocessor
    print("\n[1/4] Loading and preprocessing data...")
    preprocessor = DataPreprocessor()
    
    try:
        train_generator, val_generator, test_generator = preprocessor.create_data_generators()
        print(f"✓ Training samples: {train_generator.samples}")
        print(f"✓ Validation samples: {val_generator.samples}")
        print(f"✓ Test samples: {test_generator.samples}")
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        print("\nPlease ensure you have downloaded the dataset!")
        print("Download from: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset")
        print("Place the data in the 'data/' directory with Training/ and Testing/ subdirectories")
        return None
    
    # Build model
    print("\n[2/4] Building model...")
    if model_type == 'custom':
        brain_tumor_cnn = BrainTumorCNN(model_type='custom')
    else:
        brain_tumor_cnn = BrainTumorCNN(model_type='transfer', base_model='vgg16')
    
    brain_tumor_cnn.build()
    print("✓ Model built successfully")
    print(f"✓ Total parameters: {brain_tumor_cnn.model.count_params():,}")
    
    # Train model
    print("\n[3/4] Training model...")
    print("-"*60)
    
    history = brain_tumor_cnn.train(
        train_generator,
        val_generator,
        epochs=epochs
    )
    
    print("-"*60)
    print("✓ Training completed")
    
    # Evaluate model
    print("\n[4/4] Evaluating model...")
    test_results = brain_tumor_cnn.evaluate(test_generator)
    
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Test Loss: {test_results[0]:.4f}")
    print(f"Test Accuracy: {test_results[1]:.4f}")
    if len(test_results) > 2:
        print(f"Test Precision: {test_results[2]:.4f}")
        print(f"Test Recall: {test_results[3]:.4f}")
    print("="*60)
    
    # Save model
    model_save_path = MODEL_DIR / f"brain_tumor_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5"
    brain_tumor_cnn.save(model_save_path)
    brain_tumor_cnn.save(MODEL_PATH)  # Also save as default model
    print(f"\n✓ Model saved to: {model_save_path}")
    print(f"✓ Model also saved to: {MODEL_PATH}")
    
    # Plot training history
    plot_training_history(history)
    
    return brain_tumor_cnn, history


def fine_tune_model(model_path, epochs=20):
    """Fine-tune a pre-trained model with unfrozen layers"""
    print("Loading pre-trained model for fine-tuning...")
    
    brain_tumor_cnn = BrainTumorCNN()
    brain_tumor_cnn.load(model_path)
    
    # Unfreeze some layers for fine-tuning
    if hasattr(brain_tumor_cnn.model, 'layers'):
        for layer in brain_tumor_cnn.model.layers[-10:]:
            layer.trainable = True
    
    # Recompile with lower learning rate
    from src.model import compile_model
    brain_tumor_cnn.model = compile_model(brain_tumor_cnn.model, learning_rate=1e-5)
    
    # Continue training
    preprocessor = DataPreprocessor()
    train_gen, val_gen, _ = preprocessor.create_data_generators()
    
    history = brain_tumor_cnn.train(train_gen, val_gen, epochs=epochs)
    
    return brain_tumor_cnn, history


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Brain Tumor Detection Model')
    parser.add_argument('--model-type', type=str, default='custom', 
                       choices=['custom', 'transfer'],
                       help='Model type: custom CNN or transfer learning')
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                       help='Number of training epochs')
    parser.add_argument('--fine-tune', type=str, default=None,
                       help='Path to model to fine-tune')
    
    args = parser.parse_args()
    
    if args.fine_tune:
        model, history = fine_tune_model(args.fine_tune, epochs=args.epochs)
    else:
        model, history = train_model(
            model_type=args.model_type,
            epochs=args.epochs
        )
