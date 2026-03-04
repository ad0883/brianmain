"""
Debug script to verify model loading and predictions
"""
import os
import sys
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from pathlib import Path
from PIL import Image
import random

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.model_pytorch import BrainTumorResNet
from src.config import TEST_DIR, MODEL_DIR, IMG_SIZE, CLASS_NAMES

def main():
    print("=" * 60)
    print("BRAIN TUMOR MODEL DEBUGGING")
    print("=" * 60)
    
    # 1. Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[1] Device: {device}")
    
    # 2. Load model
    print("\n[2] Loading model...")
    model_path = MODEL_DIR / 'brain_tumor_model_pytorch_best.pth'
    print(f"    Model path: {model_path}")
    print(f"    Exists: {model_path.exists()}")
    
    if not model_path.exists():
        print("ERROR: Model file not found!")
        return
    
    # Load checkpoint and inspect
    checkpoint = torch.load(model_path, map_location=device)
    print(f"    Checkpoint keys: {checkpoint.keys()}")
    
    if 'epoch' in checkpoint:
        print(f"    Trained epochs: {checkpoint['epoch']}")
    if 'best_val_acc' in checkpoint:
        print(f"    Best val accuracy: {checkpoint['best_val_acc']:.2f}%")
    if 'class_names' in checkpoint:
        print(f"    Saved class names: {checkpoint['class_names']}")
    
    # Load model
    model = BrainTumorResNet(num_classes=4, pretrained=False).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("    ✓ Model loaded successfully")
    
    # 3. Check ImageFolder class ordering
    print("\n[3] Checking dataset class ordering...")
    test_dataset = datasets.ImageFolder(str(TEST_DIR))
    print(f"    ImageFolder class_to_idx: {test_dataset.class_to_idx}")
    print(f"    Config CLASS_NAMES: {CLASS_NAMES}")
    
    # Verify they match
    imagefolder_order = sorted(test_dataset.class_to_idx.keys())
    config_order = CLASS_NAMES
    print(f"    ImageFolder order (sorted): {imagefolder_order}")
    print(f"    Config order: {config_order}")
    
    if imagefolder_order == config_order:
        print("    ✓ Class ordering MATCHES")
    else:
        print("    ✗ WARNING: Class ordering MISMATCH!")
    
    # 4. Test transforms - compare web app vs training
    print("\n[4] Transform comparison:")
    
    # Web app transform (from app.py)
    web_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    print("    Web app transform: Resize -> ToTensor -> Normalize")
    
    # Training validation transform  
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    print("    Training val transform: Resize -> ToTensor -> Normalize")
    print("    ✓ Transforms MATCH")
    
    # 5. Test predictions on sample images from each class
    print("\n[5] Testing predictions on sample images from each class...")
    print("-" * 60)
    
    results = {name: {'correct': 0, 'total': 0} for name in CLASS_NAMES}
    
    for class_name in CLASS_NAMES:
        class_dir = TEST_DIR / class_name
        if not class_dir.exists():
            print(f"    WARNING: {class_dir} not found!")
            continue
        
        images = list(class_dir.glob("*"))[:10]  # Test first 10 images per class
        print(f"\n    Testing class '{class_name}' ({len(images)} images):")
        
        for img_path in images:
            try:
                image = Image.open(img_path).convert('RGB')
                input_tensor = web_transform(image).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    output = model(input_tensor)
                    probs = F.softmax(output, dim=1)
                    pred_idx = output.argmax(1).item()
                    confidence = probs[0, pred_idx].item() * 100
                
                pred_class = CLASS_NAMES[pred_idx]
                is_correct = pred_class == class_name
                results[class_name]['total'] += 1
                if is_correct:
                    results[class_name]['correct'] += 1
                
                status = "✓" if is_correct else "✗"
                print(f"      {status} {img_path.name}: pred={pred_class} ({confidence:.1f}%) | true={class_name}")
                
            except Exception as e:
                print(f"      ERROR processing {img_path.name}: {e}")
    
    # 6. Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    total_correct = sum(r['correct'] for r in results.values())
    total_tested = sum(r['total'] for r in results.values())
    
    print(f"\nPer-class accuracy:")
    for class_name in CLASS_NAMES:
        r = results[class_name]
        if r['total'] > 0:
            acc = 100 * r['correct'] / r['total']
            print(f"  {class_name}: {r['correct']}/{r['total']} = {acc:.1f}%")
        else:
            print(f"  {class_name}: No images tested")
    
    if total_tested > 0:
        overall_acc = 100 * total_correct / total_tested
        print(f"\nOverall: {total_correct}/{total_tested} = {overall_acc:.1f}%")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
