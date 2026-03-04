"""
Flask Web Application for Brain Tumor Detection
Using PyTorch Model
"""

import os
import sys
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import numpy as np
from datetime import datetime
from PIL import Image

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from torchvision import transforms

# Configuration
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']
IMG_SIZE = 224
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')

# Initialize Flask app
app = Flask(__name__)
CORS(app)
app.secret_key = 'brain_tumor_detection_secret_key_drdo_2024'

# Configuration
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Create upload folder
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global model
model = None
device = None


def get_tumor_info(tumor_class):
    """Get information about a tumor type"""
    info = {
        'glioma': {
            'name': 'Glioma',
            'description': 'A tumor that arises from glial cells in the brain or spine. Glial cells support and protect neurons.',
            'severity': 'high',
            'common_symptoms': [
                'Headaches (often worse in the morning)',
                'Seizures',
                'Memory problems',
                'Personality or behavior changes',
                'Vision problems',
                'Speech difficulties'
            ],
            'recommendation': 'Immediate consultation with a neuro-oncologist is recommended.'
        },
        'meningioma': {
            'name': 'Meningioma', 
            'description': 'A tumor that forms on the membranes (meninges) covering the brain and spinal cord. Usually benign and slow-growing.',
            'severity': 'moderate',
            'common_symptoms': [
                'Gradual onset headaches',
                'Vision changes or loss',
                'Hearing loss or ringing',
                'Weakness in arms or legs',
                'Memory loss'
            ],
            'recommendation': 'Consultation with a neurologist recommended for monitoring and treatment options.'
        },
        'pituitary': {
            'name': 'Pituitary Tumor',
            'description': 'A tumor that develops in the pituitary gland at the base of the brain. Can affect hormone production.',
            'severity': 'moderate',
            'common_symptoms': [
                'Vision problems (peripheral vision loss)',
                'Hormonal imbalances',
                'Fatigue',
                'Unexplained weight changes',
                'Mood changes'
            ],
            'recommendation': 'Consultation with an endocrinologist and neurologist recommended.'
        },
        'notumor': {
            'name': 'No Tumor Detected',
            'description': 'The MRI scan appears normal with no visible tumor masses.',
            'severity': 'low',
            'common_symptoms': [],
            'recommendation': 'Regular health checkups recommended. Consult a doctor if symptoms persist.'
        }
    }
    return info.get(tumor_class, info['notumor'])


def load_model():
    """Load the PyTorch model"""
    global model, device
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Try to load PyTorch model
    model_path = os.path.join(MODEL_DIR, 'brain_tumor_model_pytorch_best.pth')
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return False
    
    try:
        # Import model class
        from src.model_pytorch import BrainTumorResNet
        
        model = BrainTumorResNet(num_classes=4, pretrained=False).to(device)
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print(f"✓ Model loaded successfully from {model_path}")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False


def predict_image(image_path):
    """Make prediction on an image"""
    global model, device
    
    # Image transforms
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = F.softmax(output, dim=1)
        predicted_idx = output.argmax(1).item()
        confidence = probabilities[0, predicted_idx].item()
    
    # Get all probabilities
    all_probs = {CLASS_NAMES[i]: probabilities[0, i].item() for i in range(len(CLASS_NAMES))}
    
    return {
        'predicted_class': CLASS_NAMES[predicted_idx],
        'confidence': confidence,
        'is_tumor_detected': CLASS_NAMES[predicted_idx] != 'notumor',
        'all_probabilities': all_probs
    }


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    global model
    
    # Check if model is loaded
    if model is None:
        if not load_model():
            return jsonify({
                'error': 'Model not loaded. Please train the model first.',
                'instruction': 'Run: python src/train_pytorch.py --model resnet'
            }), 500
    
    # Check if file is present
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Allowed: png, jpg, jpeg, gif, bmp, tiff'}), 400
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Make prediction
        results = predict_image(filepath)
        
        # Get tumor information
        tumor_info = get_tumor_info(results['predicted_class'])
        
        # Prepare response
        response = {
            'success': True,
            'prediction': {
                'class': results['predicted_class'],
                'confidence': round(results['confidence'] * 100, 2),
                'is_tumor_detected': results['is_tumor_detected'],
                'probabilities': {
                    k: round(v * 100, 2) 
                    for k, v in results['all_probabilities'].items()
                }
            },
            'tumor_info': tumor_info,
            'filename': filename,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions (same as /predict but clearly marked as API)"""
    return predict()


@app.route('/health')
def health_check():
    """Health check endpoint"""
    global model
    model_loaded = model is not None
    
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_loaded,
        'device': str(device) if device else 'not initialized',
        'classes': CLASS_NAMES,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/info')
def api_info():
    """API information endpoint"""
    return jsonify({
        'name': 'Brain Tumor Detection API',
        'version': '1.0.0',
        'organization': 'DRDO',
        'endpoints': {
            '/': 'Web interface',
            '/predict': 'POST - Upload MRI image for prediction',
            '/api/predict': 'POST - API endpoint for predictions',
            '/health': 'GET - Health check',
            '/api/info': 'GET - API information'
        },
        'supported_formats': list(ALLOWED_EXTENSIONS),
        'max_file_size': '16MB',
        'classes': CLASS_NAMES
    })


@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return jsonify({'error': 'File too large. Maximum size is 16MB'}), 413


@app.errorhandler(500)
def server_error(e):
    """Handle server errors"""
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    print("="*60)
    print("BRAIN TUMOR DETECTION - WEB APPLICATION")
    print("="*60)
    
    # Load model at startup
    print("Loading model...")
    if load_model():
        print("✓ Model ready for predictions")
    else:
        print("⚠ Model not found - predictions will fail")
        print("  Train first: python src/train_pytorch.py --model resnet")
    
    print("="*60)
    print(f"Starting server at http://localhost:5000")
    print("="*60)
    
    app.run(host='0.0.0.0', port=5000, debug=True)
