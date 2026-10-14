"""Gradio/ZeroGPU deployment of brianmain's existing MRI classifier."""

import os

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

# Import spaces before torch so ZeroGPU can enable its CUDA emulation.
import spaces
import torch
import gradio as gr
import hashlib
import urllib.request
from pathlib import Path
from PIL import Image
from torchvision import transforms

from model_pytorch import BrainTumorResNet

SOURCE_REVISION = "54baeb9acc4b4fa61906d7c8281077518bc2cd3e"
MODEL_SHA256 = "35de51a812a9be0b6f871e54757851e7d4e231651521fd3473b82a1f6568a546"
MODEL_SIZE = 180408506
MODEL_URL = (
    "https://media.githubusercontent.com/media/ad0883/brianmain/"
    f"{SOURCE_REVISION}/models/brain_tumor_model_pytorch_best.pth"
)
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
DISPLAY_NAMES = ["Glioma", "Meningioma", "No tumor", "Pituitary"]
MODEL_PATH = Path(
    os.environ.get(
        "MODEL_PATH",
        str(Path(__file__).resolve().parent / "models/brain_tumor_model_pytorch_best.pth"),
    )
)


def verified_checkpoint(path):
    if not path.is_file() or path.stat().st_size != MODEL_SIZE:
        return False
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest() == MODEL_SHA256


def download_checkpoint():
    if verified_checkpoint(MODEL_PATH):
        return MODEL_PATH
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = MODEL_PATH.with_suffix(".download")
    print("Downloading the trained MRI classifier from GitHub...", flush=True)
    with urllib.request.urlopen(MODEL_URL, timeout=60) as response:
        with temporary.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
    if not verified_checkpoint(temporary):
        temporary.unlink(missing_ok=True)
        raise RuntimeError("The trained checkpoint did not match its Git LFS checksum")
    temporary.replace(MODEL_PATH)
    return MODEL_PATH


checkpoint = torch.load(download_checkpoint(), map_location="cpu", weights_only=True)
if checkpoint.get("class_names", CLASS_NAMES) != CLASS_NAMES:
    raise RuntimeError("The checkpoint's label order differs from this classifier")

model = BrainTumorResNet(num_classes=len(CLASS_NAMES), pretrained=False)
model.load_state_dict(checkpoint["model_state_dict"])
del checkpoint
model.eval()

# ZeroGPU allows CUDA placement at module startup through emulation. Local
# verification falls back to CPU; GPU execution is requested only for inference.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
)


@spaces.GPU(duration=10)
def classify(image: Image.Image):
    if image is None:
        raise gr.Error("Upload an MRI image first.")
    image_tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.inference_mode():
        probabilities = model(image_tensor).softmax(dim=1)[0].cpu().tolist()
    scores = dict(zip(DISPLAY_NAMES, probabilities))
    predicted_index = max(range(len(probabilities)), key=probabilities.__getitem__)
    summary = (
        f"### Model prediction: {DISPLAY_NAMES[predicted_index]}\n"
        f"Confidence: **{probabilities[predicted_index] * 100:.2f}%**"
    )
    details = {
        "class": CLASS_NAMES[predicted_index],
        "confidence": round(probabilities[predicted_index] * 100, 2),
        "is_tumor_detected": CLASS_NAMES[predicted_index] != "notumor",
        "probabilities": {
            label: round(probability * 100, 2)
            for label, probability in zip(CLASS_NAMES, probabilities)
        },
    }
    return scores, summary, details


with gr.Blocks(title="Brain Tumor MRI Classifier") as demo:
    gr.Markdown(
        "# 🧠 Brain Tumor MRI Classifier\n"
        "Upload a brain MRI image to see the predicted class and probabilities."
    )
    with gr.Row():
        with gr.Column():
            image = gr.Image(type="pil", sources=["upload"], label="MRI image", height=360)
            classify_button = gr.Button("Classify MRI", variant="primary")
        with gr.Column():
            probabilities = gr.Label(label="Class probabilities", num_top_classes=4)
            summary = gr.Markdown("The prediction will appear here.")
            with gr.Accordion("Prediction details", open=False):
                details = gr.JSON(label="Result")
    gr.ClearButton([image, probabilities, summary, details])
    gr.Markdown(
        "For coursework and research; this demo does not replace medical diagnosis.\n\n"
        "Source: [ad0883/brianmain](https://github.com/ad0883/brianmain)."
    )
    classify_button.click(
        fn=classify,
        inputs=image,
        outputs=[probabilities, summary, details],
        api_name="classify",
        concurrency_limit=1,
    )

if __name__ == "__main__":
    demo.queue(max_size=20).launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", "7860")),
        theme=gr.themes.Soft(primary_hue="blue"),
    )
