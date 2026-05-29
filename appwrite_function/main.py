"""Appwrite Function entrypoint for brain tumor inference."""

from __future__ import annotations

import cgi
import io
import json
import os
from functools import lru_cache

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from model_pytorch import BrainTumorResNet

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
IMG_SIZE = 224
DEFAULT_MODEL_URL = os.environ.get("MODEL_URL", "").strip()
LOCAL_MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    os.path.join(os.path.dirname(__file__), "brain_tumor_model_pytorch_best.pth"),
)
CACHE_MODEL_PATH = os.environ.get("MODEL_CACHE_PATH", "/tmp/brain_tumor_model_pytorch_best.pth")


def _headers(extra: dict[str, str] | None = None) -> dict[str, str]:
    headers = {
        "content-type": "application/json",
        "access-control-allow-origin": "*",
        "access-control-allow-headers": "content-type, authorization, x-appwrite-key",
        "access-control-allow-methods": "GET, POST, OPTIONS",
    }
    if extra:
        headers.update(extra)
    return headers


def _json_response(context, payload: dict, status: int = 200):
    return context.res.text(json.dumps(payload), status, _headers())


def _load_model_file() -> str | None:
    if os.path.exists(CACHE_MODEL_PATH):
        return CACHE_MODEL_PATH

    if os.path.exists(LOCAL_MODEL_PATH):
        return LOCAL_MODEL_PATH

    if DEFAULT_MODEL_URL:
        from urllib.request import urlretrieve

        os.makedirs(os.path.dirname(CACHE_MODEL_PATH), exist_ok=True)
        urlretrieve(DEFAULT_MODEL_URL, CACHE_MODEL_PATH)
        return CACHE_MODEL_PATH

    return None


@lru_cache(maxsize=1)
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = _load_model_file()
    if not model_path:
        raise RuntimeError("MODEL_URL or MODEL_PATH must point to a trained checkpoint")

    model = BrainTumorResNet(num_classes=4, pretrained=False).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, device


def _allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in {"png", "jpg", "jpeg", "gif", "bmp", "tiff"}


def _get_tumor_info(tumor_class: str) -> dict:
    info = {
        "glioma": {
            "name": "Glioma",
            "description": "A tumor that arises from glial cells in the brain or spine.",
            "severity": "high",
            "common_symptoms": ["Headaches", "Seizures", "Memory problems"],
            "recommendation": "Immediate consultation with a neuro-oncologist is recommended.",
        },
        "meningioma": {
            "name": "Meningioma",
            "description": "A tumor that forms on the membranes covering the brain and spinal cord.",
            "severity": "moderate",
            "common_symptoms": ["Gradual headaches", "Vision changes", "Weakness"],
            "recommendation": "Consult a neurologist for monitoring and treatment options.",
        },
        "pituitary": {
            "name": "Pituitary Tumor",
            "description": "A tumor that develops in the pituitary gland at the base of the brain.",
            "severity": "moderate",
            "common_symptoms": ["Vision problems", "Hormonal imbalances", "Fatigue"],
            "recommendation": "Consult an endocrinologist and neurologist.",
        },
        "notumor": {
            "name": "No Tumor Detected",
            "description": "The MRI scan appears normal with no visible tumor masses.",
            "severity": "low",
            "common_symptoms": [],
            "recommendation": "Regular health checkups are recommended.",
        },
    }
    return info.get(tumor_class, info["notumor"])


def _predict_bytes(image_bytes: bytes) -> dict:
    model, device = load_model()
    transform = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = F.softmax(output, dim=1)
        predicted_idx = int(output.argmax(1).item())
        confidence = float(probabilities[0, predicted_idx].item())

    all_probs = {CLASS_NAMES[i]: float(probabilities[0, i].item()) for i in range(len(CLASS_NAMES))}

    return {
        "predicted_class": CLASS_NAMES[predicted_idx],
        "confidence": confidence,
        "is_tumor_detected": CLASS_NAMES[predicted_idx] != "notumor",
        "all_probabilities": all_probs,
    }


def _extract_uploaded_file(context) -> tuple[bytes, str]:
    content_type = context.req.headers.get("content-type", "")
    body = context.req.bodyBinary or context.req.bodyText or b""
    if isinstance(body, str):
        body = body.encode("utf-8")

    environ = {
        "REQUEST_METHOD": context.req.method,
        "CONTENT_TYPE": content_type,
        "CONTENT_LENGTH": str(len(body)),
    }

    form = cgi.FieldStorage(fp=io.BytesIO(body), environ=environ, keep_blank_values=True)
    if "file" not in form:
        raise ValueError("No file uploaded")

    uploaded = form["file"]
    if not getattr(uploaded, "filename", ""):
        raise ValueError("No file selected")

    return uploaded.file.read(), uploaded.filename


def main(context):
    if context.req.method == "OPTIONS":
        return context.res.text("", 204, _headers())

    path = context.req.path or "/"

    if path == "/health":
        model_loaded = False
        device = "not initialized"
        try:
            _, device_obj = load_model()
            device = str(device_obj)
            model_loaded = True
        except Exception:
            model_loaded = False

        return _json_response(
            context,
            {
                "status": "healthy",
                "model_loaded": model_loaded,
                "device": device,
                "classes": CLASS_NAMES,
            },
        )

    if path == "/api/info":
        return _json_response(
            context,
            {
                "name": "Brain Tumor Detection API",
                "version": "1.0.0",
                "endpoints": {
                    "/predict": "POST - Upload MRI image for prediction",
                    "/health": "GET - Health check",
                    "/api/info": "GET - API information",
                },
                "supported_formats": ["png", "jpg", "jpeg", "gif", "bmp", "tiff"],
                "classes": CLASS_NAMES,
            },
        )

    if path not in {"/predict", "/api/predict", "/"}:
        return _json_response(context, {"error": "Not found"}, 404)

    if context.req.method != "POST":
        return _json_response(context, {"error": "Method not allowed"}, 405)

    try:
        image_bytes, filename = _extract_uploaded_file(context)
        if not _allowed_file(filename):
            return _json_response(
                context,
                {"error": "Invalid file type. Allowed: png, jpg, jpeg, gif, bmp, tiff"},
                400,
            )

        results = _predict_bytes(image_bytes)
        tumor_info = _get_tumor_info(results["predicted_class"])

        return _json_response(
            context,
            {
                "success": True,
                "prediction": {
                    "class": results["predicted_class"],
                    "confidence": round(results["confidence"] * 100, 2),
                    "is_tumor_detected": results["is_tumor_detected"],
                    "probabilities": {k: round(v * 100, 2) for k, v in results["all_probabilities"].items()},
                },
                "tumor_info": tumor_info,
                "filename": filename,
            },
        )

    except Exception as exc:
        return _json_response(context, {"error": str(exc)}, 500)