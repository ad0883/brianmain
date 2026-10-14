"""Export the original brianmain classifier to FP32 ONNX and verify parity.

Run from a brianmain checkout after installing requirements-export.txt:
    python deploy/huggingface-static/export_model.py

No training, pretrained download, quantization, or class-order changes occur.
"""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

import numpy as np
import onnx
import onnxruntime as ort
from PIL import Image
import torch
from torchvision import transforms

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))
from model_pytorch import BrainTumorResNet

SOURCE_REVISION = "54baeb9acc4b4fa61906d7c8281077518bc2cd3e"
CHECKPOINT_SHA256 = "35de51a812a9be0b6f871e54757851e7d4e231651521fd3473b82a1f6568a546"
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
SAMPLES = {
    "glioma": "20260304_201141_Te-gl_1.jpg",
    "meningioma": "20260304_201629_Te-aug-me_80.jpg",
    "notumor": "20260304_201717_Te-no_159.jpg",
    "pituitary": "20260304_201649_Te-pi_43.jpg",
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def softmax(logits):
    values = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return values / values.sum(axis=-1, keepdims=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=REPO / "models/brain_tumor_model_pytorch_best.pth")
    parser.add_argument("--output", type=Path, default=HERE / "model.onnx")
    parser.add_argument("--samples-dir", type=Path, default=REPO / "app/uploads")
    args = parser.parse_args()
    torch.set_num_threads(2)
    checkpoint_hash = sha256(args.checkpoint)
    if checkpoint_hash != CHECKPOINT_SHA256:
        raise ValueError("Checkpoint SHA-256 does not match the verified original weights")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("class_names", CLASS_NAMES) != CLASS_NAMES:
        raise ValueError("Checkpoint class order differs from the deployed classifier")
    model = BrainTumorResNet(num_classes=4, pretrained=False).eval()
    model.load_state_dict(checkpoint["model_state_dict"])
    del checkpoint
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        torch.onnx.export(
            model,
            torch.zeros((1, 3, 224, 224), dtype=torch.float32),
            str(args.output),
            input_names=["input"],
            output_names=["logits"],
            opset_version=17,
            export_params=True,
            do_constant_folding=True,
            dynamo=False,
            external_data=False,
        )
    exported = onnx.load(str(args.output), load_external_data=False)
    if any(t.data_location == onnx.TensorProto.EXTERNAL for t in exported.graph.initializer):
        raise ValueError("Expected one self-contained ONNX file")
    onnx.helper.set_model_props(exported, {
        "source_repository": "https://github.com/ad0883/brianmain",
        "source_revision": SOURCE_REVISION,
        "checkpoint_sha256": checkpoint_hash,
        "class_names": json.dumps(CLASS_NAMES),
        "normalization_mean": json.dumps(MEAN),
        "normalization_std": json.dumps(STD),
        "input_layout": "NCHW",
        "image_resize": "RGB, 224x224, PIL bilinear (torchvision Resize)",
    })
    onnx.checker.check_model(exported)
    onnx.save_model(exported, str(args.output), save_as_external_data=False)
    del exported

    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(args.output), sess_options=options, providers=["CPUExecutionProvider"])
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])
    results = []
    for sample_class, filename in SAMPLES.items():
        path = args.samples_dir / filename
        with Image.open(path) as image:
            tensor = transform(image.convert("RGB")).unsqueeze(0)
        with torch.inference_mode():
            original_logits = model(tensor).numpy()
        exported_logits = session.run(["logits"], {"input": tensor.numpy()})[0]
        original_probs, exported_probs = softmax(original_logits), softmax(exported_logits)
        np.testing.assert_allclose(exported_logits, original_logits, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(exported_probs, original_probs, rtol=1e-4, atol=1e-6)
        original_class = CLASS_NAMES[int(original_probs.argmax())]
        exported_class = CLASS_NAMES[int(exported_probs.argmax())]
        if exported_class != original_class:
            raise AssertionError(f"Prediction changed for {filename}")
        results.append({
            "sample_class": sample_class,
            "filename": filename,
            "image_sha256": sha256(path),
            "pytorch_logits": original_logits[0].tolist(),
            "onnx_logits": exported_logits[0].tolist(),
            "pytorch_probabilities": original_probs[0].tolist(),
            "onnx_probabilities": exported_probs[0].tolist(),
            "prediction": exported_class,
            "max_absolute_logit_error": float(np.abs(exported_logits - original_logits).max()),
            "max_absolute_probability_error": float(np.abs(exported_probs - original_probs).max()),
        })
    report = {
        "source_repository": "https://github.com/ad0883/brianmain",
        "source_revision": SOURCE_REVISION,
        "checkpoint_sha256": checkpoint_hash,
        "architecture_sha256": sha256(REPO / "src/model_pytorch.py"),
        "model_file": args.output.name,
        "model_bytes": args.output.stat().st_size,
        "model_sha256": sha256(args.output),
        "format": "ONNX",
        "precision": "float32",
        "opset": 17,
        "input": {"name": "input", "shape": [1, 3, 224, 224], "type": "float32", "layout": "NCHW"},
        "output": {"name": "logits", "shape": [1, 4], "type": "float32"},
        "class_names": CLASS_NAMES,
        "preprocessing": {"color": "RGB", "resize": [224, 224], "interpolation": "PIL bilinear", "scale": "pixel/255", "mean": MEAN, "std": STD},
        "exporter": "torch.onnx.export(dynamo=False, external_data=False)",
        "versions": {name: importlib.metadata.version(name) for name in ["torch", "torchvision", "onnx", "onnxruntime", "Pillow", "numpy"]},
        "validation": {
            "runtime": "ONNX Runtime CPU",
            "logit_tolerance": {"rtol": 1e-4, "atol": 1e-4},
            "probability_tolerance": {"rtol": 1e-4, "atol": 1e-6},
            "max_absolute_logit_error": max(item["max_absolute_logit_error"] for item in results),
            "max_absolute_probability_error": max(item["max_absolute_probability_error"] for item in results),
            "samples": results,
        },
    }
    report_path = args.output.with_name("model-provenance.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ["model_file", "model_bytes", "model_sha256", "input", "output", "class_names"]}, indent=2))
    print(f"Maximum logit error: {report['validation']['max_absolute_logit_error']:.8g}")
    print(f"Maximum probability error: {report['validation']['max_absolute_probability_error']:.8g}")
    print(f"Validated {len(results)} MRI samples; report: {report_path}")


if __name__ == "__main__":
    main()
