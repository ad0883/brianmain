---
title: NeuralNet Brain Tumor Research Demo
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: static
app_file: index.html
pinned: false
tags:
  - brain-tumor
  - medical-imaging
  - onnx
  - research-demo
---

# NeuralNet Brain Tumor Research Demo

The original [ad0883/brianmain](https://github.com/ad0883/brianmain) Flask interface, deployed as a Hugging Face Static Space. The trained ResNet50 checkpoint is exported to ONNX and executed in the browser using ONNX Runtime Web 1.24.3, single-threaded WebAssembly. No inference server or API key is required.

Choose a JPG or PNG MRI image, then select **Initiate Analysis**. The first analysis downloads the approximately 99 MB model and initializes the runtime. Later analyses in the same page reuse the loaded model. Images are decoded and processed locally; the app does not upload MRI images. The app still requests model/runtime assets and display fonts from the web, so initial use requires a connection. Browser-supported image formats only; raw DICOM and most TIFF files need conversion first.

The interface retains image preview, zoom, pan, brightness/contrast, all four probability bars, chart, class information, illustrative overlay, 3D visualization, and PDF export. The overlay and 3D markers are manually generated illustrations from the original interface, **not model localization or Grad-CAM**. They do not identify a tumor's position in the selected image.

## Model and preprocessing

- Source revision: `54baeb9acc4b4fa61906d7c8281077518bc2cd3e`.
- Source checkpoint: `models/brain_tumor_model_pytorch_best.pth`.
- Architecture: `src/model_pytorch.py:BrainTumorResNet`, four-output ResNet50.
- ONNX file: `model.onnx`, FP32, opset 17, all weights embedded.
- Input: `input`, shape `[1, 3, 224, 224]`, float32 NCHW.
- Output: `logits`, shape `[1, 4]`.
- Class order: `glioma`, `meningioma`, `notumor`, `pituitary`.
- Preprocessing: RGB, antialiased bilinear resize to 224 × 224, pixel/255, ImageNet means `[0.485, 0.456, 0.406]`, standard deviations `[0.229, 0.224, 0.225]`.
- The browser uses a separable Pillow-style bilinear RGB resampler with fixed-point rounding, rather than relying on different browser canvas resize kernels. Native browser image decoding can still differ slightly from Pillow, particularly for JPEG or images with color profiles/transparency.
- Displayed probabilities are stable softmax of the real four logits; class information is copied from `app/app.py:get_tumor_info`.

`model-provenance.json` records source/checkpoint/model hashes, export dependency versions, input/output specification, and numerical comparison of PyTorch versus ONNX Runtime CPU on four existing source images. Those four export checks demonstrate numerical consistency; they are not a held-out accuracy evaluation or clinical validation.

`frontend-verification.json` records a separate check using **ONNX Runtime Web 1.24.3's real WebAssembly provider under Node.js**, the deployed model, and the actual JavaScript preprocessing. The source glioma sample produced **80.63%**, with maximum probability difference of `3.31e-6` versus the exporter’s ONNX CPU result. The deterministic RGB resampler also matched Pillow on four synthetic image sizes (maximum normalized channel difference `4.77e-7`). These checks do not exercise browser rendering or native browser JPEG decoding.

## Run locally

Serve this directory over HTTP:

```bash
python -m http.server 7860
```

Open `http://localhost:7860`. Opening `index.html` as a local file does not reliably permit downloading the model. Use a current browser with WebAssembly support and sufficient memory; the first load and inference speed depend on device performance. Downloads or initialization failures appear beside the scan input.

## Deploy

Create a **Static** Space and upload this directory, retaining `index.html`, `README.md`, `model.onnx`, `model-provenance.json`, and `static/`. Export tools are provided for reproducibility and do not run in the Space. All model and local asset paths are relative to the app directory.

ONNX Runtime's JavaScript and WebAssembly files are pinned to the same release on jsDelivr. Official setup/configuration references: [ONNX Runtime Web](https://onnxruntime.ai/docs/get-started/with-javascript/web.html), [environment flags](https://onnxruntime.ai/docs/tutorials/web/env-flags-and-session-options.html), [runtime package](https://cdn.jsdelivr.net/npm/onnxruntime-web@1.24.3/package.json).

This is an educational research demo. Class scores are model outputs, not medical diagnoses. The source project retains its existing branding; this deployment does not establish organizational endorsement, clinical approval, or a new accuracy claim.
