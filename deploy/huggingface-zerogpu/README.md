---
title: Brain Tumor MRI Classifier
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 6.30.0
python_version: "3.12"
app_file: app.py
short_description: Classify brain MRIs with the trained brianmain model
---

# Brain Tumor MRI Classifier

A Gradio interface for the existing PyTorch ResNet50 classifier from
[ad0883/brianmain](https://github.com/ad0883/brianmain).

Upload an MRI image to see the predicted class and probabilities for
**glioma**, **meningioma**, **pituitary**, and **no tumor**.
This is an academic demonstration and does not replace medical diagnosis.

Use **ZeroGPU** hardware for free hosting on eligible personal accounts
(verified email, account older than 30 days, and fewer than two existing
ZeroGPU Spaces). The Gradio SDK is required for this hardware.

The app downloads the original 180 MB trained checkpoint from GitHub at
startup and verifies its SHA-256 before loading it. It uses the same model
architecture, image resize, normalization, class order, and probability
calculation as the original Flask application.

Source revision: `54baeb9acc4b4fa61906d7c8281077518bc2cd3e`.

Place **app.py**, **model_pytorch.py**, **requirements.txt**, and **README.md**
at the Space repository root. No training data or separate model upload is
required. GPU calls use ZeroGPU's queue and visitor usage quotas.
