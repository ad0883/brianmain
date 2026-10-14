---
title: Brain Tumor MRI Classifier
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
short_description: Brain MRI classification with PyTorch and Flask
---

# Brain Tumor MRI Classifier

Upload a brain MRI image to classify it as **glioma**, **meningioma**,
**pituitary**, or **no tumor**, with a probability for each class.

This Space serves the existing Flask interface and trained PyTorch ResNet50
model from [ad0883/brianmain](https://github.com/ad0883/brianmain).
It is an academic demonstration and does not replace medical diagnosis.

The Docker build retrieves the serving files and trained checkpoint from
revision `54baeb9acc4b4fa61906d7c8281077518bc2cd3e`, checks the model's SHA-256,
and verifies that it loads before starting Gunicorn on port 7860.
The training dataset is not required for inference.

The app exposes `/health`, `/predict`, and `/api/predict`.
Uploaded images use the Space's temporary filesystem.
