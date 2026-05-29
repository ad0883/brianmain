# Appwrite Function Deployment

This folder contains the Appwrite Function entrypoint for MRI inference.

Use this folder as the function root directory when creating the Appwrite Function.

Recommended environment variables:

- `MODEL_URL`: public or signed download URL for the trained checkpoint
- `MODEL_PATH`: optional local checkpoint path if you bundle the model yourself
- `MODEL_CACHE_PATH`: cache location inside the function container

For Appwrite Sites, point the frontend at the function URL with `APPWRITE_API_BASE_URL`.