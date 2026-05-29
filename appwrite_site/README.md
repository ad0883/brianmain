# Appwrite Site: Brain Tumor Detection (Static UI)

This folder contains a static copy of the Flask-based frontend, converted so it can be hosted on Appwrite Sites.

What I added:
- `index.html` — static HTML (no Jinja) that expects `window.APPWRITE_API_BASE_URL` to be set.
- `static/js/main.js` — extracted client-side JS logic.
- `static/css/style.css` — copied stylesheet from the original Flask app.

Quick local test:

```bash
# serve locally from project root
python3 -m http.server 8001 --directory appwrite_site
# open http://localhost:8001 in your browser
```

How to configure for Appwrite
1. Deploy the inference function separately (see repository README). Obtain the function's public endpoint.
2. Edit `index.html` and set `window.APPWRITE_API_BASE_URL` near the top to your function base URL (without the `/predict` suffix). Example:

```html
<script>window.APPWRITE_API_BASE_URL = 'https://<appwrite-host>/v1/functions/<function-id>/executions';</script>
```

3. In the Appwrite Console, create a new Site and point the root directory to `appwrite_site/` (or upload the files). Appwrite Sites will host the static files.

Notes and recommendations
- The static UI expects the inference endpoint to accept a `POST` multipart/form-data body with a `file` field and return JSON `{ success: true, prediction: {...}, tumor_info: {...} }` as used by the client.
- Appwrite Sites cannot set runtime environment variables for client-side JS. Setting the `APPWRITE_API_BASE_URL` in `index.html` before upload is the simplest approach. Alternatively, you can host a tiny proxy that injects the value or use a custom build step.
