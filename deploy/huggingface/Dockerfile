# syntax=docker/dockerfile:1
FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860 \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2

WORKDIR /app

# Install the CPU wheels explicitly so pip does not download CUDA packages.
RUN pip install --no-cache-dir \
        Flask==3.1.1 flask-cors==6.0.1 gunicorn==23.0.0 \
        numpy==2.2.6 Pillow==11.3.0 \
    && pip install --no-cache-dir \
        torch==2.8.0 torchvision==0.23.0 \
        --index-url https://download.pytorch.org/whl/cpu

# Hugging Face runs Docker Spaces with UID 1000.
RUN useradd --create-home --uid 1000 appuser \
    && chown appuser:appuser /app
USER appuser

# Fetch only the serving code and the trained model from an immutable revision.
# The Space needs only this Dockerfile and README.md, not the training dataset.
RUN python - <<'PY'
import hashlib
import shutil
import urllib.request
from pathlib import Path

revision = "54baeb9acc4b4fa61906d7c8281077518bc2cd3e"
repository = "ad0883/brianmain"
root = Path.cwd()
files = [
    "app/__init__.py",
    "app/app.py",
    "app/templates/index.html",
    "app/static/css/style.css",
    "app/static/js/chart.min.js",
    "app/static/js/html2canvas.min.js",
    "app/static/js/jspdf.umd.min.js",
    "app/static/js/three.min.js",
    "app/static/js/vanilla-tilt.min.js",
    "app/static/js/vanta.net.min.js",
    "src/__init__.py",
    "src/model_pytorch.py",
]

for name in files:
    destination = root / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://raw.githubusercontent.com/{repository}/{revision}/{name}"
    with urllib.request.urlopen(url, timeout=60) as response:
        with destination.open("wb") as output:
            shutil.copyfileobj(response, output)

# GitHub raw URLs contain Git LFS pointers for .pth files. The media endpoint
# serves the actual checkpoint; verify both its size and its SHA-256 digest.
name = "models/brain_tumor_model_pytorch_best.pth"
destination = root / name
destination.parent.mkdir(parents=True, exist_ok=True)
temporary = destination.with_suffix(".download")
url = f"https://media.githubusercontent.com/media/{repository}/{revision}/{name}"
digest = hashlib.sha256()
size = 0
with urllib.request.urlopen(url, timeout=60) as response:
    with temporary.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
            digest.update(chunk)
            size += len(chunk)

expected_digest = "35de51a812a9be0b6f871e54757851e7d4e231651521fd3473b82a1f6568a546"
if size != 180408506 or digest.hexdigest() != expected_digest:
    temporary.unlink(missing_ok=True)
    raise RuntimeError("The downloaded model did not match its Git LFS checksum")
temporary.replace(destination)
(root / "app/uploads").mkdir(exist_ok=True)
print(f"Downloaded {len(files)} app files and verified the {size}-byte checkpoint")
PY

# Fail the build if the checkpoint cannot load, instead of publishing a broken app.
RUN python -c "from app.app import model; assert model is not None and not model.training, 'The trained model could not load'"

EXPOSE 7860
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "1", "--timeout", "120", "--access-logfile", "-", "--error-logfile", "-", "app.app:app"]
