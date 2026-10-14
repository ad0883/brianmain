# Deploy brianmain to Hugging Face Spaces

Three deployment options are provided:

| Option | Files | Account requirement |
| --- | --- | --- |
| Existing interface with browser inference | `deploy/huggingface-static/` | Free Static Space; suitable for a new personal account |
| Gradio with ZeroGPU | `deploy/huggingface-zerogpu/` | Free personal account with verified email, older than 30 days, and fewer than two existing ZeroGPU Spaces |
| Existing Flask UI with Docker | `deploy/huggingface/` | Hugging Face PRO for a personal account |

Hugging Face's current documentation requires a paid plan to create Docker
Spaces, even though CPU basic has no hourly hardware charge. Eligible free
accounts can host up to two Gradio ZeroGPU Spaces. Static Spaces remain free
for all accounts. The Static version includes an ONNX export of the trained
model and runs inference in the visitor's browser; the other two versions
download the original checkpoint. None needs the training dataset.

## Free option for a new account: Static Space

The Static version retains the existing web interface and uses the same
trained ResNet50 weights through ONNX Runtime Web. It runs image preprocessing
and prediction in the browser. No Python server, paid hardware, or external
prediction API is required.

1. Create a new Space with owner `ad0883`, name `brianmain`, SDK **Static**,
   and **Public** visibility. Do not select GPU hardware.
2. Upload the contents of `deploy/huggingface-static/` at the Space root,
   including `index.html`, `README.md`, `model.onnx`, and the `static/` assets.
   Replace the generated README with the supplied one.
3. Open **App**, wait for the model download to finish, then upload an MRI
   image and run the prediction. The initial model download is about 99 MB;
   later predictions in that page reuse the loaded model.

```bash
python -m pip install --upgrade huggingface_hub
hf auth login
hf repos create ad0883/brianmain --type space --space-sdk static --public
hf upload ad0883/brianmain ./deploy/huggingface-static . --type space --exclude "__pycache__/**"
```

The current `hf auth login` supports browser authorization. Follow its login
prompt locally; you do not need to paste a token into chat. The connected
ChatGPT Hugging Face app has read access, so it cannot upload this Space.

`model.onnx` is a generated binary distributed with the Space. To recreate it
from the original checkout and verified `.pth` checkpoint:

```bash
python -m pip install -r deploy/huggingface-static/requirements-export.txt
python deploy/huggingface-static/export_model.py
```

The exported FP32 model has input `input` with shape `[1,3,224,224]` and output
`logits` with shape `[1,4]`. `model-provenance.json` records the original
checkpoint hash, exported model hash, dependency versions, and numerical
comparison against PyTorch on four MRI samples.

## Free option: Gradio with ZeroGPU

1. Sign in at https://huggingface.co and open **New Space**.
2. Select owner `ad0883` and Space name `brianmain`.
3. Select **Gradio** as the SDK, **ZeroGPU** hardware, and **Public** visibility.
4. Create the Space, open **Files → Add file → Upload files**, and upload all
   four files from `deploy/huggingface-zerogpu/` at the Space root. Replace the
   generated README with the supplied README, then commit.
5. Wait for **Running**, open **App**, upload an MRI image, and click
   **Classify MRI**. The result includes the four probabilities and JSON details.

If ZeroGPU is unavailable, check the account's verified email, age, and
existing ZeroGPU Space count. A newer free account is not eligible yet.

From a local checkout containing this deployment folder:

```bash
python -m pip install --upgrade huggingface_hub
hf auth login
hf repos create ad0883/brianmain --type space --space-sdk gradio --flavor zero-a10g --public
hf upload ad0883/brianmain ./deploy/huggingface-zerogpu . --type space
```

Use the local CLI's browser authorization or enter a write-capable token only
at its local prompt. Do not create another SDK version in the same Space if
you selected this route.

## Docker option: preserve the existing Flask UI

This route requires Hugging Face PRO for a personal account. Only **Dockerfile**
and **README.md** from `deploy/huggingface/` belong at the Space root. The
Docker build downloads the Flask UI and trained checkpoint from GitHub.

1. Sign in at https://huggingface.co and open **New Space**.
2. Select owner `ad0883` and enter Space name `brianmain`.
3. Select **Docker** as the SDK and **Blank** as the template.
4. Select **CPU basic** hardware and **Public** visibility, then create the Space.
5. Open **Files**, then **Add file → Upload files**.
6. Upload `deploy/huggingface/Dockerfile` and `deploy/huggingface/README.md`
   together, at the Space repository root. Replace the generated README with
   the supplied one, and commit the files.
7. Wait for the Space status to become **Running**. The first build installs
   PyTorch and downloads the 180 MB trained checkpoint.
8. Open **App** and upload a sample MRI image. Confirm that it returns the
   predicted class and all four probabilities. The `/health` endpoint should
   report `model_loaded: true` and `device: cpu`.

### Using a terminal

From a local checkout containing this deployment folder:

```bash
python -m pip install --upgrade huggingface_hub
hf auth login
hf repos create ad0883/brianmain --type space --space-sdk docker --flavor cpu-basic --public
hf upload ad0883/brianmain ./deploy/huggingface . --type space
```

For `hf auth login`, use the local browser authorization flow or a token that
can write to your Space. Enter a token only at the local CLI prompt. The
ChatGPT Hugging Face connection can read repositories but does not grant
permission to upload Space files.

If a Space with this name already exists, skip the create command only after
checking that it is the intended Docker Space.

### Build and run locally with Docker

```bash
docker build -t brianmain-hf ./deploy/huggingface
docker run --rm -p 7860:7860 brianmain-hf
```

Open http://localhost:7860. The container uses CPU-only PyTorch and a single
Gunicorn worker, matching the intended CPU basic deployment.

## Troubleshooting

- **Configuration error:** Confirm that both files are at the Space root and
  that README.md starts with its YAML metadata (`sdk: docker`, `app_port: 7860`).
- **Download or checksum error:** Open the build logs. The app downloads its
  checkpoint from GitHub's media endpoint and verifies its original Git LFS
  checksum. Retry the build if GitHub reports a temporary network failure.
- **Model load error:** The build stops rather than serving an uninitialized
  classifier. Review the model-load message in the build logs.
- **No prediction after upload:** Leave `APPWRITE_API_BASE_URL` unset in Space
  settings so the interface calls the Flask backend in this Space.

The source revision and model hash are pinned in the Dockerfile. Updating the
GitHub app does not automatically change the deployed Space; update those
values and rebuild when you intend to deploy a newer app or checkpoint.

References: https://huggingface.co/docs/hub/spaces-sdks-docker and
https://huggingface.co/docs/huggingface_hub/en/guides/upload.
Account requirements: https://huggingface.co/docs/hub/spaces-overview and
https://huggingface.co/docs/hub/spaces-zerogpu.
Static SDK: https://huggingface.co/docs/hub/spaces-sdks-static.
