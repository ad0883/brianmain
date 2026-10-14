# Deploy brianmain to Hugging Face Spaces

Two deployment options are provided:

| Option | Files | Account requirement |
| --- | --- | --- |
| Gradio with ZeroGPU | `deploy/huggingface-zerogpu/` | Free personal account with verified email, older than 30 days, and fewer than two existing ZeroGPU Spaces |
| Existing Flask UI with Docker | `deploy/huggingface/` | Hugging Face PRO for a personal account |

Hugging Face's current documentation requires a paid plan to create Docker
Spaces, even though CPU basic has no hourly hardware charge. Eligible free
accounts can host up to two Gradio ZeroGPU Spaces. Neither option needs the
training dataset or a separate model upload.

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

Enter your write-capable Hugging Face token only at the local CLI prompt.
Do not create the Docker version in the same Space if you selected this route.

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

For `hf auth login`, use a Hugging Face token that can write to your Space.
Enter the token at the local CLI prompt. The ChatGPT Hugging Face connection
can read repositories but does not grant permission to upload Space files.

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
