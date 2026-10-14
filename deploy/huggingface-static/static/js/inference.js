/* Browser model metadata copied from app/app.py:get_tumor_info. */
const BRAIN_TUMOR_INFO = {
  "glioma": {
    "name": "Glioma",
    "description": "A tumor that arises from glial cells in the brain or spine. Glial cells support and protect neurons.",
    "severity": "high",
    "common_symptoms": [
      "Headaches (often worse in the morning)",
      "Seizures",
      "Memory problems",
      "Personality or behavior changes",
      "Vision problems",
      "Speech difficulties"
    ],
    "recommendation": "Immediate consultation with a neuro-oncologist is recommended."
  },
  "meningioma": {
    "name": "Meningioma",
    "description": "A tumor that forms on the membranes (meninges) covering the brain and spinal cord. Usually benign and slow-growing.",
    "severity": "moderate",
    "common_symptoms": [
      "Gradual onset headaches",
      "Vision changes or loss",
      "Hearing loss or ringing",
      "Weakness in arms or legs",
      "Memory loss"
    ],
    "recommendation": "Consultation with a neurologist recommended for monitoring and treatment options."
  },
  "pituitary": {
    "name": "Pituitary Tumor",
    "description": "A tumor that develops in the pituitary gland at the base of the brain. Can affect hormone production.",
    "severity": "moderate",
    "common_symptoms": [
      "Vision problems (peripheral vision loss)",
      "Hormonal imbalances",
      "Fatigue",
      "Unexplained weight changes",
      "Mood changes"
    ],
    "recommendation": "Consultation with an endocrinologist and neurologist recommended."
  },
  "notumor": {
    "name": "No Tumor Detected",
    "description": "The MRI scan appears normal with no visible tumor masses.",
    "severity": "low",
    "common_symptoms": [],
    "recommendation": "Regular health checkups recommended. Consult a doctor if symptoms persist."
  }
};

(() => {
    'use strict';
    const CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary'];
    const SIDE = 224;
    const MEAN = [0.485, 0.456, 0.406];
    const STD = [0.229, 0.224, 0.225];
    const RUNTIME_BASE = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.24.3/dist/';
    let modelSession = null;
    let sessionPromise = null;

    function setStatus(title, message, state = 'loading', progress = null) {
        const panel = document.getElementById('modelStatusPanel');
        const titleElement = document.getElementById('modelStatusTitle');
        const messageElement = document.getElementById('modelStatusText');
        const bar = document.getElementById('modelDownloadProgress');
        if (panel) panel.dataset.state = state;
        if (titleElement) titleElement.textContent = title;
        if (messageElement) messageElement.textContent = message;
        if (bar) {
            bar.hidden = progress === null;
            if (progress !== null) bar.value = progress;
        }
        const loadingText = document.getElementById('inferenceLoadingText');
        if (loadingText) loadingText.textContent = title;
        const headerText = document.getElementById('systemStatusText');
        if (headerText) headerText.textContent = state === 'error' ? 'Model unavailable' : state === 'ready' ? 'Model ready · browser' : 'Loading browser model';
    }

    function showError(error) {
        const message = error && error.message ? error.message : String(error);
        setStatus('Analysis unavailable', message + ' Please retry or reload the page.', 'error');
        console.error('Browser inference failed:', error);
    }

    async function downloadModel() {
        setStatus('Downloading trained model', 'Your image remains on this device. Downloading model.onnx...', 'loading', 0);
        const response = await fetch(new URL('./model.onnx', document.baseURI));
        if (!response.ok) throw new Error('Model download failed (HTTP ' + response.status + ').');
        if (!response.body || !response.body.getReader) return new Uint8Array(await response.arrayBuffer());
        const total = Number(response.headers.get('content-length')) || 98694867;
        const reader = response.body.getReader();
        const parts = [];
        let loaded = 0;
        while (true) {
            const {done, value} = await reader.read();
            if (done) break;
            parts.push(value);
            loaded += value.length;
            const percent = Math.min(100, loaded / total * 100);
            setStatus('Downloading trained model · ' + percent.toFixed(0) + '%', (loaded / 1048576).toFixed(1) + ' MB downloaded. Your image stays in this browser.', 'loading', percent);
        }
        if (!loaded) throw new Error('The model download was empty.');
        const bytes = new Uint8Array(loaded);
        let offset = 0;
        for (const part of parts) { bytes.set(part, offset); offset += part.length; }
        return bytes;
    }

    async function getSession() {
        if (modelSession) return modelSession;
        if (sessionPromise) return sessionPromise;
        sessionPromise = (async () => {
            if (!window.ort) throw new Error('ONNX Runtime could not load from the CDN. Check your connection or content blocker.');
            if (typeof WebAssembly === 'undefined') throw new Error('This browser does not support WebAssembly. Use a current desktop browser.');
            ort.env.wasm.numThreads = 1;
            ort.env.wasm.proxy = false;
            ort.env.wasm.wasmPaths = RUNTIME_BASE;
            const bytes = await downloadModel();
            setStatus('Initializing browser model', 'Preparing the ResNet50 model for WebAssembly inference. This first step can take a moment.');
            // Let the status paint before WebAssembly initialization.
            await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
            const session = await ort.InferenceSession.create(bytes, {executionProviders: ['wasm'], graphOptimizationLevel: 'all'});
            if (!session.inputNames.includes('input') || !session.outputNames.includes('logits')) {
                await session.release();
                throw new Error('The deployed model has unexpected input/output names.');
            }
            modelSession = session;
            setStatus('Model loaded', 'The trained model is ready. Analysis runs on this device; selected images are not uploaded.', 'ready');
            return session;
        })();
        try { return await sessionPromise; }
        catch (error) { sessionPromise = null; throw error; }
    }

    async function decodeImage(file) {
        const url = URL.createObjectURL(file);
        try {
            const img = new Image();
            img.src = url;
            await img.decode();
            if (!img.naturalWidth || !img.naturalHeight) throw new Error('The image has invalid dimensions.');
            if (img.naturalWidth * img.naturalHeight > 40000000) throw new Error('This image is too large to process in the browser. Use an image smaller than 40 megapixels.');
            const canvas = document.createElement('canvas');
            canvas.width = img.naturalWidth;
            canvas.height = img.naturalHeight;
            const context = canvas.getContext('2d', {willReadFrequently: true});
            if (!context) throw new Error('The browser image decoder is unavailable.');
            context.drawImage(img, 0, 0);
            return {width: canvas.width, height: canvas.height, rgba: context.getImageData(0, 0, canvas.width, canvas.height).data};
        } catch (error) {
            if (error.name === 'EncodingError') throw new Error('This image format could not be decoded. Please use a JPG or PNG MRI image; DICOM and TIFF are not supported by most browsers.');
            throw error;
        } finally { URL.revokeObjectURL(url); }
    }

    // Separable antialiased bilinear resize with Pillow-style 22-bit RGB rounding.
    // This avoids browser-specific canvas resizing kernels.
    function bilinearCoefficients(inputSize, outputSize) {
        const scale = inputSize / outputSize;
        const filterScale = Math.max(1, scale);
        const result = [];
        const fixedScale = 4194304;
        for (let output = 0; output < outputSize; output++) {
            const center = (output + 0.5) * scale;
            const first = Math.max(0, Math.floor(center - filterScale + 0.5));
            const last = Math.min(inputSize, Math.floor(center + filterScale + 0.5));
            const weights = [];
            let sum = 0;
            for (let input = first; input < last; input++) {
                const weight = Math.max(0, 1 - Math.abs((input - center + 0.5) / filterScale));
                weights.push(weight);
                sum += weight;
            }
            result.push({first, weights: weights.map(weight => Math.floor(weight / sum * fixedScale + 0.5))});
        }
        return result;
    }

    function preprocessImage(image) {
        const {width, height, rgba} = image;
        const horizontal = bilinearCoefficients(width, SIDE);
        const vertical = bilinearCoefficients(height, SIDE);
        const intermediate = new Uint8Array(height * SIDE * 3);
        const fixedScale = 4194304;
        for (let y = 0; y < height; y++) {
            for (let x = 0; x < SIDE; x++) {
                const coefficients = horizontal[x];
                for (let channel = 0; channel < 3; channel++) {
                    let sum = fixedScale / 2;
                    for (let n = 0; n < coefficients.weights.length; n++) sum += rgba[(y * width + coefficients.first + n) * 4 + channel] * coefficients.weights[n];
                    intermediate[(y * SIDE + x) * 3 + channel] = Math.min(255, Math.max(0, Math.floor(sum / fixedScale)));
                }
            }
        }
        const tensor = new Float32Array(3 * SIDE * SIDE);
        for (let y = 0; y < SIDE; y++) {
            const coefficients = vertical[y];
            for (let x = 0; x < SIDE; x++) {
                for (let channel = 0; channel < 3; channel++) {
                    let sum = fixedScale / 2;
                    for (let n = 0; n < coefficients.weights.length; n++) sum += intermediate[((coefficients.first + n) * SIDE + x) * 3 + channel] * coefficients.weights[n];
                    const pixel = Math.min(255, Math.max(0, Math.floor(sum / fixedScale)));
                    tensor[channel * SIDE * SIDE + y * SIDE + x] = (pixel / 255 - MEAN[channel]) / STD[channel];
                }
            }
        }
        return new ort.Tensor('float32', tensor, [1, 3, SIDE, SIDE]);
    }

    async function predict(file) {
        if (!file) throw new Error('Choose an MRI image first.');
        if (file.size > 16 * 1024 * 1024) throw new Error('The image must be smaller than 16 MB.');
        const session = await getSession();
        setStatus('Processing image in your browser', 'Decoding RGB pixels, resizing to 224 × 224, and running the trained model.');
        const image = await decodeImage(file);
        const input = preprocessImage(image);
        await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
        const result = await session.run({input});
        const logits = Array.from(result.logits.data);
        if (logits.length !== 4 || logits.some(value => !Number.isFinite(value))) throw new Error('The model returned invalid class scores.');
        const maximum = Math.max(...logits);
        const exponentials = logits.map(value => Math.exp(value - maximum));
        const denominator = exponentials.reduce((sum, value) => sum + value, 0);
        const probabilities = exponentials.map(value => value / denominator);
        let index = 0;
        for (let i = 1; i < probabilities.length; i++) if (probabilities[i] > probabilities[index]) index = i;
        const tumorClass = CLASSES[index];
        setStatus('Analysis complete · browser inference', 'Four class probabilities computed from the trained model. Your selected image was not uploaded.', 'ready');
        return {
            success: true,
            prediction: {
                class: tumorClass,
                confidence: Math.round(probabilities[index] * 10000) / 100,
                is_tumor_detected: tumorClass !== 'notumor',
                probabilities: Object.fromEntries(CLASSES.map((name, i) => [name, Math.round(probabilities[i] * 10000) / 100]))
            },
            tumor_info: BRAIN_TUMOR_INFO[tumorClass],
            filename: file.name,
            timestamp: new Date().toISOString()
        };
    }

    window.BrainTumorInference = {predict, showError};
})();
