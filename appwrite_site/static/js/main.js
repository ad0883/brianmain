// DOM Elements
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');

const previewSection = document.getElementById('previewSection');
const imagePreview = document.getElementById('imagePreview');
const fileName = document.getElementById('fileName');
const imageContainer = document.getElementById('imageContainer');
const analyzeBtn = document.getElementById('analyzeBtn');
const resetBtn = document.getElementById('resetBtn');

const emptyState = document.getElementById('emptyState');
const loadingState = document.getElementById('loadingState');
const resultsContainer = document.getElementById('resultsContainer');
const primaryResultCard = document.getElementById('primaryResultCard');

// Colors for charts
const colors = {
    'glioma': '#ef4444',     // Red
    'meningioma': '#3b82f6', // Blue
    'pituitary': '#8b5cf6',  // Purple
    'notumor': '#10b981'     // Emerald
};

let selectedFile = null;

// Visual Effects Handling
// Drag and Drop
['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, preventDefaults, false);
});

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

['dragenter', 'dragover'].forEach(eventName => {
    dropZone.addEventListener(eventName, highlight, false);
});

['dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, unhighlight, false);
});

function highlight(e) {
    dropZone.classList.add('dragover');
    dropZone.querySelector('.upload-icon').classList.add('fa-bounce');
}

function unhighlight(e) {
    dropZone.classList.remove('dragover');
    dropZone.querySelector('.upload-icon').classList.remove('fa-bounce');
}

dropZone.addEventListener('drop', handleDrop, false);

function handleDrop(e) {
    const dt = e.dataTransfer;
    const files = dt.files;
    if (files.length) handleFile(files[0]);
}

// File Input
fileInput.addEventListener('change', (e) => {
    if (e.target.files.length) handleFile(e.target.files[0]);
});

function handleFile(file) {
    if (!file.type.startsWith('image/')) {
        alert('Invalid file format. Please select an image (JPG, PNG).');
        return;
    }

    selectedFile = file;
    const reader = new FileReader();

    // Add a small delay for a smooth transition feeling
    dropZone.style.opacity = '0.5';

    reader.onload = (e) => {
        imagePreview.src = e.target.result;
        fileName.textContent = file.name;

        setTimeout(() => {
            dropZone.style.display = 'none';
            previewSection.style.display = 'flex';
            previewSection.classList.add('fade-in');
            dropZone.style.opacity = '1';

            // Reset results area
            emptyState.style.display = 'flex';
            loadingState.style.display = 'none';
            resultsContainer.style.display = 'none';
            imageContainer.classList.remove('scanning');
        }, 300);
    };

    reader.readAsDataURL(file);
}

// Analyze Action
analyzeBtn.addEventListener('click', async () => {
    if (!selectedFile) return;

    // UI state change for scanning
    analyzeBtn.disabled = true;
    analyzeBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing...';
    imageContainer.classList.add('scanning');

    emptyState.style.display = 'none';
    resultsContainer.style.display = 'none';
    loadingState.style.display = 'flex';

    const formData = new FormData();
    formData.append('file', selectedFile);
    const apiBaseUrl = (window.APPWRITE_API_BASE_URL || '').trim().replace(/\/$/, '');
    const predictUrl = apiBaseUrl ? `${apiBaseUrl}/predict` : '/predict';

    try {
        // Minimum loading time for aesthetic purposes (1.5s)
        const fetchPromise = fetch(predictUrl, { method: 'POST', body: formData });
        const timerPromise = new Promise(resolve => setTimeout(resolve, 1500));

        const [response] = await Promise.all([fetchPromise, timerPromise]);
        const data = await response.json();

        if (data.success) {
            displayResults(data);
        } else {
            throw new Error(data.error || 'Server error occurred');
        }
    } catch (error) {
        alert('Analysis Failed: ' + error.message);
        emptyState.style.display = 'flex';
        loadingState.style.display = 'none';
    } finally {
        analyzeBtn.disabled = false;
        analyzeBtn.innerHTML = '<i class="fas fa-bolt"></i> Re-Analyze Focus';
        imageContainer.classList.remove('scanning');
    }
});

// Reset Action - Test Another Image
resetBtn.addEventListener('click', () => {
    // Reset file selection
    selectedFile = null;
    fileInput.value = '';
    
    // Hide preview section, show upload area
    previewSection.style.display = 'none';
    dropZone.style.display = 'flex';
    
    // Reset image preview
    imagePreview.src = '';
    fileName.textContent = 'scan_001.jpg';
    
    // Reset results section
    resultsContainer.style.display = 'none';
    loadingState.style.display = 'none';
    emptyState.style.display = 'flex';
    
    // Reset analyze button
    analyzeBtn.innerHTML = '<i class="fas fa-bolt"></i> Initiate Analysis';
    analyzeBtn.disabled = false;
    
    // Reset view toggle if exists
    const viewToggle = document.getElementById('viewToggle');
    if (viewToggle) viewToggle.style.display = 'none';
    
    // Reset image adjustments
    const brightnessSlider = document.getElementById('brightnessSlider');
    const contrastSlider = document.getElementById('contrastSlider');
    if (brightnessSlider) brightnessSlider.value = 100;
    if (contrastSlider) contrastSlider.value = 100;
    imagePreview.style.filter = '';
    
    // Clear heatmap canvas if exists
    const heatmapCanvas = document.getElementById('heatmapCanvas');
    if (heatmapCanvas) {
        const ctx = heatmapCanvas.getContext('2d');
        ctx.clearRect(0, 0, heatmapCanvas.width, heatmapCanvas.height);
    }
});

function displayResults(data) {
    loadingState.style.display = 'none';
    resultsContainer.style.display = 'flex';
    resultsContainer.classList.add('slide-up');

    const { prediction, tumor_info } = data;

    // 1. Update Primary Card
    const statusRing = document.getElementById('statusRing');
    const predClass = document.getElementById('predictionClass');
    const confValue = document.getElementById('confidenceValue');
    const confBadge = document.getElementById('confidenceBadge');

    // Set values
    const formattedClass = prediction.class.toUpperCase().replace('_', ' ');
    predClass.textContent = formattedClass;
    confValue.textContent = prediction.confidence.toFixed(1);

    // Theming based on result
    primaryResultCard.className = 'primary-result-card'; // reset
    statusRing.className = 'status-ring'; // reset

    if (prediction.is_tumor_detected) {
        primaryResultCard.classList.add('danger-state');
        predClass.style.background = 'linear-gradient(to right, #f87171, #ef4444)';
        predClass.style.color = '#ef4444'; // Fallback color
    } else {
        primaryResultCard.classList.add('success-state');
        predClass.style.background = 'linear-gradient(to right, #34d399, #10b981)';
        predClass.style.color = '#10b981'; // Fallback color
    }
    // Apply text clip for gradient text effect
    predClass.style.backgroundClip = 'text';
    predClass.style.webkitBackgroundClip = 'text';
    predClass.style.webkitTextFillColor = 'transparent';
    predClass.style.color = 'transparent';

    // Confidence Badge Logic
    if (prediction.confidence > 95) {
        confBadge.innerHTML = '<i class="fas fa-shield-check"></i> High Confidence';
        confBadge.className = 'confidence-badge high';
    } else if (prediction.confidence > 80) {
        confBadge.innerHTML = '<i class="fas fa-exclamation-circle"></i> Moderate Confidence';
        confBadge.className = 'confidence-badge medium';
    } else {
        confBadge.innerHTML = '<i class="fas fa-question-circle"></i> Low Confidence';
        confBadge.className = 'confidence-badge low';
    }

    // 2. Build Probability Bars
    const probBars = document.getElementById('probabilityBars');
    probBars.innerHTML = '';

    // Sort probabilities highest to lowest
    const sortedProbs = Object.entries(prediction.probabilities)
        .sort(([, a], [, b]) => b - a);

    // Animate bars in with a slight delay
    sortedProbs.forEach(([className, prob], index) => {
        const item = document.createElement('div');
        item.className = 'prob-item';

        const percent = prob.toFixed(1);
        const color = colors[className] || '#a8a29e';
        const formattedName = className.charAt(0).toUpperCase() + className.slice(1);

        item.innerHTML = `
                    <div class="prob-header">
                        <span class="prob-name">${formattedName}</span>
                        <span class="prob-value">${percent}%</span>
                    </div>
                    <div class="prob-track">
                        <div class="prob-fill" style="width: 0%; background-color: ${color}; box-shadow: 0 0 10px ${color}"></div>
                    </div>
                `;
        probBars.appendChild(item);

        // Trigger animation reflow
        setTimeout(() => {
            item.querySelector('.prob-fill').style.width = `${prob}%`;
        }, 100 + (index * 150));
    });

    // 3. Update Tumor Details
    const tInfo = document.getElementById('tumorInfo');

    let symptomsHtml = '';
    if (tumor_info.common_symptoms && tumor_info.common_symptoms.length) {
        symptomsHtml = `
                    <div class="info-group">
                        <span class="info-label">Key Indicators</span>
                        <div class="symptom-tags">
                            ${tumor_info.common_symptoms.map(s => `<span class="tag">${s}</span>`).join('')}
                        </div>
                    </div>
                `;
    }

    tInfo.innerHTML = `
                <div class="info-grid">
                    <div class="info-group">
                        <span class="info-label">Classification</span>
                        <span class="info-value text-accent">${tumor_info.name}</span>
                    </div>
                    <div class="info-group">
                        <span class="info-label">Severity Index</span>
                        <span class="info-value">${tumor_info.severity}</span>
                    </div>
                </div>
                <div class="info-group full-width">
                    <span class="info-label">Pathology Profile</span>
                    <p class="info-desc">${tumor_info.description}</p>
                </div>
                ${symptomsHtml}
            `;

    // 4. Update Recommendation
    document.getElementById('recommendation').innerHTML = `
                <div class="rec-alert">
                    <i class="fas fa-stethoscope"></i>
                    ${tumor_info.recommendation}
                </div>
            `;

    // === FEATURE 1: AI Heatmap ===
    generateHeatmap(prediction.is_tumor_detected, prediction.class);
    document.getElementById('viewToggle').style.display = 'flex';
    document.getElementById('btnRawView').classList.add('active');
    document.getElementById('btnAiView').classList.remove('active');
    document.getElementById('heatmapCanvas').style.display = 'none';

    // === FEATURE 3: Polar Chart ===
    renderPolarChart(prediction.probabilities);

    // === FEATURE 4: 3D Brain ===
    render3DBrain(prediction.class, prediction.is_tumor_detected);

    // === FEATURE 5: Show Export button ===
    document.getElementById('exportPdfBtn').style.display = 'flex';
    window._lastPrediction = data; // store for PDF
}

// =============================================
// FEATURE 1: Simulated Grad-CAM Heatmap
// =============================================
function generateHeatmap(isTumor, tumorClass) {
    const canvas = document.getElementById('heatmapCanvas');
    const container = document.getElementById('imageContainer');
    // Use the displayed container size, not natural image size
    canvas.width = container.offsetWidth || 300;
    canvas.height = container.offsetHeight || 300;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (!isTumor) return; // No heatmap for clean scans

    // Generate a fake Gaussian blob for the "attention" area
    const w = canvas.width, h = canvas.height;
    // Position based on tumor type
    const positions = {
        'glioma': { cx: w * 0.45, cy: h * 0.4 },
        'meningioma': { cx: w * 0.55, cy: h * 0.35 },
        'pituitary': { cx: w * 0.5, cy: h * 0.7 }
    };
    const pos = positions[tumorClass] || { cx: w * 0.5, cy: h * 0.5 };
    const radius = Math.min(w, h) * 0.3;

    // Create radial gradient (jet colormap simulation)
    const gradient = ctx.createRadialGradient(pos.cx, pos.cy, 0, pos.cx, pos.cy, radius);
    gradient.addColorStop(0, 'rgba(255, 0, 0, 0.7)');
    gradient.addColorStop(0.3, 'rgba(255, 165, 0, 0.5)');
    gradient.addColorStop(0.6, 'rgba(255, 255, 0, 0.3)');
    gradient.addColorStop(0.85, 'rgba(0, 128, 255, 0.15)');
    gradient.addColorStop(1, 'rgba(0, 0, 255, 0)');

    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, w, h);

    // Add a second smaller hotspot for realism
    const g2 = ctx.createRadialGradient(pos.cx + radius * 0.3, pos.cy - radius * 0.2, 0,
        pos.cx + radius * 0.3, pos.cy - radius * 0.2, radius * 0.4);
    g2.addColorStop(0, 'rgba(255, 50, 50, 0.5)');
    g2.addColorStop(1, 'rgba(255, 255, 0, 0)');
    ctx.fillStyle = g2;
    ctx.fillRect(0, 0, w, h);
}

// View Toggle Logic
document.getElementById('btnRawView').addEventListener('click', () => {
    document.getElementById('heatmapCanvas').style.display = 'none';
    document.getElementById('btnRawView').classList.add('active');
    document.getElementById('btnAiView').classList.remove('active');
});
document.getElementById('btnAiView').addEventListener('click', () => {
    document.getElementById('heatmapCanvas').style.display = 'block';
    document.getElementById('btnAiView').classList.add('active');
    document.getElementById('btnRawView').classList.remove('active');
});

// =============================================
// FEATURE 2: Medical Image Viewer
// =============================================
let currentZoom = 1;
let panX = 0, panY = 0;
let isPanning = false, panStartX = 0, panStartY = 0;

function applyImageTransform() {
    const img = document.getElementById('imagePreview');
    const b = document.getElementById('brightnessSlider').value;
    const c = document.getElementById('contrastSlider').value;
    img.style.transform = `scale(${currentZoom}) translate(${panX}px, ${panY}px)`;
    img.style.filter = `brightness(${b}%) contrast(${c}%)`;
}

document.getElementById('zoomInBtn').addEventListener('click', () => {
    currentZoom = Math.min(currentZoom + 0.25, 4);
    applyImageTransform();
});
document.getElementById('zoomOutBtn').addEventListener('click', () => {
    currentZoom = Math.max(currentZoom - 0.25, 0.5);
    applyImageTransform();
});
document.getElementById('resetViewBtn').addEventListener('click', () => {
    currentZoom = 1; panX = 0; panY = 0;
    document.getElementById('brightnessSlider').value = 100;
    document.getElementById('contrastSlider').value = 100;
    applyImageTransform();
});
document.getElementById('brightnessSlider').addEventListener('input', applyImageTransform);
document.getElementById('contrastSlider').addEventListener('input', applyImageTransform);

// Pan with mouse drag
const imgContainer = document.getElementById('imageContainer');
imgContainer.addEventListener('mousedown', (e) => {
    if (currentZoom > 1) {
        isPanning = true;
        panStartX = e.clientX - panX;
        panStartY = e.clientY - panY;
        imgContainer.style.cursor = 'grabbing';
    }
});
document.addEventListener('mousemove', (e) => {
    if (isPanning) {
        panX = e.clientX - panStartX;
        panY = e.clientY - panStartY;
        applyImageTransform();
    }
});
document.addEventListener('mouseup', () => {
    isPanning = false;
    imgContainer.style.cursor = currentZoom > 1 ? 'grab' : 'default';
});

// Scroll to zoom
imgContainer.addEventListener('wheel', (e) => {
    e.preventDefault();
    currentZoom += e.deltaY < 0 ? 0.15 : -0.15;
    currentZoom = Math.max(0.5, Math.min(4, currentZoom));
    applyImageTransform();
});

// =============================================
// FEATURE 3: Chart.js Polar Area
// =============================================
let polarChartInstance = null;
function renderPolarChart(probabilities) {
    const ctx = document.getElementById('polarChart').getContext('2d');
    if (polarChartInstance) polarChartInstance.destroy();

    const labels = Object.keys(probabilities).map(k => k.charAt(0).toUpperCase() + k.slice(1));
    const data = Object.values(probabilities);
    const bgColors = Object.keys(probabilities).map(k => {
        const c = colors[k] || '#a8a29e';
        return c + '99'; // add alpha
    });
    const borderColors = Object.keys(probabilities).map(k => colors[k] || '#a8a29e');

    polarChartInstance = new Chart(ctx, {
        type: 'polarArea',
        data: {
            labels: labels,
            datasets: [{
                data: data,
                backgroundColor: bgColors,
                borderColor: borderColors,
                borderWidth: 2
            }]
        },
        options: {
            responsive: false,
            plugins: {
                legend: {
                    display: true,
                    position: 'bottom',
                    labels: { color: '#94a3b8', font: { size: 11 } }
                }
            },
            scales: {
                r: {
                    ticks: { display: false },
                    grid: { color: 'rgba(255,255,255,0.05)' }
                }
            }
        }
    });
}

// =============================================
// FEATURE 4: 3D Brain Model (Three.js)
// =============================================
let brainScene, brainCamera, brainRenderer, brainAnimId;
function render3DBrain(tumorClass, isTumor) {
    const container = document.getElementById('brainModelContainer');
    container.innerHTML = ''; // clear previous
    const containerH = 250;

    brainScene = new THREE.Scene();
    brainCamera = new THREE.PerspectiveCamera(45, container.clientWidth / containerH, 0.1, 100);
    brainCamera.position.z = 3.5;

    brainRenderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
    brainRenderer.setSize(container.clientWidth, containerH);
    brainRenderer.setPixelRatio(window.devicePixelRatio);
    container.appendChild(brainRenderer.domElement);

    // Brain shape: icosahedron with displacement
    const geometry = new THREE.IcosahedronGeometry(1.2, 3);
    // Deform vertices to look organic
    const pos = geometry.attributes.position;
    for (let i = 0; i < pos.count; i++) {
        const x = pos.getX(i), y = pos.getY(i), z = pos.getZ(i);
        const noise = 1 + Math.sin(x * 3) * 0.08 + Math.cos(y * 4) * 0.06 + Math.sin(z * 5) * 0.05;
        pos.setXYZ(i, x * noise, y * noise * 0.85, z * noise); // flatten Y slightly
    }
    geometry.computeVertexNormals();

    const brainMat = new THREE.MeshPhongMaterial({
        color: 0x8899aa,
        wireframe: false,
        transparent: true,
        opacity: 0.6,
        shininess: 80,
        side: THREE.DoubleSide
    });
    const brain = new THREE.Mesh(geometry, brainMat);
    brainScene.add(brain);

    // Wireframe overlay
    const wireMat = new THREE.MeshBasicMaterial({ color: 0x38bdf8, wireframe: true, transparent: true, opacity: 0.15 });
    const wireframe = new THREE.Mesh(geometry.clone(), wireMat);
    brainScene.add(wireframe);

    // Lighting
    const ambientLight = new THREE.AmbientLight(0x404060, 1.5);
    brainScene.add(ambientLight);
    const pointLight = new THREE.PointLight(0x38bdf8, 2, 10);
    pointLight.position.set(2, 2, 3);
    brainScene.add(pointLight);

    // Tumor marker
    if (isTumor) {
        const tumorPositions = {
            'glioma': new THREE.Vector3(0.3, 0.4, 0.8),
            'meningioma': new THREE.Vector3(-0.4, 0.6, 0.6),
            'pituitary': new THREE.Vector3(0, -0.8, 0.4)
        };
        const tPos = tumorPositions[tumorClass] || new THREE.Vector3(0, 0, 0.8);
        const tumorGeo = new THREE.SphereGeometry(0.15, 16, 16);
        const tumorMat = new THREE.MeshPhongMaterial({
            color: 0xff3333,
            emissive: 0xff0000,
            emissiveIntensity: 0.8,
            transparent: true, opacity: 0.9
        });
        const tumorMesh = new THREE.Mesh(tumorGeo, tumorMat);
        tumorMesh.position.copy(tPos);
        brain.add(tumorMesh);

        // Glow ring around tumor
        const ringGeo = new THREE.RingGeometry(0.2, 0.3, 32);
        const ringMat = new THREE.MeshBasicMaterial({ color: 0xff4444, transparent: true, opacity: 0.4, side: THREE.DoubleSide });
        const ring = new THREE.Mesh(ringGeo, ringMat);
        ring.position.copy(tPos);
        ring.lookAt(brainCamera.position);
        brain.add(ring);
    }

    // Animate
    function animate() {
        brainAnimId = requestAnimationFrame(animate);
        brain.rotation.y += 0.005;
        wireframe.rotation.y += 0.005;
        brainRenderer.render(brainScene, brainCamera);
    }
    animate();
}

// =============================================
// FEATURE 5: PDF Export
// =============================================
document.getElementById('exportPdfBtn').addEventListener('click', async () => {
    const data = window._lastPrediction;
    if (!data) return;

    const { jsPDF } = window.jspdf;
    const doc = new jsPDF('p', 'mm', 'a4');
    const pageWidth = doc.internal.pageSize.getWidth();

    // Header
    doc.setFillColor(15, 23, 42);
    doc.rect(0, 0, pageWidth, 35, 'F');
    doc.setTextColor(56, 189, 248);
    doc.setFontSize(18);
    doc.text('NeuralNet Diagnostics', 15, 15);
    doc.setFontSize(10);
    doc.setTextColor(148, 163, 184);
    doc.text(' Advanced Brain Tumor Detection System', 15, 22);
    doc.text('Diagnostic Report · ' + new Date().toLocaleString(), 15, 29);

    // Divider
    doc.setDrawColor(56, 189, 248);
    doc.line(15, 37, pageWidth - 15, 37);

    // Scan Image
    try {
        const imgEl = document.getElementById('imagePreview');
        const canvas = document.createElement('canvas');
        canvas.width = imgEl.naturalWidth || 300;
        canvas.height = imgEl.naturalHeight || 300;
        canvas.getContext('2d').drawImage(imgEl, 0, 0);
        const imgData = canvas.toDataURL('image/jpeg', 0.85);
        doc.addImage(imgData, 'JPEG', 15, 42, 60, 60);
    } catch (e) { console.warn('Could not embed image in PDF'); }

    // Diagnosis Info
    const pred = data.prediction;
    const info = data.tumor_info;
    let yPos = 45;
    const xText = 82;

    doc.setTextColor(255, 255, 255);
    doc.setFontSize(14);
    doc.text('Primary Diagnosis', xText, yPos);
    yPos += 8;

    doc.setFontSize(20);
    doc.setTextColor(pred.is_tumor_detected ? 239 : 16, pred.is_tumor_detected ? 68 : 185, pred.is_tumor_detected ? 68 : 129);
    doc.text(pred.class.toUpperCase(), xText, yPos);
    yPos += 8;

    doc.setFontSize(11);
    doc.setTextColor(200, 200, 200);
    doc.text(`Confidence: ${pred.confidence.toFixed(1)}%`, xText, yPos);
    yPos += 7;
    doc.text(`Severity: ${info.severity}`, xText, yPos);
    yPos += 7;
    doc.text(`Classification: ${info.name}`, xText, yPos);

    // Pathology
    yPos = 110;
    doc.setFontSize(12);
    doc.setTextColor(56, 189, 248);
    doc.text('Pathology Profile', 15, yPos);
    yPos += 7;
    doc.setFontSize(10);
    doc.setTextColor(200, 200, 200);
    const descLines = doc.splitTextToSize(info.description, pageWidth - 30);
    doc.text(descLines, 15, yPos);
    yPos += descLines.length * 5 + 5;

    // Probabilities
    doc.setFontSize(12);
    doc.setTextColor(56, 189, 248);
    doc.text('Probability Distribution', 15, yPos);
    yPos += 7;
    doc.setFontSize(10);
    Object.entries(pred.probabilities).forEach(([cls, prob]) => {
        doc.setTextColor(200, 200, 200);
        doc.text(`${cls.charAt(0).toUpperCase() + cls.slice(1)}: ${prob.toFixed(1)}%`, 15, yPos);
        // Draw bar
        doc.setFillColor(30, 41, 59);
        doc.rect(70, yPos - 3, 100, 4, 'F');
        const c = colors[cls] || '#a8a29e';
        doc.setFillColor(parseInt(c.slice(1, 3), 16), parseInt(c.slice(3, 5), 16), parseInt(c.slice(5, 7), 16));
        doc.rect(70, yPos - 3, prob, 4, 'F');
        yPos += 7;
    });

    // Recommendation
    yPos += 5;
    doc.setFontSize(12);
    doc.setTextColor(56, 189, 248);
    doc.text('Recommendation', 15, yPos);
    yPos += 7;
    doc.setFontSize(10);
    doc.setTextColor(200, 200, 200);
    const recLines = doc.splitTextToSize(info.recommendation, pageWidth - 30);
    doc.text(recLines, 15, yPos);

    // Footer
    const pageH = doc.internal.pageSize.getHeight();
    doc.setFontSize(8);
    doc.setTextColor(100, 100, 100);
    doc.text('Research Purposes Only. This AI system is an assistive tool and must not replace professional medical diagnosis.', 15, pageH - 10);

    doc.save('NeuralNet_Diagnostic_Report.pdf');
});

// Initialize 3D Effects
document.addEventListener('DOMContentLoaded', () => {
    // Initialize Vanilla Tilt manually to ensure it attaches
    if (typeof VanillaTilt !== 'undefined') {
        VanillaTilt.init(document.querySelectorAll(".glass-panel"), {
            max: 3,
            speed: 400,
            glare: true,
            "max-glare": 0.2,
        });
    } else {
        console.error("VanillaTilt is not loaded locally.");
    }

    // Initialize Vanta.js Neural Net Background
    setTimeout(() => {
        try {
            if (typeof VANTA !== 'undefined') {
                window.vantaEffect = VANTA.NET({
                    el: "#vanta-bg",
                    mouseControls: true,
                    touchControls: true,
                    gyroControls: false,
                    minHeight: 200.00,
                    minWidth: 200.00,
                    scale: 1.00,
                    scaleMobile: 1.00,
                    color: 0x38bdf8,
                    backgroundColor: 0x0f172a,
                    points: 12.00,
                    maxDistance: 22.00,
                    spacing: 18.00,
                    showDots: true
                });
            } else {
                console.error("VANTA is not loaded locally.");
            }
        } catch (e) {
            console.error("Vanta initialization failed:", e);
        }
    }, 500);
});
