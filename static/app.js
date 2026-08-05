const MAX_VIEWS = 8;
const MAX_FILE_BYTES = 15 * 1024 * 1024;
const SUPPORTED_TYPES = new Set(["image/jpeg", "image/png", "image/webp"]);
const files = [];

const fileInput = document.querySelector("#fileInput");
const dropzone = document.querySelector("#dropzone");
const viewList = document.querySelector("#viewList");
const viewCount = document.querySelector("#viewCount");
const generateButton = document.querySelector("#generateButton");
const endpointStatus = document.querySelector("#endpointStatus");
const statusDot = document.querySelector(".status-dot");
const settingsToggle = document.querySelector("#settingsToggle");
const settingsBody = document.querySelector("#settingsBody");
const modelViewer = document.querySelector("#modelViewer");
const emptyState = document.querySelector("#emptyState");
const processingState = document.querySelector("#processingState");
const errorState = document.querySelector("#errorState");
const errorMessage = document.querySelector("#errorMessage");
const generationMeta = document.querySelector("#generationMeta");
const downloadButton = document.querySelector("#downloadButton");
const resetCamera = document.querySelector("#resetCamera");
const tryAgainButton = document.querySelector("#tryAgainButton");
const elapsedTime = document.querySelector("#elapsedTime");
const processingKicker = document.querySelector("#processingKicker");
const processingTitle = document.querySelector("#processingTitle");
const processingCopy = document.querySelector("#processingCopy");
const uploadNotice = document.querySelector("#uploadNotice");

let quality = "1024_cascade";
let modelUrl = null;
let elapsedTimer = null;
let progressTimer = null;
let endpointReady = false;
let generationActive = false;
let previewState = "empty";
let draggedIndex = null;
let uploadNoticeTimer = null;

const phaseCopy = [
  [0, "WAKING H100", "Preparing the reconstruction", "The first request may include a cold start while model weights load."],
  [18, "ALIGNING VIEWS", "Finding shared structure", "VGGT is relating object features across every uploaded angle."],
  [65, "BUILDING SHAPE", "Resolving high-resolution geometry", "TRELLIS.2 is creating the sparse shape latent and surface topology."],
  [150, "FINISHING PBR", "Decoding materials and GLB", "Base color, metallic, roughness, and the export mesh are being packed."],
];

function prettyBytes(value) {
  if (value < 1024 * 1024) return `${Math.round(value / 1024)} KB`;
  return `${(value / (1024 * 1024)).toFixed(1)} MB`;
}

function makeId() {
  return crypto.randomUUID ? crypto.randomUUID() : `${Date.now()}-${Math.random()}`;
}

function readyMeta() {
  if (!files.length) return "AWAITING INPUT";
  return `${files.length} ${files.length === 1 ? "VIEW" : "VIEWS"} READY`;
}

function syncControls() {
  viewCount.textContent = String(files.length);
  generateButton.disabled = files.length === 0 || !endpointReady || generationActive;
  fileInput.disabled = generationActive;
  dropzone.disabled = generationActive || files.length >= MAX_VIEWS;
  for (const control of settingsBody.querySelectorAll("button, input, select")) {
    control.disabled = generationActive;
  }
  for (const card of viewList.querySelectorAll(".view-card")) {
    card.draggable = !generationActive;
    card.querySelector(".remove-view").disabled = generationActive;
  }
}

function showUploadNotice(message) {
  clearTimeout(uploadNoticeTimer);
  uploadNotice.textContent = message;
  uploadNotice.hidden = !message;
  if (message) {
    uploadNoticeTimer = setTimeout(() => {
      uploadNotice.hidden = true;
    }, 7000);
  }
}

function releaseModel() {
  if (modelUrl) {
    URL.revokeObjectURL(modelUrl);
    modelUrl = null;
  }
  modelViewer.removeAttribute("src");
  downloadButton.removeAttribute("href");
  downloadButton.classList.add("disabled");
  resetCamera.disabled = true;
}

function markInputsChanged() {
  if (previewState === "model" || previewState === "error") {
    releaseModel();
    showState("empty");
  }
  generationMeta.textContent = readyMeta();
}

function renderViews() {
  viewList.innerHTML = "";
  files.forEach((entry, index) => {
    const card = document.createElement("div");
    card.className = "view-card";
    card.draggable = !generationActive;
    card.dataset.index = String(index);
    card.innerHTML = `
      <span class="drag-handle" title="Drag to reorder">⠿</span>
      <img class="view-thumb" src="${entry.url}" alt="View ${index + 1} preview" />
      <span class="view-info">
        <b></b>
        <span></span>
      </span>
      <button class="remove-view" type="button" aria-label="Remove view ${index + 1}">
        <svg viewBox="0 0 24 24"><path d="M6 6l12 12M18 6 6 18" /></svg>
      </button>`;
    card.querySelector(".view-info b").textContent =
      `View ${String(index + 1).padStart(2, "0")} · ${entry.file.name}`;
    card.querySelector(".view-info span").textContent =
      `${entry.file.type.replace("image/", "").toUpperCase()} · ${prettyBytes(entry.file.size)}`;

    const removeButton = card.querySelector(".remove-view");
    removeButton.disabled = generationActive;
    removeButton.addEventListener("click", () => {
      if (generationActive) return;
      URL.revokeObjectURL(files[index].url);
      files.splice(index, 1);
      markInputsChanged();
      renderViews();
    });
    card.addEventListener("dragstart", (event) => {
      if (generationActive) {
        event.preventDefault();
        return;
      }
      draggedIndex = index;
      event.dataTransfer.effectAllowed = "move";
      requestAnimationFrame(() => card.classList.add("dragging"));
    });
    card.addEventListener("dragend", () => {
      draggedIndex = null;
      card.classList.remove("dragging");
    });
    card.addEventListener("dragover", (event) => {
      event.preventDefault();
      event.dataTransfer.dropEffect = "move";
    });
    card.addEventListener("drop", (event) => {
      event.preventDefault();
      if (generationActive || draggedIndex === null || draggedIndex === index) return;
      const [moved] = files.splice(draggedIndex, 1);
      files.splice(index, 0, moved);
      draggedIndex = null;
      markInputsChanged();
      renderViews();
    });
    viewList.append(card);
  });
  syncControls();
}

function addFiles(selected) {
  if (generationActive) return;
  const candidates = [...selected];
  const available = MAX_VIEWS - files.length;
  const accepted = candidates
    .filter((file) => SUPPORTED_TYPES.has(file.type) && file.size > 0 && file.size <= MAX_FILE_BYTES)
    .slice(0, available);
  const rejected = candidates.length - accepted.length;

  for (const file of accepted) {
    files.push({ id: makeId(), file, url: URL.createObjectURL(file) });
  }
  if (accepted.length) markInputsChanged();
  if (rejected) {
    showUploadNotice(
      `${rejected} ${rejected === 1 ? "file was" : "files were"} skipped. Use up to eight non-empty PNG, JPG, or WebP images of 15 MB or less.`
    );
  } else {
    showUploadNotice("");
  }
  renderViews();
}

dropzone.addEventListener("click", () => {
  if (!generationActive && files.length < MAX_VIEWS) fileInput.click();
});
fileInput.addEventListener("change", () => {
  addFiles(fileInput.files);
  fileInput.value = "";
});
for (const eventName of ["dragenter", "dragover"]) {
    dropzone.addEventListener(eventName, (event) => {
      event.preventDefault();
      if (!generationActive && files.length < MAX_VIEWS) dropzone.classList.add("dragging");
  });
}
for (const eventName of ["dragleave", "drop"]) {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.remove("dragging");
  });
}
dropzone.addEventListener("drop", (event) => {
  if (!generationActive) addFiles(event.dataTransfer.files);
});

settingsToggle.addEventListener("click", () => {
  const expanded = settingsToggle.getAttribute("aria-expanded") === "true";
  settingsToggle.setAttribute("aria-expanded", String(!expanded));
  settingsBody.classList.toggle("collapsed", expanded);
});

document.querySelectorAll("#qualityControl button").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelectorAll("#qualityControl button").forEach((candidate) => candidate.classList.remove("active"));
    button.classList.add("active");
    quality = button.dataset.value;
    markInputsChanged();
  });
});

for (const control of settingsBody.querySelectorAll("input, select")) {
  control.addEventListener("change", markInputsChanged);
}

function showState(name) {
  previewState = name;
  emptyState.hidden = name !== "empty";
  processingState.hidden = name !== "processing";
  errorState.hidden = name !== "error";
  modelViewer.style.display = name === "model" ? "block" : "none";
}

function startElapsedClock() {
  const started = Date.now();
  clearInterval(elapsedTimer);
  clearInterval(progressTimer);
  const tick = () => {
    const totalSeconds = Math.floor((Date.now() - started) / 1000);
    const minutes = String(Math.floor(totalSeconds / 60)).padStart(2, "0");
    const seconds = String(totalSeconds % 60).padStart(2, "0");
    elapsedTime.textContent = `${minutes}:${seconds}`;
  };
  const phase = () => {
    const totalSeconds = Math.floor((Date.now() - started) / 1000);
    const current = [...phaseCopy].reverse().find(([at]) => totalSeconds >= at) || phaseCopy[0];
    processingKicker.textContent = current[1];
    processingTitle.textContent = current[2];
    processingCopy.textContent = current[3];
  };
  tick();
  phase();
  elapsedTimer = setInterval(tick, 1000);
  progressTimer = setInterval(phase, 5000);
}

function stopElapsedClock() {
  clearInterval(elapsedTimer);
  clearInterval(progressTimer);
}

async function generate() {
  if (!files.length || !endpointReady || generationActive) return;
  const seedInput = document.querySelector("#seed");
  if (!seedInput.reportValidity()) return;

  generationActive = true;
  releaseModel();
  showState("processing");
  generationMeta.textContent = "GENERATION ACTIVE";
  startElapsedClock();
  syncControls();

  const form = new FormData();
  files.forEach((entry) => form.append("images", entry.file, entry.file.name));
  form.append("strategy", document.querySelector("#strategy").value);
  form.append("pipeline_type", quality);
  form.append("ss_source", document.querySelector("#ssSource").value);
  form.append("seed", seedInput.value || "0");
  form.append("decimation_target", document.querySelector("#faceTarget").value);
  form.append("texture_size", document.querySelector("#textureSize").value);

  try {
    const response = await fetch("/api/generate", { method: "POST", body: form });
    if (!response.ok) {
      let message = `Generation failed with HTTP ${response.status}.`;
      try {
        const body = await response.json();
        message = body.detail || message;
      } catch (_) {}
      throw new Error(message);
    }

    const blob = await response.blob();
    modelUrl = URL.createObjectURL(blob);
    modelViewer.src = modelUrl;
    downloadButton.href = modelUrl;
    const disposition = response.headers.get("content-disposition") || "";
    const nameMatch = disposition.match(/filename="?([^";]+)"?/i);
    downloadButton.download = nameMatch ? nameMatch[1] : "reconviagen-model.glb";
    downloadButton.classList.remove("disabled");
    resetCamera.disabled = false;
    const serverSeconds = Number(response.headers.get("x-generation-seconds"));
    const inputViews = Number(response.headers.get("x-input-views")) || files.length;
    const outputPipeline = response.headers.get("x-pipeline-type") || quality;
    const outputResolution = Number(response.headers.get("x-output-resolution"));
    const pipelineLabels = {
      "512": "512³",
      "1024": "1024³",
      "1024_cascade": "1024³ CASCADE",
      "1536_cascade": "1536³ CASCADE",
    };
    const resolutionLabel = Number.isFinite(outputResolution) && outputResolution > 0
      ? `${outputResolution}³${outputPipeline.endsWith("_cascade") ? " CASCADE" : ""}`
      : pipelineLabels[outputPipeline] || outputPipeline.toUpperCase();
    const meta = [
      `${inputViews}V`,
      resolutionLabel,
      prettyBytes(blob.size),
      "GLB",
    ];
    if (Number.isFinite(serverSeconds) && serverSeconds > 0) {
      meta.unshift(`${serverSeconds.toFixed(1)}S`);
    }
    generationMeta.textContent = meta.join(" · ");
    showState("model");
  } catch (error) {
    errorMessage.textContent = error.message || String(error);
    generationMeta.textContent = "BUILD FAILED";
    showState("error");
  } finally {
    stopElapsedClock();
    generationActive = false;
    syncControls();
  }
}

generateButton.addEventListener("click", generate);
tryAgainButton.addEventListener("click", () => {
  showState("empty");
  generationMeta.textContent = readyMeta();
});
resetCamera.addEventListener("click", () => {
  modelViewer.cameraOrbit = "0deg 75deg 105%";
  modelViewer.fieldOfView = "30deg";
  modelViewer.jumpCameraToGoal?.();
});

async function checkHealth() {
  try {
    const response = await fetch("/api/health", { cache: "no-store" });
    if (!response.ok) throw new Error(`Health check returned HTTP ${response.status}.`);
    const health = await response.json();
    endpointReady = Boolean(
      health.endpoint_configured &&
      health.auth_configured &&
      health.model === "ReconViaGen v0.5"
    );
    endpointStatus.textContent = endpointReady ? "H100 API READY" : "SETUP REQUIRED";
    endpointStatus.title = endpointReady
      ? `${health.model} · ${health.gpu}`
      : "Configure the ReconViaGen v0.5 endpoint and Modal proxy token.";
    statusDot.classList.toggle("ready", endpointReady);
    statusDot.classList.toggle("error", !endpointReady);
  } catch (_) {
    endpointReady = false;
    endpointStatus.textContent = "LOCAL API ERROR";
    endpointStatus.title = "The local ReconViaGen API could not be reached.";
    statusDot.classList.remove("ready");
    statusDot.classList.add("error");
  }
  syncControls();
}

renderViews();
checkHealth();
setInterval(checkHealth, 30000);
document.addEventListener("visibilitychange", () => {
  if (document.visibilityState === "visible") checkHealth();
});
window.addEventListener("beforeunload", () => {
  for (const entry of files) URL.revokeObjectURL(entry.url);
  if (modelUrl) URL.revokeObjectURL(modelUrl);
});
