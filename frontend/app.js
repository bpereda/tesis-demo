const form = document.querySelector("#uploadForm");
const fileInput = document.querySelector("#bagFile");
const fileLabel = document.querySelector("#fileLabel");
const sourceButtons = [...document.querySelectorAll(".source-button")];
const demoFileField = document.querySelector("#demoFileField");
const uploadField = document.querySelector("#uploadField");
const demoFileSelect = document.querySelector("#demoFileSelect");
const referenceWeightSelect = document.querySelector("#referenceWeightSelect");
const maxFramesInput = document.querySelector("#maxFrames");
const modeButtons = [...document.querySelectorAll(".mode-button")];
const submitButton = document.querySelector("#submitButton");
const healthStatus = document.querySelector("#healthStatus");
const runState = document.querySelector("#runState");
const statusText = document.querySelector("#statusText");
const progressBar = document.querySelector("#progressBar");
const stepItems = [...document.querySelectorAll(".rail-steps [data-step], .stepper [data-step]")];
const results = document.querySelector("#results");
const details = document.querySelector("#details");
const visuals = document.querySelector("#visuals");
const gallery = document.querySelector("#gallery");
const featuresBody = document.querySelector("#featuresBody");
const visualTabs = [...document.querySelectorAll("[data-visual-tab]")];
const imageDialog = document.querySelector("#imageDialog");
const dialogImage = document.querySelector("#dialogImage");
const dialogCaption = document.querySelector("#dialogCaption");
const dialogClose = document.querySelector("#dialogClose");

const fields = {
  predictedWeight: document.querySelector("#predictedWeight"),
  realWeight: document.querySelector("#realWeight"),
  weightDifference: document.querySelector("#weightDifference"),
  weightError: document.querySelector("#weightError"),
  detectedClusters: document.querySelector("#detectedClusters"),
  totalVolume: document.querySelector("#totalVolume"),
  meanDepth: document.querySelector("#meanDepth"),
};

let sourceMode = "demo";
fileInput.required = false;

function formatNumber(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  return Number(value).toLocaleString(undefined, {
    maximumFractionDigits: digits,
  });
}

function setProgress(percent, state, text) {
  progressBar.style.width = `${percent}%`;
  runState.textContent = state;
  statusText.textContent = text;
}

function setActiveStep(stepName) {
  const order = ["upload", "extract", "segment", "track", "metrics", "predict"];
  const activeIndex = order.indexOf(stepName);
  stepItems.forEach((item) => {
    const index = order.indexOf(item.dataset.step);
    item.classList.toggle("active", item.dataset.step === stepName);
    item.classList.toggle("done", index >= 0 && index < activeIndex);
  });
}

function completeSteps() {
  stepItems.forEach((item) => {
    item.classList.remove("active");
    item.classList.add("done");
  });
}

function resetSteps() {
  stepItems.forEach((item) => {
    item.classList.remove("active", "done");
  });
}

async function checkHealth() {
  try {
    const response = await fetch("/health");
    if (!response.ok) throw new Error("API no disponible");
    healthStatus.textContent = "Sistema listo";
    healthStatus.className = "status ok";
  } catch {
    healthStatus.textContent = "API sin conexión";
    healthStatus.className = "status error";
  }
}

fileInput.addEventListener("change", () => {
  const file = fileInput.files[0];
  fileLabel.textContent = file ? file.name : "Seleccionar video `.bag`";
});

sourceButtons.forEach((button) => {
  button.addEventListener("click", () => {
    sourceButtons.forEach((item) => item.classList.remove("active"));
    button.classList.add("active");
    sourceMode = button.dataset.source;
    demoFileField.hidden = sourceMode !== "demo";
    uploadField.hidden = sourceMode !== "upload";
    fileInput.required = sourceMode === "upload";
  });
});

modeButtons.forEach((button) => {
  button.addEventListener("click", () => {
    modeButtons.forEach((item) => item.classList.remove("active"));
    button.classList.add("active");
    maxFramesInput.value = button.dataset.maxFrames;
  });
});

visualTabs.forEach((button) => {
  button.addEventListener("click", () => {
    visualTabs.forEach((item) => item.classList.remove("active"));
    button.classList.add("active");
    filterGallery(button.dataset.visualTab);
  });
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const data = new FormData();
  data.append("max_frames", maxFramesInput.value || "-1");
  if (referenceWeightSelect.value) {
    data.append("reference_key", referenceWeightSelect.value);
  }

  let endpoint = "/run-demo-file";
  if (sourceMode === "upload") {
    const file = fileInput.files[0];
    if (!file) return;
    endpoint = "/upload";
    data.append("file", file);
  } else {
    if (!demoFileSelect.value) {
      setProgress(0, "Error", "No hay videos `.bag` disponibles en ~/demo_data.");
      return;
    }
    data.append("filename", demoFileSelect.value);
  }

  results.hidden = true;
  details.hidden = true;
  visuals.hidden = true;
  submitButton.disabled = true;
  resetSteps();
  setActiveStep("upload");
  setProgress(
    12,
    sourceMode === "upload" ? "Subiendo video" : "En cola",
    sourceMode === "upload" ? "Subiendo el video de la cámara RealSense a la VM." : "Usando un video `.bag` ya disponible en la VM."
  );

  try {
    const response = await fetch(endpoint, {
      method: "POST",
      body: data,
    });

    if (!response.ok) {
      let message = `La solicitud falló con estado ${response.status}`;
      try {
        const payload = await response.json();
        message = payload.detail || message;
      } catch {
        // Keep the HTTP status fallback.
      }
      throw new Error(message);
    }

    const payload = await response.json();
    await pollJob(payload.job_id);
  } catch (error) {
    setProgress(0, "Error", error.message || "El análisis falló.");
  } finally {
    submitButton.disabled = false;
  }
});

async function pollJob(jobId) {
  while (true) {
    const response = await fetch(`/jobs/${jobId}/status`, { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`No se pudo leer el estado del trabajo (${response.status})`);
    }

    const status = await response.json();
    updateFromStatus(status);

    if (status.state === "complete") {
      completeSteps();
      renderResult(status.result);
      return;
    }
    if (status.state === "error") {
      throw new Error(status.error || status.message || "El análisis falló.");
    }

    await new Promise((resolve) => window.setTimeout(resolve, 1500));
  }
}

function updateFromStatus(status) {
  if (status.stage && status.stage !== "complete" && status.stage !== "error") {
    setActiveStep(status.stage);
  }
  const label = stageLabel(status.stage, status.state);
  setProgress(status.percent ?? 0, label, status.message || "Análisis en ejecución.");
}

function stageLabel(stage, state) {
  if (state === "queued") return "En cola";
  if (state === "complete") return "Completo";
  if (state === "error") return "Error";
  const labels = {
    upload: "Preparando video",
    extract: "Leyendo imagen y profundidad",
    segment: "Detectando racimos",
    track: "Evitando conteos repetidos",
    metrics: "Calculando medidas",
    predict: "Estimando peso",
  };
  return labels[stage] || "En ejecución";
}

function renderResult(result) {
  fields.predictedWeight.textContent = `${formatNumber(result.predicted_weight, 2)} kg`;
  const comparison = result.real_weight_comparison;
  if (comparison) {
    fields.realWeight.textContent = `${formatNumber(comparison.real_weight_kg, 2)} kg`;
    fields.weightDifference.textContent =
      comparison.error_kg === undefined ? "-" : `${comparison.error_kg >= 0 ? "+" : ""}${formatNumber(comparison.error_kg, 2)} kg`;
    fields.weightError.textContent =
      comparison.error_percent === undefined || comparison.error_percent === null ? "-" : `${formatNumber(comparison.error_percent, 1)}%`;
  } else {
    fields.realWeight.textContent = "-";
    fields.weightDifference.textContent = "-";
    fields.weightError.textContent = "-";
  }
  fields.detectedClusters.textContent = formatNumber(result.detected_clusters, 0);
  fields.totalVolume.textContent = `${formatNumber(result.total_estimated_volume_cm3 / 1000, 3)} L`;
  fields.meanDepth.textContent = `${formatNumber(result.mean_depth_m, 3)} m`;

  featuresBody.replaceChildren();
  Object.entries(result.model_features || {}).forEach(([key, value]) => {
    const row = document.createElement("tr");
    const name = document.createElement("td");
    const val = document.createElement("td");
    name.textContent = labelFeature(key);
    val.textContent = formatNumber(value, 6);
    row.append(name, val);
    featuresBody.append(row);
  });

  results.hidden = false;
  details.hidden = false;
  renderVisuals(result);
}

function labelFeature(key) {
  const labels = {
    mask_count: "Cantidad de racimos detectados",
    mask_area_m2_sum: "Área total de racimos (m2)",
    mask_area_m2_p75: "Área de racimo grande típica (m2)",
    mask_area_m2_std: "Variación del tamaño de racimos (m2)",
    liters_totales: "Volumen estimado (L)",
  };
  return labels[key] || key;
}

function addGalleryImage(title, path, jobId, kind) {
  if (!path) return;
  const card = document.createElement("figure");
  const img = document.createElement("img");
  const caption = document.createElement("figcaption");
  card.dataset.kind = kind;
  img.src = `/jobs/${jobId}/${path}`;
  img.alt = title;
  img.loading = "lazy";
  caption.textContent = title;
  card.addEventListener("click", () => openImageDialog(img.src, title));
  card.append(img, caption);
  gallery.append(card);
}

function renderVisuals(result) {
  gallery.replaceChildren();
  const jobId = result.job_id;
  const visualData = result.visuals || {};

  addGalleryImage("Imagen RGB del video", visualData.rgb_preview, jobId, "rgb");
  (visualData.tracking_overlays || []).forEach((path, index) => {
    addGalleryImage(`Racimos detectados ${index + 1}`, path, jobId, "tracking");
  });
  addGalleryImage("Racimos elegidos para medir", visualData.representative_masks, jobId, "masks");

  visualTabs.forEach((button) => button.classList.toggle("active", button.dataset.visualTab === "all"));
  filterGallery("all");
  visuals.hidden = gallery.children.length === 0;
}

function filterGallery(kind) {
  [...gallery.children].forEach((card) => {
    card.hidden = kind !== "all" && card.dataset.kind !== kind;
  });
}

checkHealth();
loadDemoFiles();
loadRealWeights();

async function loadDemoFiles() {
  try {
    const response = await fetch("/demo-files", { cache: "no-store" });
    if (!response.ok) throw new Error("No se pudieron listar los archivos de demo");
    const payload = await response.json();
    demoFileSelect.replaceChildren();
    if (!payload.files || payload.files.length === 0) {
      const option = document.createElement("option");
      option.value = "";
      option.textContent = "No se encontraron videos .bag en ~/demo_data";
      demoFileSelect.append(option);
      return;
    }
    payload.files.forEach((file) => {
      const option = document.createElement("option");
      option.value = file.name;
      option.textContent = `${file.name} (${formatBytes(file.size_bytes)})`;
      demoFileSelect.append(option);
    });
  } catch {
    demoFileSelect.replaceChildren();
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "No se pudieron cargar los videos .bag de la VM";
    demoFileSelect.append(option);
  }
}

async function loadRealWeights() {
  try {
    const response = await fetch("/real-weights", { cache: "no-store" });
    if (!response.ok) throw new Error("No se pudieron cargar los pesos reales");
    const payload = await response.json();
    referenceWeightSelect.replaceChildren();

    const emptyOption = document.createElement("option");
    emptyOption.value = "";
    emptyOption.textContent = "Sin comparación con peso real";
    referenceWeightSelect.append(emptyOption);

    (payload.entries || []).forEach((entry) => {
      const option = document.createElement("option");
      option.value = entry.key;
      const paired = entry.paired_id ? ` / ${entry.paired_id}` : "";
      option.textContent = `${entry.primary_id}${paired} - ${formatNumber(entry.real_weight_kg, 3)} kg`;
      referenceWeightSelect.append(option);
    });
  } catch {
    referenceWeightSelect.replaceChildren();
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "No se pudieron cargar los pesos reales";
    referenceWeightSelect.append(option);
  }
}

function formatBytes(bytes) {
  if (!Number.isFinite(Number(bytes))) return "";
  const units = ["B", "KB", "MB", "GB"];
  let value = Number(bytes);
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value.toFixed(unit === 0 ? 0 : 1)} ${units[unit]}`;
}

function openImageDialog(src, title) {
  dialogImage.src = src;
  dialogImage.alt = title;
  dialogCaption.textContent = title;
  if (typeof imageDialog.showModal === "function") {
    imageDialog.showModal();
  } else {
    window.open(src, "_blank", "noreferrer");
  }
}

dialogClose.addEventListener("click", () => imageDialog.close());
imageDialog.addEventListener("click", (event) => {
  if (event.target === imageDialog) {
    imageDialog.close();
  }
});
