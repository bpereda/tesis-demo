const form = document.querySelector("#uploadForm");
const fileInput = document.querySelector("#bagFile");
const fileLabel = document.querySelector("#fileLabel");
const maxFramesInput = document.querySelector("#maxFrames");
const modeButtons = [...document.querySelectorAll(".mode-button")];
const submitButton = document.querySelector("#submitButton");
const healthStatus = document.querySelector("#healthStatus");
const runState = document.querySelector("#runState");
const statusText = document.querySelector("#statusText");
const progressBar = document.querySelector("#progressBar");
const stepItems = [...document.querySelectorAll("[data-step]")];
const results = document.querySelector("#results");
const details = document.querySelector("#details");
const visuals = document.querySelector("#visuals");
const gallery = document.querySelector("#gallery");
const featuresBody = document.querySelector("#featuresBody");
const visualTabs = [...document.querySelectorAll("[data-visual-tab]")];

const fields = {
  predictedWeight: document.querySelector("#predictedWeight"),
  detectedClusters: document.querySelector("#detectedClusters"),
  totalVolume: document.querySelector("#totalVolume"),
  meanDepth: document.querySelector("#meanDepth"),
};

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
    if (!response.ok) throw new Error("API unavailable");
    healthStatus.textContent = "System Ready";
    healthStatus.className = "status ok";
  } catch {
    healthStatus.textContent = "API Offline";
    healthStatus.className = "status error";
  }
}

fileInput.addEventListener("change", () => {
  const file = fileInput.files[0];
  fileLabel.textContent = file ? file.name : "Select a `.bag` file";
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
  const file = fileInput.files[0];
  if (!file) return;

  const data = new FormData();
  data.append("file", file);
  data.append("max_frames", maxFramesInput.value || "-1");

  results.hidden = true;
  details.hidden = true;
  visuals.hidden = true;
  submitButton.disabled = true;
  resetSteps();
  setActiveStep("upload");
  setProgress(12, "Uploading", "Uploading the RealSense bag to the GPU VM.");

  const timers = [
    window.setTimeout(() => {
      setActiveStep("extract");
      setProgress(28, "Extracting", "Reading RGB frames and aligning depth data from the RealSense recording.");
    }, 1000),
    window.setTimeout(() => {
      setActiveStep("segment");
      setProgress(46, "Segmenting", "Running the segmentation model to isolate visible grape clusters.");
    }, 4500),
    window.setTimeout(() => {
      setActiveStep("track");
      setProgress(62, "Tracking", "Assigning stable IDs so repeated grape-cluster detections are counted consistently.");
    }, 9000),
    window.setTimeout(() => {
      setActiveStep("metrics");
      setProgress(78, "Measuring", "Combining representative masks with depth to compute area and volume features.");
    }, 13000),
    window.setTimeout(() => {
      setActiveStep("predict");
      setProgress(90, "Predicting", "Passing aggregated features through the trained yield model.");
    }, 17000),
  ];

  try {
    const response = await fetch("/upload", {
      method: "POST",
      body: data,
    });
    timers.forEach((timer) => window.clearTimeout(timer));

    if (!response.ok) {
      let message = `Request failed with ${response.status}`;
      try {
        const payload = await response.json();
        message = payload.detail || message;
      } catch {
        // Keep the HTTP status fallback.
      }
      throw new Error(message);
    }

    setProgress(100, "Complete", "Prediction and visual evidence are ready.");
    completeSteps();
    renderResult(await response.json());
  } catch (error) {
    timers.forEach((timer) => window.clearTimeout(timer));
    setProgress(0, "Error", error.message || "Pipeline failed.");
  } finally {
    submitButton.disabled = false;
  }
});

function renderResult(result) {
  fields.predictedWeight.textContent = formatNumber(result.predicted_weight, 2);
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
    mask_count: "Cluster mask count",
    mask_area_m2_sum: "Aggregate mask area (m2)",
    mask_area_m2_p75: "Mask area p75 (m2)",
    mask_area_m2_std: "Mask area std (m2)",
    liters_totales: "Total volume estimate (L)",
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
  card.append(img, caption);
  gallery.append(card);
}

function renderVisuals(result) {
  gallery.replaceChildren();
  const jobId = result.job_id;
  const visualData = result.visuals || {};

  addGalleryImage("RGB preview", visualData.rgb_preview, jobId, "rgb");
  (visualData.tracking_overlays || []).forEach((path, index) => {
    addGalleryImage(`Segmentation and tracking ${index + 1}`, path, jobId, "tracking");
  });
  addGalleryImage("Representative masks", visualData.representative_masks, jobId, "masks");

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
