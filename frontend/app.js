const form = document.querySelector("#uploadForm");
const fileInput = document.querySelector("#bagFile");
const fileLabel = document.querySelector("#fileLabel");
const maxFramesInput = document.querySelector("#maxFrames");
const submitButton = document.querySelector("#submitButton");
const healthStatus = document.querySelector("#healthStatus");
const runState = document.querySelector("#runState");
const statusText = document.querySelector("#statusText");
const progressBar = document.querySelector("#progressBar");
const results = document.querySelector("#results");
const details = document.querySelector("#details");
const visuals = document.querySelector("#visuals");
const gallery = document.querySelector("#gallery");
const featuresBody = document.querySelector("#featuresBody");

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

async function checkHealth() {
  try {
    const response = await fetch("/health");
    if (!response.ok) throw new Error("API unavailable");
    healthStatus.textContent = "API Ready";
    healthStatus.className = "status ok";
  } catch {
    healthStatus.textContent = "API Offline";
    healthStatus.className = "status error";
  }
}

fileInput.addEventListener("change", () => {
  const file = fileInput.files[0];
  fileLabel.textContent = file ? file.name : "Select a .bag recording";
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
  setProgress(12, "Uploading", "Uploading the RealSense bag to the VM.");

  const timer = window.setTimeout(() => {
    setProgress(54, "Running", "Pipeline is extracting frames, segmenting clusters, tracking IDs, and computing depth metrics.");
  }, 1400);

  try {
    const response = await fetch("/upload", {
      method: "POST",
      body: data,
    });
    window.clearTimeout(timer);

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

    setProgress(100, "Complete", "Prediction and intermediate outputs are ready.");
    renderResult(await response.json());
  } catch (error) {
    window.clearTimeout(timer);
    setProgress(0, "Error", error.message || "Pipeline failed.");
  } finally {
    submitButton.disabled = false;
  }
});

function renderResult(result) {
  fields.predictedWeight.textContent = formatNumber(result.predicted_weight, 2);
  fields.detectedClusters.textContent = formatNumber(result.detected_clusters, 0);
  fields.totalVolume.textContent = `${formatNumber(result.total_estimated_volume_cm3, 1)} cm3`;
  fields.meanDepth.textContent = `${formatNumber(result.mean_depth_m, 3)} m`;

  featuresBody.replaceChildren();
  Object.entries(result.model_features || {}).forEach(([key, value]) => {
    const row = document.createElement("tr");
    const name = document.createElement("td");
    const val = document.createElement("td");
    name.textContent = key;
    val.textContent = formatNumber(value, 6);
    row.append(name, val);
    featuresBody.append(row);
  });

  results.hidden = false;
  details.hidden = false;
  renderVisuals(result);
}

function addGalleryImage(title, path, jobId) {
  if (!path) return;
  const card = document.createElement("figure");
  const img = document.createElement("img");
  const caption = document.createElement("figcaption");
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

  addGalleryImage("RGB preview", visualData.rgb_preview, jobId);
  (visualData.tracking_overlays || []).forEach((path, index) => {
    addGalleryImage(`Segmentation and tracking ${index + 1}`, path, jobId);
  });
  addGalleryImage("Representative masks", visualData.representative_masks, jobId);

  visuals.hidden = gallery.children.length === 0;
}

checkHealth();
