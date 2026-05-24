let currentResumeId = null;

const RECOMMENDED_RESUME_FLOW = [
  "summary",
  "skills",
  "experience",
  "projects",
  "education",
  "courses",
  "languages",
];

const uploadBtn = document.getElementById("uploadBtn");
const analyzeBtn = document.getElementById("analyzeBtn");
const resumeFileInput = document.getElementById("resumeFile");
const uploadStatus = document.getElementById("uploadStatus");
const fileNameDisplay = document.getElementById("selectedFileName");
const uploadBox = document.querySelector(".upload-box");
const uploadButtonLabel = document.querySelector("label[for='resumeFile']");
const jobDescriptionSelect = document.getElementById("jobDescriptionSelect");
const resultSection = document.getElementById("resultSection");

let jobDescriptionOptions = [];

async function loadJobDescriptionTemplates() {
  try {
    const response = await fetch("/job_descriptions", { method: "GET" });
    const payload = await response.json();
    if (!response.ok) {
      return;
    }

    jobDescriptionOptions = Array.isArray(payload.items) ? payload.items : [];
    jobDescriptionOptions.forEach((item) => {
      const option = document.createElement("option");
      option.value = item.id;
      option.textContent = `${item.title} (${item.domain})`;
      jobDescriptionSelect.appendChild(option);
    });
  } catch (error) {
    // Keep manual textarea flow working even if template endpoint is unavailable.
  }
}

async function uploadSelectedResume() {
  const file = resumeFileInput.files[0];
  if (!file) {
    uploadStatus.textContent = "Select a file first.";
    return null;
  }

  uploadStatus.textContent = "Uploading...";

  const formData = new FormData();
  formData.append("resume", file);

  const response = await fetch("/upload_resume", {
    method: "POST",
    body: formData,
  });
  const data = await response.json();

  if (!response.ok) {
    uploadStatus.textContent = data.error || "Upload failed.";
    return null;
  }

  currentResumeId = data.resume_id;
  uploadStatus.textContent = `Uploaded: ${data.file_name}`;
  return currentResumeId;
}

function setList(id, values) {
  const node = document.getElementById(id);
  node.innerHTML = "";

  if (!values || values.length === 0) {
    const li = document.createElement("li");
    li.textContent = "No items";
    node.appendChild(li);
    return;
  }

  values.forEach((value) => {
    const li = document.createElement("li");
    li.textContent = value;
    node.appendChild(li);
  });
}

function setProgress(id, value) {
  const bar = document.getElementById(id);
  const safeValue = Math.max(0, Math.min(100, value));
  bar.style.width = `${safeValue}%`;
}

function setPredictionList(id, values, showProbability = true) {
  const node = document.getElementById(id);
  node.innerHTML = "";
  if (!values || values.length === 0) {
    const li = document.createElement("li");
    li.textContent = "No prediction data";
    node.appendChild(li);
    return;
  }

  values.forEach((item) => {
    const li = document.createElement("li");
    if (showProbability) {
      const probability = confidenceToPercent(item.probability);
      li.textContent = `${item.label} (${probability}%)`;
    } else {
      li.textContent = `${item.label}`;
    }
    node.appendChild(li);
  });
}

function setJobRecommendations(id, values) {
  const node = document.getElementById(id);
  node.innerHTML = "";

  if (!values || values.length === 0) {
    const li = document.createElement("li");
    li.textContent = "No job matches found";
    node.appendChild(li);
    return;
  }

  values.forEach((item) => {
    const li = document.createElement("li");
    const title = document.createElement("strong");
    title.textContent = item.title || "Unknown role";

    const meta = document.createElement("span");
    const score = confidenceToPercent(item.confidence ?? item.score ?? 0);
    meta.textContent = `${item.domain || "Unknown domain"} • ${score}% match`;

    const reason = document.createElement("span");
    reason.className = "li-meta";
    reason.textContent = item.reason || "Role fit based on resume signals and skill overlap.";

    li.appendChild(title);
    li.appendChild(meta);
    li.appendChild(reason);
    node.appendChild(li);
  });
}

function toPercent(value) {
  const asNumber = Number(value || 0);
  return Math.max(0, Math.min(100, Math.round(asNumber)));
}

function confidenceToPercent(value) {
  const asNumber = Number(value || 0);
  return Math.max(0, Math.min(100, Math.round(asNumber * 100)));
}

function formatFlowList(items) {
  if (!Array.isArray(items) || items.length === 0) {
    return "Not found";
  }
  return items.join(" -> ");
}

jobDescriptionSelect.addEventListener("change", () => {
  const selectedId = jobDescriptionSelect.value;
  const selected = jobDescriptionOptions.find((item) => item.id === selectedId);
  if (selected) {
    uploadStatus.textContent = `Selected JD: ${selected.title}`;
  }
});

resumeFileInput.addEventListener("change", () => {
  currentResumeId = null;
  const file = resumeFileInput.files[0];
  if (file) {
    uploadStatus.textContent = "Resume selected. You can upload or click Analyze directly.";
    if (fileNameDisplay) {
      fileNameDisplay.textContent = file.name;
    }
    if (uploadButtonLabel) {
      uploadButtonLabel.textContent = "Replace file";
    }
    if (uploadBox) {
      uploadBox.classList.add("has-file");
    }
  } else {
    uploadStatus.textContent = "Select a file first.";
    if (fileNameDisplay) {
      fileNameDisplay.textContent = "No file selected.";
    }
    if (uploadButtonLabel) {
      uploadButtonLabel.textContent = "Select file";
    }
    if (uploadBox) {
      uploadBox.classList.remove("has-file");
    }
  }
});

uploadBtn.addEventListener("click", async () => {
  try {
    await uploadSelectedResume();
  } catch (error) {
    uploadStatus.textContent = "Upload error. Try again.";
  }
});

analyzeBtn.addEventListener("click", async () => {
  const selectedId = jobDescriptionSelect.value;
  const selectedTemplate = jobDescriptionOptions.find((item) => item.id === selectedId);
  if (!selectedTemplate || !selectedTemplate.description) {
    alert("Please select a job description from the dropdown.");
    return;
  }
  const jd = selectedTemplate.description;

  if (!currentResumeId) {
    try {
      const uploadedId = await uploadSelectedResume();
      if (!uploadedId) {
        alert("Please select a valid resume file before analyzing.");
        return;
      }
    } catch (error) {
      alert("Resume upload failed. Try again.");
      return;
    }
  }

  analyzeBtn.disabled = true;
  analyzeBtn.textContent = "Analyzing...";

  try {
    const response = await fetch("/analyze", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        resume_id: currentResumeId,
        job_description: jd,
      }),
    });

    const result = await response.json();
    if (!response.ok) {
      alert(result.details ? `${result.error}\n${result.details}` : (result.error || "Analysis failed."));
      return;
    }

    resultSection.classList.remove("hidden");
    const overallPercent = toPercent(result.score);
    const similarityPercent = confidenceToPercent(result.similarity_score);
    const skillPercent = confidenceToPercent(result.skill_match_ratio);
    const jdFitPercent = toPercent(result.legacy_overall_score);
    const flowScorePercent = toPercent(result?.quality_details?.format_flow_score);
    const flowFeedback = result?.quality_details?.flow_feedback || "No flow feedback available.";
    const sectionSequence = Array.isArray(result?.quality_details?.section_sequence)
      ? result.quality_details.section_sequence
      : [];

    const uncertaintyMessages = [];
    // Intentionally omit low-confidence text from the banner.

    document.getElementById("overallValue").textContent = `${overallPercent}%`;
    document.getElementById("similarityValue").textContent = `${similarityPercent}%`;
    document.getElementById("skillValue").textContent = `${skillPercent}%`;
    document.getElementById("jdFitValue").textContent = `${jdFitPercent}%`;

    setProgress("overallBar", overallPercent);
    setProgress("similarityBar", similarityPercent);
    setProgress("skillBar", skillPercent);
    setProgress("jdFitBar", jdFitPercent);

    const uncertaintyBanner = document.getElementById("uncertaintyBanner");
    if (uncertaintyMessages.length > 0) {
      uncertaintyBanner.classList.remove("hidden");
      uncertaintyBanner.textContent = uncertaintyMessages.join(" ");
    } else {
      uncertaintyBanner.classList.add("hidden");
      uncertaintyBanner.textContent = "";
    }

    document.getElementById("flowScoreText").textContent = `Flow score: ${flowScorePercent}%`;
    document.getElementById("flowMeaningText").textContent =
      "Flow in a resume means the order of sections from top to bottom.";
    document.getElementById("flowCurrentText").textContent =
      `Current flow: ${formatFlowList(sectionSequence)}`;
    document.getElementById("flowRecommendedText").textContent =
      `Recommended flow: ${RECOMMENDED_RESUME_FLOW.join(" -> ")}`;
    document.getElementById("flowFeedbackText").textContent = flowFeedback;

    setPredictionList("top2Quality", result?.top2_predictions?.quality || []);
    setPredictionList("top2Domain", result?.top2_predictions?.domain || [], false);
    setJobRecommendations("jobRecommendations", result?.job_recommendations || []);

    const bestJob = result?.best_suitable_job || {};
    const bestJobConfidence = confidenceToPercent(bestJob.confidence ?? result?.confidence?.recommendation ?? 0);
    document.getElementById("bestJobTitle").textContent = bestJob.title || "Unknown role";
    const bestJobScore = confidenceToPercent(bestJob.score ?? 0);
    document.getElementById("bestJobMeta").textContent = `${bestJob.domain || "Unknown domain"} • Score ${bestJobScore}% • Confidence ${bestJobConfidence}%`;
    document.getElementById("bestJobReason").textContent = bestJob.reason || "This role is the closest overall fit for the uploaded resume.";
    document.getElementById("bestJobConfidence").textContent = `${bestJobConfidence}%`;
    setProgress("bestJobBar", bestJobConfidence);

    document.getElementById(
      "categoryText"
    ).textContent = `Predicted domain: ${result.predicted_category} • Model confidence ${confidenceToPercent(result?.confidence?.domain)}%`;

    setList("strengths", result.strengths);
    setList("weaknesses", result.weaknesses);
    setList("suggestions", result.suggestions);
  } catch (error) {
    alert("Unexpected error while analyzing the resume.");
  } finally {
    analyzeBtn.disabled = false;
    analyzeBtn.textContent = "Analyze";
  }
});

loadJobDescriptionTemplates();
