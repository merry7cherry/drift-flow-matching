"use strict";

(() => {
  const byId = id => document.getElementById(id);
  const setText = (id, text) => { byId(id).textContent = text; };
  const setImage = (id, source, alt) => {
    byId(id).src = source;
    byId(id).alt = alt;
  };
  const setPressed = (buttons, selected) => {
    buttons.forEach(button => button.setAttribute("aria-pressed", String(button === selected)));
  };

  // Read the generated table: results.json is the source for every displayed metric.
  const ffhqMetrics = Object.fromEntries([...document.querySelectorAll("#image-results .ours")].map(row => [
    row.dataset.nfe,
    { fid: row.querySelector(".ffhq-fid").textContent, emd: row.querySelector(".ffhq-emd").textContent },
  ]));
  const imageButtons = [...document.querySelectorAll("[data-image-nfe]")];
  imageButtons.forEach(button => button.addEventListener("click", () => {
    const nfe = button.dataset.imageNfe;
    setPressed(imageButtons, button);
    setImage("ffhq-image", `assets/ffhq-nfe-${nfe}.jpg`, `FFHQ class-conditional samples from the paper at ${nfe} NFE`);
    setText("image-caption", `FFHQ · ${nfe} NFE`);
    const metric = ffhqMetrics[nfe];
    setText("ffhq-metric", metric
      ? `FFHQ · ${nfe} NFE · FID ${metric.fid} · EMD ${metric.emd}`
      : "10 NFE · Qualitative samples; FID and EMD are not reported at this NFE.");
  }));

  const datasetSelect = byId("dataset");
  const dfmNfeSelect = byId("dfm-nfe");
  const referenceMethodSelect = byId("reference-method");
  const referenceNfeSelect = byId("reference-nfe");
  const modeButtons = [...document.querySelectorAll("[data-trajectory-mode]")];
  const datasets = { moon: "Two moons", checkerboard_grid: "Checkerboard", letter_f: "Letter F", letter_m: "Letter M" };
  const methods = {
    mean_flow: { label: "MeanFlow", nfe: ["1", "20"], asset: nfe => `mean_flow_steps_${nfe}` },
    flow_matching: { label: "Flow Matching", nfe: ["50"], asset: () => "flow_matching" },
    drift: { label: "Drift Model", nfe: ["1"], asset: () => "drift" },
  };
  const rememberedNfe = { mean_flow: "20", flow_matching: "50", drift: "1" };
  let mode = "steps";

  function populateReferenceNfe() {
    const key = referenceMethodSelect.value;
    referenceNfeSelect.replaceChildren(...methods[key].nfe.map(nfe => {
      const option = document.createElement("option");
      option.value = nfe;
      option.textContent = `${nfe} NFE`;
      return option;
    }));
    referenceNfeSelect.value = rememberedNfe[key];
    referenceNfeSelect.disabled = methods[key].nfe.length === 1;
  }
  function updateTrajectories() {
    const sceneKey = datasetSelect.value;
    const scene = datasets[sceneKey];
    const stepComparison = mode === "steps";
    const method = stepComparison
      ? { label: "DFM", asset: nfe => `drift_flow_matching_steps_${nfe}` }
      : methods[referenceMethodSelect.value];
    const referenceNfe = stepComparison ? "1" : referenceNfeSelect.value;
    const dfmNfe = stepComparison ? "20" : dfmNfeSelect.value;
    const referenceLabel = `${method.label} · ${referenceNfe} NFE`;
    byId("method-controls").hidden = stepComparison;
    byId("dfm-control").hidden = stepComparison;
    setText("reference-title", referenceLabel);
    setText("dfm-title", `DFM · ${dfmNfe} NFE`);
    setImage("reference-image", `assets/${sceneKey}-${method.asset(referenceNfe)}.jpg`, `${referenceLabel} generation trajectories toward ${scene}`);
    setImage("dfm-image", `assets/${sceneKey}-drift_flow_matching_steps_${dfmNfe}.jpg`, `DFM · ${dfmNfe} NFE generation trajectories toward ${scene}`);
    setImage("target-image", `assets/${sceneKey}-ground_truth.jpg`, `${scene} Ground Truth from the paper`);
    setText("target-caption", `${scene} · Target samples and reference interpolations from the paper.`);
    setText("mode-description", stepComparison ? "Compare DFM at 1 and 20 inference steps." : "Choose a reference method and each method’s inference budget.");
    setText("comparison-budget", stepComparison
      ? "DFM · 1 NFE → 20 NFE"
      : `${referenceNfe === dfmNfe ? "Equal" : "Different"} NFE · ${method.label}: ${referenceNfe} · DFM: ${dfmNfe}`);
    setText("trajectory-caption", `${scene} · ${referenceLabel} compared with DFM · ${dfmNfe} NFE.`);
  }
  modeButtons.forEach(button => button.addEventListener("click", () => {
    mode = button.dataset.trajectoryMode;
    setPressed(modeButtons, button);
    updateTrajectories();
  }));
  datasetSelect.addEventListener("change", updateTrajectories);
  dfmNfeSelect.addEventListener("change", updateTrajectories);
  referenceMethodSelect.addEventListener("change", () => { populateReferenceNfe(); updateTrajectories(); });
  referenceNfeSelect.addEventListener("change", () => {
    rememberedNfe[referenceMethodSelect.value] = referenceNfeSelect.value;
    updateTrajectories();
  });
  populateReferenceNfe();
  updateTrajectories();

  const trainingSelect = byId("training-pair");
  const trainingPairs = { early: ["0.05", "0.30"], wide: ["0.20", "0.80"], middle: ["0.35", "0.65"], late: ["0.70", "0.95"] };
  function updateTrainingPair() {
    const pair = trainingSelect.value;
    const [t, r] = trainingPairs[pair];
    setImage("training-image", `assets/group-${pair}.jpg`, `Grouped training drift for time pair t = ${t}, r = ${r}`);
    setText("training-caption", `(t, r) = (${t}, ${r}) · Arrows show the training drift V. Inference uses the learned transport u.`);
  }
  trainingSelect.addEventListener("change", updateTrainingPair);
  updateTrainingPair();

  byId("copy-citation").addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(byId("bibtex").textContent);
      setText("copy-status", "BibTeX copied.");
    } catch {
      setText("copy-status", "Select and copy the BibTeX text above.");
    }
  });
})();
