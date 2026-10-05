"use strict";

(() => {
  const byId = id => document.getElementById(id);
  const setText = (id, text) => {
    const element = byId(id);
    if (element) element.textContent = text;
  };
  const setImage = (id, source, alt) => {
    const element = byId(id);
    if (!element) return;
    element.src = source;
    element.alt = alt;
  };
  const setPressed = (buttons, selected) => {
    buttons.forEach(button => button.setAttribute("aria-pressed", String(button === selected)));
  };

  // The FFHQ photographs and the table contain different sets of reported NFE.
  const imageButtons = [...document.querySelectorAll("[data-image-nfe]")];
  const ffhqMetrics = {
    "1": { fid: "116.2", emd: "225.5" },
    "2": { fid: "80.4", emd: "215.7" },
    "5": { fid: "75.9", emd: "196.2" },
  };
  function updateImageGeneration(button) {
    const nfe = button.dataset.imageNfe;
    if (!["1", "2", "5", "10"].includes(nfe)) return;
    setPressed(imageButtons, button);
    setImage("ffhq-image", `assets/ffhq-nfe-${nfe}.jpg`,
      `Archived FFHQ class-conditional generated samples at ${nfe} NFE`);
    setText("image-caption", `FFHQ · ${nfe} NFE`);
    const metric = ffhqMetrics[nfe];
    setText("ffhq-metric", metric
      ? `Paper-reported FFHQ results at ${nfe} NFE: FID ${metric.fid} · EMD ${metric.emd}.`
      : "10 NFE: generated examples are available; quantitative FID and EMD at this NFE are not reported in the paper’s comparison table.");
  }
  imageButtons.forEach(button => button.addEventListener("click", () => updateImageGeneration(button)));
  const initialImageButton = imageButtons.find(button => button.getAttribute("aria-pressed") === "true") || imageButtons[0];
  if (initialImageButton) updateImageGeneration(initialImageButton);

  const datasetSelect = byId("dataset");
  const dfmNfeSelect = byId("dfm-nfe");
  const referenceMethodSelect = byId("reference-method");
  const referenceNfeSelect = byId("reference-nfe");
  const modeButtons = [...document.querySelectorAll("[data-trajectory-mode]")];
  const presetButtons = [...document.querySelectorAll("[data-comparison-preset]")];
  const datasets = {
    moon: "Two moons",
    checkerboard_grid: "Checkerboard",
    letter_f: "Letter F",
    letter_m: "Letter M",
  };
  const methods = {
    drift_flow_matching: { label: "DFM", nfe: ["1", "20"], asset: nfe => `drift_flow_matching_steps_${nfe}` },
    mean_flow: { label: "MeanFlow", nfe: ["1", "20"], asset: nfe => `mean_flow_steps_${nfe}` },
    flow_matching: { label: "Flow Matching", nfe: ["50"], asset: () => "flow_matching" },
    drift: { label: "Drift Model", nfe: ["1"], asset: () => "drift" },
  };
  const presets = {
    one: { dfm: "1", method: "drift", reference: "1" },
    twenty: { dfm: "20", method: "mean_flow", reference: "20" },
    flow: { dfm: "20", method: "flow_matching", reference: "50" },
  };
  const rememberedNfe = { drift_flow_matching: "1", mean_flow: "20", flow_matching: "50", drift: "1" };
  let mode = "steps";

  function selectedReferenceMethod() {
    const value = referenceMethodSelect?.value;
    return Object.hasOwn(methods, value) ? value : "drift_flow_matching";
  }
  function populateReferenceNfe(preferred) {
    if (!referenceNfeSelect) return;
    const method = selectedReferenceMethod();
    const available = methods[method].nfe;
    const candidate = preferred || rememberedNfe[method];
    const selected = available.includes(candidate) ? candidate : available[0];
    const options = available.map(nfe => {
      const option = document.createElement("option");
      option.value = nfe;
      option.textContent = `${nfe} NFE`;
      return option;
    });
    referenceNfeSelect.replaceChildren(...options);
    referenceNfeSelect.value = selected;
    referenceNfeSelect.disabled = available.length === 1;
    rememberedNfe[method] = selected;
  }
  function updateTrajectories() {
    if (!datasetSelect || !dfmNfeSelect) return;
    const sceneKey = Object.hasOwn(datasets, datasetSelect.value) ? datasetSelect.value : "moon";
    const scene = datasets[sceneKey];
    const dfmNfe = methods.drift_flow_matching.nfe.includes(dfmNfeSelect.value) ? dfmNfeSelect.value : "20";
    dfmNfeSelect.value = dfmNfe;
    const methodKey = mode === "steps" ? "drift_flow_matching" : selectedReferenceMethod();
    const referenceMethod = methods[methodKey];
    const requestedNfe = referenceNfeSelect?.value;
    const referenceNfe = mode === "steps" ? "1"
      : referenceMethod.nfe.includes(requestedNfe) ? requestedNfe : referenceMethod.nfe[0];
    const referenceLabel = `${referenceMethod.label} · ${referenceNfe} NFE`;
    const dfmLabel = `DFM · ${dfmNfe} NFE`;

    const methodControls = byId("method-controls");
    if (methodControls) methodControls.hidden = mode === "steps";
    modeButtons.forEach(button => button.setAttribute("aria-pressed", String(button.dataset.trajectoryMode === mode)));
    presetButtons.forEach(button => {
      const preset = presets[button.dataset.comparisonPreset];
      const selected = mode === "methods" && preset && preset.dfm === dfmNfe
        && preset.method === methodKey && preset.reference === referenceNfe;
      button.setAttribute("aria-pressed", String(Boolean(selected)));
    });

    setText("reference-title", referenceLabel);
    setText("dfm-title", dfmLabel);
    setImage("reference-image", `assets/${sceneKey}-${referenceMethod.asset(referenceNfe)}.jpg`,
      `${referenceMethod.label} generation trajectories at ${referenceNfe} NFE toward ${scene}`);
    setImage("dfm-image", `assets/${sceneKey}-drift_flow_matching_steps_${dfmNfe}.jpg`,
      `DFM generation trajectories at ${dfmNfe} NFE toward ${scene}`);
    setImage("target-image", `assets/${sceneKey}-ground_truth.jpg`,
      `${scene} target distribution and reference interpolation from the source`);
    setText("target-caption", `${scene} target distribution. Red endpoints show target samples; connecting lines are reference interpolations, not a unique learned generation path.`);

    if (mode === "steps") {
      setText("mode-description", "Compare DFM at different inference budgets using the paper’s recorded 1- and 20-step generation trajectories.");
      setText("comparison-budget", dfmNfe === "1"
        ? "Both panels show DFM at 1 NFE. Choose 20 NFE to compare inference budgets."
        : "Same method · DFM at 1 NFE and 20 NFE.");
    } else {
      setText("mode-description", "Compare the paper’s recorded methods. Each method offers only the NFE values available in the archived figures.");
      setText("comparison-budget", referenceNfe === dfmNfe
        ? `Equal NFE · ${referenceMethod.label} and DFM each use ${dfmNfe} model evaluation${dfmNfe === "1" ? "" : "s"} per sample.`
        : `Different NFE · ${referenceMethod.label} uses ${referenceNfe}; DFM uses ${dfmNfe} model evaluation${dfmNfe === "1" ? "" : "s"} per sample.`);
    }
    setText("trajectory-caption", `${scene} · ${referenceLabel} compared with ${dfmLabel}. Controls switch archived paper figures; they do not run model inference. Historical checkpoint and paired-seed provenance have not been recovered.`);
  }
  function setMode(value) {
    if (value !== "steps" && value !== "methods") return;
    mode = value;
    updateTrajectories();
  }
  modeButtons.forEach(button => button.addEventListener("click", () => setMode(button.dataset.trajectoryMode)));
  datasetSelect?.addEventListener("change", updateTrajectories);
  dfmNfeSelect?.addEventListener("change", updateTrajectories);
  referenceMethodSelect?.addEventListener("change", () => {
    populateReferenceNfe();
    updateTrajectories();
  });
  referenceNfeSelect?.addEventListener("change", () => {
    rememberedNfe[selectedReferenceMethod()] = referenceNfeSelect.value;
    updateTrajectories();
  });
  presetButtons.forEach(button => button.addEventListener("click", () => {
    const preset = presets[button.dataset.comparisonPreset];
    if (!preset || !dfmNfeSelect || !referenceMethodSelect) return;
    dfmNfeSelect.value = preset.dfm;
    referenceMethodSelect.value = preset.method;
    populateReferenceNfe(preset.reference);
    setMode("methods");
  }));
  populateReferenceNfe(referenceNfeSelect?.value);
  updateTrajectories();

  const trainingSelect = byId("training-pair");
  const trainingPairs = {
    early: ["0.05", "0.30"],
    wide: ["0.20", "0.80"],
    middle: ["0.35", "0.65"],
    late: ["0.70", "0.95"],
  };
  function updateTrainingPair() {
    if (!trainingSelect) return;
    const pair = trainingSelect.value;
    if (!Object.hasOwn(trainingPairs, pair)) return;
    const [t, r] = trainingPairs[pair];
    setImage("training-image", `assets/group-${pair}.jpg`,
      `Grouped training drift for time pair t = ${t}, r = ${r}`);
    setText("training-caption", `Training time pair (t, r) = (${t}, ${r}). The drift V provides the distribution-level training correction; it is not the model’s inference velocity u.`);
  }
  trainingSelect?.addEventListener("change", updateTrainingPair);
  updateTrainingPair();

  byId("copy-citation")?.addEventListener("click", async () => {
    const citation = byId("bibtex");
    if (!citation) return;
    try {
      await navigator.clipboard.writeText(citation.textContent);
      setText("copy-status", "BibTeX copied.");
    } catch {
      const selection = window.getSelection();
      if (selection) {
        const range = document.createRange();
        range.selectNodeContents(citation);
        selection.removeAllRanges();
        selection.addRange(range);
        setText("copy-status", "Citation selected. Use your browser’s Copy command.");
      } else {
        setText("copy-status", "Copy the BibTeX text above.");
      }
    }
  });
})();
