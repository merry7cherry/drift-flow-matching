"use strict";
const image = document.querySelector("#ffhq-image");
document.querySelectorAll("[data-image-nfe]").forEach(button => {
  button.addEventListener("click", () => {
    const nfe = button.dataset.imageNfe;
    document.querySelectorAll("[data-image-nfe]").forEach(other => other.setAttribute("aria-pressed", String(other === button)));
    image.src = `assets/ffhq-nfe-${nfe}.jpg`;
    image.alt = `FFHQ class-conditional generated samples at ${nfe} inference steps`;
    document.querySelector("#image-caption").textContent = `FFHQ · ${nfe} NFE`;
  });
});
const dataset = document.querySelector("#dataset");
const reference = document.querySelector("#reference");
function updateTrajectories() {
  const scene = dataset.selectedOptions[0].textContent;
  const label = reference.selectedOptions[0].textContent;
  document.querySelector("#reference-title").textContent = label;
  const left = document.querySelector("#reference-image");
  left.src = `assets/${dataset.value}-${reference.value}.jpg`;
  left.alt = `${label} trajectories toward ${scene}`;
  const right = document.querySelector("#dfm-image");
  right.src = `assets/${dataset.value}-drift_flow_matching_steps_20.jpg`;
  right.alt = `DFM at 20 NFE toward ${scene}`;
  document.querySelector("#trajectory-caption").textContent = `${scene} · ${label} compared with DFM at 20 NFE · archived paper visualization. These controls switch recorded figures; they do not run a model in your browser.`;
}
dataset.addEventListener("change", updateTrajectories);
reference.addEventListener("change", updateTrajectories);
document.querySelector("#copy-citation").addEventListener("click", async () => {
  const status = document.querySelector("#copy-status");
  try {
    await navigator.clipboard.writeText(document.querySelector("#bibtex").textContent);
    status.textContent = "BibTeX copied.";
  } catch {
    const range = document.createRange();
    range.selectNodeContents(document.querySelector("#bibtex"));
    const selection = window.getSelection();
    selection.removeAllRanges(); selection.addRange(range);
    status.textContent = "Citation selected. Use your browser’s Copy command.";
  }
});
