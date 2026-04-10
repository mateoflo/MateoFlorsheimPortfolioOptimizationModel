const form = document.getElementById("optimizer-form");
const assetTable = document.getElementById("asset-table");
const assetTemplate = document.getElementById("asset-row-template");
const addAssetButton = document.getElementById("add-asset-button");
const loadExampleButton = document.getElementById("load-example-button");
const statusPill = document.getElementById("status-pill");
const statusText = document.getElementById("status-text");
const resultsBody = document.querySelector("#results-table tbody");

const EXAMPLE_ASSETS = [
  { ticker: "SPY", max_weight: "45" },
  { ticker: "QQQ", max_weight: "35" },
  { ticker: "XLF", max_weight: "20" },
  { ticker: "XLV", max_weight: "20" },
];

function setStatus(text, tone = "ready") {
  statusPill.textContent = text;
  statusPill.classList.remove("loading", "error");
  if (tone === "loading") {
    statusPill.classList.add("loading");
  }
  if (tone === "error") {
    statusPill.classList.add("error");
  }
}

function createAssetRow(ticker = "", maxWeight = "25") {
  const fragment = assetTemplate.content.cloneNode(true);
  const row = fragment.querySelector(".asset-row");
  row.querySelector(".ticker-input").value = ticker;
  row.querySelector(".max-weight-input").value = maxWeight;
  row.querySelector(".remove-button").addEventListener("click", () => {
    row.remove();
    ensureAtLeastOneRow();
    syncMaxWeightMode();
  });
  return fragment;
}

function appendAssetRow(ticker = "", maxWeight = "25") {
  assetTable.appendChild(createAssetRow(ticker, maxWeight));
}

function assetRows() {
  return Array.from(assetTable.querySelectorAll(".asset-row"));
}

function ensureAtLeastOneRow() {
  if (assetRows().length === 0) {
    appendAssetRow();
  }
}

function collectPayload() {
  const formData = new FormData(form);
  const settings = Object.fromEntries(formData.entries());
  settings.auto_treasury_bill_yield = formData.has("auto_treasury_bill_yield");
  const assets = assetRows().map((row) => ({
    ticker: row.querySelector(".ticker-input").value.trim(),
    max_weight: row.querySelector(".max-weight-input").value.trim(),
  }));
  return { settings, assets };
}

function updateSummary(summary) {
  document.querySelectorAll("[data-summary-key]").forEach((node) => {
    const key = node.dataset.summaryKey;
    node.textContent = summary[key] || "--";
  });
}

function updateTable(assetRowsPayload) {
  if (!assetRowsPayload.length) {
    resultsBody.innerHTML = '<tr class="placeholder-row"><td colspan="8">No asset rows were returned.</td></tr>';
    return;
  }

  resultsBody.innerHTML = "";
  assetRowsPayload.forEach((row) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${row.ticker}</td>
      <td>${row.price}</td>
      <td>${row.expected_return}</td>
      <td>${row.volatility}</td>
      <td>${row.target_weight}</td>
      <td>${row.recommended_shares}</td>
      <td>${row.invested_dollars}</td>
      <td>${row.realized_weight}</td>
    `;
    resultsBody.appendChild(tr);
  });
}

function syncMaxWeightMode() {
  const mode = form.querySelector('[name="max_allocation_mode"]').value;
  const disabled = mode === "Auto";
  assetRows().forEach((row) => {
    row.querySelector(".max-weight-input").disabled = disabled;
  });
}

async function submitForm(event) {
  event.preventDefault();
  setStatus("Optimizing", "loading");
  statusText.textContent = "Submitting the portfolio to the Python research engine.";

  try {
    const response = await fetch("/api/optimize", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(collectPayload()),
    });
    const data = await response.json();
    if (!response.ok || !data.ok) {
      throw new Error(data.error || "Optimization failed.");
    }

    updateSummary(data.result.summary);
    updateTable(data.result.asset_rows);
    setStatus("Complete");
    statusText.textContent = "Optimization completed with the current browser inputs.";
  } catch (error) {
    setStatus("Error", "error");
    statusText.textContent = error.message;
    resultsBody.innerHTML = `<tr class="placeholder-row"><td colspan="8">${error.message}</td></tr>`;
  }
}

function loadExample() {
  assetRows().forEach((row) => row.remove());
  EXAMPLE_ASSETS.forEach((asset) => appendAssetRow(asset.ticker, asset.max_weight));
  syncMaxWeightMode();
  setStatus("Example loaded");
  statusText.textContent = "Example tickers loaded. Adjust any setting and run the optimizer.";
}

addAssetButton.addEventListener("click", () => {
  appendAssetRow();
  syncMaxWeightMode();
});

loadExampleButton.addEventListener("click", loadExample);
form.addEventListener("submit", submitForm);
form.querySelector('[name="max_allocation_mode"]').addEventListener("change", syncMaxWeightMode);

assetRows().forEach((row) => {
  row.querySelector(".remove-button").addEventListener("click", () => {
    row.remove();
    ensureAtLeastOneRow();
    syncMaxWeightMode();
  });
});

syncMaxWeightMode();
