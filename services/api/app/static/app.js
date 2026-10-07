/* Housing Lab - vanilla JS front-end for the model service. */
"use strict";

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
// Escape any server-provided text before it goes into innerHTML.
const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const usd = (v) => (v == null ? "-" : "$" + Math.round(v).toLocaleString("en-US"));
const NS = "http://www.w3.org/2000/svg";

async function api(path, options = {}) {
  const res = await fetch(path, { headers: { "Content-Type": "application/json" }, ...options });
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.detail ? (typeof body.detail === "string" ? body.detail : body.detail.map((d) => `${(d.loc || []).slice(1).join(".")}: ${d.msg}`).join("; ")) : res.statusText);
  return body;
}

/* ---------- tabs ---------- */
$$(".tab[data-tab]").forEach((btn) => btn.addEventListener("click", () => {
  $$(".tab[data-tab]").forEach((b) => { b.classList.toggle("active", b === btn); b.setAttribute("aria-selected", b === btn); });
  $$(".panel").forEach((p) => { const on = p.id === `tab-${btn.dataset.tab}`; p.hidden = !on; p.classList.toggle("active", on); });
  if (btn.dataset.tab === "pipeline") loadPipeline();
  if (btn.dataset.tab === "monitor") loadMonitoring();
}));

/* ---------- health + model info ---------- */
const state = { info: null, sourceKind: "ui" };

async function refreshHealth() {
  const el = $("#status");
  try {
    const h = await api("/health");
    el.className = "status " + (h.status === "ok" ? "ok" : "bad");
    $("#status-text").textContent = h.status === "ok"
      ? `@${h.alias}: model v${h.model_version} · ${h.feature_set}`
      : "no model deployed yet - run the pipeline";
    if (h.status === "ok" && !state.info) await loadInfo();
  } catch (e) {
    el.className = "status bad";
    $("#status-text").textContent = "service unreachable";
  }
}

async function loadInfo() {
  state.info = await api("/model-info");
  $("#ocean").innerHTML = state.info.categories.map((c) => `<option>${esc(c)}</option>`).join("");
  drawGrid(null);
  if (!$("#form [name=longitude]").value) await loadSample("clean");
}

/* ---------- form <-> request ---------- */
const NUMERIC = ["longitude", "latitude", "housing_median_age", "median_income", "total_rooms", "total_bedrooms", "population", "households"];

function readForm() {
  const row = {};
  for (const name of NUMERIC) {
    const raw = $(`#form [name=${name}]`).value.trim();
    row[name] = raw === "" ? null : Number(raw);
  }
  row.ocean_proximity = $("#ocean").value;
  return row;
}

function fillForm(row) {
  for (const name of NUMERIC) $(`#form [name=${name}]`).value = row[name] ?? "";
  const sel = $("#ocean");
  if (![...sel.options].some((o) => o.value === row.ocean_proximity)) sel.add(new Option(row.ocean_proximity, row.ocean_proximity));
  sel.value = row.ocean_proximity;
}

async function loadSample(kind) {
  $$(".chip[data-kind]").forEach((c) => c.classList.toggle("on", c.dataset.kind === kind));
  try {
    const s = await api(`/samples?kind=${kind}`);
    fillForm(s.row);
    state.trueValue = s.true_price_usd;
    state.sourceKind = `sample-${s.kind}`;
    await estimate();
  } catch (e) { showError(e.message); }
}
$$(".chip[data-kind]").forEach((chip) => chip.addEventListener("click", () => loadSample(chip.dataset.kind)));
$("#form").addEventListener("input", () => { state.trueValue = null; state.sourceKind = "ui"; $$(".chip[data-kind]").forEach((c) => c.classList.remove("on")); });
$("#estimate").addEventListener("click", estimate);
$("#form").addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); estimate(); } });

function showError(msg) {
  $("#result-empty").hidden = true; $("#result").hidden = false;
  $("#price").textContent = "-";
  $("#price-sub").textContent = "";
  $("#flags").innerHTML = `<li class="warn"><span>&#9888;</span><span>${esc(msg)}</span></li>`;
  $("#meta").textContent = "";
}

/* ---------- estimate ---------- */
async function estimate() {
  const row = readForm();
  $$("#form input, #form select").forEach((el) => el.classList.remove("flagged"));
  $("#estimate").disabled = true;
  try {
    const r = await api("/predict", { method: "POST", body: JSON.stringify({ ...row, source: state.sourceKind }) });
    renderResult(r);
    renderSeen(r.seen, row);
    r.quality.missing.concat(r.quality.out_of_range).forEach((n) => $(`#form [name=${n}]`)?.classList.add("flagged"));
    if (r.quality.unknown_category) $("#ocean").classList.add("flagged");
  } catch (e) { showError(e.message); }
  finally { $("#estimate").disabled = false; }
}

function renderResult(r) {
  $("#result-empty").hidden = true; $("#result").hidden = false;
  $("#price").textContent = usd(r.price_usd);
  if (state.trueValue != null) {
    const err = r.price_usd - state.trueValue, pct = 100 * err / state.trueValue;
    $("#price-sub").innerHTML = `actual ${usd(state.trueValue)} &middot; <span class="${err > 0 ? "err-pos" : "err-neg"}">${err > 0 ? "+" : "-"}${usd(Math.abs(err))} (${pct > 0 ? "+" : ""}${pct.toFixed(0)}%)</span>`;
  } else {
    $("#price-sub").textContent = "predicted median house value of the block";
  }
  const q = r.quality, items = [];
  if (!q.missing.length && !q.unknown_category && !q.out_of_range.length) {
    items.push(`<li class="ok"><span>&#10003;</span><span>All values present, known area type, every number inside the training range.</span></li>`);
  }
  if (q.missing.length) items.push(`<li class="warn"><span>&#9888;</span><span><b>Missing:</b> ${esc(q.missing.join(", "))} was replaced by its training median, as the training pipeline does.</span></li>`);
  if (q.unknown_category) items.push(`<li class="warn"><span>&#9888;</span><span><b>Unseen area type.</b> The lookup layer sends it to its unknown bucket (index 0), which the model never trained on: treat the estimate with caution.</span></li>`);
  if (q.out_of_range.length) items.push(`<li class="warn"><span>&#9888;</span><span><b>Outside the training range:</b> ${esc(q.out_of_range.join(", "))}. A neural network extrapolates unpredictably there.</span></li>`);
  $("#flags").innerHTML = items.join("");
  $("#meta").textContent = `model v${r.model_version} · ${r.latency_ms} ms · logged as request #${r.prediction_id ?? "n/a"}`;
}

/* ---------- the 20 x 20 grid ---------- */
function drawGrid(seen, row) {
  const svg = $("#grid"), g = state.info?.grid;
  if (!g) return;
  const n = g.size, size = 16.5, pad = 0;
  svg.innerHTML = "";
  const counts = g.cell_counts || [];
  for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) {
    const idx = r * n + c, rect = document.createElementNS(NS, "rect");
    rect.setAttribute("x", pad + c * size); rect.setAttribute("y", pad + (n - 1 - r) * size);   // north is up
    rect.setAttribute("width", size); rect.setAttribute("height", size);
    rect.setAttribute("class", "cell" + (counts[idx] > 0 ? " seen" : "") + (seen && seen.grid_cell === idx ? " on" : ""));
    const t = document.createElementNS(NS, "title");
    t.textContent = `cell ${idx} (row ${r}, column ${c}): ${counts[idx] ?? 0} training blocks`;
    rect.appendChild(t);
    svg.appendChild(rect);
  }
  if (seen && row) {
    const frac = (v, b, k) => (k <= 0 || k >= n - 1 ? 0.5 : Math.min(1, Math.max(0, (v - b[k - 1]) / (b[k] - b[k - 1]))));
    const fx = frac(row.longitude, g.lon_boundaries, seen.grid_col), fy = frac(row.latitude, g.lat_boundaries, seen.grid_row);
    const dot = document.createElementNS(NS, "circle");
    dot.setAttribute("cx", pad + (seen.grid_col + fx) * size); dot.setAttribute("cy", pad + (n - seen.grid_row - fy) * size);
    dot.setAttribute("r", 3.4); dot.setAttribute("class", "dot");
    svg.appendChild(dot);
  }
}

function renderSeen(seen, row) {
  $("#seen-empty").hidden = true; $("#seen").hidden = false;
  drawGrid(seen, row);
  $("#cell-label").textContent = `grid cell ${seen.grid_cell} (row ${seen.grid_row}, column ${seen.grid_col}) · income bucket ${seen.income_bucket + 1} of 5`;
  $("#zbars").innerHTML = Object.entries(seen.z_scores).map(([k, z]) => {
    const w = Math.min(Math.abs(z), 3) / 3 * 50;
    return `<div class="zrow"><span class="name">${esc(k.replaceAll("_", " "))}</span><div class="track"><span class="fill${z < 0 ? " neg" : ""}" style="${z < 0 ? "right" : "left"}:50%;width:${w}%"></span></div><span class="val">${z > 0 ? "+" : ""}${z.toFixed(2)}</span></div>`;
  }).join("");
}

/* ---------- pipeline results ---------- */
function hbars(el, rows, spec) {
  // rows: [{name, values: [number...], labels: [string...]}]; spec.max: array of scales per series
  el.innerHTML = rows.map((r) => `<div class="hrow"><span class="name">${esc(r.name)}</span><div class="bars2">${
    r.values.map((v, i) => `<div class="b"><i class="${i ? "alt" : ""}" style="width:${Math.max(1, 100 * v / spec.max[i] * 0.78)}%"></i><span>${esc(r.labels[i])}</span></div>`).join("")}</div></div>`).join("");
}

let pipelineLoaded = false;
async function loadPipeline() {
  if (pipelineLoaded) return;
  try {
    const { tables: t } = await api("/pipeline/summary");
    $("#pipe-empty").hidden = true; $("#pipe-grid").hidden = false;
    pipelineLoaded = true;
    if (t.csv_ablation) {
      const strict = t.csv_ablation.filter((r) => r.regime === "strict");
      const names = [...new Set(strict.map((r) => r.variant))].sort();
      const pick = (name, step) => strict.find((r) => r.variant === name && r.step_ms === step)?.examples_per_second ?? 0;
      const slow = Math.max(...strict.map((r) => r.step_ms));
      const rows = names.map((n) => ({ name: n.replace(/^\d\. /, ""), values: [pick(n, 0), pick(n, slow)], labels: [Math.round(pick(n, 0)).toLocaleString(), Math.round(pick(n, slow)).toLocaleString()] }));
      const mx = Math.max(...rows.flatMap((r) => r.values));
      hbars($("#p-csv"), rows, { max: [mx, mx] });
      $("#p-csv-note").textContent = `Strict mode switches off tf.data's own automatic optimisations so each of the book's steps can be seen on its own; the second bar adds a simulated ${slow} ms training step, where prefetching starts to matter.`;
    }
    if (t.format_benchmark) {
      const f = t.format_benchmark;
      hbars($("#p-formats"), f.map((r) => ({ name: r.format, values: [r.size_vs_csv, r.examples_per_second / 1000], labels: [`${r.size_vs_csv.toFixed(2)}x`, `${(r.examples_per_second / 1000).toFixed(0)}k`] })),
        { max: [Math.max(...f.map((r) => r.size_vs_csv)), Math.max(...f.map((r) => r.examples_per_second / 1000))] });
    }
    if (t.scale_test) {
      const s = t.scale_test;
      hbars($("#p-scale"), s.map((r) => ({ name: `${r.rows.toLocaleString()} rows`, values: [r.stream_growth_mb, r.pandas_growth_mb], labels: [`${r.stream_growth_mb.toFixed(0)} MB`, `${r.pandas_growth_mb.toFixed(0)} MB`] })),
        { max: [Math.max(...s.map((r) => r.pandas_growth_mb)), Math.max(...s.map((r) => r.pandas_growth_mb))] });
    }
    if (t.shuffle_quality) {
      const s = t.shuffle_quality;
      hbars($("#p-shuffle"), s.map((r) => ({ name: r.label, values: [r.batch_diversity], labels: [r.batch_diversity.toFixed(2)] })), { max: [1] });
    }
    if (t.feature_summary) {
      const fs = t.feature_summary, sets = [...new Set(fs.map((r) => r.feature_set))];
      const heads = [...new Set(fs.map((r) => r.head))];
      const get = (set, head) => fs.find((r) => r.feature_set === set && r.head === head);
      const rows = sets.map((n) => ({ name: n, values: heads.map((h) => get(n, h)?.valid_rmse ?? 0), labels: heads.map((h) => { const r = get(n, h); return r ? `$${Math.round(r.valid_rmse).toLocaleString()}` : "-"; }) }));
      const mx = Math.max(...rows.flatMap((r) => r.values));
      hbars($("#p-features"), rows, { max: heads.map(() => mx) });
      $("#p-features-note").textContent = "Mean over several seeds. The linear model gains a lot from the engineered features; the small neural network learns most of it from the raw numbers (see the report for the confidence intervals).";
    }
  } catch (e) {
    $("#pipe-grid").hidden = true; $("#pipe-empty").hidden = false;
    $("#pipe-empty").textContent = e.message;
  }
}

/* ---------- monitoring ---------- */
async function loadMonitoring() {
  try {
    const m = await api("/monitoring/summary");
    const pct = (v) => (v == null ? "-" : `${v}%`);
    $("#mon-window").textContent = `· last ${m.window_hours / 24} days`;
    $("#k-n").textContent = m.n_predictions.toLocaleString();
    $("#k-unknown").textContent = pct(m.unknown_category_pct);
    $("#k-range").textContent = pct(m.out_of_range_pct);
    $("#k-missing").textContent = pct(m.missing_pct);
    $("#k-missing-sub").textContent = m.expected_missing_pct == null ? "" : `training: ${m.expected_missing_pct}% of rows`;
    $("#k-price").textContent = usd(m.mean_prediction_usd);
    $("#k-price-sub").textContent = m.training_mean_usd ? `training mean ${usd(m.training_mean_usd)}` : "";
    $("#k-lat").textContent = m.mean_latency_ms == null ? "-" : `${m.mean_latency_ms} ms`;
    $("#source-table tbody").innerHTML = Object.entries(m.by_source).map(([s, v]) =>
      `<tr><td>${esc(s)}</td><td class="num">${v.n}</td><td class="num">${v.flagged}</td><td class="num">${(100 * v.flagged / v.n).toFixed(0)}%</td></tr>`).join("")
      || `<tr><td colspan="4" class="muted">no requests yet</td></tr>`;
    $("#recent").innerHTML = m.recent.map((r) => {
      const flags = [r.unknown_category ? "unseen area" : "", r.n_missing ? "missing value" : "", r.n_out_of_range ? "out of range" : ""].filter(Boolean);
      return `<div class="rec"><div><span class="big">${usd(r.prediction_usd)}</span>${flags.length ? ` <span class="flag">${esc(flags.join(", "))}</span>` : ""}
        <small>${esc(r.ocean_proximity)} · income ${esc(r.median_income)}</small><small>${esc(r.source)}</small></div></div>`;
    }).join("") || '<span class="muted">no requests yet</span>';
  } catch (e) { $("#recent").innerHTML = `<span class="muted">${esc(e.message)}</span>`; }
}
$("#refresh-mon").addEventListener("click", loadMonitoring);

refreshHealth();
setInterval(refreshHealth, 30000);
