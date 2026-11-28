// StronghouseAI web UI. Plain JavaScript, no build step (see docs/adr/0004-plain-web-ui.md).
"use strict";

const ENGINE_POLL_MS = 3000;
const DOWNLOAD_POLL_MS = 1500;
const $ = (selector) => document.querySelector(selector);

// ---------- helpers ----------

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (ch) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch]
  ));
}

function formatBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes <= 0) return "0 B";
  const units = ["B", "KB", "MB", "GB", "TB"];
  const i = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  return `${(bytes / 1024 ** i).toFixed(i >= 3 ? 1 : 0)} ${units[i]}`;
}

const OFFLINE_MESSAGE = "Can't reach StronghouseAI. Is the app still running in your terminal?";

// kind: "error" (default, stays longer) or "info".
function toast(message, kind = "error") {
  const el = $("#toast");
  el.textContent = message;
  el.className = `toast ${kind}`;
  el.hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => { el.hidden = true; }, kind === "error" ? 7000 : 4000);
}

// The server sends {"detail": "<sentence>"}; older or proxy errors may not, so fall back.
async function errorFromResponse(response) {
  const body = await response.json().catch(() => ({}));
  const detail = typeof body.detail === "string" ? body.detail : "";
  return new Error(detail || `${response.status} ${response.statusText || "error"}`);
}

async function api(path, options = {}) {
  let response;
  try {
    response = await fetch(`/api${path}`, {
      headers: { "Content-Type": "application/json" },
      ...options,
      body: options.body ? JSON.stringify(options.body) : undefined,
    });
  } catch (error) {
    if (error.name === "AbortError") throw error;
    throw new Error(OFFLINE_MESSAGE);
  }
  if (!response.ok) throw await errorFromResponse(response);
  return response.json();
}

async function withErrors(action) {
  try { return await action(); } catch (error) { toast(error.message); return undefined; }
}

// Run an action with its button disabled, so a double click cannot send it twice.
async function whileBusy(button, action) {
  if (button.disabled) return undefined;
  button.disabled = true;
  try { return await action(); } finally { button.disabled = false; }
}

// Put text on the clipboard. navigator.clipboard needs a secure page (127.0.0.1 is one);
// if the app is opened by another address, fall back to the older copy command.
async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    const area = document.createElement("textarea");
    area.value = text;
    area.setAttribute("readonly", "");
    area.style.position = "fixed";
    area.style.opacity = "0";
    document.body.appendChild(area);
    area.select();
    const copied = document.execCommand("copy");
    area.remove();
    return copied;
  }
}

async function copyWithToast(text, what) {
  if (await copyText(text)) toast(`${what} copied. Paste it into your chat with Claude.`, "info");
  else toast(`Could not copy the ${what.toLowerCase()}. Select the text and press Ctrl+C instead.`);
}

// Update a live log box without breaking what the user is doing in it: leave it alone
// while text in it is selected (so it can be copied), and only follow the end of the
// log if the user had scrolled to the bottom.
function setLogText(el, text) {
  if (el.textContent === text) return;
  const selection = window.getSelection();
  if (selection && !selection.isCollapsed && el.contains(selection.anchorNode)) return;
  const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 24;
  el.textContent = text;
  if (atBottom) el.scrollTop = el.scrollHeight;
}

// In-app replacement for prompt() and confirm(). Resolves to the typed text (with
// `input`), true (confirm), or null when cancelled.
function ask({ title, message = "", input = null, confirmText = "OK", danger = false }) {
  const dialog = $("#ask-dialog");
  const field = $("#ask-input");
  $("#ask-title").textContent = title;
  $("#ask-message").textContent = message;
  $("#ask-ok").textContent = confirmText;
  $("#ask-ok").className = danger ? "danger" : "";
  field.hidden = input === null;
  field.value = input ?? "";
  dialog.returnValue = "";
  dialog.showModal();
  if (input !== null) field.select();
  return new Promise((resolve) => {
    dialog.addEventListener("close", () => {
      if (dialog.returnValue !== "ok") resolve(null);
      else resolve(input === null ? true : field.value);
    }, { once: true });
  });
}
$("#ask-input").addEventListener("keydown", (event) => {
  if (event.key === "Enter") { event.preventDefault(); $("#ask-dialog").close("ok"); }
});

function emptyState(text) {
  return `<p class="empty">${escapeHtml(text)}</p>`;
}

function loadingState(text) {
  return `<p class="empty"><span class="thinking"><span class="spinner" aria-hidden="true"></span>${escapeHtml(text)}</span></p>`;
}

// Last line of defence: an unexpected script error becomes a message, not a dead page.
window.addEventListener("unhandledrejection", (event) => toast(event.reason?.message || String(event.reason)));

// ---------- tabs ----------

const TAB_LOADERS = { library: () => loadLibrary(), run: () => loadRunModels(), system: () => loadSystem() };

function showTab(name) {
  const button = document.querySelector(`.tabs button[data-tab="${name}"]`);
  if (!button) return;
  document.querySelectorAll(".tabs button").forEach((b) => {
    b.classList.toggle("active", b === button);
    b.setAttribute("aria-selected", String(b === button));
  });
  document.querySelectorAll(".panel").forEach((p) => p.classList.toggle("active", p.id === `tab-${name}`));
  // Remember the page in the address, so a reload stays where you were.
  if (location.hash !== `#${name}`) history.replaceState(null, "", `#${name}`);
  TAB_LOADERS[name]?.();
}

document.querySelectorAll(".tabs button").forEach((button) => {
  button.addEventListener("click", () => showTab(button.dataset.tab));
});

// ---------- vault (library) ----------

async function loadLibrary() {
  const search = encodeURIComponent($("#library-search").value);
  const entries = await withErrors(() => api(`/library?search=${search}`));
  if (!entries) return;
  const list = $("#library-list");
  if (entries.length === 0) {
    list.innerHTML = $("#library-search").value
      ? emptyState("No models match that search.")
      : emptyState("Your vault is empty. Archive a model from Discover, or put .gguf files in "
        + "<archive>/<owner>/<repo>/ and press Rescan.");
    return;
  }
  list.innerHTML = entries.map((entry) => `
    <div class="card">
      <h3>${escapeHtml(entry.filename)}</h3>
      <div class="meta">${escapeHtml(entry.repo_id)} · ${formatBytes(entry.size_bytes)}</div>
      <div class="badges">
        ${entry.is_moe ? '<span class="badge moe">MoE</span>' : ""}
        ${entry.architecture ? `<span class="badge">${escapeHtml(entry.architecture)}</span>` : ""}
        ${entry.license ? `<span class="badge">${escapeHtml(entry.license)}</span>` : ""}
        ${entry.sha256 ? '<span class="badge ok" title="SHA-256 checked after download">verified</span>' : ""}
        ${(entry.tags ?? []).map((t) => `<span class="badge">#${escapeHtml(t)}</span>`).join("")}
      </div>
      <div class="row">
        <button data-run="${entry.id}">Run</button>
        <button class="secondary" data-tags="${entry.id}" data-current="${escapeHtml((entry.tags ?? []).join(", "))}">Tags</button>
        <button class="danger" data-forget="${entry.id}" data-name="${escapeHtml(entry.filename)}"
          title="Remove from this list. The file stays on disk.">Forget</button>
      </div>
    </div>`).join("");
}

$("#library-list").addEventListener("click", async (event) => {
  const target = event.target.closest("button");
  if (!target) return;
  if (target.dataset.run) {
    showTab("run");
    await loadRunModels(target.dataset.run);
  } else if (target.dataset.tags) {
    const answer = await ask({
      title: "Tags", message: "Separate tags with commas.", input: target.dataset.current, confirmText: "Save",
    });
    if (answer === null) return;
    await withErrors(() => api(`/library/${target.dataset.tags}/tags`, {
      method: "PUT", body: { tags: answer.split(",") },
    }));
    loadLibrary();
  } else if (target.dataset.forget) {
    const sure = await ask({
      title: `Forget ${target.dataset.name}?`,
      message: "It is removed from the vault list only. The file stays on disk, and Rescan brings it back.",
      confirmText: "Forget",
    });
    if (!sure) return;
    const removed = await withErrors(() => api(`/library/${target.dataset.forget}`, { method: "DELETE" }));
    if (removed) toast("Removed from the list. The file is still on disk.", "info");
    loadLibrary();
  }
});

$("#library-search").addEventListener("input", () => {
  clearTimeout(loadLibrary.timer);
  loadLibrary.timer = setTimeout(loadLibrary, 250);
});
$("#library-rescan").addEventListener("click", (event) => whileBusy(event.currentTarget, async () => {
  const result = await withErrors(() => api("/library/rescan", { method: "POST" }));
  if (result) toast(`Indexed ${result.indexed} model(s).`, "info");
  await loadLibrary();
}));

// ---------- discover (Hugging Face) ----------

const KIND_NAMES = { "text-generation": "chat", "image-text-to-text": "vision", "feature-extraction": "embedding" };
const FIT_NAMES = {
  gpu: ["fits in GPU", "The whole file fits in graphics memory: fastest."],
  offload: ["GPU + RAM", "Runs with part of it in system RAM (MoE offload helps here)."],
  too_big: ["too big", "Larger than graphics memory and RAM together."],
};
let hubFiles = { repo: null, files: [] };

function hubSearchParams() {
  const params = new URLSearchParams({
    q: $("#hub-query").value,
    size: $("#hub-size").value,
    kind: $("#hub-kind").value,
    sort: $("#hub-sort").value,
  });
  if ($("#hub-moe").checked) params.set("moe", "true");
  return params;
}

async function searchHub() {
  $("#hub-results").innerHTML = loadingState("Searching Hugging Face…");
  const models = await withErrors(() => api(`/hub/search?${hubSearchParams()}`));
  if (!models) { $("#hub-results").innerHTML = emptyState("Search failed. Check your connection and try again."); return; }
  $("#hub-results").innerHTML = models.map((model) => `
    <div class="item selectable${model.repo_id === hubFiles.repo ? " selected" : ""}" data-repo="${escapeHtml(model.repo_id)}">
      <div><div class="name">${escapeHtml(model.repo_id)}</div>
        <div class="badges">
        ${model.size_label ? `<span class="badge">${escapeHtml(model.size_label)}</span>` : ""}
        ${model.is_moe ? '<span class="badge moe">MoE</span>' : ""}
        ${KIND_NAMES[model.kind] ? `<span class="badge">${KIND_NAMES[model.kind]}</span>` : ""}</div></div>
      <span class="side" title="Downloads">${(model.downloads ?? 0).toLocaleString()} downloads</span>
    </div>`).join("") || emptyState("No GGUF models match. Try fewer filters.");
}

const UNKNOWN_FIT = ["fit unknown", "Could not compare this file with your hardware."];

function renderHubFiles() {
  const { repo, files } = hubFiles;
  if (!repo) return;
  const hideTooBig = $("#hub-fits").checked;
  const visible = files.filter((f) => !(hideTooBig && f.fit === "too_big"));
  const hidden = files.length - visible.length;
  $("#hub-files").innerHTML = visible.map((file) => {
    const [fitLabel, fitHelp] = FIT_NAMES[file.fit] ?? UNKNOWN_FIT;
    return `
    <div class="item">
      <div><div class="name">${escapeHtml(file.filename)}</div>
        <div class="badges"><span class="badge fit-${escapeHtml(file.fit)}" title="${fitHelp}">${fitLabel}</span></div></div>
      <span class="side">${formatBytes(file.size_bytes)}
        <button class="small" data-download="${escapeHtml(file.filename)}" data-repo="${escapeHtml(repo)}">Archive</button></span>
    </div>`;
  }).join("")
    + (hidden ? `<p class="muted">${hidden} file(s) too big for this computer are hidden.</p>` : "")
    || emptyState("No .gguf files in this repo.");
}

$("#hub-form").addEventListener("submit", (event) => { event.preventDefault(); searchHub(); });
["#hub-size", "#hub-kind", "#hub-sort", "#hub-moe"].forEach((sel) =>
  $(sel).addEventListener("change", () => { if ($("#hub-query").value || $("#hub-results").children.length) searchHub(); }));
$("#hub-fits").addEventListener("change", renderHubFiles);

$("#hub-results").addEventListener("click", async (event) => {
  const row = event.target.closest("[data-repo]");
  if (!row) return;
  const repo = row.dataset.repo;
  $("#hub-results").querySelectorAll(".item").forEach((item) => item.classList.toggle("selected", item === row));
  $("#hub-files").innerHTML = loadingState("Loading files…");
  const files = await withErrors(() => api(`/hub/files?repo_id=${encodeURIComponent(repo)}`));
  if (!files) { $("#hub-files").innerHTML = emptyState("Could not load the files. Try again."); return; }
  hubFiles = { repo, files };
  renderHubFiles();
});

$("#hub-files").addEventListener("click", async (event) => {
  const button = event.target.closest("[data-download]");
  if (!button) return;
  if (button.disabled) return;
  button.disabled = true;
  button.textContent = "Archiving…";
  const job = await withErrors(() => api("/downloads", {
    method: "POST", body: { repo_id: button.dataset.repo, filename: button.dataset.download },
  }));
  if (!job) {
    button.disabled = false;
    button.textContent = "Archive";
    return;
  }
  toast("Download started.", "info");
  pollDownloads();
});

async function pollDownloads() {
  const jobs = await withErrors(() => api("/downloads"));
  if (!jobs) return;
  $("#download-list").innerHTML = jobs.map((job) => {
    const percent = job.bytes_total ? Math.floor((100 * job.bytes_done) / job.bytes_total) : 0;
    return `<div class="item"><div class="grow-fill">
      <div class="name">${escapeHtml(job.repo_id)} / ${escapeHtml(job.filename)}</div>
      <div class="muted num">${escapeHtml(job.state)} · ${formatBytes(job.bytes_done)} of ${formatBytes(job.bytes_total)}
        ${job.error ? ` · <span class="error">${escapeHtml(job.error)}</span>` : ""}</div>
      <div class="progress"><div style="width:${percent}%"></div></div></div>
      <span class="side">${percent}%</span></div>`;
  }).join("") || emptyState("No downloads yet.");
  const active = jobs.some((job) => ["queued", "downloading", "verifying"].includes(job.state));
  clearTimeout(pollDownloads.timer);
  if (active) pollDownloads.timer = setTimeout(pollDownloads, DOWNLOAD_POLL_MS);
}

// ---------- run (planner + engine) ----------

async function loadRunModels(selectId) {
  const entries = await withErrors(() => api("/library"));
  if (!entries) return;
  const select = $("#run-model");
  const keep = selectId ?? select.value;
  select.innerHTML = entries.length
    ? entries.map((e) => `<option value="${e.id}">${escapeHtml(e.filename)}${e.is_moe ? " (MoE)" : ""}</option>`).join("")
    : '<option value="">No models in your vault yet</option>';
  if (keep) select.value = keep;
  showProfile();
}

async function showProfile() {
  const id = $("#run-model").value;
  if (!id) { $("#run-profile").textContent = "No models in your vault yet."; return; }
  const p = await withErrors(() => api(`/library/${id}/profile`));
  if (!p) return;
  $("#run-profile").innerHTML = `${escapeHtml(p.architecture)} · ${p.block_count} blocks ·
    trained context ${(p.context_length ?? 0).toLocaleString()}<br>
    ${p.is_moe ? `MoE: ${p.expert_count} experts, ${p.experts_used} active per token ·
    experts ${formatBytes(p.expert_bytes)}, shared ${formatBytes(p.dense_bytes)}` : `Dense model · ${formatBytes(p.dense_bytes)}`}`;
}
$("#run-model").addEventListener("change", showProfile);

// Extra llama-server flags are remembered per browser, so a fast terminal setup
// only has to be pasted once.
const EXTRA_FLAGS_KEY = "stronghouse.extraFlags";
function rememberedExtraFlags() {
  try { return localStorage.getItem(EXTRA_FLAGS_KEY) ?? ""; } catch { return ""; }
}
$("#run-extra").value = rememberedExtraFlags();
$("#run-extra").addEventListener("change", () => {
  try { localStorage.setItem(EXTRA_FLAGS_KEY, $("#run-extra").value); } catch { /* private window */ }
});

function planBody() {
  const optionalInt = (sel) => ($(sel).value === "" ? null : Number($(sel).value));
  const extra = $("#run-extra").value.trim();
  return {
    context: Number($("#run-context").value),
    kv_cache_type: $("#run-kv").value,
    n_cpu_moe: optionalInt("#run-ncpumoe"),
    n_gpu_layers: optionalInt("#run-ngl"),
    threads: optionalInt("#run-threads"),
    batch_size: optionalInt("#run-batch"),
    ubatch_size: optionalInt("#run-ubatch"),
    // Empty box: use the app-wide default flags (STRONGHOUSE_EXTRA_ARGS), if any.
    extra_args: extra === "" ? null : extra,
  };
}

const STRATEGY_NAMES = { full_gpu: "Full GPU", moe_split: "MoE split", partial_layers: "Partial layers", cpu_only: "CPU only" };

// Same rounding as the server's estimates: 125, 4.2 or 0.24 tokens/s.
const formatRate = (rate) => (rate >= 10 ? rate.toFixed(0) : rate >= 1 ? rate.toFixed(1) : rate.toFixed(2));

function renderForecast(forecast, plan) {
  if (!forecast) { $("#run-forecast").innerHTML = ""; return; }
  const streaming = plan.memory_mode === "stream_from_disk";
  const quants = forecast.quant_options.map((o) => `
    <span class="badge ${o.memory_mode === "in_memory" ? "fit-gpu" : "fit-offload"} ${o.is_current ? "current" : ""}"
      title="${o.memory_mode === "in_memory" ? "fits in GPU + RAM" : "streams from disk"} · about ${formatBytes(o.estimated_file_bytes)}">
      ${escapeHtml(o.label)} · ${formatRate(o.tokens_per_second)} t/s</span>`).join("");
  $("#run-forecast").innerHTML = `
    <div class="speed">${streaming ? '<span class="badge fit-offload">Big model mode</span> ' : ""}Expected speed:
      <strong>${escapeHtml(forecast.wait)}</strong></div>
    <div class="muted">A rough estimate from memory speeds. The speed test gives the real number.
      ${streaming ? "Too slow to watch? Queue it on the Jobs page and walk away." : ""}</div>
    ${forecast.quant_tip ? `<div class="notice warn">${escapeHtml(forecast.quant_tip)}</div>` : ""}
    ${quants ? `<details><summary>Other quantizations of this model</summary>
      <div class="quant-grid">${quants}</div>
      <div class="muted">Green fits in GPU + RAM; amber streams from disk. Download another quantization from Discover.</div></details>` : ""}`;
}

function renderPlan({ plan, command, forecast }) {
  renderForecast(forecast, plan);
  const total = plan.estimated_gpu_bytes + plan.estimated_cpu_bytes || 1;
  const gpuShare = (100 * plan.estimated_gpu_bytes) / total;
  $("#run-plan-view").innerHTML = `
    <div class="plan-head num"><strong>${escapeHtml(STRATEGY_NAMES[plan.strategy] ?? plan.strategy)}</strong>
      <span>context ${plan.context_size.toLocaleString()}</span><span>GPU layers ${plan.n_gpu_layers}</span>
      <span>experts on CPU ${plan.n_cpu_moe}</span></div>
    <div class="meter"><div class="gpu" style="width:${gpuShare}%"></div><div class="cpu" style="flex:1"></div></div>
    <div class="muted num">GPU ≈ ${formatBytes(plan.estimated_gpu_bytes)} · RAM ≈ ${formatBytes(plan.estimated_cpu_bytes)} (estimates)</div>
    ${plan.notes.length ? `<ul class="plan-notes">${plan.notes.map((n) => `<li>${escapeHtml(n)}</li>`).join("")}</ul>` : ""}
    <details><summary>Command</summary><code>${escapeHtml(command.join(" "))}</code></details>`;
}

function selectedModelId() {
  const id = $("#run-model").value;
  if (!id) toast("Pick a model first. Your vault is empty? Archive one from Discover.");
  return id;
}

$("#run-plan").addEventListener("click", (event) => whileBusy(event.currentTarget, async () => {
  const id = selectedModelId();
  if (!id) return;
  const result = await withErrors(() => api(`/library/${id}/plan`, { method: "POST", body: planBody() }));
  if (result) renderPlan(result);
}));

$("#run-start").addEventListener("click", (event) => whileBusy(event.currentTarget, async () => {
  const id = selectedModelId();
  if (!id) return;
  const status = await withErrors(() => api("/engine/start", {
    method: "POST", body: { model_id: Number(id), ...planBody() },
  }));
  if (status) { toast("Loading model… this can take a minute for big models.", "info"); pollEngine(); }
}));

$("#run-stop").addEventListener("click", (event) => whileBusy(event.currentTarget, async () => {
  await withErrors(() => api("/engine/stop", { method: "POST" }));
  pollEngine();
}));

$("#run-bench").addEventListener("click", (event) => whileBusy(event.currentTarget, async () => {
  $("#run-bench-view").innerHTML = '<div class="bench thinking"><span class="spinner" aria-hidden="true"></span>Running…</div>';
  const result = await withErrors(() => api("/engine/benchmark", { method: "POST" }));
  $("#run-bench-view").innerHTML = result
    ? `<div class="bench"><strong>${result.generated_tokens_per_second}</strong> tokens/s generating ·
       ${result.prompt_tokens_per_second} tokens/s reading the prompt</div>`
    : "";
}));

$("#run-log-copy").addEventListener("click", (event) => whileBusy(event.currentTarget, async () => {
  // The server hides tokens and passwords, so the copy is safe to paste anywhere.
  const engineLog = await withErrors(() => api("/logs/engine?lines=500"));
  if (engineLog) await copyWithToast(engineLog.copy_text, "Engine log");
}));

function showEngineState(state, model = "", title = "") {
  const pill = $("#engine-pill");
  pill.className = `pill ${state}`;
  pill.title = title;
  pill.querySelector(".state").textContent = state === "offline" ? "App offline" : state;
  $("#engine-model").textContent = model;
}

async function pollEngine() {
  clearTimeout(pollEngine.timer);
  pollEngine.timer = setTimeout(pollEngine, ENGINE_POLL_MS);
  const status = await api("/engine").catch(() => null);
  if (!status) { showEngineState("offline", "", OFFLINE_MESSAGE); return; }
  const model = status.model_path ? status.model_path.split("/").pop() : "";
  showEngineState(status.state, model);
  const logTail = status.log_tail ?? [];
  setLogText($("#run-log"), logTail.join("\n"));
  const outOfMemory = status.state === "crashed" && logTail.some((l) => /failed to allocate/i.test(l));
  if (outOfMemory && !pollEngine.warned) {
    pollEngine.warned = true;
    toast("Ran out of GPU memory. Try a smaller context, a q8_0 KV cache, or more experts on CPU.");
  }
  if (status.state !== "crashed") pollEngine.warned = false;
}

// ---------- chat ----------

const chatHistory = [];  // sent to the model: role and content only
const chatTurns = [];    // shown and saved: also each reply's thinking, for Copy chat and Export

function addMessage(role, text) {
  const el = document.createElement("div");
  el.className = `msg ${role}`;
  el.textContent = text;
  $("#chat-log").appendChild(el);
  el.scrollIntoView({ block: "end" });
  return el;
}

// A reply bubble with a live "thinking" area. Reasoning models (Qwen3, DeepSeek R1)
// think before they answer, sometimes for minutes on a laptop; showing that text
// as it arrives makes clear the model is working, not frozen.
function replyView(bubble) {
  const started = Date.now();
  bubble.innerHTML = `
    <details class="thought" open>
      <summary><span class="spinner" aria-hidden="true"></span><span class="label">Thinking… 0s</span></summary>
      <div class="thought-text"></div>
    </details>
    <div class="answer"></div>`.replace(/>\s+</g, "><").trim();
  const details = bubble.querySelector(".thought");
  const label = bubble.querySelector(".label");
  const thoughtText = bubble.querySelector(".thought-text");
  const answer = bubble.querySelector(".answer");
  let thought = "";
  let answering = false;
  const seconds = () => Math.round((Date.now() - started) / 1000);
  const words = () => (thought.match(/\S+/g) || []).length;
  const tick = () => {
    if (!answering) label.textContent = `Thinking… ${seconds()}s${thought ? ` · ${words()} words` : ""}`;
  };
  const timer = setInterval(tick, 1000);
  const finishThinking = () => {
    if (answering) return;
    answering = true;
    clearInterval(timer);
    details.querySelector(".spinner")?.remove();
    if (thought.trim()) {
      label.textContent = `Thought for ${seconds()}s (${words()} words)`;
      details.open = false;
    } else {
      details.remove();
    }
  };
  return {
    addThought(text) {
      thought += text;
      thoughtText.append(text);
      thoughtText.scrollTop = thoughtText.scrollHeight;
      tick();
    },
    // append() adds a new text node instead of rewriting the text, so a selection the
    // user made in the reply survives while more of it streams in.
    addAnswer(text) { finishThinking(); answer.append(text); },
    answer: () => answer.textContent,
    thought: () => thought,
    end(note) { finishThinking(); clearInterval(timer); if (note) answer.append(note); addCopyButton(bubble, answer); },
    // The model asked to go online instead of answering: keep what it said or thought,
    // drop the bubble if there is nothing, and let the next reply continue below.
    handOff() {
      finishThinking();
      clearInterval(timer);
      if (!answer.textContent.trim() && !bubble.querySelector(".thought")) bubble.remove();
    },
    fail(message) {
      finishThinking();
      clearInterval(timer);
      answer.classList.add("error");
      answer.append(`${answer.textContent ? "\n\n" : ""}${message}`);
    },
  };
}

// A Copy button under a finished reply. It sits outside the answer text, so selecting
// the reply by hand does not pick up the button's label.
function addCopyButton(bubble, answer) {
  const tools = document.createElement("div");
  tools.className = "msg-tools";
  const button = document.createElement("button");
  button.type = "button";
  button.className = "secondary small";
  button.textContent = "Copy";
  button.title = "Copy this reply";
  button.addEventListener("click", async () => {
    if (await copyText(answer.textContent)) toast("Reply copied.", "info");
    else toast("Could not copy the reply. Select the text and press Ctrl+C instead.");
  });
  tools.appendChild(button);
  bubble.appendChild(tools);
}

// Some models write their reasoning inline as <think>...</think> instead of in a
// separate field; route that part to the thinking area too.
function inlineThinkSplitter(view) {
  let mode = "start";  // start -> (think) -> answer
  let pending = "";
  return (text) => {
    pending += text;
    if (mode === "start") {
      const trimmed = pending.trimStart();
      if ("<think>".startsWith(trimmed)) return;  // could still become <think>
      if (trimmed.startsWith("<think>")) { mode = "think"; pending = trimmed.slice(7); }
      else { mode = "answer"; }
    }
    if (mode === "think") {
      const end = pending.indexOf("</think>");
      if (end === -1) {
        const safe = pending.length - 8;  // keep a possible partial closing tag
        if (safe > 0) { view.addThought(pending.slice(0, safe)); pending = pending.slice(safe); }
        return;
      }
      view.addThought(pending.slice(0, end));
      pending = pending.slice(end + 8).trimStart();
      mode = "answer";
    }
    if (mode === "answer" && pending) { view.addAnswer(pending); pending = ""; }
  };
}

// One "data: {...}" line of the server-sent stream, or null for anything else.
// A garbled line is skipped rather than ending the whole reply.
function parseStreamLine(line) {
  if (!line.startsWith("data: ") || line.includes("[DONE]")) return null;
  try { return JSON.parse(line.slice(6)); } catch { return null; }
}

let chatAbort = null;

function setChatBusy(busy) {
  $("#chat-form button[type=submit]").hidden = busy;
  $("#chat-stop").hidden = !busy;
}

// ----- internet access: the model may ask for the web tools, the user allows each use -----

const MAX_WEB_STEPS = 6;  // searches and page reads one reply may make (matches core/chat_web.py)
const WEB_DENIED = "The user did not allow this. Do not ask again; answer from what you know and say you could not check online.";
const WEB_ACTIONS = {
  web_search: (args) => `Search the web for “${args.query ?? ""}”`,
  fetch_web_page: (args) => `Open the page ${args.url ?? ""}`,
  read_news_feed: (args) => `Read the news feed ${args.url ?? ""}`,
};

function describeWebCall(call) {
  let args = {};
  try { args = JSON.parse(call.arguments || "{}"); } catch { /* shown as the raw name below */ }
  const describe = WEB_ACTIONS[call.name];
  return { args, text: describe ? describe(args) : `Use ${call.name}` };
}

// A card in the chat that asks before the model goes online. Resolves true (Allow) or
// false (Deny); rejects with an AbortError when the user presses Stop instead.
function webStepCard(text) {
  const card = document.createElement("div");
  card.className = "msg web-step";
  card.innerHTML = `
    <div class="web-head"><svg class="icon" aria-hidden="true"><use href="#i-globe"/></svg><span class="web-title">The model wants to use the internet</span></div>
    <div class="web-what"></div>
    <div class="web-buttons"><button type="button" class="small">Allow</button><button type="button" class="secondary small">Deny</button></div>`
    .replace(/>\s+</g, "><").trim();
  card.querySelector(".web-what").textContent = text;
  $("#chat-log").appendChild(card);
  card.scrollIntoView({ block: "end" });
  const title = card.querySelector(".web-title");
  const buttons = card.querySelector(".web-buttons");
  return {
    decide(signal) {
      const [allow, deny] = buttons.querySelectorAll("button");
      allow.focus();
      return new Promise((resolve, reject) => {
        const finish = (allowed) => { buttons.remove(); resolve(allowed); };
        allow.addEventListener("click", () => finish(true));
        deny.addEventListener("click", () => finish(false));
        signal.addEventListener("abort", () => {
          buttons.remove();
          title.textContent = "Stopped before going online";
          reject(new DOMException("Stopped", "AbortError"));
        }, { once: true });
      });
    },
    status(text) { title.textContent = text; },
    showResult(result, ok) {
      const found = document.createElement("details");
      found.className = "web-found";
      found.innerHTML = `<summary></summary><div class="web-text"></div>`;
      found.querySelector("summary").textContent = ok ? `What it found (${result.length.toLocaleString()} characters)` : "It did not work";
      found.querySelector(".web-text").textContent = result;
      card.appendChild(found);
    },
  };
}

// Ask about one web call, run it if allowed, and return the text the model gets back.
async function runWebCall(call, signal) {
  const { args, text } = describeWebCall(call);
  const card = webStepCard(text);
  if (!(await card.decide(signal))) {
    card.status("You said no");
    return { text, result: WEB_DENIED };
  }
  card.status("Going online…");
  const response = await fetch("/api/chat/web", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name: call.name, arguments: args }),
    signal,
  });
  if (!response.ok) throw await errorFromResponse(response);
  const { result, ok } = await response.json();
  card.status(ok ? "Used the internet" : "Could not reach it");
  card.showResult(result, ok);
  return { text, result };
}

// Stream one model reply into its bubble. Returns the tool calls the model asked for
// (empty for a plain answer). Tool calls arrive in pieces, keyed by their index.
async function streamReply(view, body, signal) {
  const addContent = inlineThinkSplitter(view);
  const response = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  if (!response.ok) throw await errorFromResponse(response);
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  const calls = [];
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop();
    for (const line of lines) {
      const event = parseStreamLine(line);
      if (!event) continue;
      if (event.error) throw new Error(event.error.message || "The engine reported an error.");
      const delta = event.choices?.[0]?.delta ?? {};
      if (delta.reasoning_content) view.addThought(delta.reasoning_content);
      if (delta.content) addContent(delta.content);
      for (const part of delta.tool_calls ?? []) {
        const call = (calls[part.index ?? 0] ??= { id: "", name: "", arguments: "" });
        if (part.id) call.id = part.id;
        if (part.function?.name) call.name += part.function.name;
        if (part.function?.arguments) call.arguments += part.function.arguments;
      }
    }
  }
  return calls.filter((call) => call && call.name)
    .map((call, index) => ({ ...call, id: call.id || `call_${index}` }));
}

async function sendChat(text) {
  // Everything this message adds to the history, so a failed send can be undone as one.
  const historyStart = chatHistory.length;
  const turnsStart = chatTurns.length;
  chatHistory.push({ role: "user", content: text });
  chatTurns.push({ role: "user", content: text });
  addMessage("user", text);
  const useWeb = $("#chat-web").checked;
  chatAbort = new AbortController();
  setChatBusy(true);
  let view = null;
  try {
    for (let step = 0; ; step++) {
      view = replyView(addMessage("assistant", ""));
      const body = { messages: chatHistory };
      if (!$("#chat-think").checked) {
        // Understood by Qwen3 and other templates with a thinking switch; others ignore it.
        body.chat_template_kwargs = { enable_thinking: false };
      }
      // After MAX_WEB_STEPS the tools are taken away, so the model has to answer.
      const offerWeb = useWeb && step < MAX_WEB_STEPS;
      if (offerWeb) body.web = true;
      const calls = await streamReply(view, body, chatAbort.signal);
      if (!offerWeb || !calls.length) break;
      view.handOff();
      const said = view.answer();
      view = null;  // a Stop from here on has no reply bubble to finish
      chatHistory.push({
        role: "assistant",
        content: said,
        tool_calls: calls.map((c) => ({ id: c.id, type: "function", function: { name: c.name, arguments: c.arguments } })),
      });
      for (const call of calls) {
        const { text: asked, result } = await runWebCall(call, chatAbort.signal);
        chatHistory.push({ role: "tool", tool_call_id: call.id, content: result });
        chatTurns.push({ role: "tool", content: `${asked}\n\n${result}` });
      }
    }
    view.end(view.answer() ? "" : "(The model sent an empty answer.)");
    chatHistory.push({ role: "assistant", content: view.answer() });
    chatTurns.push({ role: "assistant", content: view.answer(), thinking: view.thought() });
  } catch (error) {
    const stopped = error.name === "AbortError";
    // Between replies (while asking or going online) there is no bubble yet, so start one.
    if (!view && !stopped) view = replyView(addMessage("assistant", ""));
    if (!stopped) view.fail(error.message);
    else view?.end(view.answer() ? "\n\n(Stopped.)" : "(Stopped.)");
    if (stopped && view && view.answer() !== "(Stopped.)") {
      chatHistory.push({ role: "assistant", content: view.answer() });
      chatTurns.push({ role: "assistant", content: view.answer(), thinking: view.thought() });
    } else {
      chatHistory.length = historyStart;
      chatTurns.length = turnsStart;
    }
  } finally {
    chatAbort = null;
    setChatBusy(false);
  }
}

$("#chat-form").addEventListener("submit", (event) => {
  event.preventDefault();
  if (chatAbort) return;
  const text = $("#chat-input").value.trim();
  if (!text) return;
  $("#chat-input").value = "";
  sendChat(text);
});
$("#chat-stop").addEventListener("click", () => chatAbort?.abort());
$("#chat-input").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); $("#chat-form").requestSubmit(); }
});
$("#chat-clear").addEventListener("click", () => {
  chatHistory.length = 0;
  chatTurns.length = 0;
  $("#chat-log").innerHTML = "";
});

// The conversation as plain text (built by the server, so the format is tested there).
async function chatTranscript() {
  if (!chatTurns.length) {
    toast("Nothing to save yet. Send a message first.", "info");
    return null;
  }
  const response = await fetch("/api/chat/transcript", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ messages: chatTurns }),
  });
  if (!response.ok) throw await errorFromResponse(response);
  return response.json();
}

// Save text as a .txt file through the browser's own download, so nothing leaves the computer.
function downloadText(text, filename) {
  const url = URL.createObjectURL(new Blob([text], { type: "text/plain;charset=utf-8" }));
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

$("#chat-copy-all").addEventListener("click", (event) => whileBusy(event.currentTarget, async () => {
  try {
    const transcript = await chatTranscript();
    if (!transcript) return;
    if (await copyText(transcript.text)) toast("Chat copied.", "info");
    else toast("Could not copy the chat. Use Export .txt instead.");
  } catch (error) { toast(error.message); }
}));
$("#chat-export").addEventListener("click", (event) => whileBusy(event.currentTarget, async () => {
  try {
    const transcript = await chatTranscript();
    if (!transcript) return;
    downloadText(transcript.text, transcript.filename);
    toast(`Saved ${transcript.filename} to your Downloads folder.`, "info");
  } catch (error) { toast(error.message); }
}));

// ---------- system ----------

function statCard(label, value, detail, usedShare = null) {
  const meter = usedShare === null ? ""
    : `<div class="meter"><div class="gpu" style="width:${Math.round(usedShare * 100)}%"></div></div>`;
  return `<div class="stat"><div class="label">${escapeHtml(label)}</div>
    <div class="value">${escapeHtml(value)}</div>${meter}<div class="detail">${escapeHtml(detail)}</div></div>`;
}

const usedShare = (free, total) => (total > 0 ? Math.min(1, Math.max(0, 1 - free / total)) : null);

async function loadSystem() {
  $("#system-view").innerHTML = loadingState("Checking this computer…");
  const [hw, engine] = await Promise.all([withErrors(() => api("/hardware")), api("/engine").catch(() => null)]);
  if (!hw) { $("#system-view").innerHTML = emptyState("Could not read the hardware."); return; }
  const machine = hw.machine || {};
  const backendNames = { cuda: "CUDA", vulkan: "Vulkan", rocm: "ROCm", cpu: "CPU only" };
  const diskNames = { nvme: "NVMe", sata_ssd: "SATA SSD", hdd: "Hard disk" };
  $("#system-view").innerHTML = [
    hw.gpu_name
      ? statCard("GPU memory", `${formatBytes(hw.gpu_free_bytes)} free`,
        `${hw.gpu_name} · ${formatBytes(hw.gpu_total_bytes)} total`, usedShare(hw.gpu_free_bytes, hw.gpu_total_bytes))
      : statCard("GPU", "Not detected",
        "No NVIDIA or AMD GPU found (nvidia-smi, or the amdgpu driver). Plans use the CPU only."),
    statCard("GPU backend", backendNames[machine.backend] || "Unknown",
      (machine.warnings || []).join(" ")
        || "How llama-server talks to the GPU. Set STRONGHOUSE_GPU_BACKEND to change it."),
    statCard("Vault drive", diskNames[hw.archive_disk] || "Unknown",
      `Models bigger than ${formatBytes(machine.in_memory_limit_bytes || 0)} stream from here at about `
        + `${((machine.memory_speeds?.disk_bytes_per_second || 0) / 1e9).toFixed(2)} GB/s. `
        + "Set STRONGHOUSE_DISK_GBPS if you measured yours."),
    statCard("Model size limits", `${formatBytes(machine.in_memory_limit_bytes || 0)} in memory`,
      `Up to ${formatBytes(machine.gpu_limit_bytes || 0)} runs fully on the GPU. Bigger than `
        + `${formatBytes(machine.in_memory_limit_bytes || 0)} streams from disk: slow, but it runs.`),
    statCard("System memory", `${formatBytes(hw.ram_available_bytes)} free`,
      `${formatBytes(hw.ram_total_bytes)} total`, usedShare(hw.ram_available_bytes, hw.ram_total_bytes)),
    statCard("CPU", `${hw.cpu_threads} threads`, "Used for experts and layers kept in RAM"),
    statCard("Engine", engine?.version ? "llama-server found" : "Not found",
      engine?.version || "Install llama.cpp (see README), or set STRONGHOUSE_LLAMA_SERVER."),
  ].join("");
}

// ---------- start ----------
const startTab = location.hash.slice(1);
if (startTab && startTab !== "library" && document.querySelector(`.tabs button[data-tab="${startTab}"]`)) {
  // guild.js adds its own tab loader, so wait until it has run.
  window.addEventListener("DOMContentLoaded", () => showTab(startTab));
}
loadLibrary();
pollEngine();
pollDownloads();
