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

function planBody() {
  const optionalInt = (sel) => ($(sel).value === "" ? null : Number($(sel).value));
  return {
    context: Number($("#run-context").value),
    kv_cache_type: $("#run-kv").value,
    n_cpu_moe: optionalInt("#run-ncpumoe"),
    n_gpu_layers: optionalInt("#run-ngl"),
  };
}

const STRATEGY_NAMES = { full_gpu: "Full GPU", moe_split: "MoE split", partial_layers: "Partial layers", cpu_only: "CPU only" };

function renderPlan({ plan, command }) {
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
  $("#run-log").textContent = logTail.join("\n");
  const outOfMemory = status.state === "crashed" && logTail.some((l) => /failed to allocate/i.test(l));
  if (outOfMemory && !pollEngine.warned) {
    pollEngine.warned = true;
    toast("Ran out of GPU memory. Try a smaller context, a q8_0 KV cache, or more experts on CPU.");
  }
  if (status.state !== "crashed") pollEngine.warned = false;
}

// ---------- chat ----------

const chatHistory = [];

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
      thoughtText.textContent = thought;
      thoughtText.scrollTop = thoughtText.scrollHeight;
      tick();
    },
    addAnswer(text) { finishThinking(); answer.textContent += text; },
    answer: () => answer.textContent,
    end(note) { finishThinking(); clearInterval(timer); if (note) answer.textContent += note; },
    fail(message) {
      finishThinking();
      clearInterval(timer);
      answer.classList.add("error");
      answer.textContent += `${answer.textContent ? "\n\n" : ""}${message}`;
    },
  };
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

async function sendChat(text) {
  chatHistory.push({ role: "user", content: text });
  addMessage("user", text);
  const view = replyView(addMessage("assistant", ""));
  const addContent = inlineThinkSplitter(view);
  chatAbort = new AbortController();
  setChatBusy(true);
  const body = { messages: chatHistory };
  if (!$("#chat-think").checked) {
    // Understood by Qwen3 and other templates with a thinking switch; others ignore it.
    body.chat_template_kwargs = { enable_thinking: false };
  }
  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: chatAbort.signal,
    });
    if (!response.ok) throw await errorFromResponse(response);
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
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
      }
    }
    view.end(view.answer() ? "" : "(The model sent an empty answer.)");
    chatHistory.push({ role: "assistant", content: view.answer() });
  } catch (error) {
    const stopped = error.name === "AbortError";
    if (!stopped) view.fail(error.message);
    else view.end(view.answer() ? "\n\n(Stopped.)" : "(Stopped.)");
    if (stopped && view.answer() !== "(Stopped.)") chatHistory.push({ role: "assistant", content: view.answer() });
    else chatHistory.pop();
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
$("#chat-clear").addEventListener("click", () => { chatHistory.length = 0; $("#chat-log").innerHTML = ""; });

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
  $("#system-view").innerHTML = [
    hw.gpu_name
      ? statCard("GPU memory", `${formatBytes(hw.gpu_free_bytes)} free`,
        `${hw.gpu_name} · ${formatBytes(hw.gpu_total_bytes)} total`, usedShare(hw.gpu_free_bytes, hw.gpu_total_bytes))
      : statCard("GPU", "Not detected", "No NVIDIA GPU found (is nvidia-smi working?). Plans use the CPU only."),
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
