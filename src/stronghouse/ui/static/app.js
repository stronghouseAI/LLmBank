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
  if (!bytes) return "0 B";
  const units = ["B", "KB", "MB", "GB", "TB"];
  const i = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  return `${(bytes / 1024 ** i).toFixed(i >= 3 ? 1 : 0)} ${units[i]}`;
}

function toast(message) {
  const el = $("#toast");
  el.textContent = message;
  el.hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => { el.hidden = true; }, 5000);
}

async function api(path, options = {}) {
  const response = await fetch(`/api${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
    body: options.body ? JSON.stringify(options.body) : undefined,
  });
  if (!response.ok) {
    const detail = await response.json().catch(() => ({}));
    throw new Error(detail.detail || `${response.status} ${response.statusText}`);
  }
  return response.json();
}

async function withErrors(action) {
  try { return await action(); } catch (error) { toast(error.message); return undefined; }
}

// ---------- tabs ----------

document.querySelectorAll(".tabs button").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelectorAll(".tabs button").forEach((b) => b.classList.toggle("active", b === button));
    document.querySelectorAll(".panel").forEach((p) => p.classList.toggle("active", p.id === `tab-${button.dataset.tab}`));
    if (button.dataset.tab === "library") loadLibrary();
    if (button.dataset.tab === "run") loadRunModels();
    if (button.dataset.tab === "system") loadSystem();
  });
});

// ---------- vault (library) ----------

async function loadLibrary() {
  const search = encodeURIComponent($("#library-search").value);
  const entries = await withErrors(() => api(`/library?search=${search}`));
  if (!entries) return;
  const list = $("#library-list");
  if (entries.length === 0) {
    list.innerHTML = `<p class="muted">Your vault is empty. Find a model in Discover, or put .gguf files in
      &lt;archive&gt;/&lt;owner&gt;/&lt;repo&gt;/ and press Rescan.</p>`;
    return;
  }
  list.innerHTML = entries.map((entry) => `
    <div class="card">
      <h4>${escapeHtml(entry.filename)}</h4>
      <div class="meta">${escapeHtml(entry.repo_id)} · ${formatBytes(entry.size_bytes)}</div>
      <div>
        ${entry.is_moe ? '<span class="badge moe">MoE</span>' : ""}
        ${entry.architecture ? `<span class="badge">${escapeHtml(entry.architecture)}</span>` : ""}
        ${entry.license ? `<span class="badge">license: ${escapeHtml(entry.license)}</span>` : ""}
        ${entry.sha256 ? '<span class="badge">verified</span>' : ""}
        ${entry.tags.map((t) => `<span class="badge">#${escapeHtml(t)}</span>`).join("")}
      </div>
      <div class="row">
        <button class="secondary" data-run="${entry.id}">Run</button>
        <button class="secondary" data-tags="${entry.id}" data-current="${escapeHtml(entry.tags.join(", "))}">Tags</button>
        <button class="danger" data-forget="${entry.id}">Forget</button>
      </div>
    </div>`).join("");
}

$("#library-list").addEventListener("click", async (event) => {
  const target = event.target;
  if (target.dataset.run) {
    await loadRunModels(target.dataset.run);
    document.querySelector('.tabs button[data-tab="run"]').click();
  } else if (target.dataset.tags) {
    const answer = prompt("Tags, separated by commas", target.dataset.current);
    if (answer === null) return;
    await withErrors(() => api(`/library/${target.dataset.tags}/tags`, {
      method: "PUT", body: { tags: answer.split(",") },
    }));
    loadLibrary();
  } else if (target.dataset.forget) {
    await withErrors(() => api(`/library/${target.dataset.forget}`, { method: "DELETE" }));
    toast("Removed from the list. The file is still on disk.");
    loadLibrary();
  }
});

$("#library-search").addEventListener("input", () => {
  clearTimeout(loadLibrary.timer);
  loadLibrary.timer = setTimeout(loadLibrary, 250);
});
$("#library-rescan").addEventListener("click", async () => {
  const result = await withErrors(() => api("/library/rescan", { method: "POST" }));
  if (result) toast(`Indexed ${result.indexed} model(s).`);
  loadLibrary();
});

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
  $("#hub-results").innerHTML = '<p class="muted">Searching…</p>';
  const models = await withErrors(() => api(`/hub/search?${hubSearchParams()}`));
  if (!models) { $("#hub-results").innerHTML = ""; return; }
  $("#hub-results").innerHTML = models.map((model) => `
    <div class="item selectable" data-repo="${escapeHtml(model.repo_id)}">
      <span><span class="name">${escapeHtml(model.repo_id)}</span><br>
        ${model.size_label ? `<span class="badge">${escapeHtml(model.size_label)}</span>` : ""}
        ${model.is_moe ? '<span class="badge moe">MoE</span>' : ""}
        ${KIND_NAMES[model.kind] ? `<span class="badge">${KIND_NAMES[model.kind]}</span>` : ""}</span>
      <span class="muted">${model.downloads.toLocaleString()} ↓</span>
    </div>`).join("") || '<p class="muted">No GGUF models match. Try fewer filters.</p>';
}

function renderHubFiles() {
  const { repo, files } = hubFiles;
  if (!repo) return;
  const hideTooBig = $("#hub-fits").checked;
  const visible = files.filter((f) => !(hideTooBig && f.fit === "too_big"));
  const hidden = files.length - visible.length;
  $("#hub-files").innerHTML = visible.map((file) => `
    <div class="item">
      <span><span class="name">${escapeHtml(file.filename)}</span><br>
        <span class="badge fit-${file.fit}" title="${FIT_NAMES[file.fit][1]}">${FIT_NAMES[file.fit][0]}</span></span>
      <span>${formatBytes(file.size_bytes)}
        <button data-download="${escapeHtml(file.filename)}" data-repo="${escapeHtml(repo)}">Archive</button></span>
    </div>`).join("")
    + (hidden ? `<p class="muted">${hidden} file(s) too big for this computer are hidden.</p>` : "")
    || '<p class="muted">No .gguf files in this repo.</p>';
}

$("#hub-form").addEventListener("submit", (event) => { event.preventDefault(); searchHub(); });
["#hub-size", "#hub-kind", "#hub-sort", "#hub-moe"].forEach((sel) =>
  $(sel).addEventListener("change", () => { if ($("#hub-query").value || $("#hub-results").children.length) searchHub(); }));
$("#hub-fits").addEventListener("change", renderHubFiles);

$("#hub-results").addEventListener("click", async (event) => {
  const row = event.target.closest("[data-repo]");
  if (!row) return;
  const repo = row.dataset.repo;
  $("#hub-files").innerHTML = '<p class="muted">Loading files…</p>';
  const files = await withErrors(() => api(`/hub/files?repo_id=${encodeURIComponent(repo)}`));
  if (!files) { $("#hub-files").innerHTML = ""; return; }
  hubFiles = { repo, files };
  renderHubFiles();
});

$("#hub-files").addEventListener("click", async (event) => {
  const button = event.target.closest("[data-download]");
  if (!button) return;
  button.disabled = true;
  button.textContent = "Archiving…";
  await withErrors(() => api("/downloads", {
    method: "POST", body: { repo_id: button.dataset.repo, filename: button.dataset.download },
  }));
  toast("Download started.");
  pollDownloads();
});

async function pollDownloads() {
  const jobs = await withErrors(() => api("/downloads"));
  if (!jobs) return;
  $("#download-list").innerHTML = jobs.map((job) => {
    const percent = job.bytes_total ? Math.floor((100 * job.bytes_done) / job.bytes_total) : 0;
    return `<div class="item"><div style="flex:1">
      <div class="name">${escapeHtml(job.repo_id)} / ${escapeHtml(job.filename)}</div>
      <div class="muted">${escapeHtml(job.state)} · ${formatBytes(job.bytes_done)} of ${formatBytes(job.bytes_total)}
        ${job.error ? ` · <span style="color:var(--danger)">${escapeHtml(job.error)}</span>` : ""}</div>
      <div class="progress"><div style="width:${percent}%"></div></div></div></div>`;
  }).join("") || '<p class="muted">No downloads yet.</p>';
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
  select.innerHTML = entries.map((e) =>
    `<option value="${e.id}">${escapeHtml(e.filename)}${e.is_moe ? " (MoE)" : ""}</option>`).join("");
  if (keep) select.value = keep;
  showProfile();
}

async function showProfile() {
  const id = $("#run-model").value;
  if (!id) { $("#run-profile").textContent = "No models in your vault yet."; return; }
  const p = await withErrors(() => api(`/library/${id}/profile`));
  if (!p) return;
  $("#run-profile").innerHTML = `${escapeHtml(p.architecture)} · ${p.block_count} blocks ·
    trained context ${p.context_length.toLocaleString()}<br>
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

function renderPlan({ plan, command }) {
  const total = plan.estimated_gpu_bytes + plan.estimated_cpu_bytes || 1;
  const gpuShare = (100 * plan.estimated_gpu_bytes) / total;
  $("#run-plan-view").innerHTML = `
    <div><strong>${escapeHtml(plan.strategy.replace("_", " "))}</strong> ·
      context ${plan.context_size.toLocaleString()} · GPU layers ${plan.n_gpu_layers} ·
      experts on CPU ${plan.n_cpu_moe}</div>
    <div class="meter"><div class="gpu" style="width:${gpuShare}%"></div><div class="cpu" style="flex:1"></div></div>
    <div class="muted">GPU ≈ ${formatBytes(plan.estimated_gpu_bytes)} · RAM ≈ ${formatBytes(plan.estimated_cpu_bytes)} (estimates)</div>
    <ul>${plan.notes.map((n) => `<li>${escapeHtml(n)}</li>`).join("")}</ul>
    <details><summary>Command</summary><code>${escapeHtml(command.join(" "))}</code></details>`;
}

$("#run-plan").addEventListener("click", async () => {
  const id = $("#run-model").value;
  if (!id) return;
  const result = await withErrors(() => api(`/library/${id}/plan`, { method: "POST", body: planBody() }));
  if (result) renderPlan(result);
});

$("#run-start").addEventListener("click", async () => {
  const id = $("#run-model").value;
  if (!id) return;
  const status = await withErrors(() => api("/engine/start", {
    method: "POST", body: { model_id: Number(id), ...planBody() },
  }));
  if (status) { toast("Loading model… this can take a minute for big models."); pollEngine(); }
});

$("#run-stop").addEventListener("click", async () => {
  await withErrors(() => api("/engine/stop", { method: "POST" }));
  pollEngine();
});

$("#run-bench").addEventListener("click", async () => {
  $("#run-bench-view").textContent = "Running…";
  const result = await withErrors(() => api("/engine/benchmark", { method: "POST" }));
  $("#run-bench-view").innerHTML = result
    ? `<strong>${result.generated_tokens_per_second} tokens/s</strong> generating ·
       ${result.prompt_tokens_per_second} tokens/s reading the prompt`
    : "";
});

async function pollEngine() {
  const status = await api("/engine").catch(() => null);
  clearTimeout(pollEngine.timer);
  pollEngine.timer = setTimeout(pollEngine, ENGINE_POLL_MS);
  if (!status) return;
  const pill = $("#engine-pill");
  pill.className = `pill ${status.state}`;
  const model = status.model_path ? status.model_path.split("/").pop() : "";
  pill.textContent = `engine: ${status.state}${model ? ` · ${model}` : ""}`;
  $("#run-log").textContent = status.log_tail.join("\n");
  const outOfMemory = status.state === "crashed" && status.log_tail.some((l) => /failed to allocate/i.test(l));
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

async function sendChat(text) {
  chatHistory.push({ role: "user", content: text });
  addMessage("user", text);
  const bubble = addMessage("assistant", "…");
  let reply = "";
  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages: chatHistory }),
    });
    if (!response.ok) {
      const detail = await response.json().catch(() => ({}));
      throw new Error(detail.detail || response.statusText);
    }
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
        if (!line.startsWith("data: ") || line.includes("[DONE]")) continue;
        const delta = JSON.parse(line.slice(6)).choices?.[0]?.delta?.content;
        if (delta) { reply += delta; bubble.textContent = reply; }
      }
    }
    chatHistory.push({ role: "assistant", content: reply });
  } catch (error) {
    bubble.textContent = `Error: ${error.message}`;
    chatHistory.pop();
  }
}

$("#chat-form").addEventListener("submit", (event) => {
  event.preventDefault();
  const text = $("#chat-input").value.trim();
  if (!text) return;
  $("#chat-input").value = "";
  sendChat(text);
});
$("#chat-input").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); $("#chat-form").requestSubmit(); }
});
$("#chat-clear").addEventListener("click", () => { chatHistory.length = 0; $("#chat-log").innerHTML = ""; });

// ---------- system ----------

async function loadSystem() {
  const [hw, engine] = await Promise.all([withErrors(() => api("/hardware")), api("/engine").catch(() => null)]);
  if (!hw) return;
  $("#system-view").innerHTML = `
    <h3>GPU</h3><div>${hw.gpu_name ? `${escapeHtml(hw.gpu_name)} · ${formatBytes(hw.gpu_free_bytes)} free of ${formatBytes(hw.gpu_total_bytes)}` : "No NVIDIA GPU detected"}</div>
    <h3>Memory</h3><div>${formatBytes(hw.ram_available_bytes)} free of ${formatBytes(hw.ram_total_bytes)}</div>
    <h3>CPU</h3><div>${hw.cpu_threads} threads</div>
    <h3>Engine</h3><div>${engine && engine.version ? escapeHtml(engine.version) : "llama-server not found. Install llama.cpp (see README)."}</div>`;
}

// ---------- start ----------
loadLibrary();
pollEngine();
pollDownloads();
