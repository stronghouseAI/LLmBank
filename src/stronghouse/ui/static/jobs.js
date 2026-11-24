// Jobs page: queue prompts for a big, slow model and read the answers when they are done.
// Uses helpers from app.js: $, api, withErrors, whileBusy, toast, escapeHtml, emptyState, planBody.
"use strict";

const JOB_POLL_MS = 3000;
const ACTIVE_JOB_STATES = ["queued", "loading", "running"];
const JOB_STATE_NAMES = {
  queued: "waiting", loading: "loading model", running: "writing", done: "done", failed: "failed", cancelled: "cancelled",
};
const JOB_STATE_BADGES = { done: "ok", failed: "fit-too_big", running: "fit-offload", loading: "fit-offload" };
const jobs = { selected: null };

async function loadJobs() {
  const entries = await withErrors(() => api("/library"));
  if (entries) {
    const select = $("#job-model");
    const keep = select.value;
    select.innerHTML = '<option value="">The loaded model</option>'
      + entries.map((e) => `<option value="${e.id}">${escapeHtml(e.filename)}${e.is_moe ? " (MoE)" : ""}</option>`).join("");
    select.value = keep;
  }
  pollJobs();
}
TAB_LOADERS.jobs = loadJobs;

function jobsTabOpen() {
  return $("#tab-jobs").classList.contains("active");
}

async function pollJobs() {
  clearTimeout(pollJobs.timer);
  const list = await api("/jobs").catch(() => null);
  if (list) renderJobList(list);
  if (jobs.selected) await showJob(jobs.selected);
  const busy = (list ?? []).some((j) => ACTIVE_JOB_STATES.includes(j.state));
  if (jobsTabOpen() && busy) pollJobs.timer = setTimeout(pollJobs, JOB_POLL_MS);
}

function renderJobList(list) {
  $("#job-list").innerHTML = list.map((j) => `
    <div class="item selectable" data-job="${j.id}">
      <span class="name">${escapeHtml(j.title)}</span>
      <span class="side"><span class="badge ${JOB_STATE_BADGES[j.state] ?? ""}">${escapeHtml(JOB_STATE_NAMES[j.state] ?? j.state)}</span></span></div>`).join("")
    || emptyState("No jobs yet. Queue a prompt and come back later.");
}

async function showJob(jobId) {
  const job = await withErrors(() => api(`/jobs/${jobId}`));
  if (!job) return;
  jobs.selected = job.id;
  $("#job-view").hidden = false;
  $("#job-title").textContent = job.title;
  const facts = [JOB_STATE_NAMES[job.state] ?? job.state];
  if (job.generated_tokens) facts.push(`${job.generated_tokens} tokens`);
  if (job.tokens_per_second) facts.push(`${job.tokens_per_second} tokens/s`);
  if (job.output_path) facts.push(`saved to ${job.output_path}`);
  $("#job-meta").innerHTML = escapeHtml(facts.join(" · "))
    + (job.error ? `<div class="notice warn">${escapeHtml(job.error)}</div>` : "");
  $("#job-output").textContent = job.output || (job.state === "queued" ? "Waiting for earlier jobs to finish." : "Nothing written yet.");
  $("#job-cancel").hidden = !ACTIVE_JOB_STATES.includes(job.state);
  $("#job-retry").hidden = ACTIVE_JOB_STATES.includes(job.state);
}

$("#job-list").addEventListener("click", (event) => {
  const row = event.target.closest("[data-job]");
  if (row) showJob(Number(row.dataset.job));
});

$("#job-form").addEventListener("submit", (event) => {
  event.preventDefault();
  whileBusy(event.submitter ?? $("#job-form button[type=submit]"), async () => {
    const prompt = $("#job-prompt").value.trim();
    if (!prompt) { toast("Write a prompt for the job first."); return; }
    const modelId = $("#job-model").value;
    const maxTokens = $("#job-max-tokens").value;
    const body = {
      // The chosen model is loaded with the Run page's settings (context, KV cache, flags).
      ...(modelId ? { ...planBody(), model_id: Number(modelId) } : {}),
      prompt,
      think: $("#job-think").checked,
      max_tokens: maxTokens === "" ? null : Number(maxTokens),
    };
    const job = await withErrors(() => api("/jobs", { method: "POST", body }));
    if (!job) return;
    $("#job-prompt").value = "";
    toast(`Job queued. You will get a desktop notification when it is done.`, "info");
    jobs.selected = job.id;
    pollJobs();
  });
});

$("#job-cancel").addEventListener("click", (event) => whileBusy(event.currentTarget, async () => {
  if (!jobs.selected) return;
  await withErrors(() => api(`/jobs/${jobs.selected}/cancel`, { method: "POST" }));
  pollJobs();
}));

$("#job-retry").addEventListener("click", (event) => whileBusy(event.currentTarget, async () => {
  if (!jobs.selected) return;
  const job = await withErrors(() => api(`/jobs/${jobs.selected}/retry`, { method: "POST" }));
  if (job) { jobs.selected = job.id; pollJobs(); }
}));
