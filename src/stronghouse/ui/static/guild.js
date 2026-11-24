// Agents page (code name "guild"): create agents, run tasks ("quests" in the API), approve actions.
// Uses helpers from app.js: $, api, withErrors, toast, escapeHtml.
"use strict";

const QUEST_POLL_MS = 1200;
const QUEST_RETRY_MS = 3000;
const ACTIVE_QUEST_STATES = ["running", "waiting_approval"];
const SKILL_CATEGORY_NAMES = {
  files: "Files", web: "Web", documents: "Documents", desktop: "Desktop", utility: "Utility",
};
const AVATAR_COLOURS = 6;
const QUEST_STATE_NAMES = {
  running: "running", waiting_approval: "waiting for you", done: "done", failed: "failed", cancelled: "cancelled",
};
const guild = { catalog: null, agents: [], editing: null, questId: null };

async function loadGuild() {
  if (!guild.catalog) {
    guild.catalog = await withErrors(() => api("/agents/catalog"));
    if (!guild.catalog) return;
    renderCatalogOptions();
  }
  await loadAgents();
  loadQuests();
}

TAB_LOADERS.guild = loadGuild;

// Initials in a coloured circle; the colour is stable for a given name.
function avatar(name, size = "") {
  const words = String(name).trim().split(/\s+/).filter(Boolean);
  const initials = (words.length > 1 ? words[0][0] + words[1][0] : (words[0] ?? "?")[0]).toUpperCase();
  const hash = [...String(name)].reduce((sum, ch) => (sum * 31 + ch.codePointAt(0)) >>> 0, 7);
  return `<span class="avatar c${(hash % AVATAR_COLOURS) + 1} ${size}" aria-hidden="true">${escapeHtml(initials)}</span>`;
}

const questStateName = (state) => QUEST_STATE_NAMES[state] ?? String(state).replaceAll("_", " ");

// ---------- agents ----------

async function loadAgents() {
  guild.agents = (await withErrors(() => api("/agents"))) || [];
  $("#agent-list").innerHTML = guild.agents.map(agentCard).join("")
    || emptyState("No agents yet. Press New agent and give it only the skills its job needs.");
  const options = guild.agents.map((a) => `<option value="${a.id}">${escapeHtml(a.name)}</option>`).join("");
  const [worker, reviewer] = [$("#quest-worker").value, $("#quest-reviewer").value];
  $("#quest-worker").innerHTML = options;
  $("#quest-reviewer").innerHTML = `<option value="">No reviewer</option>${options}`;
  if (guild.agents.some((a) => String(a.id) === worker)) $("#quest-worker").value = worker;
  $("#quest-reviewer").value = guild.agents.some((a) => String(a.id) === reviewer) ? reviewer : "";
  updateQuestHint();
}

// Tasks that clearly need a kind of skill, so we can warn before a quest is wasted.
const TASK_NEEDS = [
  { pattern: /https?:\/\/|\brss\b|\bfeed\b|\bwebsite\b|\bweb\b|\bonline\b|\binternet\b|\bheadlines?\b|\bnews\b|\bsearch the\b/i,
    skills: ["fetch_web_page", "read_news_feed", "web_search"], label: "web" },
  { pattern: /\.pdf\b|\bpdf\b/i, skills: ["read_pdf"], label: "Read PDF" },
];

function updateQuestHint() {
  const hint = $("#quest-skill-hint");
  const agent = guild.agents.find((a) => String(a.id) === $("#quest-worker").value);
  if (!agent) { hint.textContent = ""; return; }
  const skills = agent.skills.map(skillTitle).join(", ") || "no skills";
  const task = $("#quest-task").value;
  const missing = TASK_NEEDS.find((need) => need.pattern.test(task) && !need.skills.some((s) => agent.skills.includes(s)));
  if (!missing) {
    hint.innerHTML = `${escapeHtml(agent.name)} can: ${escapeHtml(skills)}`;
    return;
  }
  const able = guild.agents.filter((a) => missing.skills.some((s) => a.skills.includes(s))).map((a) => a.name);
  const suggestion = able.length ? `Pick ${escapeHtml(able.join(" or "))}, or press` : "Press";
  hint.innerHTML = `<div class="notice warn">${escapeHtml(agent.name)} has no ${escapeHtml(missing.label)} skill for this task
    (can: ${escapeHtml(skills)}). ${suggestion} Edit to add it.</div>`;
}

$("#quest-worker").addEventListener("change", updateQuestHint);
$("#quest-task").addEventListener("input", updateQuestHint);

function agentCard(agent) {
  const done = agent.tasks_done ?? 0;
  const skills = agent.skills.length;
  return `
    <div class="card">
      <div class="agent-head">
        ${avatar(agent.name)}
        <div><h3>${escapeHtml(agent.name)}</h3>
          <div class="meta">${done} task${done === 1 ? "" : "s"} done · ${skills} skill${skills === 1 ? "" : "s"}</div></div>
      </div>
      <div class="badges">${agent.skills.map((s) => `<span class="badge">${escapeHtml(skillTitle(s))}</span>`).join("")}</div>
      <div class="row">
        <button class="secondary small" data-edit="${agent.id}">Edit</button>
        <button class="secondary small" data-journal="${agent.id}">Notes</button>
        <button class="secondary small" data-export="${agent.id}">Export</button>
        <button class="danger small" data-delete="${agent.id}">Delete</button>
      </div>
    </div>`;
}

function skillTitle(name) {
  return guild.catalog?.skills.find((s) => s.name === name)?.title ?? name;
}

function renderCatalogOptions() {
  const { skills } = guild.catalog;
  const categories = [...new Set(skills.map((s) => s.category))];
  $("#agent-skills").innerHTML = categories.map((category) => `
    <div class="skill-category">${escapeHtml(SKILL_CATEGORY_NAMES[category] ?? category)}</div>
    ${skills.filter((s) => s.category === category).map((s) => `
    <label><input type="checkbox" value="${escapeHtml(s.name)}">
      <span>${escapeHtml(s.title)}${s.needs_approval ? '<span class="tag ask">asks first</span>' : ""}<small>${escapeHtml(s.description)}</small></span>
    </label>`).join("")}`).join("");
}

function openSheet(agent) {
  guild.editing = agent ?? null;
  $("#agent-form-title").textContent = agent ? `Edit ${agent.name}` : "New agent";
  $("#agent-name").value = agent?.name ?? "";
  $("#agent-personality").value = agent?.personality ?? "";
  $("#agent-creativity").value = agent?.creativity ?? 3;
  $("#agent-diligence").value = agent?.diligence ?? 5;
  document.querySelectorAll("#agent-skills input").forEach((box) => {
    box.checked = agent ? agent.skills.includes(box.value) : false;
  });
  updateSheetHints();
  $("#agent-form").hidden = false;
  $("#agent-name").focus();
}

function updateSheetHints() {
  const advised = guild.catalog.recommended_max_skills;
  const used = document.querySelectorAll("#agent-skills input:checked").length;
  $("#skill-slots").textContent = `${used} selected`;
  $("#skill-advice").hidden = used <= advised;
  $("#skill-advice").textContent =
    `Small models can get confused with more than ${advised} skills. If this agent picks the wrong tool, give it fewer.`;
  const { temperature_by_creativity: temps, max_steps_by_diligence: steps } = guild.catalog;
  $("#creativity-value").textContent = temps?.[$("#agent-creativity").value] ?? $("#agent-creativity").value;
  $("#diligence-value").textContent = steps?.[$("#agent-diligence").value] ?? $("#agent-diligence").value;
}

["#agent-skills", "#agent-creativity", "#agent-diligence"].forEach((sel) =>
  $(sel).addEventListener("input", updateSheetHints));

$("#agent-new").addEventListener("click", () => openSheet(null));
$("#agent-cancel").addEventListener("click", () => { $("#agent-form").hidden = true; });

$("#agent-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const editing = guild.editing;
  const body = {
    name: $("#agent-name").value,
    portrait: editing?.portrait ?? "",  // kept for exported cards; no longer shown
    personality: $("#agent-personality").value,
    skills: [...document.querySelectorAll("#agent-skills input:checked")].map((b) => b.value),
    creativity: Number($("#agent-creativity").value),
    diligence: Number($("#agent-diligence").value),
  };
  const path = editing ? `/agents/${editing.id}` : "/agents";
  const saved = await withErrors(() => api(path, { method: editing ? "PUT" : "POST", body }));
  if (!saved) return;
  toast(`${saved.name} saved.`, "info");
  $("#agent-form").hidden = true;
  loadAgents();
});

$("#agent-list").addEventListener("click", async (event) => {
  const button = event.target.closest("button");
  if (!button) return;
  const { edit, journal, delete: dismiss } = button.dataset;
  const exportId = button.dataset.export;
  if (edit) openSheet(guild.agents.find((a) => a.id === Number(edit)));
  if (journal) openJournal(Number(journal));
  if (exportId) {
    const card = await withErrors(() => api(`/agents/${exportId}/card`));
    if (card) downloadJson(card, `${card.name.replace(/\W+/g, "_")}.agent.json`);
  }
  if (dismiss) {
    const agent = guild.agents.find((a) => a.id === Number(dismiss));
    const sure = await ask({
      title: `Delete ${agent.name}?`, message: "Their saved notes are deleted too. Export first if you may want them back.",
      confirmText: "Delete agent", danger: true,
    });
    if (!sure) return;
    await withErrors(() => api(`/agents/${dismiss}`, { method: "DELETE" }));
    loadAgents();
  }
});

$("#agent-import").addEventListener("change", async (event) => {
  const file = event.target.files[0];
  event.target.value = "";
  if (!file) return;
  let card;
  try { card = JSON.parse(await file.text()); } catch { toast("That file is not valid JSON."); return; }
  const agent = await withErrors(() => api("/agents/import", { method: "POST", body: card }));
  if (agent) { toast(`${agent.name} imported.`, "info"); loadAgents(); }
});

function downloadJson(data, filename) {
  const url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], { type: "application/json" }));
  const link = Object.assign(document.createElement("a"), { href: url, download: filename });
  link.click();
  URL.revokeObjectURL(url);
}

// ---------- journal ----------

async function openJournal(agentId) {
  const agent = guild.agents.find((a) => a.id === agentId);
  $("#journal-title").textContent = `${agent.name}'s notes`;
  await renderJournal(agentId);
  $("#journal-dialog").showModal();
}

async function renderJournal(agentId) {
  const notes = (await withErrors(() => api(`/agents/${agentId}/journal`))) || [];
  $("#journal-list").innerHTML = notes.map((n) => `
    <div class="item"><span>${escapeHtml(n.note)}</span>
      <button class="danger small" data-agent="${agentId}" data-note="${n.id}">Forget</button></div>`).join("")
    || emptyState("Nothing written yet. Agents with the Take note skill fill this in.");
}

$("#journal-list").addEventListener("click", async (event) => {
  const { agent, note } = event.target.closest("button")?.dataset ?? {};
  if (!note) return;
  await withErrors(() => api(`/agents/${agent}/journal/${note}`, { method: "DELETE" }));
  renderJournal(Number(agent));
});
$("#journal-close").addEventListener("click", () => $("#journal-dialog").close());

// ---------- quests ----------

$("#quest-start").addEventListener("click", (event) => whileBusy(event.currentTarget, async () => {
  const task = $("#quest-task").value.trim();
  const worker = Number($("#quest-worker").value);
  const reviewer = Number($("#quest-reviewer").value);
  if (!task || !worker) { toast("Pick an agent and describe the task."); return; }
  if ($("#quest-skill-hint .warn")) {
    const sure = await ask({
      title: "Start anyway?", message: "This agent seems to lack a skill this task needs.", confirmText: "Start task",
    });
    if (!sure) return;
  }
  const agentIds = reviewer && reviewer !== worker ? [worker, reviewer] : [worker];
  const quest = await withErrors(() => api("/quests", { method: "POST", body: { task, agent_ids: agentIds } }));
  if (!quest) return;
  $("#quest-task").value = "";
  updateQuestHint();
  showQuest(quest.id);
}));

async function loadQuests() {
  const quests = (await withErrors(() => api("/quests"))) || [];
  $("#quest-list").innerHTML = quests.map((q) => `
    <div class="item selectable" data-quest="${q.id}">
      <span class="name">${escapeHtml(q.task.slice(0, 80))}</span>
      <span class="side">${escapeHtml(questStateName(q.state))}</span></div>`).join("")
    || emptyState("No tasks yet.");
}

$("#quest-list").addEventListener("click", (event) => {
  const row = event.target.closest("[data-quest]");
  if (row) showQuest(Number(row.dataset.quest));
});

function showQuest(questId) {
  guild.questId = questId;
  $("#quest-view").hidden = false;
  pollQuest();
}

async function pollQuest() {
  clearTimeout(pollQuest.timer);
  if (!guild.questId) return;
  const questId = guild.questId;
  const quest = await api(`/quests/${questId}`).catch(() => null);
  if (questId !== guild.questId) return;  // another quest was opened meanwhile
  if (!quest) {
    // Keep trying: giving up here could leave an approval request unseen.
    pollQuest.timer = setTimeout(pollQuest, QUEST_RETRY_MS);
    return;
  }
  renderQuest(quest);
  if (ACTIVE_QUEST_STATES.includes(quest.state)) {
    pollQuest.timer = setTimeout(pollQuest, QUEST_POLL_MS);
  } else {
    loadQuests();
    loadAgents();
  }
}

function renderQuest(quest) {
  $("#quest-title").textContent = `Task #${quest.id} · ${questStateName(quest.state)}`;
  $("#quest-cancel").hidden = !ACTIVE_QUEST_STATES.includes(quest.state);
  const log = $("#quest-log");
  const atBottom = log.scrollTop + log.clientHeight >= log.scrollHeight - 20;
  log.innerHTML = quest.steps.map((s) => `
    <div class="step ${escapeHtml(s.kind)}"><span class="who">${escapeHtml(s.agent_name)}</span>${escapeHtml(s.text)}</div>`).join("")
    + (quest.result && quest.state === "done" ? `<div class="step review"><span class="who">Result</span>${escapeHtml(quest.result)}</div>` : "")
    + (quest.state === "running"
      ? '<div class="thinking"><span class="spinner" aria-hidden="true"></span>Working on it…</div>' : "");
  if (atBottom) log.scrollTop = log.scrollHeight;

  const box = $("#quest-approval");
  const pending = quest.pending;
  box.hidden = !pending;
  if (pending && box.dataset.shown !== JSON.stringify(pending)) {
    box.dataset.shown = JSON.stringify(pending);
    box.innerHTML = `
      <div class="title">${escapeHtml(pending.agent_name)} wants to ${escapeHtml(skillTitle(pending.skill).toLowerCase())}</div>
      <pre>${escapeHtml(describeAction(pending))}</pre>
      <div class="row"><button data-decide="yes">Allow once</button>
        <button class="danger" data-decide="no">Deny</button></div>`;
  }
  if (!pending) box.dataset.shown = "";
}

function describeAction({ skill, arguments: args }) {
  if (skill === "run_command") return `$ ${args.command ?? ""}`;
  if (skill === "write_file") return `File: ${args.path ?? ""}\n\n${args.content ?? ""}`;
  return Object.entries(args).map(([k, v]) => `${k}: ${v}`).join("\n");
}

$("#quest-approval").addEventListener("click", async (event) => {
  const choice = event.target.closest("[data-decide]")?.dataset.decide;
  if (!choice) return;
  const buttons = $("#quest-approval").querySelectorAll("button");
  if ([...buttons].some((b) => b.disabled)) return;  // one decision per request
  buttons.forEach((b) => { b.disabled = true; });
  const sent = await withErrors(() => api(`/quests/${guild.questId}/decision`, {
    method: "POST", body: { approve: choice === "yes" },
  }));
  // On failure let the person try again; on success the next render replaces the box.
  if (!sent) buttons.forEach((b) => { b.disabled = false; });
  pollQuest();
});

$("#quest-cancel").addEventListener("click", (event) => whileBusy(event.currentTarget, async () => {
  await withErrors(() => api(`/quests/${guild.questId}/cancel`, { method: "POST" }));
  pollQuest();
}));
